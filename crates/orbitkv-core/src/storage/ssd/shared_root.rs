use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Error, ErrorKind, Seek, Write};
use std::num::NonZeroU64;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::time::{Duration, Instant};

use super::config::MAX_SSD_SHARDS;
use super::persistence::{frame, read_frame};
use super::{files, persistence};

pub(super) const PARTITION_PREFIX: &str = "orbitkv-ssd-";
const LIMIT_FILE: &str = ".orbitkv-ssd-capacity";
const JOURNAL_NAME: &str = "shard-000000.dat.index";
const MAX_DECIMAL_U64: usize = 20;

fn partition_namespace(name: &OsStr) -> Option<&str> {
    name.to_str()?
        .strip_prefix(PARTITION_PREFIX)
        .filter(|s| orbitkv_state::is_storage_namespace(s))
}

pub(super) fn reserved_name(name: &OsStr) -> bool {
    name == LIMIT_FILE || partition_namespace(name).is_some()
}

fn shard_index(name: &OsStr) -> io::Result<Option<usize>> {
    // The index and its checkpoint are outside the payload budget.
    if name == JOURNAL_NAME || name == "shard-000000.dat.index.tmp" {
        return Ok(None);
    }
    name.to_str()
        .and_then(|s| s.strip_prefix("shard-"))
        .and_then(|s| s.strip_suffix(".dat"))
        .filter(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|s| s.parse::<usize>().ok())
        .map(Some)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                "unexpected file in shared SSD partition",
            )
        })
}

fn lock_root(root: &Path, try_only: bool) -> io::Result<File> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY)
        .open(root)?;
    if try_only {
        directory.try_lock().map_err(Error::from)?;
    } else {
        directory.lock()?;
    }
    Ok(directory)
}

/// Maintenance holds an exclusive flock on this private namespace directory.
pub(super) fn release_requested(directory: &File) -> io::Result<bool> {
    match directory.try_lock_shared() {
        Ok(()) => directory.unlock().map(|()| false),
        Err(fs::TryLockError::WouldBlock) => Ok(true),
        Err(fs::TryLockError::Error(error)) => Err(error),
    }
}

fn stored_limit(root: &Path) -> io::Result<Option<u64>> {
    let mut file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(root.join(LIMIT_FILE))
    {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "invalid shared SSD root capacity file",
        ));
    }
    let limit = read_frame::<NonZeroU64>(&mut file, MAX_DECIMAL_U64)?.ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            "empty shared SSD root capacity file",
        )
    })?;
    if file.stream_position()? != meta.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "trailing shared SSD root capacity data",
        ));
    }
    Ok(Some(limit.get()))
}

fn stage_limit(directory: &File, limit: u64) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    // An interrupted initialization leaves no temporary name or partially written policy.
    let mut file = File::from(openat(
        directory,
        ".",
        OFlags::TMPFILE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )?);
    file.write_all(&frame(&limit, MAX_DECIMAL_U64)?)?;
    file.sync_all()?;
    Ok(file)
}

fn required_payload(root: &Path, own: &str, shards: usize, shard_bytes: u64) -> io::Result<u64> {
    let overflow = || {
        Error::new(
            ErrorKind::InvalidData,
            "shared SSD root payload size overflow",
        )
    };
    let mut required = shard_bytes
        .checked_mul(shards as u64)
        .ok_or_else(overflow)?;
    for partition in fs::read_dir(root)? {
        let partition = partition?;
        let name = partition.file_name();
        let Some(namespace) = partition_namespace(&name) else {
            continue;
        };
        if !partition.file_type()?.is_dir() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid shared SSD partition directory",
            ));
        }
        for shard in fs::read_dir(partition.path())? {
            let shard = shard?;
            let Some(index) = shard_index(&shard.file_name())? else {
                continue;
            };
            let meta = shard.metadata()?;
            if !meta.is_file() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "invalid shared SSD shard",
                ));
            }
            if meta.nlink() != 1 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "shared SSD shard has another hard link",
                ));
            }
            // The requested partition is already charged above. Keep any excess old length
            // until the journal and the resize have committed, including partially resized shards.
            let extra = if namespace == own && index < shards {
                meta.len().saturating_sub(shard_bytes)
            } else {
                meta.len()
            };
            required = required.checked_add(extra).ok_or_else(overflow)?;
        }
    }
    Ok(required)
}

/// Holds the directory lock through journal recovery and shard sizing.
#[derive(Debug)]
pub(super) struct Admission {
    directory: File,
    pending_limit: Option<File>,
}

impl Admission {
    /// Publish a new limit after validating the partition and before changing any shard length.
    pub(super) fn publish_limit(&mut self) -> io::Result<()> {
        use rustix::fs::{AtFlags, CWD, linkat};

        if let Some(file) = self.pending_limit.take() {
            // Linking through procfs works for unprivileged Managers and cannot replace an existing entry.
            linkat(
                CWD,
                format!("/proc/self/fd/{}", file.as_raw_fd()),
                &self.directory,
                LIMIT_FILE,
                AtFlags::SYMLINK_FOLLOW,
            )?;
            self.directory.sync_all()?;
        }
        Ok(())
    }
}

pub(super) fn admit(
    root: &Path,
    namespace: &str,
    shards: usize,
    shard_bytes: u64,
    configured: Option<u64>,
) -> io::Result<Admission> {
    let directory = lock_root(root, false)?;
    let stored = stored_limit(root)?;
    if let (Some(existing), Some(requested)) = (stored, configured)
        && existing != requested
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("shared SSD root capacity is {existing} bytes, requested {requested}"),
        ));
    }
    let mut pending_limit = None;
    if let Some(limit) = stored.or(configured) {
        let needed = required_payload(root, namespace, shards, shard_bytes)?;
        if needed > limit {
            return Err(Error::new(
                ErrorKind::StorageFull,
                format!(
                    "shared SSD root capacity exceeded: {needed} payload bytes required, limit {limit}"
                ),
            ));
        }
        if stored.is_none() {
            pending_limit = Some(stage_limit(&directory, limit)?);
        }
    }
    Ok(Admission {
        directory,
        pending_limit,
    })
}

/// Evict one partition and return the sum of the shard lengths released.
/// A zero wait requires an offline owner. A nonzero wait asks the live Manager
/// to relinquish this SSD tier and bounds waiting for admission and file users.
pub fn reclaim_shared_ssd_partition(
    root: &Path,
    namespace: &str,
    wait: Duration,
) -> io::Result<u64> {
    if !orbitkv_state::is_storage_namespace(namespace) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "expected an exact orbitkv:v2 storage namespace",
        ));
    }
    files::outside_shared_partition(root)?;
    let partition = root.join(format!("{PARTITION_PREFIX}{namespace}"));
    let journal_path = partition.join(JOURNAL_NAME);
    let deadline =
        if wait.is_zero() {
            None
        } else {
            Some(Instant::now().checked_add(wait).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "SSD release wait is too large")
            })?)
        };
    let _request = if deadline.is_some() {
        files::check_shared_directory(&partition)?;
        let request = File::open(&partition)?;
        wait_for_owner(deadline, || request.try_lock().map_err(Error::from))?;
        Some(request)
    } else {
        None
    };
    let (_root_lock, mut locked) = wait_for_owner(deadline, || {
        let directory = lock_root(root, deadline.is_some())?;
        files::check_shared_directory(&partition)?;
        let files = lock_partition_files(&partition, &journal_path)?;
        Ok((directory, files))
    })?;
    let released = locked.identities.iter().try_fold(0u64, |total, meta| {
        total
            .checked_add(meta.len())
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "shared SSD payload size overflow"))
    })?;
    persistence::discard_partition_records(&mut locked.journal, namespace, &locked.identities)?;
    persistence::sync_parent(&journal_path)?;
    persistence::fault("ssd_reclaim_journal_synced");
    for file in &locked.shards {
        file.set_len(0)?;
        file.sync_all()?;
        persistence::fault("ssd_reclaim_shard_synced");
    }
    Ok(released)
}

/// Retry only lock/lease preflight. The caller owns every lock before any
/// durable mutation and never retries journal or shard I/O here.
fn wait_for_owner<T>(
    deadline: Option<Instant>,
    mut attempt: impl FnMut() -> io::Result<T>,
) -> io::Result<T> {
    let Some(deadline) = deadline else {
        return attempt();
    };
    loop {
        match attempt() {
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                std::thread::sleep(remaining.min(Duration::from_millis(50)));
                if Instant::now() >= deadline {
                    return Err(Error::new(
                        ErrorKind::TimedOut,
                        format!("timed out waiting for shared SSD ownership: {error}"),
                    ));
                }
            }
            result => return result,
        }
    }
}

struct LockedPartition {
    shards: Vec<File>,
    identities: Vec<fs::Metadata>,
    journal: File,
}

fn lock_partition_files(partition: &Path, journal_path: &Path) -> io::Result<LockedPartition> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(partition)? {
        let entry = entry?;
        if let Some(index) = shard_index(&entry.file_name())? {
            if index >= MAX_SSD_SHARDS {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "invalid shared SSD shard index",
                ));
            }
            paths.push((index, entry.path()));
        }
    }
    paths.sort_unstable_by_key(|(index, _)| *index);
    if paths.is_empty() || paths.iter().enumerate().any(|(i, (index, _))| i != *index) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "incomplete shared SSD shard set",
        ));
    }
    let mut shards = Vec::with_capacity(paths.len());
    let mut identities = Vec::with_capacity(paths.len());
    for (_, path) in paths {
        let (file, meta) = open_partition_file(&path)?;
        files::lock_shard(&file, &path)?;
        files::previous_file_users_drained(&file)?;
        shards.push(file);
        identities.push(meta);
    }
    let (journal, _) = open_partition_file(journal_path)?;
    files::previous_file_users_drained(&journal)?;
    Ok(LockedPartition {
        shards,
        identities,
        journal,
    })
}

fn open_partition_file(path: &Path) -> io::Result<(File, fs::Metadata)> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.nlink() != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "shared SSD reclamation requires regular, singly linked files",
        ));
    }
    Ok((file, meta))
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/shared_root.rs"]
mod tests;
