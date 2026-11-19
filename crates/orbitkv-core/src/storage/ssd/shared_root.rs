use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Error, ErrorKind, Seek, Write};
use std::num::NonZeroU64;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use super::persistence::{frame, read_frame};

pub(super) const PARTITION_PREFIX: &str = "orbitkv-ssd-";
const LIMIT_FILE: &str = ".orbitkv-ssd-capacity";
const LIMIT_TEMP: &str = ".orbitkv-ssd-capacity.tmp";
const MAX_DECIMAL_U64: usize = 20;

fn partition_namespace(name: &OsStr) -> Option<&str> {
    name.to_str()?
        .strip_prefix(PARTITION_PREFIX)
        .filter(|s| orbitkv_state::is_storage_namespace(s))
}

pub(super) fn reserved_name(name: &OsStr) -> bool {
    name == LIMIT_FILE || name == LIMIT_TEMP || partition_namespace(name).is_some()
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

fn install_limit(root: &Path, directory: &File, limit: u64) -> io::Result<()> {
    let temp = root.join(LIMIT_TEMP);
    // A killed initializer may have left this unpublished file. Never truncate an existing inode.
    match fs::remove_file(&temp) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)?;
    file.write_all(&frame(&limit, MAX_DECIMAL_U64)?)?;
    file.sync_all()?;
    fs::rename(temp, root.join(LIMIT_FILE))?;
    directory.sync_all()
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
            let name = shard.file_name();
            // Checkpoints can be replaced during this scan. Their bytes are outside the payload limit.
            if name == "shard-000000.dat.index" || name == "shard-000000.dat.index.tmp" {
                continue;
            }
            let index = name
                .to_str()
                .and_then(|s| s.strip_prefix("shard-"))
                .and_then(|s| s.strip_suffix(".dat"))
                .filter(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        "unexpected file in shared SSD partition",
                    )
                })?;
            let meta = shard.metadata()?;
            if !meta.is_file() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "invalid shared SSD shard",
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

/// Hold the directory lock through journal recovery and shard sizing. Live transfers never take it.
pub(super) fn admit(
    root: &Path,
    namespace: &str,
    shards: usize,
    shard_bytes: u64,
    configured: Option<u64>,
) -> io::Result<File> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY)
        .open(root)?;
    directory.lock()?;
    let stored = stored_limit(root)?;
    if let (Some(existing), Some(requested)) = (stored, configured)
        && existing != requested
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("shared SSD root capacity is {existing} bytes, requested {requested}"),
        ));
    }
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
            install_limit(root, &directory, limit)?;
        }
    }
    Ok(directory)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/shared_root.rs"]
mod tests;
