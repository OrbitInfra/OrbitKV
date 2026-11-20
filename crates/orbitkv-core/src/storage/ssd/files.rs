use std::path::{Path, PathBuf};

/// Reserve capacity before admitting GPU I/O. Allocation failures must not
/// silently turn a capacity error into sparse, best-effort writes.
pub(super) fn reserve_cache_space<F: std::borrow::Borrow<std::fs::File>>(
    files: &[F],
    capacity: u64,
    preserve: bool,
) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;

    let length = i64::try_from(capacity).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD shard capacity exceeds off_t",
        )
    })?;
    for (shard, file) in files.iter().enumerate() {
        let file = file.borrow();
        // SAFETY: each fd is owned and no I/O has been admitted. Use the native
        // allocation operation, without a libc zero-fill emulation on failure.
        if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, length) } != 0 {
            let error = std::io::Error::last_os_error();
            // These files were just truncated for this startup. Release even a
            // partially allocated failing shard before returning the error.
            if !preserve {
                for (index, file) in files.iter().enumerate() {
                    if let Err(cleanup) = file.borrow().set_len(0) {
                        log::warn!(
                            "Failed to release SSD shard {index} after allocation failure: {cleanup}"
                        );
                    }
                }
            }
            return Err(std::io::Error::new(
                error.kind(),
                format!("failed to reserve {capacity} bytes for SSD shard {shard}: {error}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn aligned_shard_capacity(
    capacity_bytes: u64,
    shard_count: usize,
    alignment: usize,
) -> std::io::Result<u64> {
    if shard_count == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache requires at least one shard",
        ));
    }
    let shard_count = shard_count as u64;
    let raw = capacity_bytes / shard_count;
    // cuFile expands edge reads to 4 KiB, including the last block in a shard.
    let alignment = alignment as u64;
    let capacity = raw / alignment * alignment;
    if capacity == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache capacity is too small for the requested shard count",
        ));
    }
    Ok(capacity)
}

/// Reject a managed directory anywhere in an operator-supplied path before creating children.
pub(super) fn outside_shared_partition(path: &Path) -> std::io::Result<()> {
    let in_partition = |path: &Path| {
        path.components()
            .any(|part| super::shared_root::reserved_name(part.as_os_str()))
    };
    let absolute = std::path::absolute(path)?;
    if !in_partition(&absolute) {
        // The leaf may not exist yet. Resolve its closest existing ancestor to cover directory aliases.
        for ancestor in absolute.ancestors() {
            match ancestor.canonicalize() {
                Ok(existing) if in_partition(&existing) => break,
                Ok(_) => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "cache path or shared root is inside a managed SSD partition or uses a reserved root capacity file",
    ))
}

/// Keep the first shard and its journal at the same names for every shard count.
pub(super) fn open_shared_partition(
    root: &Path,
    namespace: &str,
    shards: usize,
    shard_capacity: u64,
    root_capacity: Option<u64>,
) -> std::io::Result<(PathBuf, super::shared_root::Admission)> {
    use std::fs::DirBuilder;
    use std::os::unix::fs::DirBuilderExt;

    outside_shared_partition(root)?;
    let lock = super::shared_root::admit(root, namespace, shards, shard_capacity, root_capacity)?;
    let directory = root.join(format!(
        "{}{namespace}",
        super::shared_root::PARTITION_PREFIX
    ));
    match DirBuilder::new().mode(0o700).create(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            check_shared_directory(&directory)?;
        }
        Err(error) => return Err(error),
    }
    if shards == 1 {
        Ok((directory.join("shard-000000.dat"), lock))
    } else {
        Ok((directory, lock))
    }
}

pub(super) fn check_shared_directory(directory: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(directory)?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "shared SSD namespace needs a private directory",
        ));
    }
    Ok(())
}

/// Under the first shard lock, account for every data file even if metadata is missing.
fn shared_partition_has_payload(paths: &[PathBuf], journal: &Path) -> std::io::Result<bool> {
    use std::io::{Error, ErrorKind};

    let directory = journal
        .parent()
        .ok_or_else(|| Error::other("missing SSD partition directory"))?;
    let checkpoint = journal.with_extension("index.tmp");
    let mut has_payload = false;
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "unexpected entry in shared SSD partition",
            ));
        }
        if path == journal || path == checkpoint {
            continue;
        }
        if !paths.contains(&path) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "unexpected file or shard count change in shared SSD partition",
            ));
        }
        has_payload |= entry.metadata()?.len() > 0;
    }
    Ok(has_payload)
}

pub(super) fn open_cache_files(
    cache_paths: &[PathBuf],
    shards_per_path: usize,
    shard_capacity: u64,
    shared_namespace: Option<&str>,
    preserve: bool,
) -> std::io::Result<Vec<std::fs::File>> {
    use std::fs;
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = fs::OpenOptions::new();
    options
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT | libc::O_NOFOLLOW | libc::O_CLOEXEC);

    if cache_paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache paths cannot be empty",
        ));
    }

    if shared_namespace.is_none() {
        for path in cache_paths {
            outside_shared_partition(path)?;
        }
    }
    let total_shards = cache_paths.len() * shards_per_path;

    // Open and lock every inode before modifying any of them.
    let mut paths = Vec::with_capacity(total_shards);
    if total_shards == 1 {
        if let Some(parent) = cache_paths[0].parent() {
            fs::create_dir_all(parent)?;
        }
        paths.push(cache_paths[0].clone());
    } else {
        // Multi-path or multi-shard: each path must be a directory.
        for path in cache_paths {
            if path.exists() && !path.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "SSD cache path {} must be a directory with multiple shards",
                        path.display()
                    ),
                ));
            }
            fs::create_dir_all(path)?;
        }
        for (path_id, path) in cache_paths.iter().enumerate() {
            for local_shard in 0..shards_per_path {
                paths.push(path.join(format!(
                    "shard-{:06}.dat",
                    path_id * shards_per_path + local_shard
                )));
            }
        }
    }
    let metadata = journal_path(cache_paths, shards_per_path)?;
    let mut files = Vec::with_capacity(total_shards);
    for path in &paths {
        let file = options.open(path)?;
        lock_shard(&file, path)?;
        if files.is_empty() && (shared_namespace.is_some() || !preserve) {
            let has_payload =
                shared_namespace.is_some() && shared_partition_has_payload(&paths, &metadata)?;
            super::persistence::check_partition_file(
                &metadata,
                shared_namespace,
                total_shards,
                has_payload,
            )?;
        }
        files.push(file);
    }
    if preserve || metadata.exists() {
        for file in &files {
            previous_file_users_drained(file)?;
        }
    }
    if !preserve {
        match fs::remove_file(&metadata) {
            Ok(()) => super::persistence::sync_parent(&metadata)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        for file in &files {
            file.set_len(0)?;
            file.set_len(shard_capacity)?;
        }
    }
    if preserve {
        for path in &paths {
            // Persist newly created names and their ancestor directory entries.
            for ancestor in path.ancestors().take_while(|p| !p.as_os_str().is_empty()) {
                super::persistence::sync_parent(ancestor)?;
            }
        }
    }

    Ok(files)
}

pub(super) fn lock_shard(file: &std::fs::File, path: &Path) -> std::io::Result<()> {
    file.try_lock().map_err(|err| {
        std::io::Error::other(format!(
            "SSD shard is already owned: {}: {err}",
            path.display()
        ))
    })
}

pub(super) fn previous_file_users_drained(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    // F_SETSIG is Linux UAPI 10 on the supported x86_64/aarch64 targets;
    // libc exposes F_SETLEASE but not F_SETSIG on those GNU targets.
    const F_SETSIG: libc::c_int = 10;
    // SAFETY: fd is owned. Change only this fresh description's lease signal
    // to ignored-by-default SIGURG; no process-wide signal handler is changed.
    if unsafe { libc::fcntl(fd, F_SETSIG, libc::SIGURG) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: this fresh owned description is not yet used by OrbiTE. Linux
    // grants a write lease only when its inode has no other read/write opens,
    // including file descriptions retained by old kernel io_uring requests.
    if unsafe { libc::fcntl(fd, libc::F_SETLEASE, libc::F_WRLCK) } < 0 {
        let error = std::io::Error::last_os_error();
        return Err(std::io::Error::new(
            error.kind(),
            format!(
                "SSD ownership requires exclusive file-lease support and drained prior file users: {error}"
            ),
        ));
    }
    // SAFETY: release only the lease just acquired on this owned description.
    // The separate lifetime flock still excludes another cooperating Manager.
    if unsafe { libc::fcntl(fd, libc::F_SETLEASE, libc::F_UNLCK) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

pub(super) fn journal_path(paths: &[PathBuf], shards: usize) -> std::io::Result<PathBuf> {
    let first = paths
        .first()
        .ok_or_else(|| std::io::Error::other("missing SSD path"))?;
    let data = if paths.len() == 1 && shards == 1 {
        first.clone()
    } else {
        first.join("shard-000000.dat")
    };
    let mut name = data.into_os_string();
    name.push(".index");
    Ok(PathBuf::from(name))
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/files.rs"]
mod tests;
