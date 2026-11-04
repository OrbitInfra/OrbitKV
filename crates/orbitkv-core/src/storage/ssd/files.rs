use std::path::PathBuf;

/// Reserve capacity before admitting GPU I/O. Allocation failures must not
/// silently turn a capacity error into sparse, best-effort writes.
pub(super) fn reserve_cache_space<F: std::borrow::Borrow<std::fs::File>>(
    files: &[F],
    capacity: u64,
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
            for (index, file) in files.iter().enumerate() {
                if let Err(cleanup) = file.borrow().set_len(0) {
                    log::warn!(
                        "Failed to release SSD shard {index} after allocation failure: {cleanup}"
                    );
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
    let shard_count = u64::try_from(shard_count).expect("usize fits into u64");
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

pub(super) fn open_cache_files(
    cache_paths: &[PathBuf],
    shards_per_path: usize,
    shard_capacity: u64,
    options: &mut std::fs::OpenOptions,
) -> std::io::Result<Vec<std::fs::File>> {
    use std::fs;
    use std::os::unix::fs::OpenOptionsExt;

    options
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT);

    if cache_paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache paths cannot be empty",
        ));
    }

    let total_shards = cache_paths.len() * shards_per_path;

    // A single shard uses a file path; multiple shards use directories.
    if total_shards == 1 {
        if let Some(parent) = cache_paths[0].parent() {
            fs::create_dir_all(parent)?;
        }
        let file = options.open(&cache_paths[0])?;
        file.set_len(shard_capacity)?;
        return Ok(vec![file]);
    }

    // Multi-path or multi-shard: each path must be a directory.
    for path in cache_paths {
        if path.exists() && !path.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "SSD cache path {} must be a directory when using multiple paths or shards",
                    path.display()
                ),
            ));
        }
        fs::create_dir_all(path)?;
    }

    let mut files = Vec::with_capacity(total_shards);
    for (path_id, path) in cache_paths.iter().enumerate() {
        for local_shard in 0..shards_per_path {
            let global_shard_id = path_id * shards_per_path + local_shard;
            let file_path = path.join(format!("shard-{global_shard_id:06}.dat"));
            let file = options.open(&file_path)?;
            file.set_len(shard_capacity)?;
            files.push(file);
        }
    }

    Ok(files)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/files.rs"]
mod tests;
