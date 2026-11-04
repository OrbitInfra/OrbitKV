//! The fork's file receipts are independent of generic TENT task status.

use std::ffi::{c_char, c_int, c_void};

use libloading::os::unix::Library;

use super::{NativeEngine, api, set_config, symbol};

pub const IOURING: c_int = 6;
pub const GDS: c_int = 5;
pub const READ_WRITE: c_int = 3;
pub const UNQUIESCED: c_int = 8;

#[repr(C)]
pub struct FileOptions {
    pub struct_size: u32,
    pub io_uring_enabled: u32,
    pub gds_mode: c_int,
    pub io_uring_queue_depth: u32,
    pub gds_queue_depth: u32,
}

#[repr(C)]
pub struct FileRange {
    pub address: *mut c_void,
    pub length: u64,
}

#[repr(C)]
pub struct FileOwner {
    pub context: *mut c_void,
    pub retain: unsafe extern "C" fn(*mut c_void),
    pub release: unsafe extern "C" fn(*mut c_void),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FileResult {
    pub outcome: c_int,
    pub drained: u32,
    pub transferred_bytes: u64,
}

impl Default for FileResult {
    fn default() -> Self {
        Self {
            outcome: UNQUIESCED,
            drained: 0,
            transferred_bytes: 0,
        }
    }
}

/// Fallible native boundary, also used by CPU controls of the actual Rust owners.
#[derive(Clone, Copy)]
pub struct FileApi {
    pub create: unsafe extern "C" fn(*const FileOptions, *mut NativeEngine) -> c_int,
    pub open: unsafe extern "C" fn(
        NativeEngine,
        *const c_char,
        c_int,
        c_int,
        *mut u64,
        *mut u64,
    ) -> c_int,
    pub submit: unsafe extern "C" fn(
        NativeEngine,
        u64,
        c_int,
        u64,
        *const FileRange,
        usize,
        *const FileOwner,
        *mut u64,
    ) -> c_int,
    pub drain: unsafe extern "C" fn(NativeEngine, u64, u64, c_int, *mut FileResult) -> c_int,
    pub release: unsafe extern "C" fn(NativeEngine, u64) -> c_int,
    pub close: unsafe extern "C" fn(NativeEngine, u64, u64, *mut FileResult) -> c_int,
    pub destroy: unsafe extern "C" fn(NativeEngine),
}

impl FileApi {
    pub(super) unsafe fn load(library: &Library) -> Result<Self, String> {
        // SAFETY: the pinned C header defines these signatures; Api owns the library.
        unsafe {
            Ok(Self {
                create: symbol(library, b"tent_create_engine_with_files\0")?,
                open: symbol(library, b"tent_file_open\0")?,
                submit: symbol(library, b"tent_file_submit\0")?,
                drain: symbol(library, b"tent_file_drain\0")?,
                release: symbol(library, b"tent_file_release_batch\0")?,
                close: symbol(library, b"tent_file_close\0")?,
                destroy: symbol(library, b"tent_destroy_engine\0")?,
            })
        }
    }
}

pub fn file_api() -> Result<FileApi, String> {
    super::ensure_loaded()?;
    api().files.clone()
}

/// Configure only the calling thread's next engine. Peer construction resets
/// its own settings; file transports never enter the peer/source-drain engine.
///
/// # Safety
/// The runtime must be loaded, and engine creation must be serialized with
/// peer construction, which temporarily configures process environment values.
pub unsafe fn configure_file_engine() {
    // SAFETY: static C strings and thread-local native configuration. The caller
    // serializes engine creation against the existing environment-scoped peer path.
    unsafe {
        set_config(c"metadata_type", c"p2p");
        set_config(c"metadata_servers", c"P2PHANDSHAKE");
        set_config(c"local_segment_name", c"127.0.0.1:0");
        set_config(c"rpc_server_hostname", c"127.0.0.1");
        set_config(c"rpc_server_port", c"0");
        set_config(c"metrics/enabled", c"false");
        set_config(c"use_legacy_transport_selection", c"false");
        set_config(c"enable_runtime_queue", c"false");
        set_config(c"enable_auto_failover_on_poll", c"false");
        for transport in [
            c"transports/tcp/enable",
            c"transports/shm/enable",
            c"transports/rdma/enable",
            c"transports/nvlink/enable",
            c"transports/mnnvl/enable",
            c"transports/hp_tcp/enable",
            c"transports/fabric/enable",
            c"transports/mpcomm/enable",
        ] {
            set_config(transport, c"false");
        }
        set_config(
            c"transports/io_uring/pinned_staging",
            if cfg!(feature = "cuda") {
                c"true"
            } else {
                c"false"
            },
        );
        set_config(c"transports/io_uring/staging_slots", c"4");
        set_config(c"transports/io_uring/staging_slot_kb", c"16384");
    }
}
