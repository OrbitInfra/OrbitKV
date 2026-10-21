//! Raw dynamic bindings for the pinned Mooncake TENT C ABI.
//!
//! The sys crate builds only TENT from the pinned Mooncake submodule. Runtime
//! loading stays dynamic so the native dependency graph is not linked into
//! every Rust binary or Python extension.

#![allow(
    clippy::missing_safety_doc,
    reason = "crate-private C ABI shims share the safety contract documented by orbitkv-transfer"
)]

use std::env;
use std::ffi::{CStr, CString, OsString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

pub const SOURCE_VERSION: &str = "v0.3.13.post1+orbitkv.1";
pub const SOURCE_REVISION: &str = "12cbf549b6351c800de0f94a480276d216b9a43b";
#[cfg(feature = "cuda")]
const WORKSPACE_RUNTIME_LIB_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.orbitkv/mooncake/cuda/lib"
);
#[cfg(not(feature = "cuda"))]
const WORKSPACE_RUNTIME_LIB_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.orbitkv/mooncake/cpu/lib"
);

pub type NativeEngine = *mut c_void;
pub type SegmentId = u64;
pub type BatchId = u64;

pub const INVALID_SEGMENT: SegmentId = u64::MAX;
pub const INVALID_BATCH: BatchId = 0;

const GLOBAL_READ_WRITE: c_int = 2;
const TRANSPORT_UNSPECIFIED: c_int = 0;

struct SuspendedTentConfig(Option<OsString>);

impl SuspendedTentConfig {
    fn for_forced_tcp() -> Option<Self> {
        env::var_os("MC_FORCE_TCP")?;
        let previous = env::var_os("MC_TENT_CONF");
        unsafe { env::remove_var("MC_TENT_CONF") };
        Some(Self(previous))
    }
}

impl Drop for SuspendedTentConfig {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => unsafe { env::set_var("MC_TENT_CONF", value) },
            None => unsafe { env::remove_var("MC_TENT_CONF") },
        }
    }
}

#[repr(C)]
pub struct TransferRequest {
    pub opcode: c_int,
    pub source: *mut c_void,
    pub target_id: SegmentId,
    pub target_offset: u64,
    pub length: u64,
    pub priority: c_int,
    pub transport_hint: c_int,
    pub expected_source_epoch: [c_char; 33],
}

#[repr(C)]
pub struct TransferStatus {
    pub status: c_int,
    pub transferred_bytes: u64,
}

#[repr(C)]
pub struct MemoryOptions {
    pub location: [c_char; 64],
    pub permission: c_int,
    pub transport_type: c_int,
    pub shm_path: [c_char; 256],
    pub shm_offset: usize,
    pub internal: c_int,
}

#[repr(C)]
pub struct NotificationRecord {
    pub handle: SegmentId,
    pub name: [c_char; 256],
    pub message: [c_char; 4096],
}

#[repr(C)]
pub struct NotificationInfo {
    pub count: c_int,
    pub records: *mut NotificationRecord,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct NicLoadStat {
    pub device_name: [c_char; 64],
    pub inflight_bytes: u64,
    pub ewma_bandwidth_bps: f64,
}

type SetConfig = unsafe extern "C" fn(*const c_char, *const c_char);
type Create = unsafe extern "C" fn() -> NativeEngine;
type Destroy = unsafe extern "C" fn(NativeEngine);
type Available = unsafe extern "C" fn(NativeEngine) -> c_int;
type LocalEndpoint = unsafe extern "C" fn(NativeEngine, *mut c_char, usize) -> c_int;
type OpenSegment = unsafe extern "C" fn(NativeEngine, *mut SegmentId, *const c_char) -> c_int;
type CloseSegment = unsafe extern "C" fn(NativeEngine, SegmentId) -> c_int;
type RegisterMemory =
    unsafe extern "C" fn(NativeEngine, *mut c_void, usize, *mut MemoryOptions) -> c_int;
type UnregisterMemory = unsafe extern "C" fn(NativeEngine, *mut c_void, usize) -> c_int;
type CloseSource = unsafe extern "C" fn(NativeEngine, u64, *mut c_char, *mut u64) -> c_int;
type SourceEpoch = unsafe extern "C" fn(NativeEngine, *mut c_char, usize) -> c_int;
type ValidateRemoteEpoch = unsafe extern "C" fn(NativeEngine, SegmentId, *const c_char) -> c_int;
type AllocateBatch = unsafe extern "C" fn(NativeEngine, usize) -> BatchId;
type Submit = unsafe extern "C" fn(NativeEngine, BatchId, *mut TransferRequest, usize) -> c_int;
type SubmitWithNotify = unsafe extern "C" fn(
    NativeEngine,
    BatchId,
    *mut TransferRequest,
    usize,
    *const c_char,
    *const c_char,
) -> c_int;
type TransferStatusFn =
    unsafe extern "C" fn(NativeEngine, BatchId, usize, *mut TransferStatus) -> c_int;
type BatchStatusFn = unsafe extern "C" fn(NativeEngine, BatchId, *mut TransferStatus) -> c_int;
type CancelTask = unsafe extern "C" fn(NativeEngine, BatchId, usize) -> c_int;
type FreeBatch = unsafe extern "C" fn(NativeEngine, BatchId) -> c_int;
type ReceiveNotifications = unsafe extern "C" fn(NativeEngine, *mut NotificationInfo) -> c_int;
type FreeNotifications = unsafe extern "C" fn(*mut NotificationInfo);
type SendNotification =
    unsafe extern "C" fn(NativeEngine, SegmentId, *const c_char, *const c_char) -> c_int;
type GetNicLoadStats = unsafe extern "C" fn(NativeEngine, *mut NicLoadStat, *mut usize) -> c_int;

struct Api {
    _dependencies: Vec<Library>,
    _library: Library,
    set_config: SetConfig,
    create: Create,
    destroy: Destroy,
    available: Available,
    local_endpoint: LocalEndpoint,
    open_segment: OpenSegment,
    close_segment: CloseSegment,
    register_memory: RegisterMemory,
    unregister_memory: UnregisterMemory,
    close_source: CloseSource,
    source_epoch: SourceEpoch,
    validate_remote_epoch: ValidateRemoteEpoch,
    allocate_batch: AllocateBatch,
    submit: Submit,
    submit_with_notify: SubmitWithNotify,
    transfer_status: TransferStatusFn,
    batch_status: BatchStatusFn,
    cancel_task: CancelTask,
    free_batch: FreeBatch,
    receive_notifications: ReceiveNotifications,
    free_notifications: FreeNotifications,
    send_notification: SendNotification,
    get_nic_load_stats: GetNicLoadStats,
}

static API: LazyLock<Result<Api, String>> = LazyLock::new(load_api);

pub fn ensure_loaded() -> Result<(), String> {
    API.as_ref().map(|_| ()).map_err(Clone::clone)
}

fn api() -> &'static Api {
    API.as_ref()
        .unwrap_or_else(|error| panic!("Mooncake TENT runtime unavailable: {error}"))
}

fn load_api() -> Result<Api, String> {
    let mut attempted = Vec::new();
    for directory in candidate_directories() {
        match unsafe { load_api_from(&directory) } {
            Ok(api) => return Ok(api),
            Err(error) => attempted.push(format!("{} ({error})", directory.display())),
        }
    }
    Err(format!(
        "could not load libtent_shared.so; searched {}",
        attempted.join(", "),
    ))
}

fn candidate_directories() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("ORBITKV_MOONCAKE_LIB_DIR") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = module_directory() {
        candidates.push(path);
    }
    if let Ok(executable) = env::current_exe()
        && let Some(parent) = executable.parent()
    {
        candidates.push(parent.to_path_buf());
    }
    candidates.push(PathBuf::from(WORKSPACE_RUNTIME_LIB_DIR));
    candidates.dedup();
    candidates
}

#[inline(never)]
fn module_directory() -> Option<PathBuf> {
    let mut info = libc::Dl_info {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    let found = unsafe { libc::dladdr(module_directory as *const () as *const c_void, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) };
    Path::new(path.to_str().ok()?)
        .parent()
        .map(Path::to_path_buf)
}

unsafe fn load_api_from(directory: &Path) -> Result<Api, String> {
    let flags = RTLD_NOW | RTLD_GLOBAL;
    let asio = unsafe { Library::open(Some(directory.join("libasio.so")), flags) }
        .map_err(|error| error.to_string())?;
    let common = unsafe { Library::open(Some(directory.join("libmooncake_common.so")), flags) }
        .map_err(|error| error.to_string())?;
    let library = unsafe { Library::open(Some(directory.join("libtent_shared.so")), flags) }
        .map_err(|error| error.to_string())?;

    Ok(Api {
        set_config: unsafe { symbol(&library, b"tent_set_config\0")? },
        create: unsafe { symbol(&library, b"tent_create_engine\0")? },
        destroy: unsafe { symbol(&library, b"tent_destroy_engine\0")? },
        available: unsafe { symbol(&library, b"tent_available\0")? },
        local_endpoint: unsafe { symbol(&library, b"tent_segment_name\0")? },
        open_segment: unsafe { symbol(&library, b"tent_open_segment\0")? },
        close_segment: unsafe { symbol(&library, b"tent_close_segment\0")? },
        register_memory: unsafe { symbol(&library, b"tent_register_memory_ex\0")? },
        unregister_memory: unsafe { symbol(&library, b"tent_unregister_memory\0")? },
        close_source: unsafe { symbol(&library, b"tent_close_source\0")? },
        source_epoch: unsafe { symbol(&library, b"tent_source_epoch\0")? },
        validate_remote_epoch: unsafe { symbol(&library, b"tent_validate_remote_epoch\0")? },
        allocate_batch: unsafe { symbol(&library, b"tent_allocate_batch\0")? },
        submit: unsafe { symbol(&library, b"tent_submit\0")? },
        submit_with_notify: unsafe { symbol(&library, b"tent_submit_notif\0")? },
        transfer_status: unsafe { symbol(&library, b"tent_task_status\0")? },
        batch_status: unsafe { symbol(&library, b"tent_overall_status\0")? },
        cancel_task: unsafe { symbol(&library, b"tent_cancel_task\0")? },
        free_batch: unsafe { symbol(&library, b"tent_free_batch\0")? },
        receive_notifications: unsafe { symbol(&library, b"tent_recv_notifs\0")? },
        free_notifications: unsafe { symbol(&library, b"tent_free_notifs\0")? },
        send_notification: unsafe { symbol(&library, b"tent_send_notifs\0")? },
        get_nic_load_stats: unsafe { symbol(&library, b"tent_get_nic_load_stats\0")? },
        _dependencies: vec![asio, common],
        _library: library,
    })
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| format!("missing Mooncake TENT symbol: {error}"))
}

unsafe fn set_config(key: &CStr, value: &CStr) {
    unsafe { (api().set_config)(key.as_ptr(), value.as_ptr()) };
}

pub unsafe fn create(
    metadata: *const c_char,
    local_server_name: *const c_char,
    bind_host: *const c_char,
    rpc_port: u64,
) -> NativeEngine {
    // Explicit forced-TCP qualification must not be undone by a general TENT
    // JSON config. TransferEngine::new serializes creation around this scope.
    let _tent_config = SuspendedTentConfig::for_forced_tcp();
    let Ok(port) = CString::new(rpc_port.to_string()) else {
        return std::ptr::null_mut();
    };
    unsafe {
        set_config(c"metadata_type", c"p2p");
        set_config(c"metadata_servers", CStr::from_ptr(metadata));
        set_config(c"local_segment_name", CStr::from_ptr(local_server_name));
        set_config(c"rpc_server_hostname", CStr::from_ptr(bind_host));
        set_config(c"rpc_server_port", &port);
        set_config(c"metrics/enabled", c"false");
        set_config(c"use_legacy_transport_selection", c"false");
        set_config(c"enable_runtime_queue", c"false");
        // A successful retry cannot prove that the preceding failed physical
        // attempt has stopped touching its source and destination buffers.
        set_config(c"enable_auto_failover_on_poll", c"false");
        set_config(c"transports/tcp/enable", c"true");
        set_config(c"transports/shm/enable", c"false");
        set_config(c"transports/gds/enable", c"false");
        set_config(c"transports/io_uring/enable", c"false");
        let native_paths = if env::var_os("MC_FORCE_TCP").is_some() {
            c"false"
        } else {
            c"true"
        };
        set_config(c"transports/rdma/enable", native_paths);
        set_config(c"transports/nvlink/enable", native_paths);
        set_config(c"transports/mnnvl/enable", native_paths);
        let engine = (api().create)();
        if engine.is_null() || (api().available)(engine) != 1 {
            if !engine.is_null() {
                (api().destroy)(engine);
            }
            return std::ptr::null_mut();
        }
        engine
    }
}

pub unsafe fn destroy(engine: NativeEngine) {
    unsafe { (api().destroy)(engine) };
}

pub unsafe fn local_ip_and_port(engine: NativeEngine, output: *mut c_char, len: usize) -> c_int {
    unsafe { (api().local_endpoint)(engine, output, len) }
}

pub unsafe fn open_segment(engine: NativeEngine, name: *const c_char) -> SegmentId {
    let mut segment = 0;
    if unsafe { (api().open_segment)(engine, &mut segment, name) } == 0 {
        segment
    } else {
        INVALID_SEGMENT
    }
}

pub unsafe fn close_segment(engine: NativeEngine, segment: SegmentId) -> c_int {
    unsafe { (api().close_segment)(engine, segment) }
}

pub unsafe fn register_memory(
    engine: NativeEngine,
    address: *mut c_void,
    length: usize,
    location: *const c_char,
) -> c_int {
    let location = unsafe { CStr::from_ptr(location) }.to_bytes();
    if location.len() >= 64 {
        return -1;
    }
    let mut options = MemoryOptions {
        location: [0; 64],
        permission: GLOBAL_READ_WRITE,
        transport_type: TRANSPORT_UNSPECIFIED,
        shm_path: [0; 256],
        shm_offset: 0,
        internal: 0,
    };
    for (output, input) in options.location.iter_mut().zip(location) {
        *output = *input as c_char;
    }
    unsafe { (api().register_memory)(engine, address, length, &mut options) }
}

pub unsafe fn unregister_memory(
    engine: NativeEngine,
    address: *mut c_void,
    length: usize,
) -> c_int {
    unsafe { (api().unregister_memory)(engine, address, length) }
}

pub unsafe fn close_source(
    engine: NativeEngine,
    timeout_ms: u64,
    epoch: &mut [c_char; 33],
    active: &mut u64,
) -> c_int {
    unsafe { (api().close_source)(engine, timeout_ms, epoch.as_mut_ptr(), active) }
}

pub unsafe fn source_epoch(engine: NativeEngine, epoch: &mut [c_char; 33]) -> c_int {
    unsafe { (api().source_epoch)(engine, epoch.as_mut_ptr(), epoch.len()) }
}

pub unsafe fn validate_remote_epoch(
    engine: NativeEngine,
    segment: SegmentId,
    epoch: &[c_char; 33],
) -> c_int {
    unsafe { (api().validate_remote_epoch)(engine, segment, epoch.as_ptr()) }
}

pub unsafe fn allocate_batch(engine: NativeEngine, batch_size: usize) -> BatchId {
    unsafe { (api().allocate_batch)(engine, batch_size) }
}

pub unsafe fn submit(
    engine: NativeEngine,
    batch: BatchId,
    requests: &mut [TransferRequest],
) -> c_int {
    unsafe { (api().submit)(engine, batch, requests.as_mut_ptr(), requests.len()) }
}

pub unsafe fn submit_with_notify(
    engine: NativeEngine,
    batch: BatchId,
    requests: &mut [TransferRequest],
    name: *const c_char,
    message: *const c_char,
) -> c_int {
    unsafe {
        (api().submit_with_notify)(
            engine,
            batch,
            requests.as_mut_ptr(),
            requests.len(),
            name,
            message,
        )
    }
}

pub unsafe fn transfer_status(
    engine: NativeEngine,
    batch: BatchId,
    task: usize,
    status: &mut TransferStatus,
) -> c_int {
    unsafe { (api().transfer_status)(engine, batch, task, status) }
}

pub unsafe fn batch_status(
    engine: NativeEngine,
    batch: BatchId,
    status: &mut TransferStatus,
) -> c_int {
    unsafe { (api().batch_status)(engine, batch, status) }
}

pub unsafe fn cancel_task(engine: NativeEngine, batch: BatchId, task: usize) -> c_int {
    unsafe { (api().cancel_task)(engine, batch, task) }
}

pub unsafe fn free_batch(engine: NativeEngine, batch: BatchId) -> c_int {
    unsafe { (api().free_batch)(engine, batch) }
}

pub unsafe fn receive_notifications(engine: NativeEngine, info: &mut NotificationInfo) -> c_int {
    unsafe { (api().receive_notifications)(engine, info) }
}

pub unsafe fn free_notifications(info: &mut NotificationInfo) {
    unsafe { (api().free_notifications)(info) };
}

pub unsafe fn notify(
    engine: NativeEngine,
    target: SegmentId,
    name: *const c_char,
    message: *const c_char,
) -> c_int {
    unsafe { (api().send_notification)(engine, target, name, message) }
}

pub unsafe fn nic_load_stats(
    engine: NativeEngine,
    stats: *mut NicLoadStat,
    count: &mut usize,
) -> c_int {
    unsafe { (api().get_nic_load_stats)(engine, stats, count) }
}

#[cfg(test)]
#[path = "../tests/unit/lib.rs"]
mod tests;
