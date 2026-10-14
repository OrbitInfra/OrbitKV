//! OrbitKV-facing Mooncake TENT lifecycle and batch execution.
//!
//! OrbitKV owns state identity, leases, generation checks, and transfer plans.
//! Mooncake exclusively owns transport mechanics: registered segments, RDMA/TCP
//! selection, rails, endpoints, retries, and batch completion.

use std::collections::HashMap;
use std::ffi::{CString, OsString, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::ptr::NonNull;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::metrics::{Counter, UpDownCounter};
use orbitkv_mooncake_sys as native;

use crate::error::{MooncakeError, Result};
use crate::notification::{NotificationMailbox, NotificationMatch};
use crate::types::{
    INVALID_BATCH, NicLoadStat, Notification, STATUS_CANCELED, STATUS_COMPLETED, STATUS_FAILED,
    STATUS_INVALID, STATUS_PENDING, STATUS_TIMEOUT, STATUS_WAITING, TransferOp, TransferSlice,
};

static ENGINE_CREATE_LOCK: Mutex<()> = Mutex::new(());
const MAX_NATIVE_BATCHES: usize = 1024;
const MAX_NATIVE_REQUEST_BYTES: usize = 32 * 1024 * 1024;

struct TransferMetrics {
    active: UpDownCounter<i64>,
    request_bytes: UpDownCounter<i64>,
    retained: UpDownCounter<i64>,
    retained_bytes: UpDownCounter<i64>,
    rejections: Counter<u64>,
}

static TRANSFER_METRICS: LazyLock<TransferMetrics> = LazyLock::new(|| {
    let meter = opentelemetry::global::meter("orbitkv-transfer");
    TransferMetrics {
        active: meter
            .i64_up_down_counter("orbitkv_native_transfer_batches")
            .build(),
        request_bytes: meter
            .i64_up_down_counter("orbitkv_native_transfer_request_bytes")
            .build(),
        retained: meter
            .i64_up_down_counter("orbitkv_native_transfer_retained_batches")
            .build(),
        retained_bytes: meter
            .i64_up_down_counter("orbitkv_native_transfer_retained_descriptor_bytes")
            .build(),
        rejections: meter
            .u64_counter("orbitkv_native_transfer_admission_rejections")
            .build(),
    }
});

#[derive(Default)]
struct BatchAdmission {
    closed: bool,
    active: usize,
    retained: usize,
    request_bytes: usize,
    logical_bytes: i64,
}

struct AdmittedBatch<'a> {
    admission: &'a Mutex<BatchAdmission>,
    descriptor_bytes: i64,
    request_bytes: usize,
    retained: bool,
}

impl BatchAdmission {
    fn acquire(
        admission: &Mutex<Self>,
        tasks: usize,
        descriptor_bytes: u64,
    ) -> Result<AdmittedBatch<'_>> {
        let bytes =
            i64::try_from(descriptor_bytes).map_err(|_| MooncakeError::BatchAdmissionExhausted)?;
        let request_bytes = tasks
            .checked_mul(std::mem::size_of::<native::TransferRequest>())
            .ok_or(MooncakeError::BatchAdmissionExhausted)?;
        let mut state = admission.lock().unwrap_or_else(|error| error.into_inner());
        let logical_bytes = state.logical_bytes.checked_add(bytes);
        let total_request_bytes = state.request_bytes.checked_add(request_bytes);
        if state.closed
            || state.active >= MAX_NATIVE_BATCHES
            || logical_bytes.is_none()
            || total_request_bytes.is_none_or(|total| total > MAX_NATIVE_REQUEST_BYTES)
        {
            TRANSFER_METRICS.rejections.add(1, &[]);
            return Err(MooncakeError::BatchAdmissionExhausted);
        }
        state.active += 1;
        state.request_bytes = total_request_bytes.ok_or(MooncakeError::BatchAdmissionExhausted)?;
        state.logical_bytes = logical_bytes.ok_or(MooncakeError::BatchAdmissionExhausted)?;
        TRANSFER_METRICS.active.add(1, &[]);
        TRANSFER_METRICS
            .request_bytes
            .add(request_bytes as i64, &[]);
        Ok(AdmittedBatch {
            admission,
            descriptor_bytes: bytes,
            request_bytes,
            retained: false,
        })
    }
}

impl AdmittedBatch<'_> {
    fn retain(&mut self) {
        if !self.retained {
            self.retained = true;
            self.admission
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .retained += 1;
            TRANSFER_METRICS.retained.add(1, &[]);
            TRANSFER_METRICS
                .retained_bytes
                .add(self.descriptor_bytes, &[]);
        }
    }
}

impl Drop for AdmittedBatch<'_> {
    fn drop(&mut self) {
        let mut state = self
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.active -= 1;
        state.request_bytes -= self.request_bytes;
        state.logical_bytes -= self.descriptor_bytes;
        TRANSFER_METRICS.active.add(-1, &[]);
        TRANSFER_METRICS
            .request_bytes
            .add(-(self.request_bytes as i64), &[]);
        if self.retained {
            state.retained -= 1;
            TRANSFER_METRICS.retained.add(-1, &[]);
            TRANSFER_METRICS
                .retained_bytes
                .add(-self.descriptor_bytes, &[]);
        }
    }
}

pub struct TransferEngine {
    native: native::NativeEngine,
    source_epoch: String,
    segments: Mutex<HashMap<String, native::SegmentId>>,
    notification_poll: Mutex<()>,
    notification_mailbox: Mutex<NotificationMailbox>,
    admission: Mutex<BatchAdmission>,
}

/// Owns one Mooncake memory registration. The caller must retain the actual
/// CPU/GPU allocation separately until this token is dropped or unregistered.
pub struct MemoryRegistration {
    engine: Arc<TransferEngine>,
    address: NonNull<u8>,
    length: usize,
    active: bool,
}

// SAFETY: The token never dereferences the address. Its owner guarantees the
// registered allocation remains valid, and TransferEngine serializes native state.
unsafe impl Send for MemoryRegistration {}
unsafe impl Sync for MemoryRegistration {}

impl MemoryRegistration {
    pub fn address(&self) -> NonNull<u8> {
        self.address
    }

    pub fn try_unregister(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let result = unsafe { self.engine.unregister_memory(self.address, self.length) };
        if result.is_ok() {
            self.active = false;
        }
        result
    }

    pub fn unregister(mut self) -> Result<()> {
        self.try_unregister()
    }
}

impl Drop for MemoryRegistration {
    fn drop(&mut self) {
        if self.active {
            let _ = unsafe { self.engine.unregister_memory(self.address, self.length) };
        }
    }
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new(
        metadata: &str,
        local_server_name: &str,
        bind_host: &str,
        rpc_port: u64,
        nic_filter: &[String],
    ) -> Result<Self> {
        native::ensure_loaded().map_err(MooncakeError::NativeRuntime)?;
        // The upstream C ABI takes its NIC filter from a process-global
        // environment variable. Serialize initialization and restore the
        // caller's environment so multiple engines cannot leak filters into
        // one another.
        let _create_guard = ENGINE_CREATE_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _filter_guard = ScopedNicFilter::apply(nic_filter)?;
        let metadata = CString::new(metadata)?;
        let local_server_name = CString::new(local_server_name)?;
        let bind_host = CString::new(bind_host)?;
        let native = unsafe {
            native::create(
                metadata.as_ptr(),
                local_server_name.as_ptr(),
                bind_host.as_ptr(),
                rpc_port,
            )
        };
        if native.is_null() {
            return Err(MooncakeError::Create);
        }
        let mut epoch = [0 as c_char; 33];
        if let Err(error) = check("sourceEpoch", unsafe {
            native::source_epoch(native, &mut epoch)
        }) {
            unsafe { native::destroy(native) };
            return Err(error);
        }
        let source_epoch = fixed_c_string(&epoch);
        if let Err(error) = source_epoch_bytes(&source_epoch) {
            unsafe { native::destroy(native) };
            return Err(error);
        }
        Ok(Self {
            native,
            source_epoch,
            segments: Mutex::new(HashMap::new()),
            notification_poll: Mutex::new(()),
            notification_mailbox: Mutex::new(NotificationMailbox::default()),
            admission: Mutex::new(BatchAdmission::default()),
        })
    }

    pub fn local_segment_name(&self) -> Result<String> {
        let mut output = [0 as c_char; 256];
        check("getLocalIpAndPort", unsafe {
            native::local_ip_and_port(self.native, output.as_mut_ptr(), output.len())
        })?;
        Ok(fixed_c_string(&output))
    }

    pub fn source_epoch(&self) -> &str {
        &self.source_epoch
    }

    /// Permanently stop admission and await source CPU/TCP pointer handlers.
    /// Device/RDMA paths and active requester batches cannot produce a receipt.
    /// On any error the caller must retain registrations and allocation owners.
    pub fn close_source(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout.min(Duration::from_secs(5));
        {
            let mut admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
            admission.closed = true;
        }
        let mut epoch = [0 as c_char; 33];
        let mut active = u64::MAX;
        let status = unsafe {
            native::close_source(
                self.native,
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_millis() as u64,
                &mut epoch,
                &mut active,
            )
        };
        check("closeSource", status)?;
        if active != 0
            || epoch[32] != 0
            || epoch[..32]
                .iter()
                .any(|c| !matches!(*c as u8, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(MooncakeError::Operation {
                operation: "closeSourceReceipt",
                status: -1,
            });
        }
        let admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        if admission.active != 0 {
            return Err(MooncakeError::Operation {
                operation: "closeSourceRequesterStillOwned",
                status: 1,
            });
        }
        Ok(())
    }

    /// Register memory under a Mooncake topology location such as `cpu:0`.
    ///
    /// # Safety
    ///
    /// The memory must remain valid and pinned until it is unregistered.
    unsafe fn register_memory(
        &self,
        address: NonNull<u8>,
        length: usize,
        location: &str,
    ) -> Result<()> {
        let admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        if admission.closed {
            return Err(MooncakeError::Operation {
                operation: "registerSourceClosed",
                status: 1,
            });
        }
        let location = CString::new(location)?;
        check("registerLocalMemory", unsafe {
            native::register_memory(
                self.native,
                address.as_ptr().cast::<c_void>(),
                length,
                location.as_ptr(),
            )
        })
    }

    /// Register memory and return a token that unregisters it on drop.
    ///
    /// # Safety
    ///
    /// The allocation must outlive the returned token and every submitted
    /// transfer using it. `location` must describe the actual memory domain,
    /// for example `cpu:0` or `cuda:0`.
    pub unsafe fn register_memory_owned(
        self: &Arc<Self>,
        address: NonNull<u8>,
        length: usize,
        location: &str,
    ) -> Result<MemoryRegistration> {
        unsafe { self.register_memory(address, length, location)? };
        Ok(MemoryRegistration {
            engine: Arc::clone(self),
            address,
            length,
            active: true,
        })
    }

    /// # Safety
    ///
    /// `address` must identify a currently registered region.
    unsafe fn unregister_memory(&self, address: NonNull<u8>, length: usize) -> Result<()> {
        check("unregisterLocalMemory", unsafe {
            native::unregister_memory(self.native, address.as_ptr().cast::<c_void>(), length)
        })
    }

    pub fn submit_and_wait(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
        expected_source_epoch: Option<&str>,
    ) -> Result<usize> {
        self.submit_and_wait_inner(
            operation,
            remote_segment,
            slices,
            timeout,
            None,
            expected_source_epoch,
        )
    }

    pub fn submit_and_notify(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
        notification: &Notification,
        expected_source_epoch: Option<&str>,
    ) -> Result<usize> {
        self.submit_and_wait_inner(
            operation,
            remote_segment,
            slices,
            timeout,
            Some(notification),
            expected_source_epoch,
        )
    }

    fn submit_and_wait_inner(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
        notification: Option<&Notification>,
        expected_source_epoch: Option<&str>,
    ) -> Result<usize> {
        if slices.is_empty() {
            return Ok(0);
        }
        let descriptor_bytes = slices.iter().try_fold(0u64, |total, slice| {
            total
                .checked_add(slice.length as u64)
                .ok_or(MooncakeError::BatchAdmissionExhausted)
        })?;
        let mut admission =
            BatchAdmission::acquire(&self.admission, slices.len(), descriptor_bytes)?;
        let notification = notification
            .map(|notification| {
                Ok::<_, MooncakeError>((
                    CString::new(notification.name.as_str())?,
                    CString::new(notification.message.as_str())?,
                ))
            })
            .transpose()?;
        let segment = self.open_segment(remote_segment)?;
        let expected_epoch = expected_source_epoch
            .map(source_epoch_bytes)
            .transpose()?
            .unwrap_or([0; 33]);
        if expected_source_epoch.is_some() {
            check("validateSourceEpochBeforeSubmit", unsafe {
                native::validate_remote_epoch(self.native, segment, &expected_epoch)
            })?;
        }
        let mut requests = slices
            .iter()
            .map(|slice| native::TransferRequest {
                opcode: operation.code(),
                source: slice.local.as_ptr().cast::<c_void>(),
                target_id: segment,
                target_offset: slice.remote_address,
                length: slice.length as u64,
                priority: 0,
                transport_hint: 0,
                expected_source_epoch: expected_epoch,
            })
            .collect::<Vec<_>>();
        let batch = unsafe { native::allocate_batch(self.native, slices.len()) };
        if batch == INVALID_BATCH {
            return Err(MooncakeError::InvalidBatch);
        }
        contain_native_unwind(|| {
            let submitted = match notification.as_ref() {
                Some((name, message)) => check("tent_submit_notif", unsafe {
                    native::submit_with_notify(
                        self.native,
                        batch,
                        &mut requests,
                        name.as_ptr(),
                        message.as_ptr(),
                    )
                }),
                None => check("tent_submit", unsafe {
                    native::submit(self.native, batch, &mut requests)
                }),
            };
            drain_batch(
                slices.len(),
                timeout,
                submitted,
                |task| {
                    let mut status = native::TransferStatus {
                        status: STATUS_WAITING,
                        transferred_bytes: 0,
                    };
                    check("getTransferStatus", unsafe {
                        native::transfer_status(self.native, batch, task, &mut status)
                    })?;
                    Ok(status)
                },
                |task| {
                    check("tent_cancel_task", unsafe {
                        native::cancel_task(self.native, batch, task)
                    })
                },
                || {
                    let mut status = native::TransferStatus {
                        status: STATUS_WAITING,
                        transferred_bytes: 0,
                    };
                    check("tent_overall_status", unsafe {
                        native::batch_status(self.native, batch, &mut status)
                    })?;
                    if status.status != STATUS_COMPLETED {
                        return Err(MooncakeError::TransferFailed {
                            task: 0,
                            state: status.status,
                        });
                    }
                    Ok(status.transferred_bytes as usize)
                },
                || unsafe { native::free_batch(self.native, batch) == 0 },
                |task, state, cause| {
                    admission.retain();
                    log::error!(
                        "TENT termination unproven; retaining native batch and caller buffers: batch={batch} task={task} state={} first_failure={cause:?} descriptor_bytes={descriptor_bytes}",
                        native_status_name(state),
                    );
                },
            )
        })
    }

    fn take_notifications_unlocked(&self) -> Result<Vec<Notification>> {
        let mut info = native::NotificationInfo {
            count: 0,
            records: std::ptr::null_mut(),
        };
        check("tent_recv_notifs", unsafe {
            native::receive_notifications(self.native, &mut info)
        })?;
        if info.count < 0 {
            return Err(MooncakeError::InvalidNotificationCount(info.count));
        }
        if info.count == 0 {
            return Ok(Vec::new());
        }
        if info.records.is_null() {
            return Err(MooncakeError::InvalidNotificationBuffer(info.count));
        }
        let records = unsafe { std::slice::from_raw_parts(info.records, info.count as usize) };
        let notifications = records
            .iter()
            .map(|record| Notification {
                name: fixed_c_string(&record.name),
                message: fixed_c_string(&record.message),
            })
            .collect();
        unsafe { native::free_notifications(&mut info) };
        Ok(notifications)
    }

    /// Opens a generation-fenced inbox for one logical notification name.
    pub fn open_notification_scope(&self, name: &str) -> Result<u64> {
        if name.is_empty() {
            return Err(MooncakeError::InvalidNotificationWait);
        }
        let _poll = self
            .notification_poll
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let notifications = self.take_notifications_unlocked()?;
        let mut mailbox = self
            .notification_mailbox
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mailbox.record(notifications);
        Ok(mailbox.open(name))
    }

    /// Closes the scope only if `generation` still owns `name`.
    pub fn close_notification_scope(&self, name: &str, generation: u64) {
        self.notification_mailbox
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .close(name, generation);
    }

    /// Waits without losing notifications consumed for other active scopes.
    pub fn wait_for_notification(
        &self,
        name: &str,
        generation: u64,
        expectations: &[(String, usize)],
        timeout: Duration,
    ) -> Result<Option<String>> {
        if name.is_empty()
            || generation == 0
            || expectations.is_empty()
            || expectations
                .iter()
                .any(|(message, count)| message.is_empty() || *count == 0)
            || timeout.is_zero()
        {
            return Err(MooncakeError::InvalidNotificationWait);
        }
        let started = Instant::now();
        loop {
            match self
                .notification_mailbox
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .status(name, generation, expectations)
            {
                NotificationMatch::Matched(message) => return Ok(Some(message)),
                NotificationMatch::Closed => return Ok(None),
                NotificationMatch::Pending => {}
            }

            let poll_result = {
                let _poll = self
                    .notification_poll
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                self.take_notifications_unlocked()
            };
            let notifications = match poll_result {
                Ok(notifications) => notifications,
                Err(error) => {
                    return match self
                        .notification_mailbox
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .status(name, generation, expectations)
                    {
                        NotificationMatch::Matched(message) => Ok(Some(message)),
                        NotificationMatch::Closed => Ok(None),
                        NotificationMatch::Pending => Err(error),
                    };
                }
            };
            if !notifications.is_empty() {
                self.notification_mailbox
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .record(notifications);
                continue;
            }
            if started.elapsed() >= timeout {
                return Err(MooncakeError::NotificationTimeout(name.to_string()));
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    pub fn send_notification(
        &self,
        remote_segment: &str,
        notification: &Notification,
    ) -> Result<()> {
        let segment = self.open_segment(remote_segment)?;
        let name = CString::new(notification.name.as_str())?;
        let message = CString::new(notification.message.as_str())?;
        check("tent_send_notifs", unsafe {
            native::notify(self.native, segment, name.as_ptr(), message.as_ptr())
        })
    }

    /// Snapshot live TENT RDMA rail pressure. An empty result means no RDMA
    /// rail is active; it does not prove that a particular batch used TCP.
    pub fn nic_load_stats(&self) -> Result<Vec<NicLoadStat>> {
        const INITIAL_CAPACITY: usize = 16;
        let empty = native::NicLoadStat {
            device_name: [0; 64],
            inflight_bytes: 0,
            ewma_bandwidth_bps: 0.0,
        };
        let mut stats = vec![empty; INITIAL_CAPACITY];
        loop {
            let mut count = stats.len();
            check("tent_get_nic_load_stats", unsafe {
                native::nic_load_stats(self.native, stats.as_mut_ptr(), &mut count)
            })?;
            if count > stats.len() {
                stats.resize(count, empty);
                continue;
            }
            stats.truncate(count);
            return stats
                .iter()
                .map(|stat| {
                    let device_name = fixed_c_string(&stat.device_name);
                    Ok(NicLoadStat {
                        device_name,
                        inflight_bytes: stat.inflight_bytes,
                        ewma_bandwidth_bps: stat.ewma_bandwidth_bps,
                    })
                })
                .collect();
        }
    }

    fn open_segment(&self, remote_segment: &str) -> Result<native::SegmentId> {
        let mut segments = self
            .segments
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(segment) = segments.get(remote_segment) {
            return Ok(*segment);
        }
        let name = CString::new(remote_segment)?;
        let segment = unsafe { native::open_segment(self.native, name.as_ptr()) };
        if segment == native::INVALID_SEGMENT {
            return Err(MooncakeError::Operation {
                operation: "tent_open_segment",
                status: -1,
            });
        }
        segments.insert(remote_segment.to_string(), segment);
        Ok(segment)
    }

    /// Drop a cached peer segment after a failed transfer or peer restart.
    pub fn invalidate_segment(&self, remote_segment: &str) {
        let segment = self
            .segments
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(remote_segment);
        if let Some(segment) = segment {
            let _ = unsafe { native::close_segment(self.native, segment) };
        }
    }
}

fn source_epoch_bytes(value: &str) -> Result<[c_char; 33]> {
    if value.len() != 32
        || value
            .bytes()
            .any(|byte| !matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(MooncakeError::Operation {
            operation: "invalidSourceEpoch",
            status: -1,
        });
    }
    let mut output = [0 as c_char; 33];
    for (out, byte) in output.iter_mut().zip(value.bytes()) {
        *out = byte as c_char;
    }
    Ok(output)
}

fn fixed_c_string(value: &[c_char]) -> String {
    let length = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    let bytes: Vec<_> = value[..length].iter().map(|byte| *byte as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn set_nic_filter(nics: &[String]) -> Result<()> {
    if nics.is_empty() {
        return Ok(());
    }
    let value = CString::new(nics.join(","))?;
    if unsafe { libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1) } == 0 {
        Ok(())
    } else {
        Err(MooncakeError::Operation {
            operation: "setenv(MC_TE_FILTERS)",
            status: -1,
        })
    }
}

struct ScopedNicFilter {
    previous: Option<OsString>,
    changed: bool,
}

impl ScopedNicFilter {
    fn apply(nics: &[String]) -> Result<Self> {
        let previous = std::env::var_os("MC_TE_FILTERS");
        let changed = !nics.is_empty();
        if changed {
            set_nic_filter(nics)?;
        }
        Ok(Self { previous, changed })
    }
}

impl Drop for ScopedNicFilter {
    fn drop(&mut self) {
        if !self.changed {
            return;
        }
        match self.previous.as_ref() {
            Some(previous) => {
                if let Ok(value) = CString::new(previous.as_bytes()) {
                    unsafe {
                        libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1);
                    }
                }
            }
            None => unsafe {
                libc::unsetenv(c"MC_TE_FILTERS".as_ptr());
            },
        }
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        if let Ok(segments) = self.segments.get_mut() {
            for segment in segments.values() {
                let _ = unsafe { native::close_segment(self.native, *segment) };
            }
        }
        unsafe { native::destroy(self.native) };
    }
}

// Callers keep all allocation, registration, descriptor and completion owners
// outside this unwind boundary; the closure borrows them until native drain.
fn contain_native_unwind<R>(operation: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(_panic) => {
            let message =
                b"OrbitKV: panic after native allocation; retaining all transfer owners\n";
            unsafe {
                libc::write(libc::STDERR_FILENO, message.as_ptr().cast(), message.len());
            }
            hold_without_termination();
        }
    }
}

fn hold_without_termination() -> ! {
    loop {
        unsafe {
            libc::poll(std::ptr::null_mut(), 0, 60_000);
        }
    }
}

fn native_status_name(state: i32) -> &'static str {
    match state {
        STATUS_WAITING => "waiting",
        STATUS_PENDING => "pending",
        STATUS_COMPLETED => "completed",
        STATUS_INVALID => "invalid",
        STATUS_CANCELED => "canceled",
        STATUS_TIMEOUT => "timeout",
        STATUS_FAILED => "failed",
        _ => "unknown",
    }
}

/// TENT submission can partially succeed. Cancellation is best effort, so keep
/// descriptors and caller-owned memory alive until every task completes.
/// Native failure/cancellation/invalid/unknown statuses are not physical drain
/// evidence in the pinned C ABI. Retain the calling owner after observing one.
#[allow(
    clippy::too_many_arguments,
    reason = "native batch callbacks stay with their resource owner"
)]
fn drain_batch(
    tasks: usize,
    timeout: Duration,
    submitted: Result<()>,
    mut poll: impl FnMut(usize) -> Result<native::TransferStatus>,
    mut cancel: impl FnMut(usize) -> Result<()>,
    mut completed_bytes: impl FnMut() -> Result<usize>,
    mut free: impl FnMut() -> bool,
    mut retain: impl FnMut(usize, i32, Option<&MooncakeError>),
) -> Result<usize> {
    let started = Instant::now();
    let mut completed = vec![false; tasks];
    let mut total = None;
    let mut failure = submitted.err();
    let mut timed_out = false;
    let mut cancel_sent = vec![false; tasks];
    let mut retention_reported = false;
    loop {
        if !timed_out && started.elapsed() >= timeout {
            timed_out = true;
        }
        let mut unproven = None;
        for (task, done) in completed.iter_mut().enumerate() {
            if *done {
                continue;
            }
            match poll(task) {
                Ok(status) if status.status == STATUS_COMPLETED => {
                    *done = true;
                }
                Ok(status) if matches!(status.status, STATUS_WAITING | STATUS_PENDING) => {}
                Ok(status) if status.status == STATUS_TIMEOUT => {
                    timed_out = true;
                    unproven.get_or_insert((task, status.status));
                }
                Ok(status) => {
                    failure.get_or_insert(MooncakeError::TransferFailed {
                        task,
                        state: status.status,
                    });
                    unproven.get_or_insert((task, status.status));
                }
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
            if !*done && (timed_out || failure.is_some()) && !cancel_sent[task] {
                if let Err(error) = cancel(task) {
                    failure.get_or_insert(error);
                }
                cancel_sent[task] = true;
            }
        }
        if let Some((task, state)) = unproven {
            // This callback only observes retention. It cannot authorize
            // reclamation; the stack owns native descriptors and caller buffers.
            retain(task, state, failure.as_ref());
            hold_without_termination();
        }
        if !retention_reported && (timed_out || failure.is_some()) {
            retention_reported = true;
            let task = completed.iter().position(|done| !done).unwrap_or(0);
            retain(task, STATUS_PENDING, failure.as_ref());
        }
        if completed.iter().all(|done| *done) {
            // TENT can merge public descriptors into one physical task. Its
            // per-task counters then overlap; only the batch counter is additive.
            if total.is_none() && failure.is_none() && !timed_out {
                match completed_bytes() {
                    Ok(bytes) => total = Some(bytes),
                    Err(error) => failure = Some(error),
                }
            }
            if free() {
                return match failure {
                    Some(error) => Err(error),
                    None if timed_out => Err(MooncakeError::Timeout),
                    None => Ok(total.unwrap_or(0)),
                };
            }
        }
        if started.elapsed() < Duration::from_millis(1) {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn check(operation: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(MooncakeError::Operation { operation, status })
    }
}

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
