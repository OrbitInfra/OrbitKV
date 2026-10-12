//! Tensor and stream ownership for the engine-local Restore worker.

use std::collections::HashMap;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, mpsc};
use std::time::{Duration, Instant};

use orbitkv_channel::{
    CacheClient, GrantState, RestoreHandle, RestoreResponse, RestoreState, RestoreTiming,
};
use orbitkv_core::transfer::local::{LocalRestoreExecutor, RawRestorePart};
use pyo3::prelude::*;

const MAX_PENDING: usize = 1024;

pub(crate) static RESTORE_TIMING: LazyLock<bool> = LazyLock::new(|| {
    ["ORBITKV_TRACE_TRANSFERS", "ORBITKV_COST_OBSERVATIONS"]
        .iter()
        .any(|name| std::env::var(name).as_deref() == Ok("1"))
});

/// One engine notification source signals local DMA results, independently of
/// Manager grant/reaping notifications on the channel's own eventfd.
pub(crate) struct LocalCompletions(OwnedFd);

impl LocalCompletions {
    pub(crate) fn new() -> io::Result<Self> {
        // SAFETY: eventfd returns a new owned descriptor on success.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    pub(crate) fn fd(&self) -> i32 {
        self.0.as_raw_fd()
    }

    fn notify(&self) {
        let value = 1_u64;
        loop {
            // SAFETY: value is an initialized eight-byte eventfd counter increment.
            let written = unsafe { libc::write(self.fd(), (&value as *const u64).cast(), 8) };
            if written == 8 {
                return;
            }
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            if error.kind() == io::ErrorKind::WouldBlock {
                return;
            }
            // A lost wake would strand framework page ownership indefinitely.
            std::process::abort();
        }
    }

    pub(crate) fn wait(&self, timeout: Duration) -> io::Result<bool> {
        let start = Instant::now();
        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            let millis = remaining
                .as_millis()
                .saturating_add(u128::from(
                    !remaining.subsec_nanos().is_multiple_of(1_000_000),
                ))
                .min(i32::MAX as u128) as i32;
            let mut event = libc::pollfd {
                fd: self.fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: event points to one initialized poll descriptor.
            let result = unsafe { libc::poll(&mut event, 1, millis) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if result == 0 {
                return Ok(false);
            }
            if event.revents & libc::POLLIN == 0 {
                return Err(io::Error::other(
                    "local Restore completion descriptor failed",
                ));
            }
            let mut count = 0_u64;
            // SAFETY: count is writable eight-byte storage for the eventfd read.
            unsafe { libc::read(self.fd(), (&mut count as *mut u64).cast(), 8) };
            return Ok(true);
        }
    }
}

enum RestoreResult {
    Pending { enqueued: bool },
    Ready(RestoreResponse, Option<(RestoreHandle, Instant)>),
    Consumed,
}

impl RestoreResult {
    fn take(&mut self) -> Result<Option<RestoreResponse>, &'static str> {
        match std::mem::replace(self, Self::Consumed) {
            pending @ Self::Pending { .. } => {
                *self = pending;
                Ok(None)
            }
            Self::Ready(response, observed) => {
                if let Some((handle, drained_at)) = observed {
                    orbitkv_common::timeline::record_restore(
                        "local_restore_observed",
                        orbitkv_common::timeline::RestoreTimelineFields::operation(
                            handle.session_epoch,
                            handle.session_token,
                            handle.operation_id,
                        )
                        .elapsed_ns(drained_at.elapsed().as_nanos() as u64)
                        .success(response.state == RestoreState::Succeeded),
                    );
                }
                Ok(Some(response))
            }
            Self::Consumed => Err("Restore result was already consumed"),
        }
    }
}

pub(crate) struct LocalRestore {
    result: Mutex<RestoreResult>,
    completed: Condvar,
}

impl LocalRestore {
    fn mark_enqueued(&self) {
        let mut result = self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let RestoreResult::Pending { enqueued } = &mut *result {
            *enqueued = true;
        }
        self.completed.notify_all();
    }

    pub(crate) fn wait_enqueued(&self, timeout: Duration) -> Result<bool, String> {
        let result = self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (result, _) = self
            .completed
            .wait_timeout_while(result, timeout, |result| {
                matches!(result, RestoreResult::Pending { enqueued: false })
            })
            .unwrap_or_else(|poison| poison.into_inner());
        match &*result {
            RestoreResult::Ready(response, _) if response.state != RestoreState::Succeeded => {
                Err(response.message.clone())
            }
            RestoreResult::Consumed => Err("Restore result was already consumed".into()),
            RestoreResult::Pending { enqueued } => Ok(*enqueued),
            RestoreResult::Ready(_, _) => Ok(true),
        }
    }

    pub(crate) fn poll(&self) -> Result<Option<RestoreResponse>, &'static str> {
        self.result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
    }

    pub(crate) fn wait(&self, timeout: Duration) -> Result<Option<RestoreResponse>, &'static str> {
        let result = self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (mut result, _) = self
            .completed
            .wait_timeout_while(result, timeout, |result| {
                matches!(result, RestoreResult::Pending { .. })
            })
            .unwrap_or_else(|poison| poison.into_inner());
        result.take()
    }
}

struct Job {
    handle: RestoreHandle,
    result: Arc<LocalRestore>,
    timing: Option<(Instant, RestoreTiming)>,
    layer_events: HashMap<String, u64>,
    _layer_event_owners: Vec<Py<PyAny>>,
}

/// Actual tensor exporters and CUDA mappings remain owned by the worker until
/// every accepted operation drains, even when its Python handle is dropped.
pub(crate) struct LocalRestoreWorker {
    pub(crate) manager_device_id: i32,
    executor: Arc<Mutex<Option<LocalRestoreExecutor>>>,
    jobs: mpsc::SyncSender<Job>,
    admission: Arc<LocalCompletions>,
    pending: Arc<AtomicUsize>,
    idle: Arc<(Mutex<()>, Condvar)>,
    arena_count: usize,
    stopping: Arc<AtomicBool>,
    released: Arc<(Mutex<bool>, Condvar)>,
}

impl LocalRestoreWorker {
    pub(crate) fn new(
        client: Arc<CacheClient>,
        executor: LocalRestoreExecutor,
        tensors: Vec<Py<PyAny>>,
        completions: Arc<LocalCompletions>,
        arena_count: usize,
        manager_device_id: i32,
    ) -> Result<Self, String> {
        let executor = Arc::new(Mutex::new(Some(executor)));
        let worker_executor = Arc::clone(&executor);
        let pending = Arc::new(AtomicUsize::new(0));
        let worker_pending = Arc::clone(&pending);
        let idle = Arc::new((Mutex::new(()), Condvar::new()));
        let worker_idle = Arc::clone(&idle);
        let (jobs, receiver) = mpsc::sync_channel::<Job>(MAX_PENDING);
        let admission = Arc::new(LocalCompletions::new().map_err(|error| error.to_string())?);
        let worker_admission = Arc::clone(&admission);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = Arc::clone(&stopping);
        let released = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_released = Arc::clone(&released);
        std::thread::Builder::new()
            .name("orbitkv-local-restore".into())
            .spawn(move || {
                let mut retirement = Vec::new();
                let mut active: Vec<Job> = Vec::new();
                let mut disconnected = false;
                loop {
                    loop {
                        match receiver.try_recv() {
                            Ok(mut job) => {
                                if let Some((start, timing)) = &mut job.timing {
                                    timing.dequeued_ns = start.elapsed().as_nanos() as u64;
                                }
                                active.push(job);
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                disconnected = true;
                                break;
                            }
                        }
                    }
                    let mut index = 0;
                    while index < active.len() {
                        let Some((outcome, retired)) =
                            advance(&client, &worker_executor, &mut active[index])
                        else {
                            index += 1;
                            continue;
                        };
                        let job = active.swap_remove(index);
                        if let Some(retired) = retired {
                            retirement.push(retired);
                        }
                        *job.result
                            .result
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner()) = RestoreResult::Ready(
                            outcome,
                            job.timing.and_then(|(start, timing)| {
                                (*orbitkv_common::timeline::ENABLED && timing.drained_ns != 0).then(
                                    || {
                                        (
                                            job.handle,
                                            start + Duration::from_nanos(timing.drained_ns),
                                        )
                                    },
                                )
                            }),
                        );
                        job.result.completed.notify_all();
                        completions.notify();
                        let _idle = worker_idle
                            .0
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner());
                        worker_pending.fetch_sub(1, Ordering::AcqRel);
                        worker_idle.1.notify_all();
                    }
                    reap(&client, &mut retirement);
                    if (disconnected || worker_stopping.load(Ordering::Acquire))
                        && active.is_empty()
                    {
                        break;
                    }
                    wait_for_updates(
                        &client,
                        &worker_admission,
                        !active.is_empty() || !retirement.is_empty(),
                    );
                }
                // Reaping is no longer a GPU fence: release tensor/CUDA owners first.
                // PyO3 defers decrefs until the next thread attaches to Python.
                drop(
                    worker_executor
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .take(),
                );
                drop(tensors);
                *worker_released
                    .0
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()) = true;
                worker_released.1.notify_all();
                while !retirement.is_empty() {
                    reap(&client, &mut retirement);
                    wait_for_updates(&client, &worker_admission, true);
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self {
            manager_device_id,
            executor,
            jobs,
            admission,
            pending,
            idle,
            arena_count,
            stopping,
            released,
        })
    }

    pub(crate) fn arena_count(&self) -> usize {
        self.arena_count
    }

    /// Reserve native ownership before asking the Manager to prepare a grant.
    pub(crate) fn reserve(
        &self,
        ready_stream: u64,
        layer_events: &HashMap<String, u64>,
    ) -> Result<(), String> {
        self.pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_PENDING).then_some(count + 1)
            })
            .map_err(|_| "engine-local Restore queue is full")?;
        let ready = self
            .executor
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_mut()
            .ok_or_else(|| "local Restore worker is stopped".to_string())
            .and_then(|executor| {
                executor.validate_layer_events(layer_events)?;
                executor.wait_for_destination(ready_stream)
            });
        if ready.is_err() {
            self.cancel_reservation();
        }
        ready
    }

    pub(crate) fn cancel_reservation(&self) {
        let _idle = self
            .idle
            .0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.pending.fetch_sub(1, Ordering::AcqRel);
        self.idle.1.notify_all();
    }

    pub(crate) fn wait_drained(&self) {
        let mut idle = self
            .idle
            .0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while self.pending.load(Ordering::Acquire) != 0 {
            idle = self
                .idle
                .1
                .wait(idle)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }

    pub(crate) fn shutdown(&self) {
        self.wait_drained();
        self.stopping.store(true, Ordering::Release);
        self.admission.notify();
        let mut released = self
            .released
            .0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while !*released {
            released = self
                .released
                .1
                .wait(released)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }

    pub(crate) fn submit(
        &self,
        handle: RestoreHandle,
        mut timing: Option<(Instant, RestoreTiming)>,
        layer_events: HashMap<String, u64>,
        layer_event_owners: Vec<Py<PyAny>>,
    ) -> Arc<LocalRestore> {
        if let Some((start, timing)) = &mut timing {
            timing.dispatched_ns = start.elapsed().as_nanos() as u64;
        }
        let result = Arc::new(LocalRestore {
            result: Mutex::new(RestoreResult::Pending { enqueued: false }),
            completed: Condvar::new(),
        });
        if self
            .jobs
            .send(Job {
                handle,
                result: Arc::clone(&result),
                timing,
                layer_events,
                _layer_event_owners: layer_event_owners,
            })
            .is_err()
        {
            // Losing the sole worker can lose unproven DMA. Fail closed.
            std::process::abort();
        }
        self.admission.notify();
        result
    }
}

impl Drop for LocalRestoreWorker {
    fn drop(&mut self) {
        // A Python destructor may hold the GIL. Signal only; the worker retains
        // every accepted operation and releases resources after draining them.
        self.stopping.store(true, Ordering::Release);
        self.admission.notify();
    }
}

fn advance(
    client: &CacheClient,
    executor: &Mutex<Option<LocalRestoreExecutor>>,
    job: &mut Job,
) -> Option<(RestoreResponse, Option<RestoreHandle>)> {
    let handle = job.handle;
    let failed = |message: String| RestoreResponse {
        operation_id: handle.operation_id,
        state: RestoreState::Failed,
        message,
    };
    match client.claim_local_restore(handle) {
        Ok(Some(bytes)) => {
            if let Some((start, timing)) = &mut job.timing
                && timing.claimed_ns == 0
            {
                timing.claimed_ns = start.elapsed().as_nanos() as u64;
            }
            let mut submitted_at = None;
            let final_part = matches!(
                client.restore_completions().state(handle.operation_id),
                Ok(GrantState::Active)
            );
            let result = RawRestorePart::decode(&bytes).and_then(|plan| {
                executor
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .as_mut()
                    .ok_or_else(|| "local Restore worker stopped before its operation".to_string())?
                    .execute(
                        &plan,
                        &mut job.layer_events,
                        final_part,
                        || job.result.mark_enqueued(),
                        job.timing.as_ref().map(|_| &mut submitted_at),
                    )
            });
            let timing = job.timing.as_mut().map(|(start, timing)| {
                if timing.submitted_ns == 0 {
                    timing.submitted_ns =
                        submitted_at.map_or(0, |at| at.duration_since(*start).as_nanos() as u64);
                }
                timing.drained_ns = start.elapsed().as_nanos() as u64;
                *timing
            });
            let response = match &result {
                Ok(()) => RestoreResponse {
                    operation_id: handle.operation_id,
                    state: RestoreState::Succeeded,
                    message: String::new(),
                },
                Err(message) => failed(message.clone()),
            };
            // Reaping releases source credits asynchronously, off the GPU fence.
            match client.finish_local_restore(handle, result, timing) {
                Ok(false) => None,
                Ok(true) => Some((response, Some(handle))),
                Err(error) => Some((failed(error.to_string()), Some(handle))),
            }
        }
        Ok(None) => match client.poll_restore(handle) {
            Ok(response) if response.state != RestoreState::Pending => {
                if response.state == RestoreState::Succeeded {
                    // Managed SSD/codec execution supplies only a whole-operation fence.
                    // Publish fresh events after that fence without moving its I/O ownership.
                    let empty = RawRestorePart {
                        copies: Vec::new(),
                        completed_layers: Vec::new(),
                    };
                    let result = executor
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .as_mut()
                        .ok_or_else(|| "local Restore worker stopped".to_string())
                        .and_then(|executor| {
                            executor.execute(
                                &empty,
                                &mut job.layer_events,
                                true,
                                || job.result.mark_enqueued(),
                                None,
                            )
                        });
                    if let Err(error) = result {
                        return Some((failed(error), None));
                    }
                }
                Some((response, None))
            }
            Ok(_) => None,
            Err(error) => Some((failed(error.to_string()), None)),
        },
        Err(error) => {
            // Decoding shared plan bytes can fail after Granted -> Active; no
            // GPU work was submitted here, so publishing Drained is safe.
            if matches!(
                client.restore_completions().state(handle.operation_id),
                Ok(GrantState::Active | GrantState::ActiveMore)
            ) {
                let _ = client.finish_local_restore(handle, Err(error.to_string()), None);
            }
            // A failed claim can lose the CAS to revocation. Keep its record
            // until Reaped -> Acknowledged even though no DMA was submitted.
            Some((failed(error.to_string()), Some(handle)))
        }
    }
}

fn reap(client: &CacheClient, retirement: &mut Vec<RestoreHandle>) {
    retirement.retain(|&handle| matches!(client.poll_restore(handle), Ok(response) if response.state == RestoreState::Pending));
}

fn wait_for_updates(client: &CacheClient, admission: &LocalCompletions, pending: bool) {
    let mut events = [
        libc::pollfd {
            fd: client.channel().notification_fd(),
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: admission.fd(),
            events: libc::POLLIN,
            revents: 0,
        },
    ];
    // The timeout covers peer death or another registered worker consuming a
    // shared notification. Normal admission and grant changes wake immediately.
    // SAFETY: events contains two initialized, live descriptors.
    let ready = if pending {
        unsafe { libc::poll(events.as_mut_ptr(), events.len() as libc::nfds_t, 50) }
    } else {
        // No operation or retirement owns Manager state: sleep until admission
        // or shutdown without periodic wakeups.
        unsafe { libc::poll(&mut events[1], 1, -1) }
    };
    if ready > 0 {
        for event in events {
            if event.revents & libc::POLLIN != 0 {
                let mut count = 0_u64;
                // SAFETY: eventfd consumes exactly eight bytes into valid storage.
                unsafe { libc::read(event.fd, (&mut count as *mut u64).cast(), 8) };
            }
        }
    }
}
