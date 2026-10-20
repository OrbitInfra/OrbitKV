use crate::{OrbitKVError, u64_to_usize};

use orbitkv_transfer::{
    AUTO_MEMORY_LOCATION, Notification, P2P_METADATA, TransferEngine, TransferError, TransferOp,
    TransferSlice,
};
use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyByteArray, PyDict, PyMemoryView},
};
use std::{
    collections::{HashMap, HashSet},
    ptr::NonNull,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

const MAX_REGIONS: usize = 1024;
const MAX_REGISTERED_BYTES: usize = 1 << 40;
static REGISTERED_MEMORY: Mutex<(usize, usize)> = Mutex::new((0, 0));

struct RegisteredBytes(usize);

impl RegisteredBytes {
    fn acquire(length: usize) -> PyResult<Self> {
        let mut budget = REGISTERED_MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let total = budget.1.checked_add(length);
        if budget.0 >= MAX_REGIONS || total.is_none_or(|n| n > MAX_REGISTERED_BYTES) {
            return Err(transfer_error(
                "register memory failed",
                "memory admission exhausted",
            ));
        }
        budget.0 += 1;
        budget.1 =
            total.ok_or_else(|| transfer_error("register memory failed", "byte overflow"))?;
        Ok(Self(length))
    }
}

impl Drop for RegisteredBytes {
    fn drop(&mut self) {
        let mut budget = REGISTERED_MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        budget.0 -= 1;
        budget.1 -= self.0;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RegistrationState {
    Attempted,
    Registered,
    Unregistered,
}

struct RegisteredRegion {
    // Release the export before returning its admission.
    buffer: Option<PyBuffer<u8>>,
    _admission: RegisteredBytes,
    length: usize,
    state: RegistrationState,
}

struct RetainedMemory {
    _engine: Arc<TransferEngine>,
    _regions: HashMap<u64, RegisteredRegion>,
}

static RETAINED_MEMORY: Mutex<Vec<RetainedMemory>> = Mutex::new(Vec::new());

fn transfer_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    OrbitKVError::new_err(format!("{context}: {error}"))
}

fn py_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn pointer(value: u64, field: &str) -> PyResult<NonNull<u8>> {
    NonNull::new(value as *mut u8)
        .ok_or_else(|| PyValueError::new_err(format!("{field} must be non-zero")))
}

#[pyclass(name = "MooncakeTransferEngine")]
struct PyMooncakeTransferEngine {
    engine: Arc<TransferEngine>,
    endpoint: String,
    registrations: Mutex<HashMap<u64, RegisteredRegion>>,
    retiring: AtomicBool,
}

#[pymethods]
impl PyMooncakeTransferEngine {
    #[new]
    #[pyo3(signature = (*, bind_host, nics=Vec::new()))]
    fn new(bind_host: String, nics: Vec<String>) -> PyResult<Self> {
        if bind_host.is_empty() {
            return Err(PyValueError::new_err("bind_host must not be empty"));
        }
        let local_server_name = format!("{bind_host}:0");
        let engine = TransferEngine::new(P2P_METADATA, &local_server_name, &bind_host, 0, &nics)
            .map_err(|error| transfer_error("Mooncake TENT init failed", error))?;
        let endpoint = engine
            .local_segment_name()
            .map_err(|error| transfer_error("read Mooncake endpoint failed", error))?;
        Ok(Self {
            engine: Arc::new(engine),
            endpoint,
            registrations: Mutex::new(HashMap::new()),
            retiring: AtomicBool::new(false),
        })
    }

    #[getter]
    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Retain writable contiguous byte views of built-in bytearrays.
    /// Caller mutations must remain ordered after synchronous transfers.
    fn register_memory(
        &self,
        py: Python<'_>,
        buffers: Vec<Py<PyMemoryView>>,
    ) -> PyResult<Vec<u64>> {
        if buffers.len() > MAX_REGIONS {
            return Err(PyValueError::new_err("too many memory regions"));
        }
        let mut parsed = Vec::with_capacity(buffers.len());
        for view in buffers {
            let view = view.bind(py);
            let backing = view.getattr("obj")?;
            if !backing.is_exact_instance_of::<PyByteArray>() {
                return Err(PyValueError::new_err(
                    "memory views must export built-in bytearray storage",
                ));
            }
            let buffer = PyBuffer::<u8>::get(view.as_any())?;
            if buffer.readonly() || !buffer.is_c_contiguous() || buffer.len_bytes() == 0 {
                return Err(PyValueError::new_err(
                    "memory views must be nonempty writable contiguous bytes",
                ));
            }
            let address = buffer.buf_ptr() as u64;
            let length = buffer.len_bytes();
            let capacity = backing.call_method0("__alloc__")?.extract::<usize>()?;
            parsed.push((
                address,
                "cpu:0".to_string(),
                RegisteredRegion {
                    buffer: Some(buffer),
                    _admission: RegisteredBytes::acquire(capacity)?,
                    length,
                    state: RegistrationState::Attempted,
                },
            ));
        }
        let addresses = parsed.iter().map(|(address, _, _)| *address).collect();
        self.register_regions(py, parsed)?;
        Ok(addresses)
    }

    /// Experimental borrowed pointers: the caller retains all allocations,
    /// including after registration failure, engine GC or unknown native drain.
    fn register_memory_borrowed(&self, py: Python<'_>, regions: Vec<Py<PyDict>>) -> PyResult<()> {
        if regions.len() > MAX_REGIONS {
            return Err(PyValueError::new_err("too many memory regions"));
        }
        let mut parsed = Vec::with_capacity(regions.len());
        for region in regions {
            let region = region.bind(py);
            let address = py_get(region, "addr")?;
            let length = u64_to_usize(py_get(region, "len")?, "memory region len")?;
            if length == 0 {
                return Err(PyValueError::new_err("memory region len must be positive"));
            }
            let location = region
                .get_item("location")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| AUTO_MEMORY_LOCATION.to_string());
            pointer(address, "addr")?;
            parsed.push((
                address,
                location,
                RegisteredRegion {
                    buffer: None,
                    _admission: RegisteredBytes::acquire(length)?,
                    length,
                    state: RegistrationState::Attempted,
                },
            ));
        }
        self.register_regions(py, parsed)
    }

    fn unregister_memory(&self, py: Python<'_>, addresses: Vec<u64>) -> PyResult<()> {
        let mut unique = HashSet::with_capacity(addresses.len());
        for &address in &addresses {
            pointer(address, "addr")?;
            if !unique.insert(address) {
                return Err(PyValueError::new_err(format!(
                    "memory address {address:#x} appears more than once"
                )));
            }
        }
        py.detach(move || {
            let mut registrations = self.registrations.lock().map_err(|_| {
                transfer_error("unregister memory failed", "registration lock poisoned")
            })?;
            for &address in &addresses {
                let region = registrations.get(&address).ok_or_else(|| {
                    PyValueError::new_err(format!("memory address {address:#x} is not registered"))
                })?;
                if region.buffer.is_some() {
                    return Err(PyValueError::new_err(
                        "owned CPU buffers require close() and a source drain receipt",
                    ));
                }
            }
            for &address in &addresses {
                let region = registrations.get_mut(&address).ok_or_else(|| {
                    PyValueError::new_err(format!("memory address {address:#x} is not registered"))
                })?;
                if let Err(error) = Self::unregister_region(&self.engine, address, region) {
                    self.retiring.store(true, Ordering::Release);
                    return Err(transfer_error("unregister memory failed", error));
                }
            }
            let removed: Vec<_> = addresses
                .iter()
                .filter_map(|a| registrations.remove(a))
                .collect();
            drop(registrations);
            drop(removed);
            Ok(())
        })
    }

    /// Terminal source retirement; failed close retains every buffer for retry.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut regions = self
                .registrations
                .lock()
                .map_err(|_| transfer_error("close memory failed", "registration lock poisoned"))?;
            self.close_regions(&mut regions)
                .map_err(|error| transfer_error("close memory failed", error))?;
            let released = std::mem::take(&mut *regions);
            drop(regions);
            drop(released);
            Ok(())
        })
    }

    /// Process-wide regions, conservative backing bytes and retained engines.
    #[staticmethod]
    fn memory_usage() -> (usize, usize, usize) {
        let (regions, bytes) = *REGISTERED_MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let retained = RETAINED_MEMORY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        (regions, bytes, retained)
    }

    #[pyo3(signature = (remote_endpoint, slices, timeout_s=30.0, notify_name=None, notify_message=None))]
    fn write(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
        notify_name: Option<String>,
        notify_message: Option<String>,
    ) -> PyResult<usize> {
        self.transfer(
            py,
            TransferOp::Write,
            remote_endpoint,
            slices,
            timeout_s,
            notification(notify_name, notify_message)?,
        )
    }

    #[pyo3(signature = (remote_endpoint, slices, timeout_s=30.0))]
    fn read(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
    ) -> PyResult<usize> {
        self.transfer(
            py,
            TransferOp::Read,
            remote_endpoint,
            slices,
            timeout_s,
            None,
        )
    }

    fn send_notification(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        name: String,
        message: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            engine
                .send_notification(&remote_endpoint, &Notification { name, message })
                .map_err(|error| transfer_error("send notification failed", error))
        })
    }

    fn open_notification_scope(&self, name: String) -> PyResult<u64> {
        if name.is_empty() {
            return Err(PyValueError::new_err("notification name must not be empty"));
        }
        self.engine
            .open_notification_scope(&name)
            .map_err(|error| transfer_error("open TENT notification scope failed", error))
    }

    #[pyo3(signature = (name, generation, expected_done_count=1, timeout_s=30.0))]
    fn wait_for_status(
        &self,
        py: Python<'_>,
        name: String,
        generation: u64,
        expected_done_count: usize,
        timeout_s: f64,
    ) -> PyResult<Option<String>> {
        if name.is_empty() {
            return Err(PyValueError::new_err("notification name must not be empty"));
        }
        if generation == 0 {
            return Err(PyValueError::new_err("generation must be positive"));
        }
        if expected_done_count == 0 {
            return Err(PyValueError::new_err(
                "expected_done_count must be positive",
            ));
        }
        if !timeout_s.is_finite() || timeout_s <= 0.0 {
            return Err(PyValueError::new_err("timeout_s must be positive"));
        }
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            engine
                .wait_for_notification(
                    &name,
                    generation,
                    &[
                        ("failed".to_string(), 1),
                        ("aborted".to_string(), 1),
                        ("done".to_string(), expected_done_count),
                    ],
                    Duration::from_secs_f64(timeout_s),
                )
                .map_err(|error| match error {
                    TransferError::NotificationTimeout(_) => {
                        PyTimeoutError::new_err(error.to_string())
                    }
                    error => transfer_error("wait for TENT notification failed", error),
                })
        })
    }

    fn close_notification_scope(&self, name: String, generation: u64) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyValueError::new_err("notification name must not be empty"));
        }
        self.engine.close_notification_scope(&name, generation);
        Ok(())
    }

    fn nic_load_stats(&self) -> PyResult<Vec<(String, u64, f64)>> {
        self.engine
            .nic_load_stats()
            .map(|stats| {
                stats
                    .into_iter()
                    .map(|stat| {
                        (
                            stat.device_name,
                            stat.inflight_bytes,
                            stat.ewma_bandwidth_bps,
                        )
                    })
                    .collect()
            })
            .map_err(|error| transfer_error("read TENT NIC load stats failed", error))
    }

    fn invalidate_segment(&self, remote_endpoint: String) {
        self.engine.invalidate_segment(&remote_endpoint);
    }
}

impl PyMooncakeTransferEngine {
    fn register_regions(
        &self,
        py: Python<'_>,
        parsed: Vec<(u64, String, RegisteredRegion)>,
    ) -> PyResult<()> {
        for (index, (address, location, region)) in parsed.iter().enumerate() {
            pointer(*address, "addr")?;
            let end = address
                .checked_add(region.length as u64)
                .ok_or_else(|| PyValueError::new_err("memory region address overflow"))?;
            if region.length == 0 || location.contains('\0') {
                return Err(PyValueError::new_err(
                    "invalid memory region length or location",
                ));
            }
            if parsed[..index].iter().any(|(previous, _, other)| {
                *previous < end && *address < *previous + other.length as u64
            }) {
                return Err(PyValueError::new_err("memory regions overlap"));
            }
        }
        py.detach(move || {
            let mut regions = self.registrations.lock().map_err(|_| {
                transfer_error("register memory failed", "registration lock poisoned")
            })?;
            if self.retiring.load(Ordering::Acquire) {
                return Err(transfer_error(
                    "register memory failed",
                    "memory owner is retiring",
                ));
            }
            for (address, _, region) in &parsed {
                if regions.iter().any(|(previous, existing)| {
                    *previous < *address + region.length as u64
                        && *address < *previous + existing.length as u64
                }) {
                    return Err(PyValueError::new_err(
                        "memory region overlaps an existing attempt",
                    ));
                }
                if regions
                    .values()
                    .any(|existing| existing.buffer.is_some() != region.buffer.is_some())
                {
                    return Err(PyValueError::new_err(
                        "owned and borrowed regions cannot be mixed",
                    ));
                }
            }
            for (address, location, region) in parsed {
                let region = regions.entry(address).or_insert(region);
                // The allocation and admission already have an owner if native
                // registration reports a partially retained range.
                let result = unsafe {
                    self.engine.register_memory(
                        NonNull::new_unchecked(address as *mut u8),
                        region.length,
                        &location,
                    )
                };
                if let Err(error) = result {
                    self.retiring.store(true, Ordering::Release);
                    return Err(transfer_error("register memory failed", error));
                }
                region.state = RegistrationState::Registered;
            }
            Ok(())
        })
    }

    fn unregister_region(
        engine: &TransferEngine,
        address: u64,
        region: &mut RegisteredRegion,
    ) -> orbitkv_transfer::Result<()> {
        if region.state != RegistrationState::Unregistered {
            unsafe {
                engine
                    .unregister_memory(NonNull::new_unchecked(address as *mut u8), region.length)?;
            }
            region.state = RegistrationState::Unregistered;
        }
        Ok(())
    }

    fn close_regions(
        &self,
        regions: &mut HashMap<u64, RegisteredRegion>,
    ) -> orbitkv_transfer::Result<()> {
        self.retiring.store(true, Ordering::Release);
        self.engine.close_source(Duration::from_secs(5))?;
        for (&address, region) in regions {
            Self::unregister_region(&self.engine, address, region)?;
        }
        Ok(())
    }

    fn transfer(
        &self,
        py: Python<'_>,
        operation: TransferOp,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
        notification: Option<Notification>,
    ) -> PyResult<usize> {
        if !timeout_s.is_finite() || timeout_s <= 0.0 {
            return Err(PyValueError::new_err("timeout_s must be positive"));
        }
        let slices = slices
            .into_iter()
            .map(|(local, remote, length)| {
                let length = u64_to_usize(length, "slice length")?;
                if length == 0 {
                    return Err(PyValueError::new_err("slice length must be positive"));
                }
                Ok(TransferSlice {
                    local: pointer(local, "local address")?,
                    remote_address: remote,
                    length,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            let regions = self
                .registrations
                .lock()
                .map_err(|_| transfer_error("transfer failed", "registration lock poisoned"))?;
            if self.retiring.load(Ordering::Acquire) {
                return Err(transfer_error(
                    "transfer failed",
                    "memory owner is retiring",
                ));
            }
            for slice in &slices {
                let address = slice.local.as_ptr() as u64;
                let end = address
                    .checked_add(slice.length as u64)
                    .ok_or_else(|| PyValueError::new_err("transfer address overflow"))?;
                if !regions.iter().any(|(base, region)| {
                    region.state == RegistrationState::Registered
                        && address >= *base
                        && end <= *base + region.length as u64
                }) {
                    return Err(PyValueError::new_err(
                        "transfer is outside an active local region",
                    ));
                }
            }
            let timeout = Duration::from_secs_f64(timeout_s);
            match notification.as_ref() {
                Some(notification) => engine.submit_and_notify(
                    operation,
                    &remote_endpoint,
                    &slices,
                    timeout,
                    notification,
                    None,
                ),
                None => engine.submit_and_wait(operation, &remote_endpoint, &slices, timeout, None),
            }
            .map_err(|error| transfer_error("Mooncake TENT transfer failed", error))
        })
    }
}

impl Drop for PyMooncakeTransferEngine {
    fn drop(&mut self) {
        let mut regions = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        if regions.is_empty() {
            return;
        }
        let error = if self.retiring.load(Ordering::Acquire) {
            Some("earlier native operation failed; GC will not retry cleanup".to_string())
        } else {
            self.close_regions(&mut regions)
                .err()
                .map(|e| e.to_string())
        };
        if let Some(error) = error {
            RETAINED_MEMORY
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(RetainedMemory {
                    _engine: Arc::clone(&self.engine),
                    _regions: std::mem::take(&mut *regions),
                });
            drop(regions);
            log::error!("Retained TENT engine, memory exports and admission: {error}");
        } else {
            let released = std::mem::take(&mut *regions);
            drop(regions);
            drop(released);
        }
    }
}

fn notification(name: Option<String>, message: Option<String>) -> PyResult<Option<Notification>> {
    match (name, message) {
        (None, None) => Ok(None),
        (Some(name), Some(message)) => Ok(Some(Notification { name, message })),
        _ => Err(PyValueError::new_err(
            "notify_name and notify_message must be provided together",
        )),
    }
}

pub(crate) fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMooncakeTransferEngine>()
}
