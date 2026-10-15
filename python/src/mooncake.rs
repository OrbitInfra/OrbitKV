use crate::{OrbitKVError, u64_to_usize};

use orbitkv_transfer::{
    AUTO_MEMORY_LOCATION, MemoryRegistration, Notification, P2P_METADATA, TransferEngine,
    TransferError, TransferOp, TransferSlice,
};
use pyo3::{
    exceptions::{PyTimeoutError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::{
    collections::{HashMap, HashSet},
    ptr::NonNull,
    sync::{Arc, Mutex},
    time::Duration,
};

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
    registrations: Arc<Mutex<HashMap<u64, MemoryRegistration>>>,
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
            registrations: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[getter]
    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn register_memory(&self, py: Python<'_>, regions: Vec<Py<PyDict>>) -> PyResult<()> {
        let mut parsed = Vec::with_capacity(regions.len());
        let mut addresses = HashSet::with_capacity(regions.len());
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
            if !addresses.insert(address) {
                return Err(PyValueError::new_err(format!(
                    "memory address {address:#x} appears more than once"
                )));
            }
            parsed.push((address, length, location));
        }
        let engine = Arc::clone(&self.engine);
        let registrations = Arc::clone(&self.registrations);
        py.detach(move || {
            let mut registrations = registrations.lock().map_err(|_| {
                transfer_error("register memory failed", "registration lock poisoned")
            })?;
            if let Some((address, _, _)) = parsed
                .iter()
                .find(|(address, _, _)| registrations.contains_key(address))
            {
                return Err(PyValueError::new_err(format!(
                    "memory address {address:#x} is already registered"
                )));
            }
            let mut added = Vec::with_capacity(parsed.len());
            for (address, length, location) in parsed {
                let registration = unsafe {
                    engine
                        .register_memory_owned(
                            NonNull::new_unchecked(address as *mut u8),
                            length,
                            &location,
                        )
                        .map_err(|error| transfer_error("register memory failed", error))?
                };
                added.push((address, registration));
            }
            for (address, registration) in added {
                registrations.insert(address, registration);
            }
            Ok(())
        })
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
        let registrations = Arc::clone(&self.registrations);
        py.detach(move || {
            let mut registrations = registrations.lock().map_err(|_| {
                transfer_error("unregister memory failed", "registration lock poisoned")
            })?;
            for &address in &addresses {
                let registration = registrations.get_mut(&address).ok_or_else(|| {
                    PyValueError::new_err(format!("memory address {address:#x} is not registered"))
                })?;
                registration
                    .try_unregister()
                    .map_err(|error| transfer_error("unregister memory failed", error))?;
            }
            for address in addresses {
                registrations.remove(&address);
            }
            Ok(())
        })
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
