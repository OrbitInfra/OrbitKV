use orbitkv_channel::{
    CallOptions, Command as ChannelCommand, CommandCode, QueryOutcomeCode, StatusCode,
    TransportClient,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyTuple},
};
use std::time::Duration;

#[cfg(feature = "mooncake")]
mod mooncake;

mod client;
mod local_restore;
mod recovery;
mod registration;

// Custom Python exceptions for error classification
create_exception!(orbitkv, OrbitKVError, PyException);
create_exception!(orbitkv, OrbitKVInternal, OrbitKVError);

#[pyfunction]
fn rekey_hashes(py: Python<'_>, hashes: Vec<Vec<u8>>, generation: &[u8]) -> PyResult<Py<PyTuple>> {
    let generation: &[u8; 16] = generation
        .try_into()
        .map_err(|_| PyValueError::new_err("cache generation must contain exactly 16 bytes"))?;
    let keys: Vec<_> = py.detach(|| {
        hashes
            .iter()
            .map(|hash| orbitkv_state::generation_hash(hash, generation))
            .collect()
    });
    Ok(PyTuple::new(py, keys.iter().map(|key| PyBytes::new(py, key)))?.unbind())
}

fn u64_to_usize(value: u64, field: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyRuntimeError::new_err(format!("{field}={value} exceeds usize range")))
}

#[pyclass(frozen)]
struct QueryLoading {
    #[pyo3(get)]
    admitted: bool,
}

#[pymethods]
impl QueryLoading {
    #[new]
    #[pyo3(signature = (admitted=true))]
    fn new(admitted: bool) -> Self {
        Self { admitted }
    }

    fn __repr__(&self) -> String {
        "QueryLoading()".to_string()
    }
}

#[pyclass(frozen)]
struct QueryCandidates {
    #[pyo3(get)]
    num_hit_blocks: usize,
    #[pyo3(get)]
    hit_positions: Vec<u32>,
}

#[pymethods]
impl QueryCandidates {
    #[new]
    fn new(hit_positions: Vec<u32>) -> Self {
        Self {
            num_hit_blocks: hit_positions.len(),
            hit_positions,
        }
    }
}

fn query_response(
    py: Python<'_>,
    response: orbitkv_channel::QueryBundleResponse,
) -> PyResult<Py<PyAny>> {
    match response.outcome {
        QueryOutcomeCode::Loading | QueryOutcomeCode::Busy => Py::new(
            py,
            QueryLoading {
                admitted: response.outcome == QueryOutcomeCode::Loading,
            },
        )
        .map(|value| value.into_any()),
        QueryOutcomeCode::Candidates => Py::new(
            py,
            QueryCandidates {
                num_hit_blocks: u64_to_usize(response.num_hit_blocks, "num_hit_blocks")?,
                hit_positions: response.hit_positions,
            },
        )
        .map(|value| value.into_any()),
        QueryOutcomeCode::Ready => Py::new(
            py,
            QueryReady {
                num_hit_blocks: u64_to_usize(response.num_hit_blocks, "num_hit_blocks")?,
                lease: response.lease,
                hit_positions: response.hit_positions,
            },
        )
        .map(|value| value.into_any()),
    }
}

#[pyclass(frozen)]
struct ShardedQueryReady {
    #[pyo3(get)]
    num_hit_blocks: usize,
    leases: Vec<Vec<u8>>,
    control_id: Vec<u8>,
}
#[pymethods]
impl ShardedQueryReady {
    #[getter]
    fn leases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.leases.iter().map(|lease| PyBytes::new(py, lease)))
    }
    #[getter]
    fn control_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.control_id)
    }
}
fn shard_query_response(
    py: Python<'_>,
    response: orbitkv_channel::ShardQueryResponse,
) -> PyResult<Py<PyAny>> {
    match response.outcome {
        QueryOutcomeCode::Loading | QueryOutcomeCode::Busy => Py::new(
            py,
            QueryLoading {
                admitted: response.outcome == QueryOutcomeCode::Loading,
            },
        )
        .map(|value| value.into_any()),
        QueryOutcomeCode::Ready => Py::new(
            py,
            ShardedQueryReady {
                num_hit_blocks: u64_to_usize(response.num_hit_blocks, "num_hit_blocks")?,
                leases: response.leases,
                control_id: response.control_id,
            },
        )
        .map(|value| value.into_any()),
        QueryOutcomeCode::Candidates => Err(PyValueError::new_err(
            "shard query cannot return candidates",
        )),
    }
}

#[pyclass(frozen)]
struct QueryReady {
    #[pyo3(get)]
    num_hit_blocks: usize,
    lease: Vec<u8>,
    /// Membership queries (group_id > 0) only: indices into the queried
    /// block_hashes whose block is cached; lease block i corresponds to
    /// query position hit_positions[i]. Empty for prefix queries.
    #[pyo3(get)]
    hit_positions: Vec<u32>,
}

#[pymethods]
impl QueryReady {
    #[new]
    #[pyo3(signature = (num_hit_blocks, lease, hit_positions=Vec::new()))]
    fn new(num_hit_blocks: usize, lease: Vec<u8>, hit_positions: Vec<u32>) -> Self {
        Self {
            num_hit_blocks,
            lease,
            hit_positions,
        }
    }

    #[getter]
    fn lease<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.lease)
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryReady(num_hit_blocks={}, hits={:?}, has_lease={})",
            self.num_hit_blocks,
            self.hit_positions,
            !self.lease.is_empty()
        )
    }
}

#[pyclass]
struct ChannelProbeClient {
    service_name: String,
    session_epoch: u64,
    options: CallOptions,
    client: TransportClient,
}

impl ChannelProbeClient {
    fn call(&self, py: Python<'_>, command: ChannelCommand) -> PyResult<orbitkv_channel::Response> {
        let response = py
            .detach(|| self.client.call(command, self.options))
            .map_err(|error| OrbitKVError::new_err(format!("channel probe failed: {error}")))?;
        if response.status != StatusCode::Ok {
            return Err(OrbitKVError::new_err(format!(
                "channel probe returned {:?}: client_epoch={} manager_epoch={}",
                response.status, self.session_epoch, response.session_epoch
            )));
        }
        Ok(response)
    }
}

#[pymethods]
impl ChannelProbeClient {
    #[new]
    #[pyo3(signature = (service_name, session_epoch, timeout_ms=5000, spin_iterations=64))]
    fn new(
        service_name: String,
        session_epoch: u64,
        timeout_ms: u64,
        spin_iterations: u32,
    ) -> PyResult<Self> {
        if session_epoch == 0 {
            return Err(PyValueError::new_err("session_epoch must be non-zero"));
        }
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be non-zero"));
        }
        let client = TransportClient::connect(&service_name).map_err(|error| {
            OrbitKVError::new_err(format!("channel probe connect failed: {error}"))
        })?;
        Ok(Self {
            service_name,
            session_epoch,
            options: CallOptions {
                timeout: Duration::from_millis(timeout_ms),
                spin_iterations,
            },
            client,
        })
    }

    #[getter]
    fn service_name(&self) -> &str {
        &self.service_name
    }

    #[getter]
    fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    #[pyo3(signature = (value=0, request_id=1))]
    fn ping(&self, py: Python<'_>, value: u64, request_id: u64) -> PyResult<u64> {
        let mut command = ChannelCommand::ping(request_id, self.session_epoch);
        command.arg0 = value;
        Ok(self.call(py, command)?.value0)
    }

    #[pyo3(signature = (request_id=1))]
    fn shutdown(&self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        self.call(
            py,
            ChannelCommand {
                code: CommandCode::Shutdown,
                request_id,
                ..ChannelCommand::ping(request_id, self.session_epoch)
            },
        )?;
        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn orbitkv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    orbitkv_common::logging::init_stderr("info,orbitkv_core=info");
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<ChannelProbeClient>()?;
    m.add_class::<client::PyCacheManagerClient>()?;
    m.add_class::<client::PyGpuRegistration>()?;
    m.add_class::<client::PyChannelCallObservation>()?;
    m.add_class::<client::PyPublishProgress>()?;
    m.add_class::<client::PySaveHandle>()?;
    m.add_class::<client::PyBlockHashes>()?;
    m.add_class::<client::PyRestoreHandle>()?;
    m.add_class::<client::RestoreStatus>()?;
    #[cfg(feature = "mooncake")]
    mooncake::add_classes(m)?;
    // Register custom exceptions for error classification
    m.add("OrbitKVError", m.py().get_type::<OrbitKVError>())?;
    m.add("OrbitKVInternal", m.py().get_type::<OrbitKVInternal>())?;
    m.add_class::<QueryLoading>()?;
    m.add_class::<QueryReady>()?;
    m.add_class::<ShardedQueryReady>()?;
    m.add_class::<QueryCandidates>()?;
    m.add_class::<recovery::PyRecoveryContract>()?;
    m.add_function(wrap_pyfunction!(rekey_hashes, m)?)?;

    Ok(())
}
