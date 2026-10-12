use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use orbitkv_channel::lifecycle::LifecycleCommand;
use orbitkv_channel::{
    BlockHashes, CacheClient, CallOptions, ChannelCallObservation, ChannelError, PublishLayer,
    PublishRequest, QueryIntent, RestoreHandle, RestoreLease, RestoreRequest, RestoreState,
};
use orbitkv_core::transfer::local::{LocalRestoreExecutor, LocalTensor};
use orbitkv_core::{PayloadArena, TransferMode as LocalTransferMode};
use orbitkv_proto::proto::engine::{
    ConfigureShardQueriesRequest, RegisterContextRequest, RegisteredQueryTarget, SessionRequest,
    ShardQueryTarget, TransferMode, UnregisterRequest,
};
use prost::Message;
use pyo3::{
    exceptions::{PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyBytes, PySlice},
};

use crate::local_restore::{LocalCompletions, LocalRestore, LocalRestoreWorker};
use crate::{OrbitKVError, OrbitKVInternal, query_response};

type PyLeaseLoad = (Vec<u8>, Vec<Vec<Option<u32>>>);

#[pyclass(name = "ChannelCallObservation", frozen)]
pub(crate) struct PyChannelCallObservation(ChannelCallObservation);

#[pymethods]
impl PyChannelCallObservation {
    #[getter]
    fn request_id(&self) -> u64 {
        self.0.request_id
    }

    #[getter]
    fn session_epoch(&self) -> u64 {
        self.0.session_epoch
    }

    #[getter]
    fn session_token(&self) -> u64 {
        self.0.session_token
    }

    #[getter]
    fn submitted_mono_ns(&self) -> u64 {
        self.0.submitted_mono_ns
    }

    #[getter]
    fn returned_mono_ns(&self) -> u64 {
        self.0.returned_mono_ns
    }
}

fn publish_request(
    instance_id: String,
    tp_rank: u32,
    pp_rank: u32,
    device_id: i32,
    saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
) -> PublishRequest {
    PublishRequest {
        instance_id,
        tp_rank,
        pp_rank,
        device_id,
        layers: saves
            .into_iter()
            .map(|(layer_name, block_ids, block_hashes)| PublishLayer {
                layer_name,
                block_ids,
                block_hashes,
            })
            .collect(),
    }
}

#[derive(PartialEq, Eq)]
#[pyclass(name = "BlockHashes", frozen, eq)]
pub(crate) struct PyBlockHashes(BlockHashes);

#[pymethods]
impl PyBlockHashes {
    #[new]
    fn new(hashes: Vec<Vec<u8>>) -> Self {
        Self(BlockHashes::new(hashes))
    }

    fn __len__(&self) -> usize {
        self.0.as_slice().len()
    }

    fn __getitem__(&self, slice: &Bound<'_, PySlice>) -> PyResult<Self> {
        let indices = slice.indices(self.__len__() as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err(
                "hash views require a unit slice step",
            ));
        }
        self.0
            .slice(indices.start as usize..indices.stop.max(indices.start) as usize)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err("hash view outside its batch"))
    }
}

#[pyclass(name = "RestoreHandle", frozen)]
pub(crate) struct PyRestoreHandle {
    handle: RestoreHandle,
    result: Arc<LocalRestore>,
    owner: Weak<CacheClient>,
}

#[pymethods]
impl PyRestoreHandle {
    #[getter]
    fn operation_id(&self) -> u64 {
        self.handle.operation_id
    }
    #[getter]
    fn session_epoch(&self) -> u64 {
        self.handle.session_epoch
    }
    #[getter]
    fn key(&self) -> String {
        format!(
            "manager:{}:{}:{}",
            self.handle.session_epoch, self.handle.session_token, self.handle.operation_id
        )
    }
}

#[pyclass(frozen)]
pub(crate) struct RestoreStatus {
    #[pyo3(get)]
    done: bool,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl RestoreStatus {
    #[new]
    #[pyo3(signature = (done, success, message=String::new()))]
    fn new(done: bool, success: bool, message: String) -> Self {
        Self {
            done,
            success,
            message,
        }
    }
}

impl From<orbitkv_channel::RestoreResponse> for RestoreStatus {
    fn from(response: orbitkv_channel::RestoreResponse) -> Self {
        Self {
            done: response.state != RestoreState::Pending,
            success: response.state == RestoreState::Succeeded,
            message: response.message,
        }
    }
}

fn client_error(error: ChannelError) -> PyErr {
    match error {
        ChannelError::Lifecycle { code: 1, message } => PyValueError::new_err(message),
        ChannelError::Lifecycle { code: 3, message } => OrbitKVInternal::new_err(message),
        ChannelError::RestoreTimeout { .. } => {
            PyTimeoutError::new_err("OrbitKV GPU restore timed out")
        }
        other => OrbitKVError::new_err(other.to_string()),
    }
}

fn seconds(value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err("timeout must be finite and non-negative"))
}

#[pyclass(name = "CacheManagerClient", frozen)]
pub(crate) struct PyCacheManagerClient {
    inner: Arc<CacheClient>,
    completions: Arc<LocalCompletions>,
    local: Mutex<HashMap<(String, u32, i32), Arc<LocalRestoreWorker>>>,
}

impl PyCacheManagerClient {
    fn manager_device_id(&self, instance_id: &str, tp_rank: u32, device_id: i32) -> PyResult<i32> {
        let local = self
            .local
            .lock()
            .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
        local
            .get(&(instance_id.to_string(), tp_rank, device_id))
            .map(|worker| worker.manager_device_id)
            .ok_or_else(|| PyValueError::new_err("GPU tensors must be registered before Publish"))
    }

    fn lifecycle_call(
        &self,
        py: Python<'_>,
        command: LifecycleCommand,
        payload: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.channel().lifecycle(command, &payload))
            .map(|_| ())
            .map_err(client_error)
    }
}

#[pymethods]
impl PyCacheManagerClient {
    #[new]
    #[pyo3(signature = (bootstrap_socket, *, timeout_ms=5000, spin_iterations=64))]
    fn new(
        py: Python<'_>,
        bootstrap_socket: String,
        timeout_ms: u64,
        spin_iterations: u32,
    ) -> PyResult<Self> {
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be non-zero"));
        }
        let inner = py
            .detach(|| {
                CacheClient::connect(
                    bootstrap_socket,
                    CallOptions {
                        timeout: Duration::from_millis(timeout_ms),
                        spin_iterations,
                    },
                )
            })
            .map_err(client_error)?;
        Ok(Self {
            inner: Arc::new(inner),
            local: Mutex::new(HashMap::new()),
            completions: Arc::new(
                LocalCompletions::new()
                    .map_err(|error| OrbitKVInternal::new_err(error.to_string()))?,
            ),
        })
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            let mut local = self
                .local
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for worker in local.values() {
                worker.wait_drained();
            }
            self.inner.close();
            for worker in local.values() {
                worker.shutdown();
            }
            local.clear();
        });
    }

    #[getter]
    fn transport(&self) -> &'static str {
        "iceoryx2"
    }
    #[getter]
    fn bootstrap_socket(&self) -> String {
        self.inner.socket().to_string_lossy().into_owned()
    }
    #[getter]
    fn service_name(&self) -> String {
        self.inner.channel().service_name().to_owned()
    }
    #[getter]
    fn session_epoch(&self) -> u64 {
        self.inner.channel().session_epoch()
    }
    #[getter]
    fn notification_fd(&self) -> i32 {
        self.completions.fd()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "matches framework registration metadata"
    )]
    #[pyo3(signature = (instance_id, namespace, tp_rank, pp_rank, tp_size, world_size, device_id, layer_names, wrapper_bytes_list, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list, transfer_backend, page_first, layer_group_ids=None, layer_formats=None, layer_attention=None, *, tensors))]
    fn register_context_batch(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_rank: u32,
        pp_rank: u32,
        tp_size: u32,
        world_size: u32,
        device_id: i32,
        layer_names: Vec<String>,
        wrapper_bytes_list: Vec<Vec<u8>>,
        num_blocks_list: Vec<u64>,
        bytes_per_block_list: Vec<u64>,
        kv_stride_bytes_list: Vec<u64>,
        segments_list: Vec<u32>,
        transfer_backend: &str,
        page_first: bool,
        layer_group_ids: Option<Vec<u32>>,
        layer_formats: Option<Vec<String>>,
        layer_attention: Option<Vec<(u32, String, u32, u32)>>,
        tensors: Vec<Py<PyAny>>,
    ) -> PyResult<(bool, String)> {
        let transfer_mode = match transfer_backend {
            "direct" => TransferMode::Direct,
            "kernel" => TransferMode::Kernel,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown transfer_backend '{other}' (expected 'direct' or 'kernel')"
                )));
            }
        };
        let key = (instance_id.clone(), tp_rank, device_id);
        let count = layer_names.len();
        if count == 0
            || [
                tensors.len(),
                num_blocks_list.len(),
                bytes_per_block_list.len(),
                kv_stride_bytes_list.len(),
                segments_list.len(),
            ]
            .iter()
            .any(|&length| length != count)
        {
            return Err(PyValueError::new_err(
                "local tensor registration metadata length mismatch",
            ));
        }
        let mut device = None;
        let mut bindings = Vec::with_capacity(count);
        for (index, owner) in tensors.iter().enumerate() {
            let tensor = owner.bind(py);
            let tensor_device = tensor.getattr("device")?;
            if tensor_device.getattr("type")?.extract::<String>()? != "cuda" {
                return Err(PyValueError::new_err(
                    "Restore registration requires CUDA tensors",
                ));
            }
            let ordinal = tensor_device.getattr("index")?.extract::<usize>()?;
            if device
                .replace(ordinal)
                .is_some_and(|previous| previous != ordinal)
            {
                return Err(PyValueError::new_err(
                    "one GPU registration cannot span CUDA devices",
                ));
            }
            let address = tensor.call_method0("data_ptr")?.extract::<u64>()?;
            let storage = tensor.call_method0("untyped_storage")?;
            let allocation = storage.call_method0("data_ptr")?.extract::<u64>()?;
            let size = storage.call_method0("nbytes")?.extract::<u64>()?;
            let offset = address
                .checked_sub(allocation)
                .filter(|offset| *offset < size)
                .ok_or_else(|| {
                    PyValueError::new_err("tensor view is outside its retained storage")
                })?;
            bindings.push(
                LocalTensor::new(
                    layer_names[index].clone(),
                    address,
                    crate::u64_to_usize(size - offset, "tensor bytes")?,
                    crate::u64_to_usize(allocation, "tensor allocation")?,
                    crate::u64_to_usize(num_blocks_list[index], "num_blocks")?,
                    crate::u64_to_usize(bytes_per_block_list[index], "bytes_per_block")?,
                    crate::u64_to_usize(kv_stride_bytes_list[index], "kv_stride")?,
                    segments_list[index] as usize,
                )
                .map_err(PyValueError::new_err)?,
            );
        }
        let device = device.ok_or_else(|| PyValueError::new_err("empty tensor registration"))?;
        if device_id < 0 || device_id as usize != device {
            return Err(PyValueError::new_err(
                "device_id must match the tensors' client-local CUDA ordinal",
            ));
        }
        let device_uuid = py
            .import("orbitkv.client.gpu")?
            .getattr("CudaIPCWrapper")?
            .call_method1("_get_device_uuid", (device,))?
            .extract::<String>()?;
        let request = RegisterContextRequest {
            instance_id,
            namespace,
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            tp_rank,
            tp_size,
            world_size,
            device_id,
            device_uuid,
            layer_names,
            wrapper_bytes: wrapper_bytes_list,
            num_blocks: num_blocks_list,
            bytes_per_block: bytes_per_block_list,
            kv_stride_bytes: kv_stride_bytes_list,
            segments: segments_list,
            pp_rank,
            transfer_mode: transfer_mode as i32,
            page_first,
            layer_group_ids: layer_group_ids.unwrap_or_default(),
            layer_formats: layer_formats.unwrap_or_default(),
            layer_attention: layer_attention
                .unwrap_or_default()
                .into_iter()
                .map(|(head_dim, role, layer_index, layer_count)| {
                    orbitkv_proto::proto::engine::AttentionStorageLayout {
                        head_dim,
                        role,
                        layer_index,
                        layer_count,
                    }
                })
                .collect(),
        };
        py.detach(|| {
            let mut local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            if local.contains_key(&key) {
                return Err(PyValueError::new_err(
                    "GPU context is already registered on this client",
                ));
            }
            let reply = self
                .inner
                .channel()
                .lifecycle(LifecycleCommand::Register, &request.encode_to_vec())
                .map_err(client_error)?;
            let worker = (|| -> PyResult<LocalRestoreWorker> {
                if reply.payload.len() != 4 + reply.fds.len() * 16 {
                    return Err(OrbitKVInternal::new_err(
                        "invalid payload arena registration reply",
                    ));
                }
                let manager_device_id =
                    i32::from_le_bytes(reply.payload[..4].try_into().map_err(|_| {
                        OrbitKVInternal::new_err("invalid Manager device identity")
                    })?);
                if manager_device_id < 0 {
                    return Err(OrbitKVInternal::new_err("negative Manager device ordinal"));
                }
                if local
                    .values()
                    .map(|worker| worker.arena_count())
                    .sum::<usize>()
                    + reply.fds.len()
                    > 64
                {
                    return Err(OrbitKVError::new_err(
                        "session payload arena mapping limit reached",
                    ));
                }
                let mut arenas = Vec::with_capacity(reply.fds.len());
                for (metadata, fd) in reply.payload[4..].chunks_exact(16).zip(reply.fds) {
                    let id = u64::from_le_bytes(
                        metadata[..8]
                            .try_into()
                            .map_err(|_| OrbitKVInternal::new_err("invalid arena identity"))?,
                    );
                    let size = u64::from_le_bytes(
                        metadata[8..]
                            .try_into()
                            .map_err(|_| OrbitKVInternal::new_err("invalid arena size"))?,
                    );
                    arenas.push(PayloadArena { id, size, fd });
                }
                let mode = match transfer_mode {
                    TransferMode::Direct => LocalTransferMode::Direct,
                    TransferMode::Kernel => LocalTransferMode::Kernel,
                };
                let arena_count = arenas.len();
                let executor = LocalRestoreExecutor::new(device, bindings, arenas, mode)
                    .map_err(OrbitKVInternal::new_err)?;
                LocalRestoreWorker::new(
                    Arc::clone(&self.inner),
                    executor,
                    tensors,
                    Arc::clone(&self.completions),
                    arena_count,
                    manager_device_id,
                )
                .map_err(OrbitKVInternal::new_err)
            })();
            let worker = match worker {
                Ok(worker) => worker,
                Err(error) => {
                    // Registration has already attached GPU state in the Manager.
                    // Close the session after draining existing work so a partially
                    // attached context can never accept later Restore or Publish.
                    for worker in local.values() {
                        worker.wait_drained();
                    }
                    self.inner.close();
                    for worker in local.values() {
                        worker.shutdown();
                    }
                    local.clear();
                    return Err(error);
                }
            };
            local.insert(key, Arc::new(worker));
            Ok((true, String::new()))
        })
    }

    /// Export sealed node-local query authority for released engine handshake metadata.
    fn export_query_target(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> PyResult<Py<PyBytes>> {
        let reply = py
            .detach(|| {
                self.inner.channel().lifecycle(
                    LifecycleCommand::ExportQueryTarget,
                    &SessionRequest {
                        instance_id,
                        namespace,
                        tp_size,
                        world_size,
                    }
                    .encode_to_vec(),
                )
            })
            .map_err(client_error)?;
        Ok(PyBytes::new(py, &reply.payload).unbind())
    }

    fn configure_shard_queries(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
        shards: Vec<(String, String, Vec<u8>)>,
    ) -> PyResult<()> {
        if shards.is_empty()
            || shards.len() > orbitkv_channel::ShardQueryResponse::MAX_SHARDS
            || shards
                .iter()
                .map(|(_, _, target)| target.len())
                .sum::<usize>()
                > 64 * 1024
        {
            return Err(PyValueError::new_err(
                "bounded complete shard configuration required",
            ));
        }
        let shards = shards
            .into_iter()
            .map(|(endpoint, namespace, bytes)| {
                let target = RegisteredQueryTarget::decode(bytes.as_slice())
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(ShardQueryTarget {
                    endpoint,
                    namespace,
                    target: Some(target),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let request = ConfigureShardQueriesRequest {
            local: Some(SessionRequest {
                instance_id,
                namespace,
                tp_size,
                world_size,
            }),
            shards,
        };
        if request.encoded_len() > 64 * 1024 {
            return Err(PyValueError::new_err("shard configuration exceeds 64 KiB"));
        }
        self.lifecycle_call(
            py,
            LifecycleCommand::ConfigureShardQueries,
            request.encode_to_vec(),
        )
    }

    fn query_shards(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner
                    .query_shards(instance_id, &block_hashes.0, req_id)
            })
            .map_err(client_error)?;
        crate::shard_query_response(py, response)
    }

    fn release_shard_query(&self, py: Python<'_>, control_id: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.inner.release_shard_query(control_id))
            .map_err(client_error)
    }

    fn health(&self, py: Python<'_>) -> PyResult<(bool, String)> {
        self.lifecycle_call(py, LifecycleCommand::Health, Vec::new())?;
        Ok((true, String::new()))
    }

    fn unregister_context(&self, py: Python<'_>, instance_id: String) -> PyResult<(bool, String)> {
        py.detach(|| {
            let mut local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            for ((instance, _, _), worker) in local.iter() {
                if instance == &instance_id {
                    worker.wait_drained();
                }
            }
            self.inner
                .channel()
                .lifecycle(
                    LifecycleCommand::Unregister,
                    &UnregisterRequest {
                        instance_id: instance_id.clone(),
                    }
                    .encode_to_vec(),
                )
                .map_err(client_error)?;
            for ((instance, _, _), worker) in local.iter() {
                if instance == &instance_id {
                    worker.shutdown();
                }
            }
            local.retain(|(instance, _, _), _| instance != &instance_id);
            Ok((true, String::new()))
        })
    }

    fn start_session_watcher(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> PyResult<()> {
        self.lifecycle_call(
            py,
            LifecycleCommand::Session,
            SessionRequest {
                instance_id,
                namespace,
                tp_size,
                world_size,
            }
            .encode_to_vec(),
        )
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, wait_for_full_prefix=false, group_id=0))]
    fn query_prefetch(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        wait_for_full_prefix: bool,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.query(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    group_id,
                    QueryIntent::Lookup {
                        wait_for_full_prefix,
                    },
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, wait_for_full_prefix=false, group_id=0))]
    fn query_prefetch_diagnostic(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        wait_for_full_prefix: bool,
        group_id: u32,
    ) -> PyResult<(Py<PyAny>, PyChannelCallObservation)> {
        let (response, observation) = py
            .detach(|| {
                self.inner.query_observed(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    group_id,
                    QueryIntent::Lookup {
                        wait_for_full_prefix,
                    },
                )
            })
            .map_err(client_error)?;
        Ok((
            query_response(py, response)?,
            PyChannelCallObservation(observation),
        ))
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, group_id=0))]
    fn query_candidates(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.query(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    group_id,
                    QueryIntent::Candidates,
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "native boundary for engine recovery context"
    )]
    fn read_recovery(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        contract: &crate::recovery::PyRecoveryContract,
        namespace: &str,
        start: u64,
        end: u64,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.read_recovery(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    orbitkv_channel::RecoveryRead {
                        contract: &contract.contract,
                        namespace,
                        span: orbitkv_state::TokenRange { start, end },
                        group: group_id,
                    },
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "native boundary for engine recovery context"
    )]
    fn prepare_recovery(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        contract: &crate::recovery::PyRecoveryContract,
        namespace: &str,
        start: u64,
        end: u64,
        group_id: u32,
    ) -> PyResult<bool> {
        py.detach(|| {
            self.inner.prepare_recovery(
                instance_id,
                &block_hashes.0,
                req_id,
                orbitkv_channel::RecoveryRead {
                    contract: &contract.contract,
                    namespace,
                    span: orbitkv_state::TokenRange { start, end },
                    group: group_id,
                },
            )
        })
        .map_err(client_error)
    }

    fn prepare_prefix(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
    ) -> PyResult<bool> {
        py.detach(|| {
            self.inner
                .prepare_prefix(instance_id, &block_hashes.0, req_id)
        })
        .map_err(client_error)
    }

    fn warm_prefix(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
    ) -> PyResult<bool> {
        py.detach(|| self.inner.warm_prefix(instance_id, &block_hashes.0, req_id))
            .map_err(client_error)
    }

    #[pyo3(signature = (instance_id, req_id, group_id=0))]
    fn cancel_query(
        &self,
        py: Python<'_>,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
    ) -> PyResult<()> {
        py.detach(|| self.inner.cancel_query(instance_id, req_id, group_id))
            .map_err(client_error)
    }

    fn release(&self, py: Python<'_>, lease: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.inner.release(lease))
            .map_err(client_error)
    }

    fn save(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String)> {
        let manager_device_id =
            py.detach(|| self.manager_device_id(&instance_id, tp_rank, device_id))?;
        let request = publish_request(instance_id, tp_rank, pp_rank, manager_device_id, saves);
        py.detach(|| self.inner.publish(&request))
            .map_err(client_error)?;
        Ok((true, String::new()))
    }

    fn save_diagnostic(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String, PyChannelCallObservation)> {
        let manager_device_id =
            py.detach(|| self.manager_device_id(&instance_id, tp_rank, device_id))?;
        let request = publish_request(instance_id, tp_rank, pp_rank, manager_device_id, saves);
        let observation = py
            .detach(|| self.inner.publish_observed(&request))
            .map_err(client_error)?;
        Ok((true, String::new(), PyChannelCallObservation(observation)))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Restore identifies registered destinations, leases, and the framework readiness stream"
    )]
    #[pyo3(signature = (instance_id, tp_rank, device_id, layer_groups, loads, *, ready_stream, layer_events=None))]
    fn start_restore(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        layer_groups: Vec<Vec<String>>,
        loads: Vec<PyLeaseLoad>,
        ready_stream: u64,
        layer_events: Option<Vec<(String, Py<PyAny>)>>,
    ) -> PyResult<PyRestoreHandle> {
        let mut events = HashMap::new();
        let mut event_owners = Vec::new();
        for (layer, event) in layer_events.unwrap_or_default() {
            let handle = event.bind(py).getattr("cuda_event")?.extract::<u64>()?;
            if events.insert(layer, handle).is_some() {
                return Err(PyValueError::new_err("duplicate Restore event layer"));
            }
            event_owners.push(event);
        }
        let started = (*crate::local_restore::RESTORE_TIMING).then(std::time::Instant::now);
        let loads = loads
            .into_iter()
            .map(|(lease, block_ids_by_group)| RestoreLease {
                lease,
                block_ids_by_group,
            })
            .collect();
        py.detach(|| {
            let local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            let worker = local
                .get(&(instance_id.clone(), tp_rank, device_id))
                .ok_or_else(|| {
                    PyValueError::new_err("GPU tensors must be registered before Restore")
                })?;
            worker
                .reserve(ready_stream, &events)
                .map_err(OrbitKVError::new_err)?;
            let timing = started.map(|start| {
                (
                    start,
                    orbitkv_channel::RestoreTiming {
                        readiness_ns: start.elapsed().as_nanos() as u64,
                        ..Default::default()
                    },
                )
            });
            match self.inner.start_restore(&RestoreRequest {
                instance_id,
                tp_rank,
                device_id: worker.manager_device_id,
                layer_groups,
                loads,
            }) {
                Ok(handle) => Ok(PyRestoreHandle {
                    handle,
                    result: worker.submit(handle, timing, events, event_owners),
                    owner: Arc::downgrade(&self.inner),
                }),
                Err(error) => {
                    worker.cancel_reservation();
                    Err(client_error(error))
                }
            }
        })
    }

    /// Wait for this operation's layer-event records to be submitted, not for DMA completion.
    #[pyo3(signature = (handle, *, timeout))]
    fn wait_restore_enqueued(
        &self,
        py: Python<'_>,
        handle: &PyRestoreHandle,
        timeout: f64,
    ) -> PyResult<()> {
        if !Weak::ptr_eq(&Arc::downgrade(&self.inner), &handle.owner) {
            return Err(PyValueError::new_err(
                "Restore handle belongs to another client",
            ));
        }
        let timeout = seconds(timeout)?;
        if py
            .detach(|| handle.result.wait_enqueued(timeout))
            .map_err(OrbitKVError::new_err)?
        {
            Ok(())
        } else {
            Err(PyTimeoutError::new_err(
                "OrbitKV Restore event publication timed out",
            ))
        }
    }

    fn poll_restore(&self, py: Python<'_>, handle: &PyRestoreHandle) -> PyResult<RestoreStatus> {
        if !Weak::ptr_eq(&Arc::downgrade(&self.inner), &handle.owner) {
            return Err(PyValueError::new_err(
                "Restore handle belongs to another client",
            ));
        }
        Ok(py
            .detach(|| handle.result.poll())
            .map_err(OrbitKVError::new_err)?
            .map(Into::into)
            .unwrap_or(RestoreStatus {
                done: false,
                success: false,
                message: String::new(),
            }))
    }

    #[pyo3(signature = (handle, *, timeout))]
    fn wait_restore(
        &self,
        py: Python<'_>,
        handle: &PyRestoreHandle,
        timeout: f64,
    ) -> PyResult<RestoreStatus> {
        let timeout = seconds(timeout)?;
        if !Weak::ptr_eq(&Arc::downgrade(&self.inner), &handle.owner) {
            return Err(PyValueError::new_err(
                "Restore handle belongs to another client",
            ));
        }
        py.detach(|| handle.result.wait(timeout))
            .map_err(OrbitKVError::new_err)?
            .map(Into::into)
            .ok_or_else(|| PyTimeoutError::new_err("OrbitKV GPU restore timed out"))
    }

    #[pyo3(signature = (*, timeout=0.0))]
    fn restore_completions_ready(&self, py: Python<'_>, timeout: f64) -> PyResult<bool> {
        let timeout = seconds(timeout)?;
        py.detach(|| self.completions.wait(timeout))
            .map_err(|error| OrbitKVError::new_err(error.to_string()))
    }
}
