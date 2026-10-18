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
    ConfigureShardQueriesRequest, CudaCreditDisposition, CudaIpcTensorRegistration,
    CudaRegistrationReply, RegisterContextRequest, RegisteredQueryTarget, SessionRequest,
    ShardQueryTarget, TransferMode, UnregisterRequest,
};
use prost::Message;
use pyo3::{
    exceptions::{PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyBytes, PySlice, PyTuple},
};

use crate::local_restore::{LocalCompletions, LocalRestore, LocalRestoreWorker};
use crate::{OrbitKVError, OrbitKVInternal, query_response};

type PyLeaseLoad = (Vec<u8>, Vec<Vec<Option<u32>>>);

#[pyclass(name = "PublishProgress", frozen)]
pub(crate) struct PyPublishProgress {
    #[pyo3(get)]
    active: usize,
    #[pyo3(get)]
    retained: usize,
    #[pyo3(get)]
    active_interval_seconds: f64,
    #[pyo3(get)]
    overdue: bool,
    #[pyo3(get)]
    manager_alive: Option<bool>,
}

impl From<orbitkv_channel::PublishProgress> for PyPublishProgress {
    fn from(progress: orbitkv_channel::PublishProgress) -> Self {
        Self {
            active: progress.active,
            retained: progress.retained,
            active_interval_seconds: progress.active_interval.as_secs_f64(),
            overdue: progress.overdue,
            manager_alive: progress.manager_alive,
        }
    }
}

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

fn validate_export_snapshot(
    py: Python<'_>,
    tensors: &[Py<PyAny>],
    storages: &[Py<PyAny>],
    allocations: &[u64],
    layouts: &[CudaIpcTensorRegistration],
    device: usize,
) -> PyResult<()> {
    for (((tensor, storage), allocation), layout) in
        tensors.iter().zip(storages).zip(allocations).zip(layouts)
    {
        let tensor = tensor.bind(py);
        let storage = storage.bind(py);
        let base = storage.call_method0("data_ptr")?.extract::<u64>()?;
        let pointer = tensor.call_method0("data_ptr")?.extract::<u64>()?;
        let offset = tensor.call_method0("storage_offset")?.extract::<u64>()?;
        let size = storage.call_method0("nbytes")?.extract::<u64>()?;
        let shape = tensor.getattr("shape")?.extract::<Vec<u64>>()?;
        let strides = tensor.call_method0("stride")?.extract::<Vec<u64>>()?;
        let dtype = tensor.getattr("dtype")?.str()?.to_string();
        let ordinal = tensor
            .getattr("device")?
            .getattr("index")?
            .extract::<usize>()?;
        if base != *allocation
            || size != layout.storage_size_bytes
            || base.checked_add(layout.view_offset_bytes) != Some(pointer)
            || offset.checked_mul(layout.element_size) != Some(layout.view_offset_bytes)
            || shape != layout.shape
            || strides != layout.strides
            || dtype != format!("torch.{}", layout.dtype)
            || ordinal != device
            || tensor.call_method0("element_size")?.extract::<u64>()? != layout.element_size
        {
            return Err(PyValueError::new_err(
                "CUDA tensor or retained storage changed during registration",
            ));
        }
    }
    Ok(())
}

fn publish_request(
    instance_id: String,
    tp_rank: u32,
    pp_rank: u32,
    device_id: i32,
    registration_generation: [u8; 16],
    saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
) -> PublishRequest {
    PublishRequest {
        instance_id,
        registration_generation,
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

#[pyclass(name = "GpuRegistration", frozen)]
pub(crate) struct PyGpuRegistration {
    instance_id: String,
    tp_rank: u32,
    pp_rank: u32,
    device_id: i32,
    generation: [u8; 16],
    export_owner_generation: [u8; 16],
    ipc_counters: Vec<(Vec<u8>, u64)>,
    ipc_credit_evidence: String,
    owner: Weak<CacheClient>,
    _tensors: Arc<Vec<Py<PyAny>>>,
}

#[pymethods]
impl PyGpuRegistration {
    #[getter]
    fn export_owner_generation(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.export_owner_generation).unbind()
    }

    #[getter]
    fn ipc_counters(&self) -> Vec<(Vec<u8>, u64)> {
        self.ipc_counters.clone()
    }

    #[getter]
    fn ipc_credit_evidence(&self) -> String {
        self.ipc_credit_evidence.clone()
    }

    #[getter]
    fn generation(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.generation).unbind()
    }
}

#[pyclass(name = "CacheManagerClient", frozen)]
pub(crate) struct PyCacheManagerClient {
    inner: Arc<CacheClient>,
    completions: Arc<LocalCompletions>,
    local: Mutex<HashMap<(String, u32, i32), Arc<LocalRestoreWorker>>>,
}

impl PyCacheManagerClient {
    fn registered_worker(
        &self,
        registration: &PyGpuRegistration,
    ) -> PyResult<Arc<LocalRestoreWorker>> {
        if !Weak::ptr_eq(&Arc::downgrade(&self.inner), &registration.owner) {
            return Err(PyValueError::new_err(
                "GPU registration belongs to another client",
            ));
        }
        let local = self
            .local
            .lock()
            .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
        local
            .get(&(
                registration.instance_id.clone(),
                registration.tp_rank,
                registration.device_id,
            ))
            .filter(|worker| worker.registration_generation == registration.generation)
            .cloned()
            .ok_or_else(|| PyValueError::new_err("GPU registration is no longer active"))
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

    fn retained_registration_attempts(&self) -> Vec<crate::registration::RetainedAttempt> {
        crate::registration::retained_attempts()
    }

    fn registration_retention(&self) -> (usize, u64, usize) {
        crate::registration::retention()
    }

    fn retained_credit_evidence(&self, py: Python<'_>, instance_id: &str) -> PyResult<String> {
        py.detach(|| crate::registration::retained_credit_evidence(&self.inner, instance_id))
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
    #[pyo3(signature = (instance_id, namespace, tp_rank, pp_rank, tp_size, world_size, device_id, layer_names, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list, transfer_backend, page_first, layer_group_ids=None, layer_formats=None, layer_attention=None, *, tensors))]
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
    ) -> PyResult<PyGpuRegistration> {
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
            || count > 4096
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
        let mut gpu_tensors = Vec::with_capacity(count);
        let mut storages = Vec::with_capacity(count);
        let mut allocations = Vec::with_capacity(count);
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
            let element_size = tensor.call_method0("element_size")?.extract::<u64>()?;
            let mut registration = CudaIpcTensorRegistration {
                format_version: 1,
                allocation_kind: "torch-cuda-ipc-storage".into(),
                storage_size_bytes: size,
                view_offset_bytes: offset,
                shape: tensor.getattr("shape")?.extract()?,
                strides: tensor.call_method0("stride")?.extract()?,
                dtype: tensor
                    .getattr("dtype")?
                    .str()?
                    .to_str()?
                    .strip_prefix("torch.")
                    .ok_or_else(|| PyValueError::new_err("unsupported CUDA tensor dtype"))?
                    .into(),
                element_size,
                export_owner_generation: Vec::new(),
                ..Default::default()
            };
            registration
                .validate_layout()
                .map_err(PyValueError::new_err)?;
            let declared_offset = tensor
                .call_method0("storage_offset")?
                .extract::<u64>()?
                .checked_mul(element_size)
                .ok_or_else(|| PyValueError::new_err("CUDA tensor offset overflow"))?;
            if declared_offset != offset {
                return Err(PyValueError::new_err(
                    "CUDA tensor offset differs from retained storage",
                ));
            }
            registration.view_offset_bytes = declared_offset;
            gpu_tensors.push(registration);
            storages.push(storage.unbind());
            allocations.push(allocation);
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
        let retained_bytes = crate::registration::retained_storage_bytes(
            allocations
                .iter()
                .copied()
                .zip(gpu_tensors.iter().map(|tensor| tensor.storage_size_bytes)),
        )
        .map_err(PyValueError::new_err)?;
        let device_uuid = py
            .import("orbitkv.client.gpu")?
            .call_method1("device_uuid", (device,))?
            .extract::<String>()?;
        let tensors = Arc::new(tensors);
        let mut exporter_owners = tensors
            .iter()
            .map(|tensor| tensor.clone_ref(py))
            .collect::<Vec<_>>();
        exporter_owners.extend(storages.iter().map(|storage| storage.clone_ref(py)));
        let exporter_owners = Arc::new(exporter_owners);
        let attempt = py.detach(|| {
            let local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            if local.contains_key(&key) {
                return Err(PyValueError::new_err(
                    "GPU context is already registered on this client",
                ));
            }
            crate::registration::AttemptOwner::reserve(
                Arc::clone(&self.inner),
                key.clone(),
                Arc::clone(&exporter_owners),
                retained_bytes,
            )
            .map_err(OrbitKVError::new_err)
        })?;
        let export_owner_generation = attempt.0;
        let exported = (|| -> PyResult<()> {
            validate_export_snapshot(py, &tensors, &storages, &allocations, &gpu_tensors, device)?;
            for (storage, registration) in storages.iter().zip(&mut gpu_tensors) {
                let shared = attempt.export(py, storage.bind(py))?;
                let shared = shared.bind(py).cast::<PyTuple>()?;
                if shared.len() != 8
                    || shared.get_item(0)?.extract::<usize>()? != device
                    || shared.get_item(2)?.extract::<u64>()? != registration.storage_size_bytes
                {
                    return Err(PyValueError::new_err("invalid CUDA storage export tuple"));
                }
                registration.export_owner_generation = export_owner_generation.to_vec();
                registration.device_uuid = device_uuid.clone();
                registration.allocation_handle = shared.get_item(1)?.extract()?;
                registration.storage_allocation_offset_bytes = shared.get_item(3)?.extract()?;
                registration.ref_counter_handle = shared.get_item(4)?.extract()?;
                registration.ref_counter_offset = shared.get_item(5)?.extract()?;
                registration.event_handle = shared.get_item(6)?.extract()?;
                registration.event_sync_required = shared.get_item(7)?.extract()?;
                registration
                    .validate(&device_uuid)
                    .map_err(PyValueError::new_err)?;
            }
            validate_export_snapshot(py, &tensors, &storages, &allocations, &gpu_tensors, device)?;
            Ok(())
        })();
        if let Err(error) = exported {
            attempt.rollback(py)?;
            return Err(error);
        }
        let ipc_counters = gpu_tensors
            .iter()
            .map(|tensor| (tensor.ref_counter_handle.clone(), tensor.ref_counter_offset))
            .collect();
        let snapshot_layouts = gpu_tensors.clone();
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
            gpu_tensors,
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
        let retained = Arc::clone(&exporter_owners);
        py.detach(|| {
            let mut local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            attempt.submitted()?;
            let reply = self
                .inner
                .channel()
                .lifecycle(LifecycleCommand::Register, &request.encode_to_vec())
                .map_err(client_error)?;
            let receipt = CudaRegistrationReply::decode(reply.payload.as_slice())
                .map_err(|error| OrbitKVInternal::new_err(error.to_string()))?;
            if receipt.export_owner_generation != export_owner_generation
                || receipt.credits.len() != count
            {
                return Err(OrbitKVError::new_err(
                    "CUDA registration receipt identity differs; owner retained",
                ));
            }
            if receipt.status_code > u16::MAX as u32
                || (receipt.status_code == 0
                    && receipt
                        .credits
                        .iter()
                        .any(|state| *state != CudaCreditDisposition::ImporterOwned as i32))
            {
                return Err(OrbitKVError::new_err(
                    "invalid CUDA registration receipt; owner retained",
                ));
            }
            Python::attach(|py| attempt.resolve(py, &receipt.credits))?;
            if receipt.status_code != 0 {
                return Err(client_error(ChannelError::Lifecycle {
                    code: receipt.status_code as u16,
                    message: receipt.message,
                }));
            }
            if receipt
                .credits
                .iter()
                .any(|state| *state != CudaCreditDisposition::ImporterOwned as i32)
            {
                return Err(OrbitKVError::new_err(
                    "successful CUDA registration lacks importer receipt; owner retained",
                ));
            }
            let worker = (|| -> PyResult<LocalRestoreWorker> {
                Python::attach(|py| {
                    validate_export_snapshot(
                        py,
                        &tensors,
                        &storages,
                        &allocations,
                        &snapshot_layouts,
                        device,
                    )
                })?;
                if receipt.arenas.len() != reply.fds.len() {
                    return Err(OrbitKVInternal::new_err(
                        "invalid payload arena registration reply",
                    ));
                }
                let manager_device_id = receipt.manager_device_id;
                let registration_generation: [u8; 16] = receipt
                    .registration_generation
                    .as_slice()
                    .try_into()
                    .map_err(|_| OrbitKVInternal::new_err("invalid GPU registration generation"))?;
                if registration_generation == [0; 16] || manager_device_id < 0 {
                    return Err(OrbitKVInternal::new_err(
                        "invalid Manager registration identity",
                    ));
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
                for (metadata, fd) in receipt.arenas.into_iter().zip(reply.fds) {
                    if metadata.id == 0 || metadata.size == 0 {
                        return Err(OrbitKVInternal::new_err("invalid arena bounds"));
                    }
                    arenas.push(PayloadArena {
                        id: metadata.id,
                        size: metadata.size,
                        fd,
                    });
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
                    exporter_owners,
                    Arc::clone(&self.completions),
                    arena_count,
                    manager_device_id,
                    registration_generation,
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
            let generation = worker.registration_generation;
            local.insert(key.clone(), Arc::new(worker));
            let credit_owners = attempt.accepted()?;
            let mut evidence = crate::registration::credit_evidence(&credit_owners);
            evidence["export_owner_generation"] = uuid::Uuid::from_bytes(export_owner_generation)
                .to_string()
                .into();
            evidence["registration_generation"] =
                uuid::Uuid::from_bytes(generation).to_string().into();
            let ipc_credit_evidence = evidence.to_string();
            Ok(PyGpuRegistration {
                instance_id: key.0,
                tp_rank,
                pp_rank,
                device_id,
                generation,
                export_owner_generation,
                ipc_counters,
                ipc_credit_evidence,
                owner: Arc::downgrade(&self.inner),
                _tensors: retained,
            })
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

    fn peer_alive(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.channel().peer_alive())
            .map_err(client_error)
    }

    fn publish_progress(&self, py: Python<'_>) -> PyResult<PyPublishProgress> {
        py.detach(|| self.inner.publish_progress())
            .map(Into::into)
            .map_err(client_error)
    }

    fn unregister_context(&self, py: Python<'_>, instance_id: String) -> PyResult<(bool, String)> {
        py.detach(|| {
            let mut local = self
                .local
                .lock()
                .map_err(|_| OrbitKVInternal::new_err("local executor lock poisoned"))?;
            if crate::registration::instance_pending(&self.inner, &instance_id) {
                return Err(OrbitKVError::new_err(
                    "CUDA registration attempt pending or retained; unregister refused",
                ));
            }
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
        registration: &PyGpuRegistration,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String)> {
        py.detach(|| {
            let worker = self.registered_worker(registration)?;
            let request = publish_request(
                registration.instance_id.clone(),
                registration.tp_rank,
                registration.pp_rank,
                worker.manager_device_id,
                registration.generation,
                saves,
            );
            self.inner.publish(&request).map_err(client_error)?;
            Ok((true, String::new()))
        })
    }

    fn save_diagnostic(
        &self,
        py: Python<'_>,
        registration: &PyGpuRegistration,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String, PyChannelCallObservation)> {
        py.detach(|| {
            let worker = self.registered_worker(registration)?;
            let request = publish_request(
                registration.instance_id.clone(),
                registration.tp_rank,
                registration.pp_rank,
                worker.manager_device_id,
                registration.generation,
                saves,
            );
            let observation = self
                .inner
                .publish_observed(&request)
                .map_err(client_error)?;
            Ok((true, String::new(), PyChannelCallObservation(observation)))
        })
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

#[cfg(test)]
#[path = "../tests/unit/client.rs"]
mod tests;
