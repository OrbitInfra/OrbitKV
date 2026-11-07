use orbitkv_core::transfer_engine::{CudaRegistration, TransferBackend};
use orbitkv_proto::proto::engine::{CudaCreditDisposition, CudaIpcTensorRegistration};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub device_id: i32,
    pub owner: Option<Arc<CudaRegistration>>,
}

struct LayerTensor {
    _tensor: Arc<Py<PyAny>>,
    metadata: TensorMetadata,
    credit_identity: (Vec<u8>, u64),
}

// The last actual tensor owner releases the Python CUDA IPC storage credit.
// Registry removal cannot unmap backing retained by a submitted engine READ.

struct ContextState {
    device_id: i32,
    tensors: HashMap<String, LayerTensor>,
    export_owner_generation: Option<Vec<u8>>,
}

impl ContextState {
    fn new(device_id: i32) -> Self {
        Self {
            device_id,
            tensors: HashMap::new(),
            export_owner_generation: None,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ContextKey {
    pub(crate) instance_id: String,
    pub(crate) tp_rank: u32,
    pub(crate) pp_rank: u32,
    pub(crate) device_uuid: String,
}

pub struct CudaTensorRegistry {
    contexts: HashMap<ContextKey, ContextState>,
}

impl CudaTensorRegistry {
    pub fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let torch = py.import("torch")?;
            let sys = py.import("sys")?;
            log::info!(
                "CUDA registration Python runtime: {}",
                serde_json::json!({
                    "pid": std::process::id(),
                    "torch_file": torch.getattr("__file__")?.extract::<String>()?,
                    "sys_path": sys.getattr("path")?.extract::<Vec<String>>()?,
                    "prefix": sys.getattr("prefix")?.extract::<String>()?,
                    "base_prefix": sys.getattr("base_prefix")?.extract::<String>()?,
                    "no_user_site": sys.getattr("flags")?.getattr("no_user_site")?.extract::<u8>()?,
                })
            );
            let cuda = torch.getattr("cuda")?;
            cuda.call_method0("init")?;
            Ok(Self {
                contexts: HashMap::new(),
            })
        })
    }

    pub fn empty() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    fn register_layers(
        &mut self,
        context_key: &ContextKey,
        layers: Vec<(String, CudaIpcTensorRegistration)>,
        disposition: &mut CudaCreditDisposition,
        backend: Option<Arc<dyn TransferBackend>>,
    ) -> PyResult<Vec<TensorMetadata>> {
        *disposition = CudaCreditDisposition::Unknown;
        if self.contexts.contains_key(context_key) {
            return Err(PyValueError::new_err(format!(
                "context {context_key:?} is already registered"
            )));
        }

        let mut seen_layers = HashSet::with_capacity(layers.len());
        for (layer_name, _) in &layers {
            if !seen_layers.insert(layer_name.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "layer {layer_name} appears more than once in context {context_key:?}"
                )));
            }
        }

        let device_uuid = &context_key.device_uuid;
        let mut credits: HashSet<(Vec<u8>, u64)> = self
            .contexts
            .values()
            .flat_map(|context| {
                context
                    .tensors
                    .values()
                    .map(|tensor| tensor.credit_identity.clone())
            })
            .collect();
        for (_, registration) in &layers {
            if !credits.insert((
                registration.ref_counter_handle.clone(),
                registration.ref_counter_offset,
            )) {
                return Err(PyValueError::new_err(
                    "CUDA IPC export credit already has an importer owner",
                ));
            }
        }
        let export_owner_generation = layers
            .first()
            .map(|(_, tensor)| &tensor.export_owner_generation);
        for (_, registration) in &layers {
            if Some(&registration.export_owner_generation) != export_owner_generation {
                return Err(PyValueError::new_err(
                    "CUDA export batch spans different owners",
                ));
            }
            registration
                .validate(device_uuid)
                .map_err(PyValueError::new_err)?;
        }
        *disposition = CudaCreditDisposition::NeverImported;
        let device_id = Python::attach(|py| {
            let gpu = py.import("orbitkv.client.gpu")?;
            log::info!(
                "CUDA registration Python helper: {}",
                serde_json::json!({
                    "pid": std::process::id(),
                    "gpu_module_file": gpu.getattr("__file__")?.extract::<String>()?,
                })
            );
            gpu.call_method1("device_index_from_uuid", (device_uuid,))?
                .extract::<i32>()
        })?;
        let mut context = ContextState::new(device_id);
        context.export_owner_generation = export_owner_generation.cloned();
        let mut metadatas = Vec::with_capacity(layers.len());
        for (layer_name, registration) in layers {
            *disposition = CudaCreditDisposition::Unknown;
            let mut layer_tensor = Self::materialize_tensor(device_id, device_uuid, &registration)?;
            let base = layer_tensor.metadata.data_ptr - registration.view_offset_bytes;
            let length = registration.storage_size_bytes as usize;
            let shared = self
                .contexts
                .values()
                .chain(std::iter::once(&context))
                .filter(|context| context.device_id == device_id)
                .flat_map(|context| context.tensors.values())
                .filter_map(|tensor| tensor.metadata.owner.as_ref())
                .find(|owner| owner.covers(base, length))
                .cloned();
            layer_tensor.metadata.owner = Some(match shared {
                Some(owner) => owner,
                None => {
                    let cuda = cudarc::driver::CudaContext::new(device_id as usize)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    // SAFETY: the imported tensor retains this validated storage range;
                    // native registration also owns its CUDA context through cleanup.
                    unsafe {
                        CudaRegistration::new(
                            backend.clone(),
                            base,
                            length,
                            device_id,
                            Arc::new((Arc::clone(&layer_tensor._tensor), cuda)),
                        )
                    }
                    .map_err(|error| PyValueError::new_err(error.to_string()))?
                }
            });
            let metadata = layer_tensor.metadata.clone();

            if context.device_id != metadata.device_id {
                return Err(PyValueError::new_err(format!(
                    "context {context_key:?} is pinned to device {} but got {}",
                    context.device_id, metadata.device_id
                )));
            }

            context.tensors.insert(layer_name, layer_tensor);
            metadatas.push(metadata);
        }

        self.contexts.insert(context_key.clone(), context);
        *disposition = CudaCreditDisposition::ImporterOwned;
        Ok(metadatas)
    }

    fn drop_context(&mut self, context_key: &ContextKey) -> Result<usize, String> {
        self.release_contexts(vec![context_key.clone()])
    }

    fn drop_instance(&mut self, instance_id: &str) -> Result<usize, String> {
        let keys: Vec<ContextKey> = self
            .contexts
            .keys()
            .filter(|key| key.instance_id == instance_id)
            .cloned()
            .collect();
        self.release_contexts(keys)
    }

    /// Clear all contexts and return the total number of tensors removed.
    fn clear_and_count(&mut self) -> Result<usize, String> {
        let keys: Vec<ContextKey> = self.contexts.keys().cloned().collect();
        self.release_contexts(keys)
    }

    /// Remove `keys` from the registry, returning the number of CUDA IPC tensors
    /// released.
    ///
    /// This is the single place that decides whether GIL + CUDA teardown is
    /// needed, so the decision can't drift between callers: it acquires the GIL
    /// and runs `gc.collect()` + `torch.cuda.empty_cache()` ONLY when the
    /// removed contexts actually hold live tensors. Dropping empty contexts is
    /// pure Rust, so an idle/empty registry never forces a CUDA device sync —
    /// which would block forever on a wedged GPU.
    fn release_contexts(&mut self, keys: Vec<ContextKey>) -> Result<usize, String> {
        for context in keys.iter().filter_map(|key| self.contexts.get(key)) {
            for tensor in context.tensors.values() {
                if let Some(owner) = &tensor.metadata.owner {
                    let shared = self
                        .contexts
                        .iter()
                        .filter(|(key, _)| !keys.contains(key))
                        .flat_map(|(_, context)| context.tensors.values())
                        .any(|other| {
                            other
                                .metadata
                                .owner
                                .as_ref()
                                .is_some_and(|other| Arc::ptr_eq(owner, other))
                        });
                    if !shared {
                        owner.close().map_err(|error| error.to_string())?;
                    }
                }
            }
        }
        let tensor_count: usize = keys
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .map(|ctx| ctx.tensors.len())
            .sum();

        if tensor_count == 0 {
            for key in &keys {
                self.contexts.remove(key);
            }
            return Ok(0);
        }

        // Remove contexts under the GIL so each `Py<PyAny>` is dropped (decref)
        // there, then force gc + empty_cache to actually unmap the CUDA IPC
        // memory immediately instead of letting Python's GC defer it.
        Python::attach(|py| -> PyResult<()> {
            for key in &keys {
                self.contexts.remove(key);
            }

            let gc = py.import("gc")?;
            let _ = gc.call_method0("collect");

            let torch = py.import("torch")?;
            let cuda = torch.getattr("cuda")?;
            let _ = cuda.call_method0("empty_cache");
            Ok(())
        })
        .map_err(|error| error.to_string())?;

        Ok(tensor_count)
    }

    fn materialize_tensor(
        device_id: i32,
        device_uuid: &str,
        registration: &CudaIpcTensorRegistration,
    ) -> PyResult<LayerTensor> {
        registration
            .validate(device_uuid)
            .map_err(PyValueError::new_err)?;
        Python::attach(|py| {
            let torch = py.import("torch")?;
            let cuda = torch.getattr("cuda")?;
            cuda.call_method1("set_device", (device_id,))?;
            let storage = torch.getattr("UntypedStorage")?.call_method1(
                "_new_shared_cuda",
                (
                    device_id,
                    PyBytes::new(py, &registration.allocation_handle),
                    registration.storage_size_bytes,
                    registration.storage_allocation_offset_bytes,
                    PyBytes::new(py, &registration.ref_counter_handle),
                    registration.ref_counter_offset,
                    PyBytes::new(py, &registration.event_handle),
                    registration.event_sync_required,
                ),
            )?;
            let dtype = torch.getattr(registration.dtype.as_str())?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("device", format!("cuda:{device_id}"))?;
            kwargs.set_item("dtype", dtype)?;
            let tensor = torch.getattr("empty")?.call(((0_u64,),), Some(&kwargs))?;
            tensor.call_method1(
                "set_",
                (
                    &storage,
                    registration.view_offset_bytes / registration.element_size,
                    &registration.shape,
                    &registration.strides,
                ),
            )?;
            let data_ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;
            let actual_device: i32 = tensor.getattr("device")?.getattr("index")?.extract()?;
            let actual_size: u64 = storage.call_method0("nbytes")?.extract()?;
            let base: u64 = storage.call_method0("data_ptr")?.extract()?;
            let actual_offset: u64 = tensor.call_method0("storage_offset")?.extract()?;
            let actual_dtype = tensor.getattr("dtype")?.str()?.to_string();
            let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
            let strides: Vec<usize> = tensor.call_method0("stride")?.extract()?;
            let element_size: u64 = tensor.call_method0("element_size")?.extract()?;
            if base.checked_add(registration.view_offset_bytes) != Some(data_ptr)
                || actual_offset.checked_mul(registration.element_size)
                    != Some(registration.view_offset_bytes)
                || data_ptr == 0
                || !data_ptr.is_multiple_of(registration.element_size)
                || actual_dtype != format!("torch.{}", registration.dtype)
                || actual_device != device_id
                || actual_size != registration.storage_size_bytes
                || element_size != registration.element_size
                || shape
                    .iter()
                    .map(|n| *n as u64)
                    .ne(registration.shape.iter().copied())
                || strides
                    .iter()
                    .map(|n| *n as u64)
                    .ne(registration.strides.iter().copied())
            {
                return Err(PyValueError::new_err(
                    "CUDA importer differs from validated registration",
                ));
            }
            let size_bytes = actual_size
                .checked_sub(registration.view_offset_bytes)
                .and_then(|size| usize::try_from(size).ok())
                .ok_or_else(|| PyValueError::new_err("CUDA registration view size overflow"))?;
            Ok(LayerTensor {
                _tensor: Arc::new(tensor.unbind()),
                credit_identity: (
                    registration.ref_counter_handle.clone(),
                    registration.ref_counter_offset,
                ),
                metadata: TensorMetadata {
                    owner: None,
                    dtype: format!("torch.{}", registration.dtype),
                    shape,
                    strides,
                    data_ptr,
                    size_bytes,
                    device_id,
                },
            })
        })
    }
}

#[derive(Debug)]
pub(crate) struct ImportFailure {
    pub(crate) message: String,
    pub(crate) disposition: CudaCreditDisposition,
}

/// Work submitted to the dedicated registry thread. Each carries a `oneshot`
/// the actor uses to hand the result back to the awaiting caller.
enum RegistryCommand {
    RegisterLayers {
        context_key: ContextKey,
        /// Explicit tensor/storage exports for each layer in the batch.
        layers: Vec<(String, CudaIpcTensorRegistration)>,
        backend: Option<Arc<dyn TransferBackend>>,
        // The `PyErr` is stringified on the actor thread (which holds the GIL),
        // so callers never need to touch the GIL to read an error message.
        reply: oneshot::Sender<Result<Vec<TensorMetadata>, ImportFailure>>,
    },
    DropInstance {
        instance_id: String,
        reply: oneshot::Sender<Result<usize, String>>,
    },
    DropContext {
        context_key: ContextKey,
        reply: oneshot::Sender<Result<usize, String>>,
    },
    Clear {
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

/// Async handle to a [`CudaTensorRegistry`] that lives on its own OS thread.
///
/// Every mutating op takes the GIL and may run a blocking
/// `torch.cuda.empty_cache()` that performs a CUDA device sync — which never
/// returns if the GPU is wedged. Confining the registry to one dedicated thread
/// keeps that blocking, GIL-bearing work off the async runtime *by
/// construction*: handlers only ever `.await` a reply, so a wedged CUDA call
/// pins this single thread instead of starving tokio workers (the outage where
/// a few `cleanup` calls hung every endpoint, `/health` and `/metrics`
/// included). Serializing on one thread also matches the GIL's own
/// serialization — registry ops were never able to run concurrently anyway.
#[derive(Clone)]
pub struct RegistryHandle {
    tx: mpsc::Sender<RegistryCommand>,
}

impl RegistryHandle {
    /// Move `registry` onto a dedicated `cuda-registry` thread and return an
    /// async handle to it. The thread runs until every handle is dropped.
    pub fn spawn(registry: CudaTensorRegistry) -> std::io::Result<Self> {
        // Bounds how many register/cleanup requests queue before callers await
        // for space; the actor drains them one at a time under the GIL.
        let (tx, rx) = mpsc::channel(64);
        std::thread::Builder::new()
            .name("cuda-registry".to_string())
            .spawn(move || registry_actor(registry, rx))
            .map(|_| Self { tx })
    }

    /// Materialize and register a batch of layers under `context_key`. Returns
    /// per-layer metadata in input order. The batch is transactional with
    /// respect to the registry: an existing context is rejected before any
    /// tensor is materialized, and a materialization failure does not publish a
    /// partial context.
    pub(crate) async fn register_layers(
        &self,
        context_key: ContextKey,
        layers: Vec<(String, CudaIpcTensorRegistration)>,
        backend: Option<Arc<dyn TransferBackend>>,
    ) -> Result<Vec<TensorMetadata>, ImportFailure> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::RegisterLayers {
            context_key,
            layers,
            backend,
            reply,
        })
        .await
        .map_err(|message| ImportFailure {
            message,
            disposition: CudaCreditDisposition::NeverImported,
        })?;
        rx.await.map_err(|error| ImportFailure {
            message: format!("cuda-registry reply lost: {error}"),
            disposition: CudaCreditDisposition::Unknown,
        })?
    }

    /// Drop all CUDA tensors belonging to `instance_id`; returns the count
    /// released.
    pub async fn drop_instance(&self, instance_id: String) -> Result<usize, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropInstance { instance_id, reply })
            .await?;
        rx.await
            .map_err(|error| format!("cuda-registry reply lost: {error}"))?
    }

    /// Drop exactly one context; returns the number of CUDA tensors released.
    pub(crate) async fn drop_context(&self, context_key: ContextKey) -> Result<usize, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropContext { context_key, reply })
            .await?;
        rx.await
            .map_err(|error| format!("cuda-registry reply lost: {error}"))?
    }

    /// Drop every registered tensor; returns the count released.
    pub async fn clear(&self) -> Result<usize, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::Clear { reply }).await?;
        rx.await
            .map_err(|error| format!("cuda-registry reply lost: {error}"))?
    }

    async fn dispatch(&self, cmd: RegistryCommand) -> Result<(), String> {
        self.tx
            .send(cmd)
            .await
            .map_err(|_| "cuda-registry thread is gone".into())
    }
}

/// Owns the registry and drains commands serially. Each op runs synchronously
/// on this thread, so a GIL/CUDA stall blocks only here.
fn registry_actor(mut registry: CudaTensorRegistry, mut rx: mpsc::Receiver<RegistryCommand>) {
    while let Some(cmd) = rx.blocking_recv() {
        match cmd {
            RegistryCommand::RegisterLayers {
                context_key,
                layers,
                backend,
                reply,
            } => {
                let mut disposition = CudaCreditDisposition::Unknown;
                let result = registry
                    .register_layers(&context_key, layers, &mut disposition, backend)
                    // Stringify here, on the GIL-owning thread, so the gRPC
                    // handler never needs `Python::attach` just to read the message.
                    .map_err(|err| ImportFailure {
                        message: Python::attach(|py| err.value(py).to_string()),
                        disposition,
                    });
                let _ = reply.send(result);
            }
            RegistryCommand::DropInstance { instance_id, reply } => {
                let _ = reply.send(registry.drop_instance(&instance_id));
            }
            RegistryCommand::DropContext { context_key, reply } => {
                let _ = reply.send(registry.drop_context(&context_key));
            }
            RegistryCommand::Clear { reply } => {
                let _ = reply.send(registry.clear_and_count());
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/registry.rs"]
mod tests;
