//! Engine-owned raw Restore copies from independently imported payload arenas.

use std::collections::{HashMap, HashSet, VecDeque};
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};

use super::layout::{KVCacheGeometry, KVCacheLayout};
use super::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend, TransferMode};
use crate::PayloadArena;

mod trace;
use trace::LocalCopyTrace;

pub const MAX_PLAN_BYTES: usize = 1024 * 1024;
pub const MAX_RESTORE_PLAN_BYTES: usize = 32 * MAX_PLAN_BYTES;

/// A checked range inside one live allocation of a shared payload arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRange {
    pub arena_id: u64,
    pub allocation_id: u64,
    pub allocation_offset: u64,
    pub allocation_size: u64,
    /// Absolute byte offset within the arena, not a process address.
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawCopy {
    pub source: SourceRange,
    pub layer: String,
    pub destination_offset: u64,
}

/// Pointer-free plan read only after the engine claims its source grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawRestorePlan {
    pub copies: Vec<RawCopy>,
}

/// One bounded wire part and the layers whose final ranges it contains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawRestorePart {
    pub copies: Vec<RawCopy>,
    pub completed_layers: Vec<String>,
}

impl RawRestorePlan {
    pub fn encode_parts(&self) -> Result<VecDeque<Vec<u8>>, String> {
        let last_copy: HashMap<_, _> = self
            .copies
            .iter()
            .enumerate()
            .map(|(index, copy)| (copy.layer.as_str(), index))
            .collect();
        let mut parts = VecDeque::new();
        let mut bytes = Vec::from([2, 0, 0, 0, 0, 0, 0, 0]);
        let mut count = 0_u32;
        let mut total = 8usize;
        for (index, copy) in self.copies.iter().enumerate() {
            let name_len = u16::try_from(copy.layer.len())
                .ok()
                .filter(|len| *len != 0)
                .ok_or("invalid Restore layer name length")?;
            let size = 59 + usize::from(name_len);
            if bytes.len() + size > MAX_PLAN_BYTES {
                bytes[4..8].copy_from_slice(&count.to_le_bytes());
                parts.push_back(bytes);
                bytes = Vec::from([2, 0, 0, 0, 0, 0, 0, 0]);
                count = 0;
                total += 8;
            }
            total = total
                .checked_add(size)
                .filter(|size| *size <= MAX_RESTORE_PLAN_BYTES)
                .ok_or("Restore plan exceeds the operation metadata limit")?;
            count += 1;
            let source = &copy.source;
            for value in [
                source.arena_id,
                source.allocation_id,
                source.allocation_offset,
                source.allocation_size,
                source.offset,
                source.size,
                copy.destination_offset,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.extend_from_slice(&name_len.to_le_bytes());
            bytes.extend_from_slice(copy.layer.as_bytes());
            bytes.push(u8::from(last_copy[copy.layer.as_str()] == index));
        }
        bytes[4..8].copy_from_slice(&count.to_le_bytes());
        parts.push_back(bytes);
        Ok(parts)
    }
}

impl RawRestorePart {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 || bytes.len() > MAX_PLAN_BYTES || bytes[..4] != 2_u32.to_le_bytes() {
            return Err("invalid raw Restore plan header".into());
        }
        let count =
            u32::from_le_bytes(bytes[4..8].try_into().map_err(|_| "invalid plan count")?) as usize;
        if count > (bytes.len() - 8) / 59 {
            return Err("invalid raw Restore copy count".into());
        }
        let mut copies = Vec::with_capacity(count);
        let mut completed_layers = Vec::new();
        let mut completed = HashSet::new();
        let mut remaining = &bytes[8..];
        for _ in 0..count {
            if remaining.len() < 59 {
                return Err("truncated raw Restore copy".into());
            }
            let mut words = [0_u64; 7];
            for (index, word) in words.iter_mut().enumerate() {
                *word = u64::from_le_bytes(
                    remaining[index * 8..index * 8 + 8]
                        .try_into()
                        .map_err(|_| "invalid copy word")?,
                );
            }
            let name_len = u16::from_le_bytes([remaining[56], remaining[57]]) as usize;
            if name_len == 0 || remaining.len() < 59 + name_len {
                return Err("invalid raw Restore layer name".into());
            }
            let layer = std::str::from_utf8(&remaining[58..58 + name_len])
                .map_err(|_| "Restore layer name is not UTF-8")?
                .to_owned();
            if completed.contains(&layer) {
                return Err("Restore copy follows layer completion".into());
            }
            match remaining[58 + name_len] {
                0 => {}
                1 => {
                    completed.insert(layer.clone());
                    completed_layers.push(layer.clone());
                }
                _ => return Err("invalid Restore layer completion marker".into()),
            }
            remaining = &remaining[59 + name_len..];
            copies.push(RawCopy {
                source: SourceRange {
                    arena_id: words[0],
                    allocation_id: words[1],
                    allocation_offset: words[2],
                    allocation_size: words[3],
                    offset: words[4],
                    size: words[5],
                },
                layer,
                destination_offset: words[6],
            });
        }
        if !remaining.is_empty() {
            return Err("trailing raw Restore plan bytes".into());
        }
        Ok(Self {
            copies,
            completed_layers,
        })
    }
}

/// Local binding of the exact tensor view registered with the Manager.
/// The caller must retain the tensor allocation until this binding is dropped.
pub struct LocalTensor {
    name: String,
    address: u64,
    bytes: usize,
    allocation: usize,
}

impl LocalTensor {
    #[allow(
        clippy::too_many_arguments,
        reason = "validates the registered tensor geometry"
    )]
    pub fn new(
        name: String,
        address: u64,
        bytes: usize,
        allocation: usize,
        num_blocks: usize,
        segment_bytes: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) -> Result<Self, String> {
        let geometry = KVCacheGeometry::new(
            num_blocks,
            segment_bytes,
            kv_stride_bytes,
            segments,
            None,
            1,
        )?;
        let extent = match geometry.block_ranges(num_blocks - 1)? {
            super::layout::BlockRanges::Contiguous(range) => range.end,
            super::layout::BlockRanges::Split { k, v } => k.end.max(v.end),
        };
        KVCacheLayout::bind(address, bytes, geometry)?;
        Ok(Self {
            name,
            address,
            bytes: extent,
            allocation,
        })
    }
}

struct ImportedArena {
    pointer: NonNull<u8>,
    device_pointer: u64,
    size: usize,
    _fd: OwnedFd,
    context: Arc<CudaContext>,
}

// SAFETY: the mapping is stable, shared and CUDA registered for its lifetime.
// Only the serialized executor submits reads, and drains before dropping it.
unsafe impl Send for ImportedArena {}

impl ImportedArena {
    fn new(arena: PayloadArena, context: Arc<CudaContext>) -> Result<Self, String> {
        let size = usize::try_from(arena.size).map_err(|_| "payload arena size overflow")?;
        if arena.id == 0 || size == 0 || size > isize::MAX as usize {
            return Err("invalid payload arena identity or size".into());
        }
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: stat is valid output storage; fd is independently owned.
        if unsafe { libc::fstat(arena.fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        // SAFETY: fstat initialized the full result on success.
        let stat = unsafe { stat.assume_init() };
        let required = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        // SAFETY: F_GET_SEALS does not consume or mutate the owned descriptor.
        let seals = unsafe { libc::fcntl(arena.fd.as_raw_fd(), libc::F_GET_SEALS) };
        if stat.st_size < 0
            || stat.st_size as u64 != arena.size
            || seals < 0
            || seals & required != required
        {
            return Err("payload arena size or seals do not match registration".into());
        }
        context.bind_to_thread().map_err(|e| e.to_string())?;
        // SAFETY: validated sealed file covers this nonempty mapping.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                arena.fd.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().to_string());
        }
        // SAFETY: mapping is live and writable for size bytes in the current context.
        if let Err(error) = unsafe {
            sys::cuMemHostRegister_v2(pointer, size, sys::CU_MEMHOSTREGISTER_DEVICEMAP).result()
        } {
            unsafe { libc::munmap(pointer, size) };
            return Err(format!("payload CUDA registration failed: {error}"));
        }
        let mut device_pointer = 0;
        // SAFETY: registration succeeded and output pointer is valid.
        if let Err(error) =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device_pointer, pointer, 0).result() }
        {
            unsafe {
                sys::cuMemHostUnregister(pointer);
                libc::munmap(pointer, size);
            }
            return Err(format!("payload CUDA address failed: {error}"));
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or("null payload mapping")?;
        Ok(Self {
            pointer,
            device_pointer,
            size,
            _fd: arena.fd,
            context,
        })
    }
}

impl Drop for ImportedArena {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() {
            std::process::abort();
        }
        // SAFETY: the owning executor drains every submission before drop.
        if unsafe { sys::cuMemHostUnregister(self.pointer.as_ptr().cast()).result() }.is_err() {
            std::process::abort();
        }
        unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.size) };
    }
}

/// One engine GPU's stream, tensor bindings and imported source mappings.
/// All submitted work drains inside `execute`, including partial failures.
pub struct LocalRestoreExecutor {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    readiness: CudaEvent,
    backend: Box<dyn TransferBackend>,
    tensors: HashMap<String, LocalTensor>,
    arenas: HashMap<u64, ImportedArena>,
    trace: Option<LocalCopyTrace>,
}

/// CUDA operations in registration and readiness run on a framework thread.
/// Restore its context after all temporary CUDA owners have been dropped.
struct CallerContext(Option<sys::CUcontext>);

impl CallerContext {
    fn capture() -> Result<Self, String> {
        result::ctx::get_current()
            .map(Self)
            .map_err(|error| error.to_string())
    }
}

impl Drop for CallerContext {
    fn drop(&mut self) {
        // SAFETY: this context belonged to the caller before entering native
        // work; none of these operations destroys it. Null restores no context.
        if let Err(error) =
            unsafe { result::ctx::set_current(self.0.unwrap_or(std::ptr::null_mut())) }
        {
            log::error!("Cannot restore framework CUDA context: {error}; terminating engine");
            std::process::abort();
        }
    }
}

impl LocalRestoreExecutor {
    /// Attach to the same primary context as the exporting framework tensors.
    /// `device` is the process-local CUDA ordinal, not Manager's physical ID.
    pub fn new(
        device: usize,
        tensors: Vec<LocalTensor>,
        arenas: Vec<PayloadArena>,
        mode: TransferMode,
    ) -> Result<Self, String> {
        let _caller_context = CallerContext::capture()?;
        let context = CudaContext::new(device).map_err(|e| e.to_string())?;
        let mut bindings = HashMap::new();
        for tensor in tensors {
            let mut tensor_context = std::ptr::null_mut();
            // SAFETY: CUDA validates the exported device pointer; output is a context handle.
            unsafe {
                sys::cuPointerGetAttribute(
                    (&mut tensor_context as *mut sys::CUcontext).cast(),
                    sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                    tensor.address,
                )
                .result()
            }
            .map_err(|e| format!("invalid tensor CUDA allocation: {e}"))?;
            if tensor_context != context.cu_ctx() {
                return Err(
                    "tensor allocation does not belong to the framework primary context".into(),
                );
            }
            if bindings.insert(tensor.name.clone(), tensor).is_some() {
                return Err("duplicate local tensor layer".into());
            }
        }
        if arenas.len() > 64 {
            return Err("too many payload arena mappings".into());
        }
        let mut imported = HashMap::new();
        for arena in arenas {
            let id = arena.id;
            if imported.contains_key(&id) {
                return Err("duplicate payload arena identity".into());
            }
            imported.insert(id, ImportedArena::new(arena, Arc::clone(&context))?);
        }
        let stream = context.new_stream().map_err(|e| e.to_string())?;
        let readiness = context
            .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(|e| e.to_string())?;
        let backend: Box<dyn TransferBackend> = match mode {
            TransferMode::Direct => Box::new(MemcpyBackend::new(&context)?),
            TransferMode::Kernel => Box::new(KernelBackend::new(&context)?),
        };
        let limit = orbitkv_common::timeline::diagnostic_event_limit();
        let trace = (limit > 0).then(|| {
            LocalCopyTrace::new(
                device,
                MemcpyBackend::new(&context).map(|dma| dma.max_pitch),
                limit,
            )
        });
        Ok(Self {
            context,
            stream,
            readiness,
            backend,
            tensors: bindings,
            arenas: imported,
            trace,
        })
    }

    pub fn validate_layer_events(&self, events: &HashMap<String, u64>) -> Result<(), String> {
        let mut handles = HashSet::new();
        for (layer, event) in events {
            if !self.tensors.contains_key(layer) || *event == 0 || !handles.insert(*event) {
                return Err(
                    "Restore events require unique live events for registered layers".into(),
                );
            }
        }
        Ok(())
    }

    /// Capture the caller's actual previous-user stream and fence it before
    /// requesting a plan (which may select a Manager-owned codec/SSD route).
    pub fn wait_for_destination(&mut self, ready_stream: u64) -> Result<(), String> {
        let _caller_context = CallerContext::capture()?;
        self.context.bind_to_thread().map_err(|e| e.to_string())?;
        let stream = ready_stream as sys::CUstream;
        let mut stream_context = std::ptr::null_mut();
        // SAFETY: CUDA validates the borrowed framework stream; we never destroy it.
        unsafe { sys::cuStreamGetCtx(stream, &mut stream_context).result() }
            .map_err(|e| e.to_string())?;
        if stream_context != self.context.cu_ctx() {
            return Err("Restore readiness stream belongs to another CUDA context".into());
        }
        // An idle stream already proves all preceding users have finished.
        // Do not submit an otherwise unnecessary event to the GPU in that case.
        // SAFETY: stream was validated in this context and remains caller-owned.
        match unsafe { sys::cuStreamQuery(stream) } {
            sys::CUresult::CUDA_SUCCESS => return Ok(()),
            sys::CUresult::CUDA_ERROR_NOT_READY => {}
            error => return Err(format!("Restore readiness query failed: {error:?}")),
        }
        // SAFETY: event belongs to this context and stream was verified above.
        unsafe { result::event::record(self.readiness.cu_event(), stream) }
            .map_err(|e| e.to_string())?;
        self.readiness.synchronize().map_err(|e| e.to_string())
    }

    /// Caller holds the Active grant until this method returns and publishes
    /// Drained afterwards. Sources cannot be recycled during preparation or DMA.
    pub fn execute(
        &mut self,
        plan: &RawRestorePart,
        layer_events: &mut HashMap<String, u64>,
        final_part: bool,
        enqueued: impl FnOnce(),
        submitted_at: Option<&mut Option<std::time::Instant>>,
    ) -> Result<(), String> {
        #[cfg(feature = "test-hooks")]
        crate::test_faults::pause_blocking("local_restore_claim");
        self.context.bind_to_thread().map_err(|e| e.to_string())?;
        let mut copies = Vec::with_capacity(plan.copies.len());
        let mut allocations = HashMap::new();
        for copy in &plan.copies {
            let source = &copy.source;
            let arena = self
                .arenas
                .get(&source.arena_id)
                .ok_or("Restore source arena is not registered")?;
            let allocation_end = source
                .allocation_offset
                .checked_add(source.allocation_size)
                .ok_or("source allocation overflow")?;
            let source_end = source
                .offset
                .checked_add(source.size)
                .ok_or("source copy overflow")?;
            if source.allocation_id == 0
                || source.size == 0
                || source.allocation_size == 0
                || allocation_end > arena.size as u64
                || source.offset < source.allocation_offset
                || source_end > allocation_end
            {
                return Err("Restore source exceeds its granted allocation".into());
            }
            let key = (source.arena_id, source.allocation_id);
            let bounds = (source.allocation_offset, source.allocation_size);
            let previous_bounds = allocations.entry(key).or_insert(bounds);
            if *previous_bounds != bounds {
                return Err("inconsistent Restore allocation generation bounds".into());
            }
            let tensor = self
                .tensors
                .get(&copy.layer)
                .ok_or("Restore destination layer is not registered")?;
            let end = copy
                .destination_offset
                .checked_add(source.size)
                .ok_or("destination copy overflow")?;
            if end > tensor.bytes as u64 {
                return Err("Restore destination exceeds retained tensor allocation".into());
            }
            copies.push(CopyDesc {
                device: tensor
                    .address
                    .checked_add(copy.destination_offset)
                    .ok_or("destination address overflow")?,
                // SAFETY: source offset is within the live imported arena.
                host: unsafe { arena.pointer.as_ptr().add(source.offset as usize) },
                host_device: arena
                    .device_pointer
                    .checked_add(source.offset)
                    .ok_or("source GPU address overflow")?,
                size: source.size as usize,
                device_allocation: tensor.allocation,
                host_registration: arena.pointer.as_ptr() as usize,
            });
        }
        // Validate the whole part without changing its semantic layer order.
        let mut targets: Vec<_> = copies.iter().map(|copy| (copy.device, copy.size)).collect();
        targets.sort_unstable();
        for pair in targets.windows(2) {
            if pair[0]
                .0
                .checked_add(pair[0].1 as u64)
                .is_none_or(|end| end > pair[1].0)
            {
                return Err("overlapping Restore destination ranges".into());
            }
        }
        if let Some(trace) = self.trace.as_mut() {
            trace.begin_part(copies.len(), final_part);
            for (raw, copy) in plan.copies.iter().zip(&copies) {
                if let (Some(tensor), Some(arena)) = (
                    self.tensors.get(&raw.layer),
                    self.arenas.get(&raw.source.arena_id),
                ) {
                    trace.record_geometry(raw, copy, tensor, arena.size);
                }
            }
        }
        let mut batches: Vec<(&str, Vec<CopyDesc>)> = Vec::new();
        let mut indices = HashMap::new();
        for (raw, copy) in plan.copies.iter().zip(copies) {
            let layer = if layer_events.contains_key(&raw.layer) {
                raw.layer.as_str()
            } else {
                ""
            };
            let index = *indices.entry(layer).or_insert_with(|| {
                batches.push((layer, Vec::new()));
                batches.len() - 1
            });
            batches[index].1.push(copy);
        }
        let submitted = (|| -> Result<(), String> {
            for (layer, mut copies) in batches {
                copies.sort_unstable_by_key(|copy| copy.device);
                let captured = self
                    .trace
                    .as_mut()
                    .and_then(|trace| trace.begin_batch(&copies, self.backend.name()));
                let started = captured.map(|_| std::time::Instant::now());
                let enqueue_result = (|| -> Result<(), String> {
                    #[cfg(feature = "test-hooks")]
                    if !copies.is_empty()
                        && (crate::test_faults::active("local_restore_dma")
                            || crate::test_faults::active("local_restore_error"))
                    {
                        self.backend.h2d(&copies[..1], &self.stream)?;
                        crate::test_faults::pause_blocking("local_restore_dma");
                        if crate::test_faults::active("local_restore_error") {
                            return Err("injected failure after first local Restore enqueue".into());
                        }
                        self.backend.h2d(&copies[1..], &self.stream)?;
                    } else {
                        self.backend.h2d(&copies, &self.stream)?;
                    }
                    #[cfg(not(feature = "test-hooks"))]
                    self.backend.h2d(&copies, &self.stream)?;
                    Ok(())
                })();
                if let (Some(trace), Some(index), Some(started)) =
                    (self.trace.as_mut(), captured, started)
                {
                    trace.finish_batch(
                        index,
                        started.elapsed().as_nanos() as u64,
                        enqueue_result.is_ok(),
                    );
                }
                enqueue_result?;
                if plan.completed_layers.iter().any(|name| name == layer)
                    && let Some(event) = layer_events.remove(layer)
                {
                    // The caller retains the framework event through this operation's drain.
                    unsafe {
                        result::event::record(event as sys::CUevent, self.stream.cu_stream())
                    }
                    .map_err(|error| error.to_string())?;
                }
            }
            if final_part {
                // Layers with no required bytes still need a fresh generation of their event.
                for (_, event) in layer_events.drain() {
                    unsafe {
                        result::event::record(event as sys::CUevent, self.stream.cu_stream())
                    }
                    .map_err(|error| error.to_string())?;
                }
                enqueued();
            }
            Ok(())
        })();
        if let Some(submitted_at) = submitted_at {
            *submitted_at = Some(std::time::Instant::now());
        }
        let drain_started = self.trace.as_ref().map(|_| std::time::Instant::now());
        if let Err(error) = self.stream.synchronize() {
            log::error!("Cannot establish engine-local Restore drain: {error}; terminating engine");
            std::process::abort();
        }
        if let (Some(trace), Some(started)) = (self.trace.as_mut(), drain_started) {
            trace.finish_part(started.elapsed().as_nanos() as u64, submitted.is_ok());
        }
        submitted
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/local.rs"]
mod tests;
