//! Temporary GPU gather/scatter owners for engine file I/O; removed by A2.

use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};

use super::GpuIo;
use super::io::{EngineFile, Receipt};
use super::layout::{IoBatch, STAGING_BYTES};
use crate::cost::{CostEstimateKey, CostObservationKind, Observation, Outcome, Representation};
use crate::metrics::core_metrics;
use crate::transfer::finish_gpu_transfer;
use crate::transfer_engine::{ErrorKind, FileMemory, FileTransport, TransferOp, TransferSlice};

pub(crate) struct GpuFile {
    pub(super) file: Arc<EngineFile>,
    pub(super) transport: FileTransport,
    pub(crate) gpu_io: Arc<GpuIo>,
}

pub(crate) struct GpuSlot {
    context: Arc<CudaContext>,
    slot: Option<EngineSlot>,
}

impl GpuSlot {
    pub(crate) fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        context.bind_to_thread().map_err(|e| e.to_string())?;
        Ok(Self {
            context: Arc::clone(context),
            slot: None,
        })
    }

    pub(crate) fn submit(
        &mut self,
        file: Arc<GpuFile>,
        batch: IoBatch,
        write: bool,
        extent: Arc<dyn Send + Sync>,
    ) -> Result<(), String> {
        if self.slot.is_none() {
            self.slot = Some(EngineSlot::new(&self.context)?);
        }
        self.slot
            .as_mut()
            .ok_or("missing GPU SSD slot")?
            .submit(file, batch, write, extent)
    }

    pub(crate) fn poll(&mut self) -> Option<Result<(), String>> {
        self.slot.as_mut()?.poll()
    }
}

struct Buffer {
    pointer: u64,
    context: Arc<CudaContext>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the slot observes its CUDA event and engine receipt before
        // reuse/drop; registration retains this Arc on uncertain native work.
        let result = self
            .context
            .bind_to_thread()
            .and_then(|()| unsafe { result::free_sync(self.pointer) });
        if let Err(error) = result {
            log::error!("Cannot release GPU SSD staging allocation: {error}");
            std::process::abort();
        }
        crate::metrics::core_metrics()
            .ssd_gpu_staging_bytes
            .add(-(STAGING_BYTES as i64), &[]);
    }
}

enum Phase {
    Gather,
    File(Receipt),
    Scatter,
}

struct Pending {
    file: Arc<GpuFile>,
    batch: IoBatch,
    write: bool,
    extent: Arc<dyn Send + Sync>,
    phase: Phase,
    started: Instant,
    observation: Observation,
}

impl Pending {
    fn finish(self, success: bool, retained: bool) {
        self.observation.finish(
            if success {
                Outcome::Completed
            } else {
                Outcome::Failed
            },
            success.then_some(self.batch.bytes as u64),
        );
        let metrics = core_metrics();
        let (seconds, bytes, failures) = if self.write {
            (
                &metrics.ssd_cufile_write_seconds,
                &metrics.ssd_cufile_write_bytes,
                &metrics.ssd_cufile_write_failures,
            )
        } else {
            (
                &metrics.ssd_cufile_read_seconds,
                &metrics.ssd_cufile_read_bytes,
                &metrics.ssd_cufile_read_failures,
            )
        };
        seconds.record(self.started.elapsed().as_secs_f64(), &[]);
        if success {
            bytes.add(self.batch.bytes as u64, &[]);
        } else {
            failures.add(1, &[]);
        }
        if !retained {
            metrics.ssd_cufile_inflight_batches.add(-1, &[]);
        }
    }
}

struct EngineSlot {
    stream: Arc<CudaStream>,
    event: CudaEvent,
    buffer: Arc<Buffer>,
    pending: Option<Pending>,
    poisoned: bool,
}

impl EngineSlot {
    fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let stream = context.new_stream().map_err(|e| e.to_string())?;
        let event = context.new_event(None).map_err(|e| e.to_string())?;
        // SAFETY: this exact context owns the slot through event and engine drain.
        let pointer = unsafe { result::malloc_sync(STAGING_BYTES) }.map_err(|e| e.to_string())?;
        crate::metrics::core_metrics()
            .ssd_gpu_staging_bytes
            .add(STAGING_BYTES as i64, &[]);
        Ok(Self {
            stream,
            event,
            buffer: Arc::new(Buffer {
                pointer,
                context: Arc::clone(context),
            }),
            pending: None,
            poisoned: false,
        })
    }

    fn submit(
        &mut self,
        file: Arc<GpuFile>,
        batch: IoBatch,
        write: bool,
        extent: Arc<dyn Send + Sync>,
    ) -> Result<(), String> {
        if self.pending.is_some() || self.poisoned {
            return Err("GPU SSD slot is busy or retained after unknown drain".into());
        }
        batch.validate()?;
        let mut observation = Observation::new(
            CostEstimateKey::new(
                if write {
                    CostObservationKind::SsdCufileWrite
                } else {
                    CostObservationKind::SsdCufileRead
                },
                file.file.cost_resource,
                Representation::Unknown,
                batch.bytes as u64,
                batch.copies.len(),
            ),
            None,
        );
        observation.admitted();
        observation.submitted();
        core_metrics().ssd_cufile_inflight_batches.add(1, &[]);
        self.pending = Some(Pending {
            file,
            batch,
            write,
            extent,
            phase: Phase::Gather,
            started: Instant::now(),
            observation,
        });
        let result = (|| {
            let pending = self.pending.as_ref().ok_or("missing GPU SSD operation")?;
            if write {
                // SAFETY: the worker retains immutable source pages and this slot
                // exclusively owns the bounded staging allocation through its event.
                unsafe {
                    result::memset_d8_async(
                        self.buffer.pointer,
                        0,
                        pending.batch.bytes,
                        self.stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string())?;
                for copy in &pending.batch.copies {
                    // SAFETY: batch.validate bounds every copy within this slot.
                    unsafe {
                        result::memcpy_dtod_async(
                            self.buffer.pointer + copy.file_offset - pending.batch.file_offset,
                            copy.device,
                            copy.bytes,
                            self.stream.cu_stream(),
                        )
                    }
                    .map_err(|e| e.to_string())?;
                }
            }
            self.event.record(&self.stream).map_err(|e| e.to_string())
        })();
        if let Err(error) = result {
            let _ = finish_gpu_transfer(&self.stream, Ok(()));
            if let Some(pending) = self.pending.take() {
                pending.finish(false, false);
            }
            return Err(error);
        }
        Ok(())
    }

    fn poll(&mut self) -> Option<Result<(), String>> {
        let mut pending = self.pending.take()?;
        let result = match &mut pending.phase {
            Phase::File(receiver) => {
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active(if pending.write {
                    "cufile_write_completion"
                } else {
                    "cufile_read_completion"
                }) {
                    self.pending = Some(pending);
                    return None;
                }
                match receiver.try_recv() {
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        self.pending = Some(pending);
                        return None;
                    }
                    Err(error) => {
                        self.poisoned = true;
                        Err(error.to_string())
                    }
                    Ok(Err(error)) => {
                        self.poisoned = error.kind == ErrorKind::Unquiesced;
                        Err(error.to_string())
                    }
                    Ok(Ok(bytes)) if bytes != pending.batch.bytes => {
                        Err("short GPU SSD transfer".into())
                    }
                    Ok(Ok(_)) if pending.write => {
                        #[cfg(feature = "test-hooks")]
                        if crate::test_faults::active("cufile_write_error") {
                            Err("injected GPU file write error after physical receipt".into())
                        } else {
                            Ok(())
                        }
                        #[cfg(not(feature = "test-hooks"))]
                        Ok(())
                    }
                    Ok(Ok(_)) => {
                        let result = self.scatter(&pending.batch);
                        if result.is_ok() {
                            pending.phase = Phase::Scatter;
                            self.pending = Some(pending);
                            return None;
                        }
                        result
                    }
                }
            }
            Phase::Gather | Phase::Scatter => {
                // SAFETY: the event remains owned and was recorded on this stream.
                match unsafe { result::event::query(self.event.cu_event()) } {
                    Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {
                        self.pending = Some(pending);
                        return None;
                    }
                    Err(error) => Err(error.to_string()),
                    Ok(()) if matches!(pending.phase, Phase::Scatter) => Ok(()),
                    Ok(()) => {
                        let submitted = self.submit_file(&pending);
                        match submitted {
                            Ok(receipt) => {
                                pending.phase = Phase::File(receipt);
                                self.pending = Some(pending);
                                return None;
                            }
                            Err(error) => Err(error),
                        }
                    }
                }
            }
        };
        if let Err(error) = &result {
            // This drains only gather/scatter. An unknown file operation retains
            // its own registration, buffer, file and extent in EngineFile.
            let _ = finish_gpu_transfer(&self.stream, Ok(()));
            pending.file.gpu_io.failed(error);
        }
        pending.finish(result.is_ok(), self.poisoned);
        Some(result)
    }

    fn submit_file(&self, pending: &Pending) -> Result<Receipt, String> {
        let local = NonNull::new(self.buffer.pointer as usize as *mut u8)
            .ok_or("null GPU staging pointer")?;
        pending
            .file
            .file
            .submit(
                if pending.write {
                    TransferOp::Write
                } else {
                    TransferOp::Read
                },
                pending.file.transport,
                vec![TransferSlice {
                    local,
                    remote_address: pending.batch.file_offset,
                    length: pending.batch.bytes,
                }],
                FileMemory::Cuda {
                    address: self.buffer.pointer,
                    length: STAGING_BYTES,
                    context: Arc::clone(&self.buffer.context),
                    _owner: self.buffer.clone(),
                },
                Arc::clone(&pending.extent),
            )
            .map_err(|e| e.to_string())
    }

    fn scatter(&self, batch: &IoBatch) -> Result<(), String> {
        for copy in &batch.copies {
            // SAFETY: physical file completion precedes scatter; the worker
            // retains all destination pages until the following CUDA event.
            unsafe {
                result::memcpy_dtod_async(
                    copy.device,
                    self.buffer.pointer + copy.file_offset - batch.file_offset,
                    copy.bytes,
                    self.stream.cu_stream(),
                )
            }
            .map_err(|e| e.to_string())?;
        }
        self.event.record(&self.stream).map_err(|e| e.to_string())
    }
}

impl Drop for EngineSlot {
    fn drop(&mut self) {
        let _ = finish_gpu_transfer(&self.stream, Ok(()));
        // Detached engine work owns its own Arc<Buffer> and immutable extent.
    }
}
