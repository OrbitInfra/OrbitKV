use std::sync::{Arc, mpsc};

use cudarc::driver::{CudaStream, result};

use crate::codec::{
    EncodedSegment,
    gpu::{DecodeError, DecodeInput, GpuCodec},
};
use crate::metrics::core_metrics;
use crate::transfer::finish_gpu_transfer;
use crate::transfer_engine::{CudaRegistration, TransferEngine};

struct DeviceInput {
    stream: Arc<CudaStream>,
    pointer: u64,
    bytes: usize,
}

impl DeviceInput {
    fn new(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self, String> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| error.to_string())?;
        // SAFETY: owned until all cuFile scatters and codec work have completed.
        let pointer = unsafe { result::malloc_sync(bytes) }.map_err(|error| error.to_string())?;
        core_metrics()
            .storage_codec_workspace_bytes
            .add(bytes as i64, &[]);
        core_metrics()
            .storage_codec_workspace_allocations
            .add(1, &[]);
        Ok(Self {
            stream: Arc::clone(stream),
            pointer,
            bytes,
        })
    }
}

impl Drop for DeviceInput {
    fn drop(&mut self) {
        let _ = finish_gpu_transfer(&self.stream, Ok(()));
        // SAFETY: the queue sends the next preparation (or closes the worker)
        // only after every slot writing this allocation has completed.
        if let Err(error) = unsafe { result::free_sync(self.pointer) } {
            log::error!("Cannot release SSD codec input: {error}");
            std::process::abort();
        }
        core_metrics()
            .storage_codec_workspace_bytes
            .add(-(self.bytes as i64), &[]);
    }
}

pub(super) struct DecodeRange {
    pub(super) source: u64,
    pub(super) target: u64,
    pub(super) meta: EncodedSegment,
}

pub(super) enum DecodeCommand {
    Prepare {
        bytes: usize,
        budget: usize,
        backend: Arc<TransferEngine>,
    },
    Decode(Vec<DecodeRange>),
}

pub(super) enum DecodeReply {
    Prepared(Result<Arc<CudaRegistration>, String>),
    Completed(Result<(), DecodeError>),
}

pub(super) fn run(
    stream: Arc<CudaStream>,
    requests: mpsc::Receiver<DecodeCommand>,
    replies: mpsc::SyncSender<DecodeReply>,
) {
    let mut input: Option<Arc<CudaRegistration>> = None;
    let mut codec: Option<GpuCodec> = None;
    let mut scratch_budget = 0;
    while let Ok(command) = requests.recv() {
        let reply = match command {
            DecodeCommand::Prepare {
                bytes,
                budget,
                backend,
            } => {
                let prepared = (|| {
                    if bytes == 0 || bytes >= budget {
                        return Err("encoded SSD input exceeds codec budget".into());
                    }
                    let limit = (budget / 2).max(bytes);
                    if input
                        .as_ref()
                        .is_some_and(|input| input.length < bytes || input.length > limit)
                    {
                        input
                            .as_ref()
                            .ok_or("missing codec input")?
                            .close()
                            .map_err(|e| e.to_string())?;
                        input = None;
                    }
                    let resident = input.as_ref().map_or(bytes, |input| input.length);
                    scratch_budget = budget - resident;
                    // Shrink old scratch before allocating input so changes in
                    // budget never produce a temporary double reservation.
                    if codec
                        .as_ref()
                        .is_some_and(|codec| codec.scratch_bytes() > scratch_budget)
                    {
                        codec = None;
                    }
                    if input.is_none() {
                        let backing = Arc::new(DeviceInput::new(&stream, bytes)?);
                        // SAFETY: registration retains the actual input and its stream/context.
                        input = Some(
                            unsafe {
                                CudaRegistration::new(
                                    Some(Arc::clone(&backend)),
                                    backing.pointer,
                                    bytes,
                                    stream.context().ordinal() as i32,
                                    backing,
                                )
                            }
                            .map_err(|e| e.to_string())?,
                        );
                    }
                    let registration = input.as_ref().ok_or("missing codec input")?;
                    let _loan = registration.loan(&backend).map_err(|e| e.to_string())?;
                    Ok(Arc::clone(registration))
                })();
                DecodeReply::Prepared(prepared)
            }
            DecodeCommand::Decode(ranges) => {
                let decoded = (|| {
                    if codec.is_none() {
                        codec =
                            Some(GpuCodec::new(stream.context()).map_err(DecodeError::Runtime)?);
                    }
                    let inputs: Vec<_> = ranges
                        .iter()
                        .map(|range| DecodeInput {
                            source: range.source,
                            source_bytes: range.meta.stored_bytes,
                            target: range.target,
                            target_bytes: range.meta.logical_bytes,
                            meta: &range.meta,
                        })
                        .collect();
                    let codec = codec
                        .as_mut()
                        .ok_or_else(|| DecodeError::Runtime("codec was not initialized".into()))?;
                    // SAFETY: the queue holds complete, disjoint input and
                    // engine destinations until this reply; no host payload.
                    unsafe { codec.decode_batch(&stream, &inputs, scratch_budget) }
                })();
                // Also cover an unsuccessful/partial submission before replying.
                let decoded = finish_gpu_transfer(&stream, Ok(()))
                    .map_err(|error| DecodeError::Runtime(error.to_string()))
                    .and(decoded);
                DecodeReply::Completed(decoded)
            }
        };
        if replies.send(reply).is_err() {
            break;
        }
    }
}
