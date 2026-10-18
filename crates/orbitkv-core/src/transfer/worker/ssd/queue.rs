use std::collections::VecDeque;
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaEvent, result, sys};
use tokio::sync::{mpsc, oneshot};

use crate::EngineError;
use crate::codec::gpu::{DecodeError, MAX_BATCH_SEGMENTS};
use crate::cost::{Observation, Outcome};
use crate::metrics::core_metrics;
use crate::storage::ssd::GpuWriteLease;
use crate::storage::ssd::cufile::{CufileFile, GpuSlot, IoBatch, STAGING_SLOTS};

use super::decode::{DecodeCommand, DecodeRange, DecodeReply};
use crate::transfer::finish_gpu_transfer;

use super::super::{
    LoadTask, SaveTask, WorkerCommand, WorkerRuntime, build_copy_descs, finish_load,
};

pub(in crate::transfer::worker) const MAX_WRITES: usize = 8;
const READ_BURST: usize = 4;

enum Task {
    Load(LoadTask),
    Save(SaveTask),
}

struct Work {
    file: Arc<CufileFile>,
    batch: IoBatch,
    write: Option<usize>,
    encoded: bool,
}

struct Write {
    lease: Option<GpuWriteLease>,
    remaining: usize,
}

#[derive(Clone, Copy)]
struct BatchOwner {
    job: u64,
    write: Option<usize>,
    encoded: bool,
}

struct Job {
    id: u64,
    task: Task,
    work: VecDeque<Work>,
    encoded: VecDeque<super::EncodedRead>,
    decoding: bool,
    writes: Vec<Write>,
    inflight: usize,
    error: Option<String>,
    prepared: bool,
    host_completion: Option<CudaEvent>,
    started: Instant,
    bytes: usize,
    observation: Observation,
    outcome: Outcome,
}

impl Job {
    fn new(id: u64, command: WorkerCommand) -> Self {
        let (task, observation) = match command {
            WorkerCommand::Load(task, observation) => (Task::Load(task), observation),
            WorkerCommand::Save(task, observation) => (Task::Save(task), observation),
            WorkerCommand::Drain(_) => unreachable!("drain is handled by the queue"),
        };
        let mut job = Self {
            id,
            task,
            work: VecDeque::new(),
            encoded: VecDeque::new(),
            decoding: false,
            writes: Vec::new(),
            inflight: 0,
            error: None,
            prepared: false,
            host_completion: None,
            started: Instant::now(),
            bytes: 0,
            observation,
            outcome: Outcome::Completed,
        };
        if let Task::Load(task) = &mut job.task {
            task.decode_ready_observation.admitted();
            task.decode_ready_observation.submitted();
        }
        let planned = (|| {
            match &mut job.task {
                Task::Load(task) => {
                    let layers = &task.layers;
                    let plans = super::plan(layers)?;
                    job.encoded = plans.encoded.into();
                    for (file, batches) in plans.raw {
                        for batch in batches {
                            job.bytes += batch.copies.iter().map(|copy| copy.bytes).sum::<usize>();
                            job.work.push_back(Work {
                                file: Arc::clone(&file),
                                batch,
                                write: None,
                                encoded: false,
                            });
                        }
                    }
                }
                Task::Save(task) => {
                    for (index, write) in
                        std::mem::take(&mut task.ssd_writes).into_iter().enumerate()
                    {
                        let file = Arc::clone(write.lease.file());
                        job.writes.push(Write {
                            lease: Some(write.lease),
                            remaining: write.batches.len(),
                        });
                        for batch in write.batches {
                            job.work.push_back(Work {
                                file: Arc::clone(&file),
                                batch,
                                write: Some(index),
                                encoded: false,
                            });
                        }
                    }
                }
            }
            Ok::<(), EngineError>(())
        })();
        if let Err(error) = planned {
            job.fail(error.to_string());
        }
        job
    }

    fn prepare(&mut self, runtime: &WorkerRuntime) -> Result<(), EngineError> {
        if self.prepared {
            return Ok(());
        }
        self.observation.admitted();
        // The route estimate starts before staging/codec work and ends only
        // at engine-visible completion, just like the io_uring host route.
        self.observation.submitted();
        let layers = match &self.task {
            Task::Load(task) => {
                let layers = &task.layers;
                self.bytes += super::super::codec::restore(
                    runtime,
                    layers,
                    task.codec_budget,
                    &mut self.observation,
                )
                .inspect_err(|_| core_metrics().storage_codec_decode_failures.add(1, &[]))?;
                layers
            }
            Task::Save(task) => &task.layers,
        };
        let (copies, bytes) = build_copy_descs(layers)?;
        if !copies.is_empty() {
            let event = runtime
                .stream
                .context()
                .new_event(None)
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            self.observation.submitted();
            let submitted = if self.is_write() {
                runtime.backend.d2h(&copies, &runtime.stream)
            } else {
                runtime.backend.h2d(&copies, &runtime.stream)
            };
            if let Err(error) = submitted.and_then(|()| {
                event
                    .record(&runtime.stream)
                    .map_err(|error| error.to_string())
            }) {
                // Partial submission still owns the job's host and GPU pages.
                return finish_gpu_transfer(&runtime.stream, Err(error));
            }
            self.host_completion = Some(event);
        }
        if !self.is_write() {
            self.bytes += bytes;
        }
        self.prepared = true;
        Ok(())
    }

    fn poll_host(&mut self, runtime: &WorkerRuntime) {
        let Some(event) = &self.host_completion else {
            return;
        };
        // This event covers only this job. A later copy on the shared stream
        // must not delay its completion or block cuFile submission/polling.
        match unsafe { result::event::query(event.cu_event()) } {
            Ok(()) => {
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active("ssd_host_completion") {
                    return;
                }
                self.host_completion = None;
            }
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {}
            Err(error) => {
                let _ = finish_gpu_transfer(&runtime.stream, Ok(()));
                self.host_completion = None;
                self.fail(error.to_string());
            }
        }
    }

    fn is_write(&self) -> bool {
        matches!(self.task, Task::Save(_))
    }

    fn fail(&mut self, error: String) {
        self.outcome = Outcome::Failed;
        self.error.get_or_insert(error);
        self.work.clear();
        self.encoded.clear();
    }

    fn cancel_abandoned(&mut self) {
        let closed = match &self.task {
            Task::Load(task) => task.completion.is_closed(),
            Task::Save(task) => task.reply.is_closed(),
        };
        if closed {
            if self.error.is_none() {
                self.outcome = Outcome::Cancelled;
                self.error = Some("GPU transfer consumer closed".into());
            }
            self.work.clear();
            self.encoded.clear();
        }
    }

    fn complete(&mut self, write: Option<usize>, result: Result<(), String>) {
        self.inflight -= 1;
        let publish = result.is_ok() && self.error.is_none();
        if let Some(index) = write {
            let write = &mut self.writes[index];
            write.remaining -= 1;
            if write.remaining == 0 && publish {
                write
                    .lease
                    .take()
                    .expect("unpublished extent owned")
                    .commit();
            }
        }
        if let Err(error) = result {
            self.fail(error);
        }
    }

    fn is_complete(&self) -> bool {
        self.inflight == 0
            && self.work.is_empty()
            && self.encoded.is_empty()
            && !self.decoding
            && self.host_completion.is_none()
    }

    fn finish(self) {
        assert_eq!(self.inflight, 0, "submitted I/O still owns this job");
        assert!(!self.decoding, "codec completion still owns this job");
        assert!(
            self.host_completion.is_none(),
            "host copy still owns this job"
        );
        let consumer_closed = match &self.task {
            Task::Load(task) => task.completion.is_closed(),
            Task::Save(task) => task.reply.is_closed(),
        };
        let outcome = if self.outcome == Outcome::Completed && consumer_closed {
            Outcome::Cancelled
        } else {
            self.outcome
        };
        // Physical cuFile bytes belong to GpuSlot; this is an inclusive job span.
        self.observation.finish(outcome, None);
        // Drop unpublished extents before telling the caller it can reuse pages.
        drop(self.writes);
        let result = self
            .error
            .map_or(Ok(()), |error| Err(EngineError::Storage(error)));
        match self.task {
            Task::Load(task) => finish_load(task, result, self.started, self.bytes, outcome),
            Task::Save(task) => {
                drop(task.ssd_admission);
                task.admission.complete();
                let _ = task.reply.send(result.map(|()| task.layers));
            }
        }
    }
}

/// One request at a time; synchronous codec fences never block the I/O queue.
/// The input arena belongs to this worker and stays stable from Prepared until
/// every cuFile fragment drains and the queue receives Completed.
struct Decoder {
    sender: Option<std_mpsc::SyncSender<DecodeCommand>>,
    replies: std_mpsc::Receiver<DecodeReply>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Decoder {
    fn new(runtime: &WorkerRuntime) -> Result<Self, String> {
        let stream = runtime
            .stream
            .context()
            .new_stream()
            .map_err(|error| error.to_string())?;
        let (sender, requests) = std_mpsc::sync_channel(1);
        let (replies, receiver) = std_mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("ssd-decode".into())
            .spawn(move || {
                super::decode::run(stream, requests, replies);
            })
            .map_err(|error| error.to_string())?;
        Ok(Self {
            sender: Some(sender),
            replies: receiver,
            thread: Some(thread),
        })
    }

    fn send(&self, command: DecodeCommand) -> Result<(), String> {
        self.sender
            .as_ref()
            .ok_or("SSD decoder is closed")?
            .try_send(command)
            .map_err(|error| format!("SSD decoder submission failed: {error}"))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.sender.take();
        if self
            .thread
            .take()
            .is_some_and(|thread| thread.join().is_err())
        {
            log::error!("SSD decoder thread failed during drain");
        }
    }
}

#[derive(PartialEq, Eq)]
enum DecodePhase {
    Preparing,
    Reading,
    Decoding,
}

struct Decode {
    job: u64,
    reads: Vec<super::EncodedRead>,
    offsets: Vec<usize>,
    base: u64,
    remaining: usize,
    phase: DecodePhase,
}

fn input_window(
    sizes: impl Iterator<Item = usize>,
    budget: usize,
) -> Result<(Vec<usize>, usize), String> {
    let mut offsets = Vec::new();
    let mut bytes = 0usize;
    for size in sizes.take(MAX_BATCH_SEGMENTS) {
        let end = bytes
            .checked_add(size)
            .and_then(|n| n.checked_next_multiple_of(4096))
            .ok_or("encoded SSD input size overflow")?;
        // Leave room for descriptors/CRC even when one segment exceeds half
        // the budget. nvCOMP's additional scratch is checked by the codec.
        if size == 0 || end.saturating_add(8192) > budget {
            break;
        }
        if !offsets.is_empty() && end > budget / 2 {
            break;
        }
        offsets.push(bytes);
        bytes = end;
    }
    if offsets.is_empty() {
        return Err("encoded SSD segment exceeds codec staging budget".into());
    }
    Ok((offsets, bytes))
}

impl Decode {
    fn begin(job: &mut Job, decoder: &Decoder) -> Result<Self, String> {
        let budget = match &job.task {
            Task::Load(task) => task.codec_budget,
            Task::Save(_) => return Err("encoded read attached to a save".into()),
        };
        let first = job.encoded.front().ok_or("missing encoded SSD read")?;
        let sizes = job
            .encoded
            .iter()
            .take_while(|read| Arc::ptr_eq(&read.source.entry.readers, &first.source.entry.readers))
            .map(|read| read.meta.stored_bytes);
        let (offsets, bytes) = input_window(sizes, budget)?;
        decoder.send(DecodeCommand::Prepare { bytes, budget })?;
        let reads = job.encoded.drain(..offsets.len()).collect();
        job.decoding = true;
        Ok(Self {
            job: job.id,
            reads,
            offsets,
            base: 0,
            remaining: 0,
            phase: DecodePhase::Preparing,
        })
    }

    fn prepared(&mut self, job: &mut Job, base: u64) -> Result<(), String> {
        self.base = base;
        let copies = self
            .reads
            .iter()
            .zip(&self.offsets)
            .map(|(read, &offset)| {
                Ok(crate::storage::ssd::cufile::CopyRange {
                    file_offset: read.file_offset,
                    device: base
                        .checked_add(offset as u64)
                        .ok_or("GPU input offset overflow")?,
                    bytes: read.meta.stored_bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let batches = crate::storage::ssd::cufile::plan_reads(copies)?;
        self.remaining = batches.len();
        for batch in batches {
            job.work.push_back(Work {
                file: Arc::clone(self.reads[0].source.file().map_err(|e| e.to_string())?),
                batch,
                write: None,
                encoded: true,
            });
        }
        self.phase = DecodePhase::Reading;
        Ok(())
    }
}

fn poll_decode(decoder: &Decoder, active: &mut Option<Decode>, jobs: &mut VecDeque<Job>) {
    let Some(mut decode) = active.take() else {
        return;
    };
    let job = jobs
        .iter_mut()
        .find(|job| job.id == decode.job)
        .expect("decoding job retained");
    if decode.phase != DecodePhase::Reading {
        match decoder.replies.try_recv() {
            Ok(DecodeReply::Prepared(result)) => {
                if result.is_ok() && job.error.is_some() {
                    job.decoding = false;
                    return;
                }
                match result.and_then(|base| decode.prepared(job, base)) {
                    Ok(()) => {}
                    Err(error) => {
                        job.fail(error);
                        job.decoding = false;
                        return;
                    }
                }
            }
            Ok(DecodeReply::Completed(result)) => {
                if let Err(error) = result {
                    core_metrics().storage_codec_decode_failures.add(1, &[]);
                    let message = match error {
                        DecodeError::Corrupt(message) => {
                            // Every input belongs to this exact generation.
                            decode.reads[0].source.invalidate_encoded();
                            message
                        }
                        DecodeError::Runtime(message) => message,
                    };
                    job.fail(message);
                } else {
                    job.bytes += decode
                        .reads
                        .iter()
                        .map(|read| read.meta.logical_bytes)
                        .sum::<usize>();
                }
                job.decoding = false;
                return;
            }
            Err(std_mpsc::TryRecvError::Empty) => {}
            Err(std_mpsc::TryRecvError::Disconnected) => {
                job.fail("SSD decoder closed before completion".into());
                job.decoding = false;
                return;
            }
        }
    }
    if decode.phase == DecodePhase::Reading {
        if job.error.is_some() {
            if job.inflight == 0 {
                job.decoding = false;
                return;
            }
        } else if decode.remaining == 0 {
            let ranges = decode
                .reads
                .iter()
                .zip(&decode.offsets)
                .map(|(read, &offset)| DecodeRange {
                    source: decode.base + offset as u64,
                    target: read.target,
                    meta: read.meta.clone(),
                })
                .collect();
            if let Err(error) = decoder.send(DecodeCommand::Decode(ranges)) {
                job.fail(error);
                job.decoding = false;
                return;
            }
            decode.phase = DecodePhase::Decoding;
        }
    }
    *active = Some(decode);
}

/// Two in-flight batches; at most one write leaves capacity for demand reads.
/// Read bursts and round-robin jobs bound starvation without preempting DMA.
pub(in crate::transfer::worker) fn run(
    mut receiver: mpsc::UnboundedReceiver<WorkerCommand>,
    runtime: WorkerRuntime,
) {
    let mut jobs: VecDeque<Job> = VecDeque::new();
    let mut slots: Vec<(GpuSlot, Option<BatchOwner>)> = Vec::new();
    let mut decoder: Option<Decoder> = None;
    let mut decoding: Option<Decode> = None;
    let mut drain: Option<oneshot::Sender<Result<(), String>>> = None;
    let mut closed = false;
    let mut next_id = 0;
    let mut read_burst = 0;
    let mut reset_slots = false;
    loop {
        // Bound CPU admission work per turn so producers cannot starve completions.
        for _ in 0..32 {
            if closed {
                break;
            }
            let command = if jobs.is_empty() {
                receiver.blocking_recv()
            } else {
                match receiver.try_recv() {
                    Ok(command) => Some(command),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => None,
                }
            };
            match command {
                Some(WorkerCommand::Drain(reply)) => {
                    drain = Some(reply);
                    closed = true;
                }
                Some(command) => {
                    jobs.push_back(Job::new(next_id, command));
                    next_id += 1;
                }
                None => closed = true,
            }
        }
        for job in &mut jobs {
            job.poll_host(&runtime);
            job.cancel_abandoned();
        }
        for (slot, owner) in &mut slots {
            if let Some(result) = slot.poll() {
                reset_slots |= result.is_err();
                let owner = owner.take().expect("submitted slot has an owner");
                if owner.encoded {
                    let decode = decoding.as_mut().expect("encoded input retained");
                    assert_eq!(decode.job, owner.job);
                    decode.remaining -= 1;
                }
                jobs.iter_mut()
                    .find(|job| job.id == owner.job)
                    .expect("submitted job retained")
                    .complete(owner.write, result);
            }
        }
        if let Some(decoder) = &decoder {
            poll_decode(decoder, &mut decoding, &mut jobs);
        }
        // Errors can leave a stream unusable. Keep every other slot alive until
        // its submitted work completes, then recreate the registered resources.
        if reset_slots && slots.iter().all(|(_, owner)| owner.is_none()) {
            slots.clear();
            reset_slots = false;
        }
        let mut index = 0;
        while index < jobs.len() {
            if jobs[index].is_complete() {
                jobs.remove(index).expect("completed job exists").finish();
            } else {
                index += 1;
            }
        }
        if jobs.is_empty() {
            if closed {
                drop(slots);
                drop(decoder);
                drop(runtime);
                if let Some(reply) = drain {
                    let _ = reply.send(Ok(()));
                }
                return;
            }
            continue;
        }
        if slots.is_empty() {
            let created = (0..STAGING_SLOTS)
                .map(|_| GpuSlot::new(runtime.stream.context()).map(|slot| (slot, None)))
                .collect::<Result<Vec<_>, _>>();
            match created {
                Ok(created) => slots = created,
                Err(error) => {
                    for job in &mut jobs {
                        for work in &job.work {
                            work.file.gpu_io.failed(&error);
                        }
                        for read in &job.encoded {
                            if let Ok(file) = read.source.file() {
                                file.gpu_io.failed(&error);
                            }
                        }
                        job.fail(error.clone());
                    }
                    continue;
                }
            }
        }
        for index in 0..slots.len() {
            if reset_slots {
                break;
            }
            if slots[index].1.is_some() {
                continue;
            }
            let writing = slots
                .iter()
                .any(|(_, owner)| owner.is_some_and(|owner| owner.write.is_some()));
            let eligible = |job: &Job| {
                if job.work.is_empty() && (job.encoded.is_empty() || decoding.is_some()) {
                    return false;
                }
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active(if job.is_write() {
                    "cufile_write"
                } else {
                    "cufile"
                }) {
                    return false;
                }
                true
            };
            let read = jobs.iter().position(|job| !job.is_write() && eligible(job));
            let write = if writing {
                None
            } else {
                jobs.iter().position(|job| job.is_write() && eligible(job))
            };
            let selected = match (read, write) {
                (Some(_), Some(write)) if read_burst >= READ_BURST => Some((write, false)),
                (Some(read), _) => Some((read, true)),
                (_, Some(write)) => Some((write, false)),
                _ => None,
            };
            let Some((position, reading)) = selected else {
                continue;
            };
            read_burst = if reading {
                (read_burst + 1).min(READ_BURST)
            } else {
                0
            };
            let mut job = jobs.remove(position).expect("selected job exists");
            if let Err(error) = job.prepare(&runtime) {
                job.fail(error.to_string());
                jobs.push_back(job);
                continue;
            }
            if job.work.is_empty() {
                if decoder.is_none() {
                    match Decoder::new(&runtime) {
                        Ok(created) => decoder = Some(created),
                        Err(error) => {
                            job.fail(error);
                            jobs.push_back(job);
                            continue;
                        }
                    }
                }
                match Decode::begin(&mut job, decoder.as_ref().expect("decoder initialized")) {
                    Ok(active) => decoding = Some(active),
                    Err(error) => job.fail(error),
                }
                jobs.push_back(job);
                continue;
            }
            let work = job.work.pop_front().expect("eligible job has work");
            let owner = BatchOwner {
                job: job.id,
                write: work.write,
                encoded: work.encoded,
            };
            job.observation.submitted();
            match slots[index].0.submit(work.file, work.batch, !reading) {
                Ok(()) => {
                    job.inflight += 1;
                    slots[index].1 = Some(owner);
                }
                Err(error) => {
                    job.fail(error);
                    reset_slots = true;
                }
            }
            jobs.push_back(job);
        }
        std::thread::sleep(Duration::from_micros(50));
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transfer/worker/ssd/queue.rs"]
mod tests;
