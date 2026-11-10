use std::collections::VecDeque;
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use crate::EngineError;
use crate::codec::gpu::{DecodeError, MAX_BATCH_SEGMENTS};
use crate::cost::{Observation, Outcome};
use crate::metrics::core_metrics;
use crate::storage::ssd::GpuFile;
use crate::storage::ssd::WriteLease;
use crate::storage::ssd::io::Receipt;
use crate::storage::ssd::layout::{FILE_INFLIGHT, IoBatch};
use crate::transfer_engine::{CudaRegistration, ErrorKind, FileMemory, TransferError, TransferOp};

use super::decode::{DecodeCommand, DecodeRange, DecodeReply};
use crate::transfer_engine::local::LocalCompletion;

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
    options: crate::transfer_engine::TransferOptions,
    file: Arc<GpuFile>,
    extent: Arc<dyn Send + Sync>,
    batch: IoBatch,
    write: Option<usize>,
    encoded: bool,
    input: Option<Arc<CudaRegistration>>,
}

struct Write {
    lease: Option<Arc<WriteLease>>,
    remaining: usize,
}

#[derive(Clone, Copy)]
struct BatchOwner {
    job: u64,
    write: Option<usize>,
    encoded: bool,
}

struct Pending {
    receipt: Receipt,
    owner: BatchOwner,
    file: Arc<GpuFile>,
}

struct Job {
    checks: Vec<(
        Arc<crate::storage::ssd::SsdReadLease>,
        Vec<(crate::storage::ssd::layout::CopyRange, u32)>,
    )>,
    id: u64,
    task: Task,
    work: VecDeque<Work>,
    encoded: VecDeque<super::EncodedRead>,
    decoding: bool,
    writes: Vec<Write>,
    inflight: usize,
    error: Option<String>,
    uncertain: bool,
    prepared: bool,
    host_completion: Option<Box<dyn LocalCompletion>>,
    started: Instant,
    bytes: usize,
    observation: Observation,
    outcome: Outcome,
}

impl Job {
    fn new(id: u64, command: WorkerCommand) -> Result<Self, EngineError> {
        let (task, observation) = match command {
            WorkerCommand::Load(task, observation) => (Task::Load(task), observation),
            WorkerCommand::Save(task, observation) => (Task::Save(task), observation),
            WorkerCommand::Drain(reply) => {
                let error = "drain cannot be submitted as a file job";
                let _ = reply.send(Err(error.into()));
                return Err(EngineError::Storage(error.into()));
            }
        };
        let mut job = Self {
            checks: Vec::new(),
            id,
            task,
            work: VecDeque::new(),
            encoded: VecDeque::new(),
            decoding: false,
            writes: Vec::new(),
            inflight: 0,
            error: None,
            uncertain: false,
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
                    let extents = Arc::new(plans.owners);
                    for (source, batches) in plans.raw {
                        let file = Arc::clone(source.file()?);
                        job.checks.push((
                            source,
                            batches
                                .iter()
                                .flat_map(|batch| {
                                    batch.copies.iter().copied().map(|copy| (copy, 0))
                                })
                                .collect(),
                        ));
                        for batch in batches {
                            job.bytes += batch.copies.iter().map(|copy| copy.bytes).sum::<usize>();
                            job.work.push_back(Work {
                                options: extents
                                    .iter()
                                    .map(|lease| lease.options)
                                    .min_by_key(|options| options.deadline)
                                    .ok_or_else(|| {
                                        EngineError::Storage("SSD read has no extent owner".into())
                                    })?,
                                file: Arc::clone(&file),
                                extent: extents.clone(),
                                batch,
                                write: None,
                                encoded: false,
                                input: None,
                            });
                        }
                    }
                }
                Task::Save(task) => {
                    for write in &task.ssd_writes {
                        if !write.lease.prepare() {
                            return Err(EngineError::Storage(
                                "SSD reservation was not durable".into(),
                            ));
                        }
                    }
                    for (index, write) in
                        std::mem::take(&mut task.ssd_writes).into_iter().enumerate()
                    {
                        let file = Arc::clone(write.lease.file());
                        job.writes.push(Write {
                            lease: Some(Arc::clone(&write.lease)),
                            remaining: write.batches.len(),
                        });
                        for batch in write.batches {
                            job.work.push_back(Work {
                                file: Arc::clone(&file),
                                extent: write.lease.clone(),
                                options: write.lease.options,
                                batch,
                                write: Some(index),
                                encoded: false,
                                input: None,
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
        Ok(job)
    }

    fn prepare(&mut self, runtime: &WorkerRuntime) -> Result<(), EngineError> {
        if self.prepared {
            return Ok(());
        }
        self.observation.admitted();
        // The route estimate starts before staging/codec work and ends only
        // at engine-visible completion, just like the io_uring host route.
        self.observation.submitted();
        // Verify before any engine destination is touched. The expected per-copy
        // CRCs are derived only from a complete, checksum-valid immutable extent.
        for (source, copies) in &mut self.checks {
            let ranges: Vec<_> = copies
                .iter()
                .map(|(copy, _)| (copy.file_offset, copy.bytes))
                .collect();
            for ((_, expected), checksum) in copies.iter_mut().zip(source.verify_ranges(&ranges)?) {
                *expected = checksum;
            }
        }
        // Encoded reads still validate the actual GPU input before decode. Also
        // validate the committed extent so raw siblings and padding stay bound.
        let mut previous = None;
        for read in &self.encoded {
            let identity = Arc::as_ptr(&read.source.entry.readers);
            if previous != Some(identity) {
                read.source.verify_ranges(&[])?;
                previous = Some(identity);
            }
        }
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
            self.observation.submitted();
            // SAFETY: the pending job retains all layer/grant owners through poll_host.
            self.host_completion = Some(
                unsafe {
                    runtime.backend.submit(
                        &copies,
                        &super::super::host_owners(layers)?,
                        !self.is_write(),
                    )
                }
                .map_err(|e| {
                    if e.kind == ErrorKind::Unquiesced {
                        crate::transfer::retain_gpu_work();
                    }
                    EngineError::Storage(e.to_string())
                })?,
            );
        }
        if !self.is_write() {
            self.bytes += bytes;
        }
        self.prepared = true;
        Ok(())
    }

    fn poll_host(&mut self, _runtime: &WorkerRuntime) {
        let Some(completion) = self.host_completion.as_mut() else {
            return;
        };
        match completion.poll() {
            Ok(true) => {
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active("ssd_host_completion") {
                    return;
                }
                self.host_completion = None;
            }
            Ok(false) => {}
            Err(error) if error.kind == ErrorKind::Unquiesced => crate::transfer::retain_gpu_work(),
            Err(error) => {
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

    fn complete(&mut self, write: Option<usize>, result: Result<(), TransferError>) {
        self.uncertain |= matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced);
        self.inflight -= 1;
        let publish = result.is_ok() && self.error.is_none();
        if let Some(index) = write {
            let write = &mut self.writes[index];
            write.remaining -= 1;
            if write.remaining == 0 && publish {
                if let Some(lease) = write.lease.take() {
                    if !lease.commit() {
                        self.fail("SSD extent was not durably committed".into());
                    }
                } else {
                    self.fail("completed extent has no publication lease".into());
                }
            }
        }
        if let Err(error) = result {
            self.fail(error.to_string());
        }
    }

    fn is_complete(&self) -> bool {
        self.inflight == 0
            && self.work.is_empty()
            && self.encoded.is_empty()
            && !self.decoding
            && self.host_completion.is_none()
    }

    fn verify_destinations(&self, runtime: &WorkerRuntime) -> Result<(), String> {
        let Task::Load(task) = &self.task else {
            return Ok(());
        };
        if self.checks.is_empty() {
            return Ok(());
        }
        let mut owner = runtime.codec.borrow_mut();
        if owner.is_none() {
            *owner = Some(crate::codec::gpu::GpuCodec::new(runtime.stream.context())?);
        }
        let codec = owner.as_mut().ok_or("missing integrity codec")?;
        for (source, copies) in &self.checks {
            for (copy, expected) in copies {
                let started = Instant::now();
                // SAFETY: all file requests have physically drained and the
                // job still owns every imported destination tensor/registration.
                let actual = unsafe {
                    codec.device_checksum(
                        &runtime.stream,
                        copy.device,
                        copy.bytes,
                        runtime.codec_budget(task.codec_budget).min(8192),
                    )
                }?;
                if !crate::integrity::checked(
                    "ssd_gpu",
                    copy.bytes as u64,
                    started,
                    actual == *expected,
                ) {
                    source.invalidate_corrupt();
                    return Err(
                        "SSD GPU destination checksum mismatch; restore invalidated".to_string()
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(
        clippy::mem_forget,
        reason = "unknown file drain retains task completion, admission and caller GPU loans"
    )]
    fn finish(self) {
        if self.uncertain {
            std::mem::forget(self);
            return;
        }
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
        // Physical file bytes belong to EngineFile; this is an inclusive job span.
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
    fn begin(job: &mut Job, decoder: &Decoder, runtime: &WorkerRuntime) -> Result<Self, String> {
        let budget = match &job.task {
            Task::Load(task) => runtime.codec_budget(task.codec_budget),
            Task::Save(_) => return Err("encoded read attached to a save".into()),
        };
        let first = job.encoded.front().ok_or("missing encoded SSD read")?;
        let sizes = job
            .encoded
            .iter()
            .take_while(|read| Arc::ptr_eq(&read.source.entry.readers, &first.source.entry.readers))
            .map(|read| read.meta.stored_bytes);
        let (offsets, bytes) = input_window(sizes, budget)?;
        let backend = Arc::clone(&first.source.file().map_err(|e| e.to_string())?.file.backend);
        decoder.send(DecodeCommand::Prepare {
            bytes,
            budget,
            backend,
        })?;
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

    fn prepared(&mut self, job: &mut Job, input: Arc<CudaRegistration>) -> Result<(), String> {
        let base = input.address;
        self.base = base;
        let copies = self
            .reads
            .iter()
            .zip(&self.offsets)
            .map(|(read, &offset)| {
                Ok(crate::storage::ssd::layout::CopyRange {
                    file_offset: read.file_offset,
                    device: base
                        .checked_add(offset as u64)
                        .ok_or("GPU input offset overflow")?,
                    bytes: read.meta.stored_bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let batches = crate::storage::ssd::layout::plan_reads(copies)?;
        self.remaining = batches.len();
        for batch in batches {
            job.work.push_back(Work {
                file: Arc::clone(self.reads[0].source.file().map_err(|e| e.to_string())?),
                extent: self.reads[0].source.clone(),
                options: self
                    .reads
                    .iter()
                    .map(|read| read.source.options)
                    .min_by_key(|options| options.deadline)
                    .ok_or("encoded read has no source deadline")?,
                batch,
                write: None,
                encoded: true,
                input: Some(Arc::clone(&input)),
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
    let Some(job) = jobs.iter_mut().find(|job| job.id == decode.job) else {
        log::error!(
            "{}",
            TransferError::new(ErrorKind::Unquiesced, "active decoder lost its job")
        );
        crate::transfer::retain_gpu_work();
    };
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
                            crate::integrity::failed("ssd_gpu", "encoding");
                            decode.reads[0].source.invalidate_corrupt();
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
    let mut slots: Vec<Option<Pending>> = (0..FILE_INFLIGHT).map(|_| None).collect();
    let mut decoder: Option<Decoder> = None;
    let mut decoding: Option<Decode> = None;
    let mut drain: Option<oneshot::Sender<Result<(), String>>> = None;
    let mut closed = false;
    let mut next_id = 0;
    let mut read_burst = 0;
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
                    let Some(command) = super::super::prepare_peer(command, &runtime) else {
                        continue;
                    };
                    match Job::new(next_id, command) {
                        Ok(job) => jobs.push_back(job),
                        Err(error) => log::error!("File job rejected: {error}"),
                    }
                    next_id += 1;
                }
                None => closed = true,
            }
        }
        for job in &mut jobs {
            job.poll_host(&runtime);
            job.cancel_abandoned();
        }
        for slot in &mut slots {
            let Some(pending) = slot.as_mut() else {
                continue;
            };
            #[cfg(feature = "test-hooks")]
            if crate::test_faults::active(if pending.owner.write.is_some() {
                "cufile_write_completion"
            } else {
                "cufile_read_completion"
            }) {
                continue;
            }
            let result = match pending.receipt.try_recv() {
                Err(oneshot::error::TryRecvError::Empty) => continue,
                Err(_) => Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "file worker closed without receipt",
                )),
                Ok(result) => result,
            };
            let Some(pending) = slot.take() else {
                continue;
            };
            let owner = pending.owner;
            if owner.encoded {
                let Some(decode) = decoding.as_mut() else {
                    log::error!(
                        "{}",
                        TransferError::new(
                            ErrorKind::Unquiesced,
                            "encoded receipt lost its decoder"
                        )
                    );
                    crate::transfer::retain_gpu_work();
                };
                assert_eq!(decode.job, owner.job);
                decode.remaining -= 1;
            }
            if matches!(&result, Err(e) if e.kind == ErrorKind::Unquiesced) {
                runtime.unquiesced.set(true);
                for job in &mut jobs {
                    job.fail("GPU file worker has no drain receipt".into());
                }
            }
            let result = result.map(|_| ());
            #[cfg(feature = "test-hooks")]
            let result = if result.is_ok()
                && owner.write.is_some()
                && crate::test_faults::active("cufile_write_error")
            {
                Err(TransferError::new(
                    ErrorKind::Backend,
                    "injected GPU file write error after physical receipt",
                ))
            } else {
                result
            };
            if let Err(error) = &result {
                pending.file.gpu_io.failed(&error.to_string());
            }
            let Some(job) = jobs.iter_mut().find(|job| job.id == owner.job) else {
                log::error!(
                    "{}",
                    TransferError::new(ErrorKind::Unquiesced, "file receipt lost its job")
                );
                crate::transfer::retain_gpu_work();
            };
            job.complete(owner.write, result);
        }
        if let Some(decoder) = &decoder {
            poll_decode(decoder, &mut decoding, &mut jobs);
        }
        let mut index = 0;
        while index < jobs.len() {
            if jobs[index].is_complete() {
                if let Some(mut job) = jobs.remove(index) {
                    if job.error.is_none()
                        && !job.uncertain
                        && let Err(error) = job.verify_destinations(&runtime)
                    {
                        job.fail(error);
                    }
                    job.finish();
                }
            } else {
                index += 1;
            }
        }
        if jobs.is_empty() {
            if closed {
                drop(slots);
                drop(decoder);
                let result = runtime.drain_transfers();
                drop(runtime);
                if let Some(reply) = drain {
                    let _ = reply.send(result);
                }
                return;
            }
            continue;
        }
        for index in 0..slots.len() {
            if slots[index].is_some() {
                continue;
            }
            let writing = slots
                .iter()
                .flatten()
                .any(|pending| pending.owner.write.is_some());
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
            let Some(mut job) = jobs.remove(position) else {
                continue;
            };
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
                let result = decoder
                    .as_ref()
                    .ok_or_else(|| "decoder was not initialized".to_string())
                    .and_then(|decoder| Decode::begin(&mut job, decoder, &runtime));
                match result {
                    Ok(active) => decoding = Some(active),
                    Err(error) => job.fail(error),
                }
                jobs.push_back(job);
                continue;
            }
            let Some(work) = job.work.pop_front() else {
                job.fail("selected file job has no work".into());
                jobs.push_back(job);
                continue;
            };
            let owner = BatchOwner {
                job: job.id,
                write: work.write,
                encoded: work.encoded,
            };
            job.observation.submitted();
            let submitted = (|| {
                let owners = if let Some(input) = work.input {
                    vec![input]
                } else {
                    let layers = match &job.task {
                        Task::Load(task) => &task.layers,
                        Task::Save(task) => &task.layers,
                    };
                    layers
                        .iter()
                        .map(|layer| {
                            layer
                                .layout
                                .owner
                                .clone()
                                .ok_or("GPU tensor has no transfer registration".to_owned())
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                work.file
                    .file
                    .submit(
                        if reading {
                            TransferOp::Read
                        } else {
                            TransferOp::Write
                        },
                        work.file.transport,
                        work.batch.into_extent()?,
                        FileMemory::RegisteredCuda {
                            owners,
                            context: Arc::clone(runtime.stream.context()),
                        },
                        work.extent,
                        work.options,
                    )
                    .map_err(|e| e.to_string())
            })();
            match submitted {
                Ok(receipt) => {
                    job.inflight += 1;
                    slots[index] = Some(Pending {
                        receipt,
                        owner,
                        file: work.file,
                    });
                }
                Err(error) => job.fail(error),
            }
            jobs.push_back(job);
        }
        std::thread::sleep(Duration::from_micros(50));
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transfer/worker/ssd/queue.rs"]
mod tests;
