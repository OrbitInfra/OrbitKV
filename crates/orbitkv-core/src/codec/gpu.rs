use std::{ops::Range, sync::Arc};

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, result};
use orbitkv_state::{AttentionRole, Scalar16, StorageFormat};

use super::{EncodedSegment, ans, turboquant_bytes};

pub(crate) const MAX_BATCH_SEGMENTS: usize = 256;
const MAX_CODEC_BYTES: usize = 16 * 1024 * 1024;
const ALIGNMENT: usize = 4096;
const CODEBOOK_BYTES: usize = 512;
const MAX_CRC_TILES: usize = 128;

pub(crate) struct EncodeInput {
    pub source: u64,
    pub bytes: usize,
    pub format: StorageFormat,
}

pub(crate) struct EncodedDeviceSegment {
    /// 4096-byte aligned. Only meta.stored_bytes bytes form the payload;
    /// callers must initialize any padding in their transfer destination.
    pub device: u64,
    pub meta: EncodedSegment,
}

pub(crate) struct EncodedBatch<'a> {
    pub outputs: Vec<Option<EncodedDeviceSegment>>,
    pub processed: usize,
    _owner: &'a GpuCodec,
}

pub(crate) struct DecodeInput<'a> {
    pub source: u64,
    pub source_bytes: usize,
    pub target: u64,
    pub target_bytes: usize,
    pub meta: &'a EncodedSegment,
}

pub(crate) struct HostDecodeInput<'a> {
    pub source: &'a [u8],
    pub target: u64,
    pub target_bytes: usize,
    pub meta: &'a EncodedSegment,
}

#[derive(Debug)]
pub(crate) enum DecodeError {
    /// The payload failed its CRC or a decoded segment failed status/size checks.
    Corrupt(String),
    /// The operation failed without establishing that the stored data is corrupt.
    Runtime(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(message) | Self::Runtime(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DecodeError {}

/// One codec belongs to one worker stream. Its only GPU allocation is a reusable
/// arena, including descriptors, codebooks, CRC reductions and nvCOMP workspace.
pub(crate) struct GpuCodec {
    ctx: Arc<CudaContext>,
    fp8_encode: CudaFunction,
    fp8_decode: CudaFunction,
    turbo_encode: CudaFunction,
    turbo_decode: CudaFunction,
    copy: CudaFunction,
    crc_parts: CudaFunction,
    crc_finish: CudaFunction,
    codebooks: Vec<f32>,
    ans: Option<ans::Library>,
    arena: Option<Arena>,
    upload: Vec<u8>,
    readback: Vec<u8>,
}

// Must match Segment in kernels.cu; all fields and padding are initialized.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DeviceSegment {
    source: u64,
    target: u64,
    logical: u64,
    stored: u64,
    capacity: u64,
    codebook: u64,
    dim: u32,
    bits: u32,
    bf: i32,
    role: i32,
    seed: u32,
    crc_tiles: u32,
    crc_offset: u32,
    reserved: u32,
}
const _: () = assert!(size_of::<DeviceSegment>() == 80);

struct Arena {
    stream: Arc<CudaStream>,
    allocation: u64,
    base: u64,
    bytes: usize,
}

impl Drop for Arena {
    fn drop(&mut self) {
        drain(&self.stream);
        // SAFETY: this is the unique owner of a synchronous CUDA allocation;
        // raw external readers must complete before their batch borrow ends.
        if let Err(error) = unsafe { result::free_sync(self.allocation) } {
            log::error!("Failed to free GPU codec arena: {error}");
            std::process::abort();
        }
        crate::metrics::core_metrics()
            .storage_codec_workspace_bytes
            .add(-(self.bytes as i64), &[]);
    }
}

fn drain(stream: &CudaStream) {
    if let Err(error) = stream.synchronize() {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr().lock(),
            "GPU codec synchronization failed: {error}"
        );
        crate::transfer::retain_gpu_work();
    }
}

struct Drain<'a> {
    stream: &'a CudaStream,
    pending: bool,
}

impl<'a> Drain<'a> {
    fn new(stream: &'a CudaStream) -> Self {
        Self {
            stream,
            pending: true,
        }
    }
    fn finish(&mut self) {
        drain(self.stream);
        self.pending = false;
    }
}

impl Drop for Drain<'_> {
    fn drop(&mut self) {
        if self.pending {
            drain(self.stream);
        }
    }
}

struct Entry {
    index: usize,
    source: u64,
    target: u64,
    logical: usize,
    stored: usize,
    capacity: usize,
    format: StorageFormat,
    offset: usize,
}

struct Group {
    range: Range<usize>,
    key: u32,
    temp: usize,
    max_bytes: usize,
}

/// Checked offsets relative to the 4096-aligned arena base. Resident allocation
/// size includes the worst-case base alignment adjustment.
struct Layout {
    descriptors: usize,
    inputs: usize,
    input_sizes: usize,
    outputs: usize,
    capacities: usize,
    sizes: usize,
    statuses: usize,
    checksums: usize,
    header_end: usize,
    partials: usize,
    temp: usize,
    payload: usize,
    end: usize,
}

fn take(end: &mut usize, bytes: usize, alignment: usize) -> Result<usize, String> {
    let offset = end
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| "GPU codec layout overflow".to_string())?;
    *end = offset
        .checked_add(bytes)
        .ok_or_else(|| "GPU codec layout overflow".to_string())?;
    Ok(offset)
}

impl Layout {
    fn new(entries: &mut [Entry], temp_bytes: usize, payload: bool) -> Result<Self, String> {
        let n = entries.len();
        let mut end = CODEBOOK_BYTES;
        let descriptors = take(&mut end, n * size_of::<DeviceSegment>(), 8)?;
        let inputs = take(&mut end, n * 8, 8)?;
        let input_sizes = take(&mut end, n * 8, 8)?;
        let outputs = take(&mut end, n * 8, 8)?;
        let capacities = take(&mut end, n * 8, 8)?;
        let sizes = take(&mut end, n * 8, 8)?;
        let statuses = take(&mut end, n * 4, 4)?;
        let checksums = take(&mut end, n * 4, 4)?;
        let header_end = end;
        let crc_tiles: usize = entries.iter().map(|e| tile_count(e.capacity)).sum();
        let partials = take(&mut end, crc_tiles * 4, 4)?;
        let temp = take(&mut end, temp_bytes, 256)?;
        let payload_start = take(&mut end, 0, ALIGNMENT)?;
        if payload {
            for e in entries {
                e.offset = take(&mut end, e.capacity, ALIGNMENT)?;
            }
        }
        Ok(Self {
            descriptors,
            inputs,
            input_sizes,
            outputs,
            capacities,
            sizes,
            statuses,
            checksums,
            header_end,
            partials,
            temp,
            payload: payload_start,
            end,
        })
    }

    fn allocation_bytes(&self) -> Result<usize, String> {
        self.end
            .checked_add(ALIGNMENT - 1)
            .filter(|&n| isize::try_from(n).is_ok())
            .ok_or_else(|| "GPU codec allocation overflow".to_string())
    }

    fn device_batch(&self, base: u64, group: &Group) -> ans::DeviceBatch {
        let start = group.range.start;
        ans::DeviceBatch {
            inputs: base + (self.inputs + start * 8) as u64,
            input_sizes: base + (self.input_sizes + start * 8) as u64,
            outputs: base + (self.outputs + start * 8) as u64,
            capacities: base + (self.capacities + start * 8) as u64,
            sizes: base + (self.sizes + start * 8) as u64,
            statuses: base + (self.statuses + start * 4) as u64,
            temp: base + self.temp as u64,
            temp_bytes: group.temp,
            count: group.range.len(),
            max_bytes: group.max_bytes,
        }
    }
}

fn tile_count(bytes: usize) -> usize {
    bytes.div_ceil(4096).clamp(1, MAX_CRC_TILES)
}

// Adjacent entries sharing a launch geometry form one GPU batch. ANS types are
// intentionally distinct: nvCOMP's compression data type is a per-call option.
fn group_key(format: StorageFormat) -> u32 {
    match format {
        StorageFormat::Exact => 0,
        StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => 1,
        StorageFormat::TurboQuant { head_dim, .. } => head_dim,
        _ => 512 + ans_type(format).unwrap_or(0) as u32,
    }
}

fn check_range(pointer: u64, bytes: usize) -> Result<(), String> {
    if pointer == 0
        || bytes == 0
        || bytes > isize::MAX as usize
        || pointer.checked_add(bytes as u64).is_none()
    {
        return Err("invalid GPU codec device range".into());
    }
    Ok(())
}

/// Validate every destination in a restore request before CPU fallback or batch
/// splitting. This checks address arithmetic and disjointness, not GPU ownership.
/// Returns sorted (start, exclusive end) ranges for subsequent alias checks.
pub(crate) fn validate_targets(
    ranges: impl IntoIterator<Item = (u64, usize)>,
) -> Result<Vec<(u64, u64)>, String> {
    let mut targets = Vec::new();
    for (pointer, bytes) in ranges {
        check_range(pointer, bytes)?;
        targets.push((pointer, pointer + bytes as u64));
    }
    targets.sort_unstable();
    if targets.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err("GPU decode target ranges overlap".into());
    }
    Ok(targets)
}

impl GpuCodec {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        use cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let major = ctx
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?;
        let minor = ctx
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?;
        let arch = if major >= 9 {
            Some("compute_90")
        } else if major == 8 && minor >= 9 {
            Some("compute_89")
        } else {
            None
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("kernels.cu"),
            cudarc::nvrtc::CompileOptions {
                arch,
                ..Default::default()
            },
        )
        .map_err(|e| format!("codec NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let function = |name| module.load_function(name).map_err(|e| e.to_string());
        let mut codebooks = Vec::new();
        for dim in [32, 64, 128, 256] {
            for bits in [3, 4] {
                codebooks.extend(centroids(dim, bits));
            }
        }
        Ok(Self {
            ctx: ctx.clone(),
            fp8_encode: function("fp8_encode_batch")?,
            fp8_decode: function("fp8_decode_batch")?,
            turbo_encode: function("turbo_encode_batch")?,
            turbo_decode: function("turbo_decode_batch")?,
            copy: function("copy_batch")?,
            crc_parts: function("crc32_parts")?,
            crc_finish: function("crc32_finish")?,
            codebooks,
            ans: None,
            arena: None,
            upload: Vec::new(),
            readback: Vec::new(),
        })
    }

    fn ans(&mut self) -> Result<&ans::Library, String> {
        if self.ans.is_none() {
            self.ans = Some(ans::Library::load()?);
        }
        self.ans
            .as_ref()
            .ok_or_else(|| "nvCOMP library unavailable".into())
    }

    fn bind(&mut self, stream: &Arc<CudaStream>, budget: usize) -> Result<(), String> {
        if stream.context() != &self.ctx {
            return Err("GPU codec used with a different CUDA context".into());
        }
        self.ctx.bind_to_thread().map_err(|e| e.to_string())?;
        if let Some(arena) = &self.arena {
            if arena.stream.cu_stream() != stream.cu_stream() {
                return Err("GPU codec must stay on its worker stream".into());
            }
            if arena.bytes > budget {
                self.arena = None;
            }
        }
        Ok(())
    }

    fn reserve(
        &mut self,
        stream: &Arc<CudaStream>,
        bytes: usize,
        budget: usize,
    ) -> Result<u64, String> {
        if bytes > budget {
            return Err("batch exceeds GPU codec budget".into());
        }
        if self.arena.as_ref().is_none_or(|a| a.bytes < bytes) {
            // Release the previous high-water allocation first so growth itself
            // never temporarily doubles the configured resident GPU budget.
            self.arena = None;
            // cuMemAlloc (not the async pool) also permits cuFileBufRegister.
            let allocation = unsafe { result::malloc_sync(bytes) }.map_err(|e| e.to_string())?;
            let base = allocation.next_multiple_of(ALIGNMENT as u64);
            self.arena = Some(Arena {
                stream: stream.clone(),
                allocation,
                base,
                bytes,
            });
            crate::metrics::core_metrics()
                .storage_codec_workspace_bytes
                .add(bytes as i64, &[]);
            crate::metrics::core_metrics()
                .storage_codec_workspace_allocations
                .add(1, &[]);
        }
        self.arena
            .as_ref()
            .map(|a| a.base)
            .ok_or_else(|| "GPU codec arena unavailable".into())
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        self.arena.as_ref().map_or(0, |a| a.bytes)
    }

    /// Query the actual single-segment staging/workspace requirement without
    /// allocating or launching GPU work. Exact copies do not need an arena.
    pub(crate) fn host_decode_fits(
        &mut self,
        meta: &EncodedSegment,
        budget: usize,
    ) -> Result<bool, String> {
        meta.validate_metadata(meta.stored_bytes)?;
        if meta.format == StorageFormat::Exact {
            return Ok(true);
        }
        let input = DecodeInput {
            source: 0,
            source_bytes: meta.stored_bytes,
            target: 0,
            target_bytes: meta.logical_bytes,
            meta,
        };
        let (_, _, layout) = self.decode_plan(&[input], true)?;
        Ok(layout.allocation_bytes()? <= budget)
    }

    fn groups(&mut self, entries: &mut [Entry], encode: bool) -> Result<Vec<Group>, String> {
        entries.sort_by_key(|e| group_key(e.format));
        let mut groups = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let key = group_key(entries[start].format);
            let end = start
                + entries[start..]
                    .iter()
                    .take_while(|e| group_key(e.format) == key)
                    .count();
            let range = start..end;
            let max_bytes = entries[range.clone()]
                .iter()
                .map(|e| e.logical)
                .max()
                .unwrap_or(0);
            let total = entries[range.clone()]
                .iter()
                .try_fold(0usize, |n, e| n.checked_add(e.logical))
                .ok_or_else(|| "GPU codec batch size overflow".to_string())?;
            let mut temp = 0;
            if let Some(data_type) = ans_type(entries[start].format) {
                let library = self.ans()?;
                temp = library.workspace(range.len(), max_bytes, total, data_type, encode)?;
                if encode {
                    temp = temp.max(library.workspace(
                        range.len(),
                        max_bytes,
                        total,
                        data_type,
                        false,
                    )?);
                    let capacity = library
                        .max_output(max_bytes, data_type)?
                        .checked_next_multiple_of(ALIGNMENT)
                        .ok_or_else(|| "ANS output capacity overflow".to_string())?;
                    for e in &mut entries[range.clone()] {
                        e.capacity = capacity;
                    }
                }
            }
            groups.push(Group {
                range,
                key,
                temp,
                max_bytes,
            });
            start = end;
        }
        Ok(groups)
    }

    fn encode_plan(
        &mut self,
        inputs: &[EncodeInput],
    ) -> Result<(Vec<Entry>, Vec<Group>, Layout), String> {
        let mut entries = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            if input.bytes == 0 || input.bytes > MAX_CODEC_BYTES {
                continue;
            }
            let stored = match input.format {
                StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
                    if input.bytes.is_multiple_of(2) && input.source.is_multiple_of(2) =>
                {
                    input.bytes / 2
                }
                StorageFormat::TurboQuant {
                    head_dim,
                    bits,
                    role,
                    ..
                } if input.source.is_multiple_of(2) => {
                    let Some(size) = turboquant_bytes(input.bytes, head_dim, bits, role) else {
                        continue;
                    };
                    size
                }
                format
                    if ans_type(format).is_some()
                        && input.bytes >= 4096
                        && input.source.is_multiple_of(8)
                        && (format != StorageFormat::Ans16 || input.bytes.is_multiple_of(2)) =>
                {
                    0
                }
                _ => continue,
            };
            check_range(input.source, input.bytes)?;
            entries.push(Entry {
                index,
                source: input.source,
                target: 0,
                logical: input.bytes,
                stored,
                capacity: stored
                    .checked_next_multiple_of(ALIGNMENT)
                    .ok_or("GPU codec size overflow")?,
                format: input.format,
                offset: 0,
            });
        }
        let groups = self.groups(&mut entries, true)?;
        let temp = groups.iter().map(|g| g.temp).max().unwrap_or(0);
        let layout = Layout::new(&mut entries, temp, true)?;
        Ok((entries, groups, layout))
    }

    /// Consume a prefix of at most MAX_BATCH_SEGMENTS. Budget pressure reduces
    /// the prefix; an individually unencodable first input is consumed as None.
    ///
    /// # Safety
    /// Sources must be readable on this stream for their declared lengths and
    /// remain leased through return. Returned device pointers borrow this codec;
    /// all D2H/cuFile accesses must finish before dropping the returned batch.
    pub(crate) unsafe fn encode_batch(
        &mut self,
        stream: &Arc<CudaStream>,
        inputs: &[EncodeInput],
        budget: usize,
    ) -> Result<EncodedBatch<'_>, String> {
        self.bind(stream, budget)?;
        let mut processed = inputs.len().min(MAX_BATCH_SEGMENTS);
        let mut plan = self.encode_plan(&inputs[..processed])?;
        let fits = |entries: &[Entry], layout: &Layout| -> Result<bool, String> {
            Ok(entries.is_empty() || layout.allocation_bytes()? <= budget)
        };
        if !fits(&plan.0, &plan.2)? {
            let mut low = 0;
            let mut high = processed;
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                let trial = self.encode_plan(&inputs[..mid])?;
                if fits(&trial.0, &trial.2)? {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            processed = low;
            plan = self.encode_plan(&inputs[..processed])?;
        }
        if processed == 0 && !inputs.is_empty() {
            return Ok(EncodedBatch {
                outputs: vec![None],
                processed: 1,
                _owner: self,
            });
        }
        let (mut entries, groups, layout) = plan;
        let mut outputs: Vec<_> = (0..processed).map(|_| None).collect();
        if !entries.is_empty() {
            let base = self.reserve(stream, layout.allocation_bytes()?, budget)?;
            for e in &mut entries {
                e.target = base + e.offset as u64;
            }
            self.prepare_header(&entries, &layout, base, true);
            let mut guard = Drain::new(stream);
            record_batch("encode", entries.len());
            self.upload_header(stream, &layout, base)?;
            // Initialize reused output storage. nvCOMP may use bytes beyond its
            // final payload length; consumers copy only the reported payload.
            unsafe {
                result::memset_d8_async(
                    base + layout.payload as u64,
                    0,
                    layout.end - layout.payload,
                    stream.cu_stream(),
                )
            }
            .map_err(|e| e.to_string())?;
            self.launch_groups(stream, &entries, &groups, &layout, base, true)?;
            self.launch_crc(stream, &entries, &layout, base, true)?;
            self.download_results(stream, &layout, base)?;
            guard.finish();
            for (i, e) in entries.iter().enumerate() {
                let status = self.status(&layout, i);
                if ans_type(e.format).is_some() && status != 0 {
                    return Err(format!(
                        "nvCOMP ANS encode segment {} status {status}",
                        e.index
                    ));
                }
                if status != 0 {
                    continue;
                }
                let stored = self.size(i);
                if stored == 0 || stored > e.capacity || stored > 32 * 1024 * 1024 {
                    return Err("GPU codec returned invalid encoded size".into());
                }
                outputs[e.index] = Some(EncodedDeviceSegment {
                    device: e.target,
                    meta: EncodedSegment {
                        version: 1,
                        format: e.format,
                        logical_bytes: e.logical,
                        stored_bytes: stored,
                        checksum: self.checksum(&layout, i),
                    },
                });
            }
        }
        Ok(EncodedBatch {
            outputs,
            processed,
            _owner: self,
        })
    }

    /// Decode device-resident encoded payloads, including cuFile destinations.
    /// The entire batch passes metadata and CRC checks before any target writes.
    ///
    /// # Safety
    /// Sources and targets must be valid on this worker's stream until return,
    /// with producer writes already ordered before this stream. Target ranges
    /// must be disjoint from each other, every source, and the codec's arena.
    /// Sources must not alias the codec's reusable arena.
    pub(crate) unsafe fn decode_batch(
        &mut self,
        stream: &Arc<CudaStream>,
        inputs: &[DecodeInput<'_>],
        budget: usize,
    ) -> Result<(), DecodeError> {
        self.validate_decode(inputs, false)
            .map_err(DecodeError::Runtime)?;
        self.bind(stream, budget).map_err(DecodeError::Runtime)?;
        if inputs.is_empty() {
            return Ok(());
        }
        let (entries, groups, layout) = self
            .decode_plan(inputs, false)
            .map_err(DecodeError::Runtime)?;
        let bytes = layout.allocation_bytes().map_err(DecodeError::Runtime)?;
        let base = self
            .reserve(stream, bytes, budget)
            .map_err(DecodeError::Runtime)?;
        self.prepare_header(&entries, &layout, base, false);
        let mut guard = Drain::new(stream);
        self.upload_header(stream, &layout, base)
            .map_err(DecodeError::Runtime)?;
        self.verify_device_crc(stream, inputs, &entries, &layout, base, &mut guard)?;
        guard.pending = true;
        record_batch("decode", entries.len());
        self.launch_groups(stream, &entries, &groups, &layout, base, false)
            .map_err(DecodeError::Runtime)?;
        self.download_results(stream, &layout, base)
            .map_err(DecodeError::Runtime)?;
        guard.finish();
        self.check_decoded(&entries, &layout)
            .map_err(DecodeError::Corrupt)
    }

    /// Host restore uses the same arena for uploads and decode scratch. Exact
    /// siblings copy directly to their targets and need no GPU staging space.
    /// Returns the consumed prefix (at most MAX_BATCH_SEGMENTS). An encoded
    /// first segment which cannot fit is an error; raw Exact needs no scratch.
    ///
    /// # Safety
    /// Target ranges must be writable on this stream until return, mutually
    /// disjoint and disjoint from this codec's arena.
    pub(crate) unsafe fn decode_host_batch(
        &mut self,
        stream: &Arc<CudaStream>,
        inputs: &[HostDecodeInput<'_>],
        budget: usize,
    ) -> Result<usize, String> {
        let mut inputs = &inputs[..inputs.len().min(MAX_BATCH_SEGMENTS)];
        let mut device_inputs: Vec<_> = inputs
            .iter()
            .map(|input| DecodeInput {
                source: 0,
                source_bytes: input.source.len(),
                target: input.target,
                target_bytes: input.target_bytes,
                meta: input.meta,
            })
            .collect();
        self.validate_decode(&device_inputs, true)?;
        self.bind(stream, budget)?;
        if inputs.is_empty() {
            return Ok(0);
        }
        let mut plan = self.decode_plan(&device_inputs, true)?;
        let fits = |entries: &[Entry], layout: &Layout| -> Result<bool, String> {
            Ok(entries.is_empty() || layout.allocation_bytes()? <= budget)
        };
        if !fits(&plan.0, &plan.2)? {
            let mut low = 0;
            let mut high = inputs.len();
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                let trial = self.decode_plan(&device_inputs[..mid], true)?;
                if fits(&trial.0, &trial.2)? {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            if low == 0 {
                return Err("first restore segment exceeds GPU codec budget".into());
            }
            inputs = &inputs[..low];
            device_inputs.truncate(low);
            plan = self.decode_plan(&device_inputs, true)?;
        }
        let (mut entries, groups, layout) = plan;
        // Only checksum the consumed prefix. Rehashing deferred siblings on
        // every budget-limited call would make host restore quadratic.
        for input in inputs {
            input.meta.validate(input.source)?;
        }
        let base = if entries.is_empty() {
            0
        } else {
            self.reserve(stream, layout.allocation_bytes()?, budget)?
        };
        for e in &mut entries {
            e.source = base + e.offset as u64;
        }
        self.prepare_header(&entries, &layout, base, false);
        let mut guard = Drain::new(stream);
        if !entries.is_empty() {
            self.upload_header(stream, &layout, base)?;
            for e in &entries {
                // SAFETY: inputs outlive the drain, and each upload has a
                // separate, checked arena slot with sufficient capacity.
                unsafe {
                    result::memcpy_htod_async(
                        e.source,
                        &inputs[e.index].source[..e.stored],
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string())?;
            }
        }
        record_batch("decode", inputs.len());
        for input in inputs {
            if input.meta.format == StorageFormat::Exact {
                unsafe {
                    result::memcpy_htod_async(
                        input.target,
                        &input.source[..input.meta.stored_bytes],
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string())?;
            }
        }
        if !entries.is_empty() {
            self.launch_groups(stream, &entries, &groups, &layout, base, false)?;
            self.download_results(stream, &layout, base)?;
        }
        guard.finish();
        self.check_decoded(&entries, &layout)?;
        Ok(inputs.len())
    }

    fn validate_decode(&self, inputs: &[DecodeInput<'_>], host: bool) -> Result<(), String> {
        if inputs.len() > MAX_BATCH_SEGMENTS {
            return Err("GPU codec batch segment limit exceeded".into());
        }
        let targets = validate_targets(
            inputs
                .iter()
                .map(|input| (input.target, input.meta.logical_bytes)),
        )?;
        for input in inputs {
            input.meta.validate_metadata(input.source_bytes)?;
            let meta = input.meta;
            if meta.logical_bytes > input.target_bytes {
                return Err("GPU decode target capacity mismatch".into());
            }
            if !host {
                check_range(input.source, meta.stored_bytes)?;
            }
            let alignment = if ans_type(meta.format).is_some() {
                8
            } else if meta.format == StorageFormat::Exact {
                1
            } else {
                2
            };
            // FP8/TurboQuant input is byte-addressed; only ANS requires aligned input.
            if !input.target.is_multiple_of(alignment)
                || (!host && ans_type(meta.format).is_some() && !input.source.is_multiple_of(8))
            {
                return Err("GPU decode device alignment mismatch".into());
            }
            if let Some(arena) = &self.arena {
                let overlap = |ptr: u64, len: usize| {
                    ptr < arena.allocation + arena.bytes as u64
                        && arena.allocation < ptr + len as u64
                };
                if overlap(input.target, meta.logical_bytes)
                    || (!host && overlap(input.source, meta.stored_bytes))
                {
                    return Err("GPU decode range aliases reusable codec arena".into());
                }
            }
        }
        if !host {
            // Sources may overlap each other (one encoded page restored into
            // multiple targets), but no decode may overwrite a live source.
            for input in inputs {
                let end = input.source + input.meta.stored_bytes as u64;
                let index = targets.partition_point(|&(start, _)| start < end);
                if index != 0 && targets[index - 1].1 > input.source {
                    return Err("GPU decode source and target ranges overlap".into());
                }
            }
        }
        Ok(())
    }

    fn decode_plan(
        &mut self,
        inputs: &[DecodeInput<'_>],
        host: bool,
    ) -> Result<(Vec<Entry>, Vec<Group>, Layout), String> {
        let mut entries = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let meta = input.meta;
            if host && meta.format == StorageFormat::Exact {
                continue;
            }
            let capacity = if host {
                meta.stored_bytes
                    .checked_next_multiple_of(ALIGNMENT)
                    .ok_or("GPU decode capacity overflow")?
            } else {
                meta.stored_bytes
            };
            entries.push(Entry {
                index,
                source: input.source,
                target: input.target,
                logical: meta.logical_bytes,
                stored: meta.stored_bytes,
                capacity,
                format: meta.format,
                offset: 0,
            });
        }
        let groups = self.groups(&mut entries, false)?;
        let temp = groups.iter().map(|g| g.temp).max().unwrap_or(0);
        let layout = Layout::new(&mut entries, temp, host)?;
        Ok((entries, groups, layout))
    }

    fn prepare_header(&mut self, entries: &[Entry], layout: &Layout, base: u64, encode: bool) {
        self.upload.resize(layout.header_end, 0);
        self.upload.fill(0);
        for (i, value) in self.codebooks.iter().enumerate() {
            self.upload[i * 4..i * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        let mut crc_offset = 0;
        for (i, e) in entries.iter().enumerate() {
            let mut desc = DeviceSegment {
                source: e.source,
                target: e.target,
                logical: e.logical as u64,
                stored: e.stored as u64,
                capacity: e.capacity as u64,
                crc_tiles: tile_count(e.capacity) as u32,
                crc_offset,
                ..Default::default()
            };
            crc_offset += desc.crc_tiles;
            match e.format {
                StorageFormat::Fp8FromBf16 => desc.bf = 1,
                StorageFormat::TurboQuant {
                    head_dim,
                    bits,
                    scalar,
                    role,
                    seed,
                } => {
                    desc.dim = head_dim;
                    desc.bits = bits as u32;
                    desc.bf = i32::from(scalar == Scalar16::Bf16);
                    desc.role = match role {
                        AttentionRole::PackedKeyValue => 2,
                        _ => i32::from(role == AttentionRole::Key),
                    };
                    desc.seed = seed;
                    // Each dimension owns 8 three-bit and 16 four-bit centroids.
                    let book = (head_dim.trailing_zeros() as usize - 5) * 24
                        + if bits == 4 { 8 } else { 0 };
                    desc.codebook = base + (book * 4) as u64;
                }
                _ => {}
            }
            // SAFETY: DeviceSegment is repr(C) and contains no implicit padding.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (&desc as *const DeviceSegment).cast::<u8>(),
                    size_of::<DeviceSegment>(),
                )
            };
            let start = layout.descriptors + i * bytes.len();
            self.upload[start..start + bytes.len()].copy_from_slice(bytes);
            for (offset, value) in [
                (layout.inputs, e.source),
                (
                    layout.input_sizes,
                    if encode { e.logical } else { e.stored } as u64,
                ),
                (layout.outputs, e.target),
                (layout.capacities, e.logical as u64),
                (
                    layout.sizes,
                    if encode { e.stored } else { e.logical } as u64,
                ),
            ] {
                self.upload[offset + i * 8..offset + i * 8 + 8]
                    .copy_from_slice(&value.to_ne_bytes());
            }
            if ans_type(e.format).is_some() {
                // Detect a backend which fails to report a result for a chunk.
                self.upload[layout.statuses + i * 4..layout.statuses + i * 4 + 4]
                    .copy_from_slice(&(-1i32).to_ne_bytes());
                if !encode {
                    self.upload[layout.sizes + i * 8..layout.sizes + i * 8 + 8].fill(0);
                }
            }
        }
    }

    fn upload_header(&self, stream: &CudaStream, layout: &Layout, base: u64) -> Result<(), String> {
        // SAFETY: self.upload stays unchanged until the enclosing drain completes.
        unsafe {
            result::memcpy_htod_async(base, &self.upload[..layout.header_end], stream.cu_stream())
        }
        .map_err(|e| e.to_string())
    }

    fn launch_groups(
        &self,
        stream: &Arc<CudaStream>,
        entries: &[Entry],
        groups: &[Group],
        layout: &Layout,
        base: u64,
        encode: bool,
    ) -> Result<(), String> {
        for group in groups {
            if let Some(data_type) = ans_type(entries[group.range.start].format) {
                let library = self.ans.as_ref().ok_or("nvCOMP not initialized")?;
                let batch = layout.device_batch(base, group);
                unsafe {
                    if encode {
                        library.encode_batch(stream, batch, data_type)?;
                    } else {
                        library.decode_batch(stream, batch, data_type)?;
                    }
                }
                continue;
            }
            let function = match (group.key, encode) {
                (0, _) => &self.copy,
                (1, true) => &self.fp8_encode,
                (1, false) => &self.fp8_decode,
                (_, true) => &self.turbo_encode,
                (_, false) => &self.turbo_decode,
            };
            let descriptors =
                base + (layout.descriptors + group.range.start * size_of::<DeviceSegment>()) as u64;
            let statuses = base + (layout.statuses + group.range.start * 4) as u64;
            let block = if group.key >= 32 { group.key } else { 256 };
            let work = if group.key >= 32 {
                group.max_bytes / (block as usize * 2)
            } else {
                group.max_bytes.div_ceil(256)
            };
            let mut launch = stream.launch_builder(function);
            launch.arg(&descriptors).arg(&statuses);
            // SAFETY: per-group descriptors have validated geometry and ranges.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (work.clamp(1, 256) as u32, group.range.len() as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn launch_crc(
        &self,
        stream: &Arc<CudaStream>,
        entries: &[Entry],
        layout: &Layout,
        base: u64,
        encode: bool,
    ) -> Result<(), String> {
        let descriptors = base + layout.descriptors as u64;
        let sizes = base + layout.sizes as u64;
        let statuses = base + layout.statuses as u64;
        let partials = base + layout.partials as u64;
        let checksums = base + layout.checksums as u64;
        let encode = i32::from(encode);
        let max_tiles = entries
            .iter()
            .map(|e| tile_count(e.capacity))
            .max()
            .unwrap_or(1) as u32;
        let mut launch = stream.launch_builder(&self.crc_parts);
        launch
            .arg(&descriptors)
            .arg(&sizes)
            .arg(&statuses)
            .arg(&partials)
            .arg(&encode);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (max_tiles, entries.len() as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| e.to_string())?;
        let mut launch = stream.launch_builder(&self.crc_finish);
        launch
            .arg(&descriptors)
            .arg(&sizes)
            .arg(&partials)
            .arg(&checksums)
            .arg(&encode);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (entries.len() as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn download_results(
        &mut self,
        stream: &CudaStream,
        layout: &Layout,
        base: u64,
    ) -> Result<(), String> {
        self.readback.resize(layout.header_end - layout.sizes, 0);
        // SAFETY: reusable host readback storage remains owned through the drain.
        unsafe {
            result::memcpy_dtoh_async(
                &mut self.readback,
                base + layout.sizes as u64,
                stream.cu_stream(),
            )
        }
        .map_err(|e| e.to_string())
    }

    fn verify_device_crc(
        &mut self,
        stream: &Arc<CudaStream>,
        inputs: &[DecodeInput<'_>],
        entries: &[Entry],
        layout: &Layout,
        base: u64,
        guard: &mut Drain<'_>,
    ) -> Result<(), DecodeError> {
        self.launch_crc(stream, entries, layout, base, false)
            .map_err(DecodeError::Runtime)?;
        self.download_results(stream, layout, base)
            .map_err(DecodeError::Runtime)?;
        guard.finish();
        for (i, e) in entries.iter().enumerate() {
            if self.checksum(layout, i) != inputs[e.index].meta.checksum {
                return Err(DecodeError::Corrupt(format!(
                    "encoded segment {} GPU checksum mismatch",
                    e.index
                )));
            }
        }
        Ok(())
    }

    fn size(&self, index: usize) -> usize {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&self.readback[index * 8..index * 8 + 8]);
        u64::from_ne_bytes(bytes) as usize
    }
    fn status(&self, layout: &Layout, index: usize) -> i32 {
        let offset = layout.statuses - layout.sizes + index * 4;
        let mut bytes = [0; 4];
        bytes.copy_from_slice(&self.readback[offset..offset + 4]);
        i32::from_ne_bytes(bytes)
    }
    fn checksum(&self, layout: &Layout, index: usize) -> u32 {
        let offset = layout.checksums - layout.sizes + index * 4;
        let mut bytes = [0; 4];
        bytes.copy_from_slice(&self.readback[offset..offset + 4]);
        u32::from_ne_bytes(bytes)
    }
    fn check_decoded(&self, entries: &[Entry], layout: &Layout) -> Result<(), String> {
        for (i, e) in entries.iter().enumerate() {
            if self.status(layout, i) != 0 || self.size(i) != e.logical {
                return Err(format!(
                    "GPU decode segment {} failed: status {}, decoded {} expected {}",
                    e.index,
                    self.status(layout, i),
                    self.size(i),
                    e.logical
                ));
            }
        }
        Ok(())
    }
}

fn record_batch(operation: &'static str, count: usize) {
    let attrs = [opentelemetry::KeyValue::new("operation", operation)];
    crate::metrics::core_metrics()
        .storage_codec_batches
        .add(1, &attrs);
    crate::metrics::core_metrics()
        .storage_codec_batch_segments
        .record(count as u64, &attrs);
}
/// Lloyd-Max Gaussian centroids, computed once per supported geometry.
/// Integration and stopping criterion match LMCache's TurboQuant MSE codebook.
fn centroids(dim: u32, bits: u8) -> Vec<f32> {
    let sigma = 1.0 / (dim as f64).sqrt();
    let limit = 3.5 * sigma;
    let count = 1usize << bits;
    let mut values: Vec<f64> = (0..count)
        .map(|i| -limit + (i as f64 + 0.5) * 2.0 * limit / count as f64)
        .collect();
    for _ in 0..200 {
        let mut edges = vec![-3.0 * limit];
        edges.extend(values.windows(2).map(|w| (w[0] + w[1]) * 0.5));
        edges.push(3.0 * limit);
        let next: Vec<f64> = edges
            .windows(2)
            .map(|w| {
                let mut mass = 0.0;
                let mut moment = 0.0;
                for i in 0..=200 {
                    let x = w[0] + (w[1] - w[0]) * i as f64 / 200.0;
                    let weight = (-0.5 * (x / sigma).powi(2)).exp()
                        * if i == 0 || i == 200 { 0.5 } else { 1.0 };
                    mass += weight;
                    moment += x * weight;
                }
                moment / mass
            })
            .collect();
        let delta = values
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        values = next;
        if delta < 1e-10 {
            break;
        }
    }
    values.into_iter().map(|x| x as f32).collect()
}

fn ans_type(format: StorageFormat) -> Option<i32> {
    match format {
        StorageFormat::Ans => Some(1),
        StorageFormat::Ans16 => Some(9),
        StorageFormat::AnsFp8 => Some(10),
        _ => None,
    }
}
