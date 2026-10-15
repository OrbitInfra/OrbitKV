//! CPU-path benchmark for long block lists.
//!
//! This exercises the core engine API with the same save -> query -> load shape
//! used by the mock vLLM RPC tests, while keeping tonic/prost out of CPU
//! profiles. The benchmark binds CUDA device 0 at runtime. Use
//! `perf record --call-graph dwarf -- cargo bench -p orbitkv-core --bench
//! cpu_path -- --profile-time 30` for profile captures.

use std::ffi::c_void;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::{CudaContext, sys};
use orbitkv_core::transfer::local::{LocalRestoreExecutor, LocalTensor, RawRestorePart};
use orbitkv_core::{
    EngineConfig, LayerSave, OrbitKVEngine, QueryLeaseId, QueryResult, RestoreExecution,
    TransferMode,
};
use tokio::runtime::Runtime;

const INSTANCE_ID: &str = "cpu-path-bench";
const NAMESPACE: &str = "cpu-path";
const LAYER_NAME: &str = "layer_0";
const DEVICE_ID: i32 = 0;
const LOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(30);
// No active membership: measure journal insertion overhead without peer I/O.
const ADVERTISE_ADDR: &str = "127.0.0.1:50055";

const BLOCK_CASES: &[usize] = &[128, 1024, 8192, 32768];
const DTOH_MULTILAYER_BLOCK_CASES: &[usize] = &[128, 1024, 8192];
const CPU_STAGE_BLOCK_CASES: &[usize] = &[1024, 8192, 32768];
const TRANSFER_FRAGMENT_BLOCK_CASES: &[usize] = &[1024, 8192];
const BYTES_PER_BLOCK: usize = 1024;
const CPU_PATH_BYTES_PER_BLOCK: usize = 1;
const MULTI_LAYER_COUNT: usize = 61;

struct BenchFixture {
    local_restore: std::sync::Mutex<LocalRestoreExecutor>,
    engine: OrbitKVEngine,
    registration_generation: [u8; 16],
    _ctx: Arc<CudaContext>,
    _gpu: GpuBuffer,
    num_blocks: usize,
    block_ids: Vec<usize>,
}

impl BenchFixture {
    fn new(num_blocks: usize, bytes_per_block: usize) -> Self {
        Self::with_config(
            num_blocks,
            bytes_per_block,
            EngineConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: Some(bytes_per_block),
                enable_numa_affinity: false,
                ..EngineConfig::default()
            },
            TransferMode::Direct,
            block_ids(num_blocks),
        )
    }

    fn with_inventory(num_blocks: usize, bytes_per_block: usize, enable_inventory: bool) -> Self {
        let membership = enable_inventory.then(|| {
            Arc::new(orbitkv_catalog::MembershipView::new(
                orbitkv_state::CacheOwner {
                    endpoint: ADVERTISE_ADDR.into(),
                    incarnation: uuid::Uuid::new_v4(),
                },
            ))
        });
        Self::with_config(
            num_blocks,
            bytes_per_block,
            EngineConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: Some(bytes_per_block),
                enable_numa_affinity: false,
                global_index: membership.as_ref().map(|view| {
                    Arc::new(orbitkv_catalog::GlobalIndex::new(
                        view.clone(),
                        1 << 20,
                        Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
                    ))
                }),
                inventory: enable_inventory.then(|| {
                    Arc::new(orbitkv_core::ResidencyInventory::new(
                        orbitkv_core::DEFAULT_INVENTORY_JOURNAL_BYTES,
                    ))
                }),
                membership,
                ..EngineConfig::default()
            },
            TransferMode::Direct,
            block_ids(num_blocks),
        )
    }

    fn with_transfer_layout(
        num_blocks: usize,
        bytes_per_block: usize,
        transfer_mode: TransferMode,
        layout: TransferFragmentLayout,
    ) -> Self {
        let block_ids = layout.block_ids(num_blocks);
        let registered_blocks = block_ids
            .iter()
            .copied()
            .max()
            .map(|id| id + 1)
            .unwrap_or(num_blocks);
        Self::with_config(
            registered_blocks,
            bytes_per_block,
            EngineConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: Some(bytes_per_block),
                enable_numa_affinity: false,
                ..EngineConfig::default()
            },
            transfer_mode,
            block_ids,
        )
    }

    fn with_config(
        registered_blocks: usize,
        bytes_per_block: usize,
        config: EngineConfig,
        transfer_mode: TransferMode,
        block_ids: Vec<usize>,
    ) -> Self {
        let ctx = CudaContext::new(DEVICE_ID as usize).expect("CUDA context");
        ctx.bind_to_thread().expect("bind CUDA context");

        let total_bytes = registered_blocks
            .checked_mul(bytes_per_block)
            .expect("registered GPU size overflow");
        let gpu = GpuBuffer::new(Arc::clone(&ctx), total_bytes);
        let mut host = vec![0u8; total_bytes];
        fill_test_pattern(&mut host, bytes_per_block);
        gpu.copy_from_host(&host);

        let pool_size = total_bytes
            .checked_mul(4)
            .and_then(|size| size.checked_add(64 << 20))
            .expect("pool size overflow");
        let engine = OrbitKVEngine::new_with_config(pool_size, false, config).expect("engine");

        let registration_generation = engine
            .register_context_layer_batch(
                INSTANCE_ID,
                NAMESPACE,
                DEVICE_ID,
                0,
                0,
                1,
                1,
                &[LAYER_NAME.to_string()],
                &[gpu.as_u64()],
                &[total_bytes],
                &[registered_blocks],
                &[bytes_per_block],
                &[0],
                &[1],
                transfer_mode,
                false,
            )
            .expect("register layer");

        let local_restore = LocalRestoreExecutor::new(
            DEVICE_ID as usize,
            vec![
                LocalTensor::new(
                    LAYER_NAME.into(),
                    gpu.as_u64(),
                    total_bytes,
                    0,
                    registered_blocks,
                    bytes_per_block,
                    0,
                    1,
                )
                .expect("local GPU binding"),
            ],
            engine.payload_arenas().expect("export payload arenas"),
            transfer_mode,
        )
        .expect("import payload arenas");
        Self {
            local_restore: std::sync::Mutex::new(local_restore),
            engine,
            registration_generation,
            _ctx: ctx,
            _gpu: gpu,
            num_blocks: block_ids.len(),
            block_ids,
        }
    }

    async fn save_and_flush(&self, hashes: Vec<Vec<u8>>) {
        self.engine
            .batch_save_kv_blocks_from_ipc(
                INSTANCE_ID,
                self.registration_generation,
                0,
                0,
                DEVICE_ID,
                vec![LayerSave {
                    layer_name: LAYER_NAME.to_string(),
                    block_ids: self.block_ids.clone(),
                    block_hashes: hashes,
                }],
            )
            .await
            .expect("save");
        self.engine.flush_saves().await;
    }

    async fn prime_cache(&self) -> Vec<Vec<u8>> {
        let hashes = make_block_hashes(self.num_blocks, 0);
        self.save_and_flush(hashes.clone()).await;
        hashes
    }

    async fn query_lease(&self, req_id: &str, hashes: &[Vec<u8>]) -> QueryLeaseId {
        {
            let QueryResult { blocks, missing } = self
                .engine
                .count_prefix_hit_blocks_with_prefetch(
                    INSTANCE_ID,
                    req_id,
                    hashes,
                    orbitkv_core::QueryMode::Demand,
                )
                .await
                .expect("query");
            assert_eq!(blocks.len(), hashes.len());
            assert_eq!(missing, 0);
            self.engine
                .create_query_lease(INSTANCE_ID, blocks)
                .expect("create query lease")
        }
    }

    async fn load_and_wait(&self, lease: QueryLeaseId) {
        let receiver = self
            .engine
            .restore(
                INSTANCE_ID,
                0,
                DEVICE_ID,
                &[vec![LAYER_NAME]],
                &[(
                    lease,
                    vec![self.block_ids.iter().copied().map(Some).collect()],
                )],
            )
            .expect("submit load");
        match receiver {
            RestoreExecution::Local(mut grant) => {
                let result = loop {
                    let (bytes, more) = grant.encoded_plan();
                    let plan = RawRestorePart::decode(bytes).expect("decode Restore plan");
                    let result = self.local_restore.lock().unwrap().execute(
                        &plan,
                        &mut Default::default(),
                        true,
                        || {},
                        None,
                    );
                    if result.is_err() || !more {
                        break result;
                    }
                    assert!(grant.advance_plan());
                };
                grant.finish(result.is_ok(), None);
                result.expect("local Restore failed");
            }
            RestoreExecution::Managed(receiver) => {
                tokio::time::timeout(LOAD_WAIT_TIMEOUT, receiver)
                    .await
                    .expect("restore timeout")
                    .expect("restore worker disappeared")
                    .result
                    .expect("restore failed");
            }
        }
    }

    fn cleanup_cache(&self) {
        black_box(self.engine.cleanup_memory_cache());
    }
}

struct MultiLayerBenchFixture {
    engine: OrbitKVEngine,
    registration_generation: [u8; 16],
    _ctx: Arc<CudaContext>,
    _gpu: GpuBuffer,
    layer_names: Vec<String>,
    num_blocks: usize,
}

#[derive(Clone, Copy)]
enum MultiLayerGpuLayout {
    SharedSource,
    DistinctLayerSources,
}

#[derive(Clone, Copy)]
enum TransferFragmentLayout {
    Contiguous,
    StridedDevice,
}

impl TransferFragmentLayout {
    fn name(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::StridedDevice => "strided_device",
        }
    }

    fn block_ids(self, num_blocks: usize) -> Vec<usize> {
        match self {
            Self::Contiguous => block_ids(num_blocks),
            Self::StridedDevice => (0..num_blocks).map(|idx| idx * 2).collect(),
        }
    }
}

fn transfer_mode_name(mode: TransferMode) -> &'static str {
    match mode {
        TransferMode::Direct => "direct",
        TransferMode::Kernel => "kernel",
    }
}

impl MultiLayerBenchFixture {
    fn new(num_blocks: usize, bytes_per_block: usize, layout: MultiLayerGpuLayout) -> Self {
        let ctx = CudaContext::new(DEVICE_ID as usize).expect("CUDA context");
        ctx.bind_to_thread().expect("bind CUDA context");

        let layer_bytes = num_blocks
            .checked_mul(bytes_per_block)
            .expect("registered GPU size overflow");
        let gpu_bytes = match layout {
            MultiLayerGpuLayout::SharedSource => layer_bytes,
            MultiLayerGpuLayout::DistinctLayerSources => layer_bytes
                .checked_mul(MULTI_LAYER_COUNT)
                .expect("multi-layer GPU size overflow"),
        };
        let gpu = GpuBuffer::new(Arc::clone(&ctx), gpu_bytes);
        let mut host = vec![0u8; gpu_bytes];
        fill_test_pattern(&mut host, bytes_per_block);
        gpu.copy_from_host(&host);

        let save_bytes = layer_bytes
            .checked_mul(MULTI_LAYER_COUNT)
            .expect("multi-layer save size overflow");
        let pool_size = save_bytes
            .checked_mul(4)
            .and_then(|size| size.checked_add(64 << 20))
            .expect("pool size overflow");
        let engine = OrbitKVEngine::new_with_config(
            pool_size,
            false,
            EngineConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: Some(bytes_per_block),
                enable_numa_affinity: false,
                ..EngineConfig::default()
            },
        )
        .expect("engine");

        let layer_names: Vec<String> = (0..MULTI_LAYER_COUNT)
            .map(|idx| format!("layer_{idx}"))
            .collect();
        let layer_ptrs: Vec<u64> = match layout {
            MultiLayerGpuLayout::SharedSource => vec![gpu.as_u64(); MULTI_LAYER_COUNT],
            MultiLayerGpuLayout::DistinctLayerSources => (0..MULTI_LAYER_COUNT)
                .map(|idx| gpu.offset_u64(idx * layer_bytes))
                .collect(),
        };
        let layer_sizes = vec![layer_bytes; MULTI_LAYER_COUNT];
        let layer_blocks = vec![num_blocks; MULTI_LAYER_COUNT];
        let block_sizes = vec![bytes_per_block; MULTI_LAYER_COUNT];
        let strides = vec![0; MULTI_LAYER_COUNT];
        let segments = vec![1; MULTI_LAYER_COUNT];

        let registration_generation = engine
            .register_context_layer_batch(
                INSTANCE_ID,
                NAMESPACE,
                DEVICE_ID,
                0,
                0,
                1,
                1,
                &layer_names,
                &layer_ptrs,
                &layer_sizes,
                &layer_blocks,
                &block_sizes,
                &strides,
                &segments,
                TransferMode::Direct,
                false,
            )
            .expect("register layers");

        Self {
            engine,
            registration_generation,
            _ctx: ctx,
            _gpu: gpu,
            layer_names,
            num_blocks,
        }
    }

    fn make_saves(&self, hashes: &[Vec<u8>]) -> Vec<LayerSave> {
        let ids = block_ids(self.num_blocks);
        self.layer_names
            .iter()
            .map(|layer_name| LayerSave {
                layer_name: layer_name.clone(),
                block_ids: ids.clone(),
                block_hashes: hashes.to_vec(),
            })
            .collect()
    }

    async fn save_and_flush(&self, saves: Vec<LayerSave>) {
        self.engine
            .batch_save_kv_blocks_from_ipc(
                INSTANCE_ID,
                self.registration_generation,
                0,
                0,
                DEVICE_ID,
                saves,
            )
            .await
            .expect("save");
        self.engine.flush_saves().await;
    }

    async fn save_only(&self, saves: Vec<LayerSave>) {
        self.engine
            .batch_save_kv_blocks_from_ipc(
                INSTANCE_ID,
                self.registration_generation,
                0,
                0,
                DEVICE_ID,
                saves,
            )
            .await
            .expect("save");
    }

    async fn flush_saves(&self) {
        self.engine.flush_saves().await;
    }

    fn cleanup_cache(&self) {
        black_box(self.engine.cleanup_memory_cache());
    }
}

fn save_flush_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_flush_unique");
    group.sample_size(10);

    for &num_blocks in BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks,
            BYTES_PER_BLOCK,
        )));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = BenchFixture::new(num_blocks, BYTES_PER_BLOCK);
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let start = Instant::now();
                        fixture.save_and_flush(hashes).await;
                        measured += start.elapsed();
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_flush_inventory_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_flush_unique_inventory");
    group.sample_size(10);

    for &num_blocks in BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks,
            BYTES_PER_BLOCK,
        )));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let _guard = rt.enter();
            let fixture = BenchFixture::with_inventory(num_blocks, BYTES_PER_BLOCK, true);
            drop(_guard);
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let start = Instant::now();
                        fixture.save_and_flush(hashes).await;
                        measured += start.elapsed();
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_flush_multilayer_cpu_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_flush_cpu_multilayer_61");
    group.sample_size(10);

    for &num_blocks in BLOCK_CASES {
        group.throughput(Throughput::Elements(
            (num_blocks * MULTI_LAYER_COUNT) as u64,
        ));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = MultiLayerBenchFixture::new(
                num_blocks,
                CPU_PATH_BYTES_PER_BLOCK,
                MultiLayerGpuLayout::SharedSource,
            );
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let saves = fixture.make_saves(&hashes);
                        let start = Instant::now();
                        fixture.save_and_flush(saves).await;
                        measured += start.elapsed();
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_submit_multilayer_cpu_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_submit_cpu_multilayer_61");
    group.sample_size(10);

    for &num_blocks in CPU_STAGE_BLOCK_CASES {
        group.throughput(Throughput::Elements(
            (num_blocks * MULTI_LAYER_COUNT) as u64,
        ));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = MultiLayerBenchFixture::new(
                num_blocks,
                CPU_PATH_BYTES_PER_BLOCK,
                MultiLayerGpuLayout::SharedSource,
            );
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let saves = fixture.make_saves(&hashes);
                        let start = Instant::now();
                        fixture.save_only(saves).await;
                        measured += start.elapsed();
                        fixture.flush_saves().await;
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_insert_flush_multilayer_cpu_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_insert_flush_cpu_multilayer_61");
    group.sample_size(10);

    for &num_blocks in CPU_STAGE_BLOCK_CASES {
        group.throughput(Throughput::Elements(
            (num_blocks * MULTI_LAYER_COUNT) as u64,
        ));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = MultiLayerBenchFixture::new(
                num_blocks,
                CPU_PATH_BYTES_PER_BLOCK,
                MultiLayerGpuLayout::SharedSource,
            );
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let saves = fixture.make_saves(&hashes);
                        fixture.save_only(saves).await;
                        let start = Instant::now();
                        fixture.flush_saves().await;
                        measured += start.elapsed();
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_flush_multilayer_dtoh_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_flush_dtoh_multilayer_61");
    group.sample_size(10);

    for &num_blocks in DTOH_MULTILAYER_BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks * MULTI_LAYER_COUNT,
            BYTES_PER_BLOCK,
        )));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = MultiLayerBenchFixture::new(
                num_blocks,
                BYTES_PER_BLOCK,
                MultiLayerGpuLayout::DistinctLayerSources,
            );
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(num_blocks, iter + 1);
                        let saves = fixture.make_saves(&hashes);
                        let start = Instant::now();
                        fixture.save_and_flush(saves).await;
                        measured += start.elapsed();
                        fixture.cleanup_cache();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn save_flush_transfer_fragmentation_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/save_flush_transfer_fragmentation");
    group.sample_size(10);

    for &num_blocks in TRANSFER_FRAGMENT_BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks,
            BYTES_PER_BLOCK,
        )));

        for transfer_mode in [TransferMode::Direct, TransferMode::Kernel] {
            for layout in [
                TransferFragmentLayout::Contiguous,
                TransferFragmentLayout::StridedDevice,
            ] {
                let id = BenchmarkId::new(
                    format!("{}_{}", transfer_mode_name(transfer_mode), layout.name()),
                    num_blocks,
                );

                group.bench_function(id, |b| {
                    let fixture = BenchFixture::with_transfer_layout(
                        num_blocks,
                        BYTES_PER_BLOCK,
                        transfer_mode,
                        layout,
                    );
                    b.iter_custom(|iters| {
                        rt.block_on(async {
                            let mut measured = Duration::ZERO;
                            for iter in 0..iters {
                                let hashes = make_block_hashes(num_blocks, iter + 1);
                                let start = Instant::now();
                                fixture.save_and_flush(hashes).await;
                                measured += start.elapsed();
                                fixture.cleanup_cache();
                            }
                            measured
                        })
                    });
                    drop(fixture);
                    std::thread::sleep(Duration::from_millis(50));
                });
            }
        }
    }

    group.finish();
}

fn load_transfer_fragmentation_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/load_transfer_fragmentation");
    group.sample_size(10);

    for &num_blocks in TRANSFER_FRAGMENT_BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks,
            BYTES_PER_BLOCK,
        )));

        for transfer_mode in [TransferMode::Direct, TransferMode::Kernel] {
            for layout in [
                TransferFragmentLayout::Contiguous,
                TransferFragmentLayout::StridedDevice,
            ] {
                let id = BenchmarkId::new(
                    format!("{}_{}", transfer_mode_name(transfer_mode), layout.name()),
                    num_blocks,
                );

                group.bench_function(id, |b| {
                    let fixture = BenchFixture::with_transfer_layout(
                        num_blocks,
                        BYTES_PER_BLOCK,
                        transfer_mode,
                        layout,
                    );
                    let hashes = rt.block_on(fixture.prime_cache());
                    b.iter_custom(|iters| {
                        rt.block_on(async {
                            let mut measured = Duration::ZERO;
                            for iter in 0..iters {
                                let req_id = format!("load-fragmentation-{iter}");
                                let lease = fixture.query_lease(&req_id, &hashes).await;
                                let start = Instant::now();
                                fixture.load_and_wait(lease).await;
                                measured += start.elapsed();
                            }
                            measured
                        })
                    });
                    drop(fixture);
                    std::thread::sleep(Duration::from_millis(50));
                });
            }
        }
    }

    group.finish();
}

fn query_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/query_prefetch_lease");
    group.sample_size(10);

    for &num_blocks in BLOCK_CASES {
        group.throughput(Throughput::Elements(num_blocks as u64));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = BenchFixture::new(num_blocks, BYTES_PER_BLOCK);
            let hashes = rt.block_on(fixture.prime_cache());
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let req_id = format!("query-{iter}");
                        let start = Instant::now();
                        let lease = fixture.query_lease(&req_id, &hashes).await;
                        measured += start.elapsed();
                        assert!(fixture.engine.release_query_lease(&lease));
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn load_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("cpu_path/load_submit_wait");
    group.sample_size(10);

    for &num_blocks in BLOCK_CASES {
        group.throughput(Throughput::Bytes(bytes_per_iter(
            num_blocks,
            BYTES_PER_BLOCK,
        )));
        group.bench_function(BenchmarkId::from_parameter(num_blocks), |b| {
            let fixture = BenchFixture::new(num_blocks, BYTES_PER_BLOCK);
            let hashes = rt.block_on(fixture.prime_cache());
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let req_id = format!("load-{iter}");
                        let lease = fixture.query_lease(&req_id, &hashes).await;
                        let start = Instant::now();
                        fixture.load_and_wait(lease).await;
                        measured += start.elapsed();
                    }
                    measured
                })
            });
            drop(fixture);
            std::thread::sleep(Duration::from_millis(50));
        });
    }

    group.finish();
}

fn block_ids(num_blocks: usize) -> Vec<usize> {
    (0..num_blocks).collect()
}

fn bytes_per_iter(num_blocks: usize, bytes_per_block: usize) -> u64 {
    num_blocks
        .checked_mul(bytes_per_block)
        .expect("bytes per iter overflow") as u64
}

fn make_block_hashes(num_blocks: usize, salt: u64) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(16);
            hash.extend_from_slice(&salt.to_le_bytes());
            hash.extend_from_slice(&(idx as u64).to_le_bytes());
            hash
        })
        .collect()
}

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    assert_eq!(host_data.len() % block_size, 0);
    for (idx, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        block.fill(((idx % 251) + 1) as u8);
    }
}

struct GpuBuffer {
    ctx: Arc<CudaContext>,
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn new(ctx: Arc<CudaContext>, len: usize) -> Self {
        assert!(len > 0);
        ctx.bind_to_thread().expect("bind CUDA context");
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ctx, ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn offset_u64(&self, offset: usize) -> u64 {
        assert!(offset <= self.len);
        self.ptr
            .checked_add(offset as u64)
            .expect("GPU pointer offset overflow")
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        self.ctx.bind_to_thread().expect("bind CUDA context");
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.ctx.bind_to_thread();
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

criterion_group!(
    benches,
    save_flush_benchmarks,
    save_flush_inventory_benchmarks,
    save_flush_multilayer_cpu_benchmarks,
    save_submit_multilayer_cpu_benchmarks,
    save_insert_flush_multilayer_cpu_benchmarks,
    save_flush_multilayer_dtoh_benchmarks,
    save_flush_transfer_fragmentation_benchmarks,
    load_transfer_fragmentation_benchmarks,
    query_benchmarks,
    load_benchmarks
);
criterion_main!(benches);
