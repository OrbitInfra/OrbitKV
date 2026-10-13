# Pinned TENT copy-status backport

`tcp-copy-status-and-cuda-context.patch` is applied only to a private build-tree
copy of pinned Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a`. Cargo never
modifies that submodule. Input file Git blob hashes must match the exact pinned source before the
zero-context patch is applied with zero fuzz. Application fails closed.
The development `ORBITKV_MOONCAKE_LIB_DIR` override skips compilation; a qualified
runtime must separately bind those bytes to a build of this patch.

The scoped patch consumes already merged upstream fixes:

- [#3476](https://github.com/kvcache-ai/Mooncake/pull/3476), merge
  `964b8cf0c8208a3fa974d1afe7b10e53e8f05070`: propagate every TCP copy status,
  prevent READ error-text/payload-length collision, choose the owning CUDA device.
- [#3973](https://github.com/kvcache-ai/Mooncake/pull/3973), merge
  `bcdde250c509fb0afa496ebeeba8a6198b772325`: driver pointer classification,
  host-to-host memcpy, and CUDA context selection/restoration.

The second backport extracts the existing copy/probe/free behavior and its driver
header; it excludes later RDMA device-synchronization and plugin changes, which
need separate lifetime gates. There is no new transport API or fault policy.
A broader pin update must remove this temporary patch after separately checking
its intervening transport changes. Upstream merge is not local qualification.

A failed copy now reaches the existing native TCP FAILED state, then the C ABI.
OrbitKV's retained native failure boundary remains unchanged: FAILED does not
release caller buffers, admission, or source owners. This backport does not
supply a physical RDMA/CUDA failure fence or authorize timeout reclamation.

Run the deterministic test helper after native compilation:

```bash
python3 crates/orbitkv-transfer/tests/native/run_copy_status.py \
  --native-build /path/to/target/debug/build/orbitkv-mooncake-sys-ID/out/native \
  --runtime /path/to/frozen/cpu/lib \
  --output /external/artifacts/new-copy-control
```

The helper replaces the Platform loader only in its own executable, keeps valid
registered host buffers, and returns a controlled copy error before submission.
It exercises the four real control-plane copy sites, native TCP status and C ABI,
full destination sentinels, normal subsequent transfers and READ error-length
collision. Run the identical helper against the original pinned library as a
negative control. A code/CPU pass does not qualify installed GPU, RDMA, Manager
SSD, adapter crash or physical termination gates.

Review follow-up preserves pointer-query errors using `cuPointerGetAttributes`:
ordinary unregistered host pointers are identified by its documented successful
zero attributes; unexpected errors never select host memcpy. Driver init failure
rejects copying. Device/context acquisition and copy errors preserve caller
device restoration, and failed restoration remains an error. These follow-ups
have controlled CUDA API tests in `tests/native/cuda_copy_errors.cu`; they are
not yet healthy-GPU qualification or an upstream merged change. A focused
upstream follow-up must be prepared separately from this scoped backport.

CUDA errors after a memcpy attempt are not physical drain evidence. The copy
owner retains its current stream, stack and callers' RPC staging buffers inline
instead of returning them. Its device-copy admission limits are 64 live plus
retained calls and 1 GiB of logical copy bytes per Platform; these bounds are
separate from Rust's 1024-batch/32 MiB request-descriptor budget and physical
source/query/pool budgets. A fixed diagnostic marks retention. Pre-submit
pointer/context/stream admission errors return normally; actual-copy errors
retain. This trades liveness for ownership safety until a native termination
primitive is independently established. CPU-only TENT is unaffected.
