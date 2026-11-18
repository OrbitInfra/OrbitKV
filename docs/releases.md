# Python releases

> OrbiTE is now the only transfer engine. GPU/RDMA/GDS and installed profiles
> require fresh hardware qualification. Earlier measurements retain their original
> revisions and backend (TENT where recorded); CPU/TCP and io_uring checks are CI
> evidence, not installed GPU qualification.

OrbitKV `0.1.0b1` was published on November 12, 2026, with Rust workspace
and tag spelling `0.1.0-beta.1`. Its [published release notes](https://github.com/OrbitInfra/OrbitKV/releases/tag/v0.1.0-beta.1)
and [frozen artifact contract](https://github.com/OrbitInfra/OrbitKV/blob/a2e879b9/docs/releases.md)
remain unchanged. Breaking API and format changes remain allowed before 0.1.0.
Changes below apply to the next pre-release; nothing new is published by this PR.

## Next pre-release (unreleased)

### Placement discovery

Manager queries now report partial coverage when any expected owner's last
inventory receipt reaches three seconds. Queries during membership replacement
report unavailable until the matching index cut is installed. Journal-reset
state retires with its owner view. The internal Rust crates removed the duplicate
`GlobalIndex::mark_syncing` and unused `GlobalIndex::reset`,
`BlockCandidates::estimated_size` and `DISCOVERY_MAX_ENDPOINT_BYTES`.
Inventory mutations, cursor reads and fence status now take the original
`MemberLease`. `MembershipView::revoke` returns the revoked lease and
`retire_owner` reports whether that lease needs bounded cleanup.
See the [lookup contract](distributed-cache.md#placement-lookup-contract).

### Optional SSD watermarks

Added paired `--ssd-high-watermark-percent` / `--ssd-low-watermark-percent`,
`SsdCacheConfig::{high_watermark_percent, low_watermark_percent}`,
`OrbitKVEngine::maintain_ssd_cache()` and `SsdCacheCleanupStats`. Exhaustive Rust
config literals must add the two fields; defaults keep watermarks disabled.
Bounded per-shard FIFO maintenance follows TTL/DRAM maintenance, retains live
read/write ranges and syncs persistent invalidation before reuse. It changes
no wire or file format. See [semantics and limits](cache-policies.md#ssd-highlow-watermarks).
Installed NVMe/GPU/GDS and serving profiles remain unqualified.

### Fixed data TTL and persistent expiry

Added `--cache-ttl-secs`, optional `EngineConfig::cache_ttl`,
`OrbitKVEngine::expire_cache()`, `CacheExpirationStats` and
`SealedBlock::cache_expiry()` / `CacheExpiry` metadata. Rust exhaustive config
literals must add the new field; defaults leave new publications without TTL.
Inherited deadlines still apply. Cleanup uses existing owners and bounded scans;
no timeout or expiry constitutes native drain. See [TTL semantics](cache-policies.md#fixed-data-ttl).

Breaking changes: SSD format 3 persists deadlines and cold-starts formats 1/2.
Peer request and response require `orbitkv/inventory-stream/v6`; upgrade all
Managers together and migrate the stopped cluster marker from v3/v4/v5.
The channel lifecycle and OrbiTE wire are unchanged. New metrics distinguish
logical expirations from physical reclamation. Installed hardware profiles
remain unqualified.

### Optional DRAM watermarks

Added paired `--cache-high-watermark-percent` / `--cache-low-watermark-percent`
flags and corresponding optional `EngineConfig` fields. Rust struct literals
that specify every field must add them; `Default` keeps both disabled.
The Manager runs bounded maintenance once per second; embedded callers schedule
the new `OrbitKVEngine::maintain_memory_cache()` method. The existing residence
histogram adds `reason="watermark"`. See [policy and lifetime semantics](cache-policies.md#dram-highlow-watermarks).
The watermark substep changes no wire or persistent format;
GPU/RDMA/GDS and installed profiles remain unqualified.

### vLLM layerwise Save

Ordinary single-group, layer-first caches now consume `save_kv_layer` readiness.
All request layers retain their page holds until their native Save receipts;
missing callbacks use step-end submission. Page-first/cross-layer/HMA,
microbatch/DBO and speculative profiles keep whole-step Save. CUDA capture
does no early admission. This changes no native API, protocol or file format.
See the [scope and ownership contract](adapters.md#layerwise-save); installed
GPU/Graph behavior and overlap performance remain unqualified.

### Breaking: required OrbiTE dependency

Removed the `orbite` Cargo feature and the unavailable-backend implementation.
Core, Manager and Python builds always include OrbiTE TCP/CUDA/io_uring;
`orbite-rdma` and `orbite-gds` remain optional transport capabilities. Remove
`orbite` from explicit Cargo feature lists. CUDA and storage SDKs are still
loaded only when a configured path needs them.

Engine construction now returns a required engine or an initialization error.
`transfer_backend`, `transfer_endpoint` and `transfer_source_epoch` no longer
return `Option`; `has_remote_transport` is removed. CUDA registrations, workers
and codecs require their engine at construction. Host codec payload uploads
require the caller's OrbiTE copy owner; standalone direct-copy fixtures live in
tests. Closed registrations and uncertain cleanup retain their existing lifetime
states. Local mode still rejects distributed admission without membership.
This changes no wire or file format, and adds no hardware qualification.

### Breaking: cost subsystem removal

Removed cost observations/selection, `ORBITKV_COST_OBSERVATIONS`,
`ORBITKV_COST_SELECTION`, `ORBITKV_CROSS_MEDIUM_SELECTION` and
`orbitkv_cost_*` metrics, and the unused `observe_completion` Rust/channel API.
Replica selection is [deterministic](architecture.md#planning); OrbiTE intent,
priority and deadline dispatch is unchanged. Command ABI 13 rejects the removed
opcode; client lifecycle 14 requires matching clients and Managers to upgrade
together. This is a pre-0.1.0 breaking change with no compatibility shim. Physical
readiness, grants, drain receipts and retained owners are unchanged. Historical
[observer verdicts](performance.md#qualification-limits) remain valid.

## Packages

`--ssd-persistent` uses local SSD format version 3 as an opt-in cache.
Keep data files and the checksummed sidecar together. Unknown or untrusted
metadata starts cold; there is no migration or compatibility alias before 0.1.0.
The default still truncates. Installed NVMe serving remains unqualified; see
[the persistence contract](storage-persistence.md).

The earlier integrity revision changed the unpublished SSD format from 1 to 2 (IEEE
CRC32 source/range references; metadata frames still use CRC32C). Version 1
starts cold. Peer protocol `orbitkv/inventory-stream/v5` requires checksum
references and binds corruption reports to drained grant tickets. Upgrade all
Managers together and replace the old cluster format marker only after stopping
them; there is no compatibility alias. The migration tool takes
`--from-protocol orbitkv/inventory-stream/v5` (or v3/v4), replacing `--direction`,
and now cuts over to v6 with the TTL contract above. The published beta uses client lifecycle 13; the unreleased
removal above advances it to 14.

The `orbitkv` PyPI wheel is OrbitKV's only distribution. All Rust crates,
including the Python extension crate, are internal and are not published.
The Manager compiles OrbiTE's Rust library into the wheel; installing the
separate `orbite` Python package does not replace that dependency.

On November 9, 2026, the unpublished distribution was renamed from `orbitkv-llm` to `orbitkv`; the import, CLI and plugin names are unchanged, and no alias package is provided.

| Distribution | Runtime | Release targets |
| --- | --- | --- |
| `orbitkv` | CUDA 13 | Linux x86_64 and aarch64, CPython 3.10–3.14 |

One distribution provides the `orbitkv` import and `orbitkv-cache-manager`
command. CUDA 12 is not a release target. There are ten wheels per release:
five CPython versions on each of two architectures; pip selects the matching wheel.
Engine extras pin vLLM 0.33.0 or SGLang 0.5.23.
Both require Python 3.11+; SGLang 0.5.23 publishes wheels for 3.11–3.13
only. Core OrbitKV retains Python 3.10+. Current installed
profiles are unqualified; [compatibility](compatibility.md) separates historical evidence.
Use separate engine environments. Build-matrix coverage does not establish
that every engine/GPU combination works on every Python version; serving
historical qualification used Python 3.11 on H20 and A100, with each result
tied to its original engine release and backend. See [engine baseline](completion-plan.md#milestones).

The wheel contains the PyO3 extension, Manager executable with OrbiTE linked in,
engine plugins, type stubs, Apache-2.0 license and third-party attribution in
`NOTICE`. The host supplies its NVIDIA
driver and compatible PyTorch/CUDA 13 environment. CUDA 13 normally requires
NVIDIA R580 or newer. Supported data-center GPUs can use NVIDIA's matching
forward-compatibility package on selected older drivers, including R535; freeze
the actual loaded driver/runtime libraries and qualify the consumed IPC, kernel
and transfer features. A compute probe alone does not qualify them. See
[NVIDIA's forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
The console preserves explicit `LD_LIBRARY_PATH` ordering and appends the
interpreter's `libpython` directory, so a selected compatibility package keeps
priority over system driver libraries. Its Python search path keeps the
interpreter's standard library before package roots, then the selected OrbitKV
package and caller dependencies before automatically discovered site-packages.
This prevents legacy site packages such as `enum34` from shadowing the standard
library during console startup or embedded Python initialization. Check the
Manager's actual loaded paths.
Tests, benchmarks and build caches are excluded from the package.

GPU ANS encoding additionally needs the separately installed nvCOMP 5.3 runtime;
it is not bundled in the wheel. FP8 and TurboQuant kernels are compiled by NVRTC
in the Manager. All storage codecs are opt-in; see [storage formats](storage-formats.md)
for dependencies, model-quality results and GPU workspace limits.

## Build a candidate

The candidate uses the exact
[`orbite` 0.1.0-beta.1 crate](https://crates.io/crates/orbite/0.1.0-beta.1),
including runtime-loaded verbs and the current caller-stream local-copy API.
OrbitKV now passes `LocalCopyOptions::default()` to its existing forced DMA or
kernel backend; cache policy, producer ordering and retirement are unchanged.
The registry requirement is `=0.1.0-beta.1`, with its checksum in `Cargo.lock`.
The tag gate rejects Git/path dependencies or a missing registry checksum.
Every release wheel is rebuilt from this lock and records the resolved dependency
in `PROVENANCE.json`; earlier Git-sourced candidates cannot be uploaded as the
registry-sourced release artifact.

The experimental vLLM V2 profile requires both `VLLM_USE_V2_MODEL_RUNNER=1` and
the boolean connector option `orbitkv.v2=true`. Ordinary V1 deployment is
unchanged. Only uniform Full attention in one group is admitted; installed GPU
qualification is pending. This adds an explicit opt-in, with no compatibility
alias or production API break. Before 0.1.0, breaking API changes are allowed;
record each break here instead of retaining compatibility shims.

V1 Full + SWA now admits multiple cache groups under the
[released HMA contract](adapters.md#released-hma-contract). Multi-group namespace
identity now includes resolved group order, layer membership and every spec field;
entries created by earlier internal hybrid experiments cannot match. There is no
alias or migration shim. Single-group identities and the V2 opt-in are unchanged;
installed Full/SWA model and fault gates remain unqualified.

SGLang Mamba checkpoints now use the released 0.5.23 component. Their identity
records conv/SSM roles and layout; ordinary MLA identity additionally records
latent/RoPE partitions. Earlier MLA entries intentionally miss after upgrading;
there is no identity alias. Full/SWA identity is unchanged. The native
`--enable-linker-mla-dedup` broadcast option now fails explicitly with OrbitKV;
per-rank MLA storage remains available. Installed GPU gates are unqualified.

From the repository root, select a Python interpreter with shared `libpython`
(required by the embedded Manager) and use a CUDA 13 toolkit:

```bash
PYO3_PYTHON=/absolute/path/to/python3.11 \
  ./scripts/build-wheel.sh --release --locked
```

CUDA 13 and OrbiTE are the default Cargo features. The script builds and stages
the Manager, builds the extension for the selected Python,
audits every bundled ELF file with auditwheel, repairs its non-host dependencies,
checks the package version/content, and imports the installed wheel in a fresh
non-editable environment. The initial maturin output uses a Linux tag; only the
whole-wheel audit selects the final manylinux tag. Auditing the extension alone
misses dependencies of the embedded-Python Manager.

A wheel can additionally enable OrbiTE RDMA:

```bash
CC=gcc CXX=g++ PYO3_PYTHON=/absolute/path/to/python3.11 \
  ./scripts/build-wheel.sh --release --features orbite-rdma --locked
python3 scripts/check-wheel.py target/wheels/*.whl --install-smoke --require-orbite-rdma
```

OrbiTE is required in every build; `orbite-rdma` adds the RDMA transport.
Use `--transfer-transport auto|tcp|rdma` to select an operator hint.
These build options do not create a second Python distribution. The installed
CLI smoke checks compiled RDMA arguments before any GPU access. Earlier
independent H20 byte/lifetime reruns covered the historical CPython 3.11 single-runtime
profile in [transport](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/transport.md#installed-client-and-manager-profile), with
explicit Torch CUDA 13.0.96 selection. That result does not qualify the current
OrbiTE path. Its private unversioned runtime alias is
needed because the Rust loader tries `libcudart.so` before Torch's `.so.13`.
Building with a CUDA 13.1 compiler does not establish a loaded CUDA 13.1 runtime
or qualify other installed environments. CI builds both the default and optional RDMA
wheel, with separate artifacts and matching Clang/libclang 18 on Ubuntu 24.04.
Use a wheel matching the target CPython ABI and audited glibc floor; the CI
Python 3.14 artifact cannot qualify a Python 3.11 execution node.

CI builds on Ubuntu with protoc 27.5 (archive SHA-256 checked before extraction), `auditwheel==6.8.2`,
`patchelf>=0.14.5`, maturin and wheel (`tomli` also on Python 3.10). Both release architectures use the same
protocol compiler; Ubuntu 22.04's system protoc cannot compile the current
proto3 optional fields without an experimental flag. Dependency copyright files are retained from the Debian/Ubuntu package
database, together with referenced common license texts.
An externally staged dependency must supply `<library-filename>.license` beside
its original shared library if it has no OS package provenance; missing notices
fail the build. The repaired artifact also retains auditwheel's dependency SBOM.

Python's shared library, CUDA driver/runtime/cuFile and RDMA core/provider
libraries remain host dependencies; the repair step must neither bundle driver
libraries nor remove the standalone Manager's `libpython` dependency. Wheels
are written to `target/wheels/`.
`maturin build` alone does not stage the Manager executable; use the wheel script.

## Validate before publishing

1. Run `python3 scripts/check-versions.py --engines --tag v0.1.0-beta.1`.
   Rust, Commitizen and the proposed tag must use `0.1.0-beta.1`; Python metadata
   must use its canonical `0.1.0b1` equivalent. All Rust crates stay private.
   Also require
   `python3 scripts/check-versions.py --publication --tag v0.1.0-beta.1` too.
2. Trigger the **Release** workflow manually on the candidate branch. It builds
   all ten wheel targets and validates their versions and contents, isolated
   imports and Manager help. The `release-artifacts` bundle contains the wheels,
   source/dependency provenance, notes and `SHA256SUMS`. Manual runs upload
   Actions artifacts only; they do not create a release or publish to PyPI.
   For example, run `gh workflow run release.yml --repo OrbitInfra/OrbitKV --ref main`,
   then download `release-artifacts` with `gh run download RUN_ID --repo OrbitInfra/OrbitKV -n release-artifacts -D /tmp/orbitkv-candidate`
   and run `sha256sum -c SHA256SUMS` in that directory.
3. Install a candidate wheel into each engine environment. From outside the
   source checkout, check the installed import path, Manager help and health,
   then verify a completion and an external restore after engine restart.
   The smoke gate excludes source overrides, imports the installed Rust extension
   and rejects removed TENT libraries. It verifies installed RECORD hashes,
   then compares cold, native HBM, full and partial recovery against native
   controls in DRAM and io_uring SSD cells. The installed console entry executes
   the Manager directly: SIGTERM must drain and exit 0; forced cleanup fails.
   Run the [release smoke and correctness gates](https://github.com/OrbitInfra/OrbitKV/blob/main/python/tests/README.md#release-smoke).
   Claiming a shared Manager additionally requires the
   [two-engine gate](https://github.com/OrbitInfra/OrbitKV/blob/main/python/tests/README.md#two-engines-sharing-one-manager),
   matching both installations to the frozen wheel and retaining concurrent
   output, each engine restart, actual SSD reads and normal drain evidence.
4. Review the final benchmark summaries and known limits. These installed GPU
   serving gates remain unrun on this candidate. A maintainer-approved beta
   must retain that limit; it cannot claim installed qualification. The
   [0.1.0 gate](completion-plan.md#m4--release-and-upstream) still requires the
   hardware batch. Candidate CI success alone never authorizes publication.

## Configure PyPI publication

Enable two-factor authentication on the PyPI account. Before the first upload,
add a GitHub pending publisher under [PyPI account publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `orbitkv` |
| GitHub owner | `OrbitInfra` |
| Repository name | `OrbitKV` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

The release job uses the GitHub `pypi` environment and OIDC Trusted Publishing;
no PyPI API token is needed. A pending publisher creates the project on first
successful upload, but does not reserve its name. See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
Protect that environment with a required maintainer reviewer and allow only
`v*` tags as deployment refs. Verify the pending
publisher fields in PyPI; a GitHub environment alone does not configure PyPI.

After OrbiTE publication, the registry-pinned rebuild, candidate acceptance and
separate publication approval, the maintainer tags the approved commit:

```bash
git tag -s v0.1.0-beta.1 APPROVED_COMMIT -m 'OrbitKV 0.1.0-beta.1'
git push origin refs/tags/v0.1.0-beta.1
```

The tag rebuilds the same ten wheels and waits for environment approval. It
publishes PyPI first, checks all filenames and SHA-256 values against the
registry receipt, then creates the GitHub prerelease with those exact assets.
If an upload or receipt check fails, inspect the registry before any recovery;
do not blindly retry uploads or rebuild a version that already has published
files. Manual runs never exercise PyPI authorization. Once publication succeeds,
install with `pip install orbitkv==0.1.0b1` in the compatible runtime environment.

## Release scope

`orbitkv` 0.1.0b1 is published; follow [M4](completion-plan.md#m4--release-and-upstream)
for official engine targets, upstream contributions and installed qualification.
Current OrbiTE GPU/RDMA/GDS and installed paths are unqualified; historical TENT
candidate checks do not carry over. Publication is the maintainer's action.
