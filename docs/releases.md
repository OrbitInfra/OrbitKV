# Python releases

> OrbiTE is now the only transfer engine. GPU/RDMA/GDS and installed profiles
> require fresh hardware qualification. Earlier measurements retain their original
> revisions and backend (TENT where recorded); CPU/TCP and io_uring checks are CI
> evidence, not installed GPU qualification.

OrbitKV 0.1.0 is being prepared for its first Python release. Build from source
until a release is published. This page describes the artifact contract and
how to validate a candidate without uploading it to PyPI.

## Packages

| Distribution | Runtime | Release targets |
| --- | --- | --- |
| `orbitkv-llm` | CUDA 13 | Linux x86_64 and aarch64, CPython 3.10–3.14 |

One distribution provides the `orbitkv` import and `orbitkv-cache-manager`
command. CUDA 12 is not a release target. There are ten wheels per release:
five CPython versions on each of two architectures; pip selects the matching wheel.
Engine extras pin vLLM 0.31.0 or SGLang 0.5.22.
SGLang requires Python 3.11+; core OrbitKV retains Python 3.10+. The selected
0.5.22 Qwen3-8B installed profile is recorded in [compatibility](compatibility.md).
Use separate engine environments. Build-matrix coverage does not establish
that every engine/GPU combination works on every Python version; serving
qualification uses Python 3.11 on H20 and A100, with each result tied to its
engine release. See [S5.1](completion-plan.md#s51--release-and-interface-audit).

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

From the repository root, select a Python interpreter with shared `libpython`
(required by the embedded Manager) and use a CUDA 13 toolkit:

```bash
PYO3_PYTHON=/absolute/path/to/python3.11 \
  ./scripts/build-wheel.sh --release
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

`orbite` is enabled by default; `orbite-rdma` adds the RDMA transport.
Use `--transfer-transport auto|tcp|rdma` to select an operator hint.
These build options do not create a second Python distribution. The installed
CLI smoke checks compiled RDMA arguments before any GPU access. Earlier
independent H20 byte/lifetime reruns covered the historical CPython 3.11 single-runtime
profile in [transport](transport.md#installed-client-and-manager-profile), with
explicit Torch CUDA 13.0.96 selection. That result does not qualify the current
OrbiTE path. Its private unversioned runtime alias is
needed because the Rust loader tries `libcudart.so` before Torch's `.so.13`.
Building with a CUDA 13.1 compiler does not establish a loaded CUDA 13.1 runtime
or qualify other installed environments. CI builds both the default and optional RDMA
wheel, with separate artifacts and matching Clang/libclang 18 on Ubuntu 24.04.
Use a wheel matching the target CPython ABI and audited glibc floor; the CI
Python 3.14 artifact cannot qualify a Python 3.11 execution node.

CI builds on Ubuntu with protoc 27.5, `auditwheel==6.8.2`,
`patchelf>=0.14.5`, maturin and wheel. Both release architectures use the same
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

1. Run `python3 scripts/check-versions.py --tag v0.1.0`. Rust, Python,
   Commitizen and the proposed tag must agree.
2. Trigger the **Release** workflow manually on the candidate branch. It builds
   all ten wheel targets and validates their versions and contents. Manual runs
   upload Actions artifacts only; they do not create a release or publish to PyPI.
3. Install a candidate wheel into each engine environment. From outside the
   source checkout, check the installed import path, Manager help and health,
   then verify a completion and an external restore after engine restart.
   The smoke gate excludes source overrides, imports the installed Rust extension
   and rejects removed TENT libraries. It verifies installed RECORD hashes,
   then compares cold, native HBM, full and partial recovery against native
   controls in DRAM and io_uring SSD cells. The installed console entry executes
   the Manager directly: SIGTERM must drain and exit 0; forced cleanup fails.
   Run the [release smoke and correctness gates](../python/tests/README.md#release-smoke).
   Claiming a shared Manager additionally requires the
   [two-engine gate](../python/tests/README.md#two-engines-sharing-one-manager),
   matching both installations to the frozen wheel and retaining concurrent
   output, each engine restart, actual SSD reads and normal drain evidence.
4. Review the final benchmark summaries and known limits. Publish only after
   these checks pass and the release is approved.

## Configure PyPI publication

Enable two-factor authentication on the PyPI account. Before the first upload,
add a GitHub pending publisher under [PyPI account publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `orbitkv-llm` |
| GitHub owner | `OrbitInfra` |
| Repository name | `OrbitKV` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

The release job uses the GitHub `pypi` environment and OIDC Trusted Publishing;
no PyPI API token is needed. A pending publisher creates the project on first
successful upload, but does not reserve its name. See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

After candidate acceptance and release approval, pushing the matching `v0.1.0`
tag runs the build/validation jobs and publishes GitHub assets and the single
PyPI distribution. Manual workflow runs only build candidates; they do not
exercise PyPI authorization. Once publication succeeds, install with
`pip install orbitkv-llm==0.1.0` in the compatible runtime environment.

## Layer-readiness candidate qualification, 2026-09-29

The following historical candidates used the earlier two-distribution matrix.
Their filenames and hashes remain evidence for those builds, not qualification
of the current single-distribution release candidate.

[Release candidate run 36513923017](https://github.com/OrbitInfra/OrbitKV/actions/runs/36513923017)
at `de451189` passes all 15 wheel targets and combined artifact validation. Its
[PR CI run](https://github.com/OrbitInfra/OrbitKV/actions/runs/36513885825)
passes all 13 checks. The manual workflow does not publish a package.

The downloaded CPython 3.11/CUDA 13/x86_64 artifact is
`orbitkv_llm_cu13-0.1.0-cp311-cp311-manylinux_2_35_x86_64.whl`, SHA256
`908b176e86e3c1af599f3a4e1ca84a5ffb97643b15f58e771f200d7d54ecf2c3`.
It passes the wheel checker and both installed-package H20 restart gates:
vLLM 0.29.0 and SGLang 0.5.20 each save and restore 90 MiB, reproduce their
initial output and release all query reservations. The historical tests use isolated Python imports, remove external TENT library
directories and verify the bundled runtime paths; see the [original record](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/releases.md#layer-readiness-candidate-qualification-2026-09-29).

Logs, installed paths, native library identities and the wheel manifest are in
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/layered-restore-20260929/ci-wheel/`. This qualifies the new
installed single-node artifact. The two-host evidence below belongs to the
previous candidate; it has not been rerun with this SHA256. No PyPI release or
RDMA qualification is claimed.

## CUDA 13 candidate qualification, 2026-09-28

[Release run 36423388612](https://github.com/OrbitInfra/OrbitKV/actions/runs/36423388612)
at commit `06e04fc3` passes all 15 build targets and the combined artifact
validation. Python 3.10–3.14, x86_64 CUDA 12/13 and aarch64 CUDA 13 artifacts
are available from that run. Its publish job is skipped because this was a
manual candidate build. The matching
[PR CI run](https://github.com/OrbitInfra/OrbitKV/actions/runs/36423395059)
also passes all 13 checks. Build coverage is separate from the GPU qualification
below.

The CPython 3.11 x86_64 candidate built with the `d936b1fa` packaging changes is
`orbitkv_llm_cu13-0.1.0-cp311-cp311-manylinux_2_35_x86_64.whl`, SHA256
`eef36f8406ae854a9ea1567da5ac09adba5a3d6f17077cca99d7be6fdd3ff163`.
The higher platform floor comes from the whole native dependency set; the old
extension-only audit had incorrectly labeled this artifact `manylinux_2_34`.

Both dedicated engine environments install this wheel non-editably. The H20
Qwen3-8B release smoke passes for vLLM 0.29.0 and SGLang 0.5.20: each restarts
the engine, restores 90 MiB to GPU, matches its initial output, and drains query
reservations. The historical test removes external TENT library paths and verifies the
three bundled primary libraries; see the [original record](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/releases.md#cuda-13-candidate-qualification-2026-09-28).

The same SHA256 is installed on the remote A100. The installed Manager console
scripts and native clients pass 8 MiB of exact GPU recovery in each direction
over IPv6 TCP, including re-serving the received replica after original-source
eviction. Every payload/gap hash matches and checked ownership counters drain.
Raw evidence, package paths and the wheel hash are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/partitioned-restore-20260928/` in `wheel-vllm/`,
`wheel-sglang/`, `installed-wheel-byte-roundtrip/` and `wheel.sha256`.

The independently downloaded CUDA 13/CPython 3.11 x86_64 artifact from Release
run `36423388612` uses the same filename; its SHA256 is
`abcdc83ab995c1c371eba574721b6cca3bc8e5fe11e53423d0661196254e2d72`.
Both engines also pass the installed-package restart gate with this CI-built
wheel: each saves and restores 90 MiB with exact outputs and no query reservation
left behind. Installing that identical CI artifact on the A100 passes the same
8 MiB GPU-byte round trip, received-replica re-serving and acknowledged release
gate. Package paths, hashes and results are recorded in `ci-wheel/` under the
artifact directory above. The initial etcd launch rejected by the environment's
HTTP proxy is retained separately; the accepted run uses direct IPv6 traffic.

These runtime results cover CUDA 13/CPython 3.11 x86_64 artifacts; they do not
qualify GPU execution on all matrix targets, RDMA, heterogeneous P/D output
equality or publication to PyPI.

## Scope of 0.1.0

The [release plan](completion-plan.md#s8--product-organization-final-artifacts-and-release)
is authoritative: no publication before current-path GPU re-qualification,
OrbiTE 0.1.0 first, installed dense and scoped-peer reruns, then 0.1.0rc1,
clean installed qualification and 0.1.0. Before release replace the OrbiTE git
pin with `orbite = "0.1"`. Publishing remains the maintainer's action.

CI builds and historical TENT/OrbiTE runs are not current hardware acceptance.
Request preparation and read cutoffs remain opt-in; their
[recorded tradeoffs](request-preparation.md#measured-results) are unchanged.
Broader hybrid, rank, HA and P/D fault claims need separate qualification.
