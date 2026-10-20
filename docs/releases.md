# Python releases

OrbitKV 0.1.0 is being prepared for its first Python release. Build from source
until a release is published. This page describes the artifact contract and
how to validate a candidate without uploading it to PyPI.

## Packages

| Distribution | Runtime | Release targets |
| --- | --- | --- |
| `orbitkv-llm` | CUDA 12 | Linux x86_64, CPython 3.10–3.14 |
| `orbitkv-llm-cu13` | CUDA 13 | Linux x86_64 and aarch64, CPython 3.10–3.14 |

Install one CUDA variant per environment; both provide the `orbitkv` import and
`orbitkv-cache-manager` command. Engine extras pin vLLM 0.31.0 or SGLang 0.5.22.
SGLang requires Python 3.11+; core OrbitKV retains Python 3.10+. The selected
0.5.22 Qwen3-8B installed profile is recorded in [compatibility](compatibility.md).
Use separate engine environments. Build-matrix coverage does not establish
that every engine/GPU combination works on every Python version; serving
qualification uses Python 3.11 on H20 and A100, with each result tied to its
engine release. See [S5.1](completion-plan.md#s51--release-and-interface-audit).

The wheel contains the PyO3 extension, Manager executable, Mooncake libraries,
engine plugins, type stubs, Apache-2.0 license and third-party attribution in
`NOTICE`. The host supplies its NVIDIA
driver and compatible PyTorch/CUDA environment. Tests, benchmarks and build
caches are excluded from the package.

GPU ANS encoding additionally needs the separately installed nvCOMP 5.3 runtime;
it is not bundled in the wheel. FP8 and TurboQuant kernels are compiled by NVRTC
in the Manager. All storage codecs are opt-in; see [storage formats](storage-formats.md)
for dependencies, model-quality results and GPU workspace limits.

## Build a candidate

From the repository root, select the Python interpreter and CUDA variant:

```bash
git submodule update --init --recursive third-party/mooncake
PYO3_PYTHON=/absolute/path/to/python3.11 \
  ./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
```

Omit the feature arguments for CUDA 12. The script builds and stages the
Manager and Mooncake libraries, builds the extension for the selected Python,
audits every bundled ELF file with auditwheel, repairs its non-host dependencies,
checks the package version/content, and imports the installed wheel in a fresh
non-editable environment. The initial maturin output uses a Linux tag; only the
whole-wheel audit selects the final manylinux tag. Auditing the extension alone
misses libraries loaded dynamically by TENT and the embedded-Python Manager.

CI builds on Ubuntu with protoc 27.5, `auditwheel==6.8.2`,
`patchelf>=0.14.5`, maturin and wheel. Both release architectures use the same
protocol compiler; Ubuntu 22.04's system protoc cannot compile the current
proto3 optional fields without an experimental flag. Dependency copyright files are retained from the Debian/Ubuntu package
database, together with referenced common license texts and Mooncake's license.
An externally staged dependency must supply `<library-filename>.license` beside
its original shared library if it has no OS package provenance; missing notices
fail the build. The repaired artifact also retains auditwheel's dependency SBOM.

Python's shared library, CUDA driver/runtime/cuFile and RDMA core/provider
libraries remain host dependencies; the repair step must neither bundle driver
libraries nor remove the standalone Manager's `libpython` dependency. Wheels
are written to `target/wheels/`.
Do not run native builds while a source-built Manager is using the staged
Mooncake libraries. `maturin build` alone does not stage the complete runtime.

## Validate before publishing

1. Run `python3 scripts/check-versions.py --tag v0.1.0`. Rust, Python,
   Commitizen and the proposed tag must agree.
2. Trigger the **Release** workflow manually on the candidate branch. It builds
   all 15 wheel targets and validates their versions and contents. Manual runs
   upload Actions artifacts only; they do not create a release or publish to PyPI.
3. Install a candidate wheel into each engine environment. From outside the
   source checkout, check the installed import path, Manager help and health,
   then verify a completion and an external restore after engine restart.
   The smoke gate removes external TENT search paths, initializes TENT, and
   checks `/proc/self/maps` to prove that its three primary libraries came from
   the installed package. It verifies pinned versions and installed RECORD hashes,
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

Pushing a matching `v*` tag runs the same build/validation jobs and then publishes
GitHub release assets and both PyPI distributions. The workflow requires the
repository's `PYPI_API_TOKEN` to authorize both names. Preparing a candidate or
running the manual workflow does not test that credential or reserve the names.

## Layer-readiness candidate qualification, 2026-09-29

[Release candidate run 36513923017](https://github.com/feichai0017/orbitkv/actions/runs/36513923017)
at `de451189` passes all 15 wheel targets and combined artifact validation. Its
[PR CI run](https://github.com/feichai0017/orbitkv/actions/runs/36513885825)
passes all 13 checks. The manual workflow does not publish a package.

The downloaded CPython 3.11/CUDA 13/x86_64 artifact is
`orbitkv_llm_cu13-0.1.0-cp311-cp311-manylinux_2_35_x86_64.whl`, SHA256
`908b176e86e3c1af599f3a4e1ca84a5ffb97643b15f58e771f200d7d54ecf2c3`.
It passes the wheel checker and both installed-package H20 restart gates:
vLLM 0.29.0 and SGLang 0.5.20 each save and restore 90 MiB, reproduce their
initial output and release all query reservations. The tests run with isolated
Python imports, remove external TENT library directories and verify loaded
runtime paths under the installed package.

Logs, installed paths, native library identities and the wheel manifest are in
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/layered-restore-20260929/ci-wheel/`. This qualifies the new
installed single-node artifact. The two-host evidence below belongs to the
previous candidate; it has not been rerun with this SHA256. No PyPI release or
RDMA qualification is claimed.

## CUDA 13 candidate qualification, 2026-09-28

[Release run 36423388612](https://github.com/feichai0017/orbitkv/actions/runs/36423388612)
at commit `06e04fc3` passes all 15 build targets and the combined artifact
validation. Python 3.10–3.14, x86_64 CUDA 12/13 and aarch64 CUDA 13 artifacts
are available from that run. Its publish job is skipped because this was a
manual candidate build. The matching
[PR CI run](https://github.com/feichai0017/orbitkv/actions/runs/36423395059)
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
reservations. The test removes external TENT library paths and verifies the
three loaded primary libraries under the installed package.

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

The initial release targets single-node DRAM/SSD cache recovery with vLLM and
SGLang. Supported hybrid layouts use compiled range selection and leased-state
validation. Cancellation, lost notifications and engine/Manager restart have
native and model-serving gates.

Request preparation and read cutoffs remain opt-in. The repeated Qwen3-8B
controls show a throughput/latency tradeoff on SGLang; see
[the policy results](request-preparation.md#measured-results).
Distributed serving, metadata HA, automatic hybrid lookahead and cross-host TP
remain outside the qualified single-node release scope. See
[deployment support](deployment.md) and [the completion plan](completion-plan.md).

The installed host-buffer TCP regression lives in
`python/tests/release/test_installed_tent_tcp.py` and is marked `release_smoke`
and `integration`. Run it from a fresh fixture directory with the installed
wheel and without source-package import paths. It checks changing READ/WRITE
payloads, untouched gaps, source stability, native mappings and explicit
unregistration. This gate requires no GPU and does not replace the healthy GPU,
RDMA, Manager SSD or adapter lifetime gates. Native source builds currently
apply the documented [upstream copy-status backport](../crates/orbitkv-mooncake-sys/patches/README.md)
in a private directory; freeze the patch and resulting bytes with the wheel.
