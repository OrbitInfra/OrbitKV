# Contributing

Start with [the repository guide](https://github.com/OrbitInfra/OrbitKV/blob/main/AGENTS.md)
and [the completion plan](completion-plan.md). The plan is the only execution
queue; architecture pages describe ownership contracts. Deliver a coherent
stage, then independently review its consumed path, gates and evidence.

## Find the owner

| Change | Location |
| --- | --- |
| Identity and recovery contracts | `crates/orbitkv-state/` |
| Cache resources, execution, GPU copies and SSD | `crates/orbitkv-core/` |
| Process channel and native client lifetime | `crates/orbitkv-channel/` |
| Manager protocol, membership and inventory streams | `crates/orbitkv-server/` |
| Local index and member admission | `crates/orbitkv-catalog/` |
| OrbiTE transfer ownership | `crates/orbitkv-core/src/peer/orbite.rs`, `transfer_engine/` |
| PyO3 and engine callbacks | `python/src/`, `python/orbitkv/` |

Python owns engine callbacks, layouts and GPU handoff. Shared state machines,
batching, planning and resource ownership stay in Rust. Read the matching
repository skill before changing a binding, engine adapter or Manager path.

## Build and validate

```bash
cargo fmt --all -- --check
cargo test -p orbitkv-state -p orbitkv-catalog
```

For native CUDA builds, follow [runtime requirements](single-node.md) and use
the CUDA feature matching the host. Run Cargo/native builds sequentially in one
checkout to keep build and runtime evidence tied to one artifact.

Use the [Python gate matrix](https://github.com/OrbitInfra/OrbitKV/blob/main/python/tests/README.md)
to choose units, integration, E2E, stress or installed-artifact checks. Default
Python tests require no GPU, engine service or native build. Heavy gates have
explicit triggers and remain unqualified when hardware is unavailable.
Local hooks use `prek run` on a topic branch.

Workspace Clippy requires a `// SAFETY:` note for each unsafe block and impl,
including test fixtures. State the actual lifetime, aliasing, alignment or FFI
invariant at the call site; if it does not hold, fix the owner before documenting
it. The strict feature gates cover no-backend, OrbiTE TCP, and OrbiTE RDMA/GDS builds.


PRs limited to Markdown, docs tooling and Python benchmark/test files run the relevant
fast checks. Native, package, CI or unrecognized inputs keep CUDA checks, wheels
and Clippy; main pushes retain the complete pipeline. Native source moves are
classified as deletions and additions, so moving a file cannot hide its check.

## Documentation

The single Sphinx site reads MyST Markdown from `docs/`. Add guides to the
appropriate `toctree` in `docs/index.md`; keep source files in place when regrouping.
The layout follows [LMCache's documentation](https://docs.lmcache.ai/) using
`sphinxawesome_theme`, with OrbitKV's own text, logo and colors. No affiliation
is implied. The landing page and README share the light/dark architecture figures.

From the repository root, with Python 3.13 and `uv`:

```bash
uv venv --python 3.13 .venv/docs
uv pip sync --python .venv/docs/bin/python docs/requirements.txt
.venv/docs/bin/sphinx-build -b dirhtml -n -W --keep-going docs /tmp/orbitkv-docs
.venv/docs/bin/python docs/check_site.py /tmp/orbitkv-docs
.venv/docs/bin/python -m unittest discover -s docs/tests
```

Keep warnings fatal. The link check verifies built pages, assets, fragments and
repository/evidence targets without fetching third-party websites. Use a full Git
history checkout so historical evidence permalinks can be verified. It also checks
legacy URL redirects and search coverage. `docs/redirects.json` preserves retired
aliases; `/docs/<page>/` redirects are derived from the maintained guides. Redirects
retain query strings and fragments. No Node/npm toolchain is needed; Mermaid
runs in the browser from a version-pinned CDN module.

Update `docs/requirements.in`, then regenerate the complete pinned Python set:

```bash
uv pip compile docs/requirements.in --python-version 3.13 -o docs/requirements.txt
```

CI builds and checks the site without publishing. The existing Pages workflow
publishes the same `/OrbitKV/` URL after a merge to main.

## Submit a change

Create a `feat/`, `fix/`, `chore/`, `refactor/`, `style/` or `ci/` branch. Use
Commitizen messages and `feichai0017 <songguocheng348@gmail.com>` for repository
commits. Keep the PR focused on the final behavior and relevant validation.

Update affected technical documentation under `docs/`; the website renders it
directly through Sphinx. Keep the Awesome theme, OrbitKV logo and palette. When shortening a
historical report, verify its archive and keep immutable evidence/reproduction
links. Generated experiment output belongs outside Git; see [evidence policy](performance.md#evidence-and-instrumentation).

Record the commit, changed contract, tests, external evidence and remaining
limits. Distinguish implemented, independently accepted and released. An upstream
issue or PR submission is not a released fix, and a missing hardware gate is
not a pass.
