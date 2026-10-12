# Contributing

Start with [the repository guide](https://github.com/feichai0017/orbitkv/blob/main/AGENTS.md)
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
| TENT wrapper and native build | `crates/orbitkv-transfer/`, `crates/orbitkv-mooncake-sys/` |
| PyO3 and engine callbacks | `python/src/`, `python/orbitkv/` |

Python owns engine callbacks, layouts and GPU handoff. Shared state machines,
batching, planning and resource ownership stay in Rust. Read the matching
repository skill before changing a binding, engine adapter or Manager path.

## Build and validate

```bash
git submodule update --init --recursive third-party/mooncake
cargo fmt --all -- --check
cargo test -p orbitkv-state -p orbitkv-catalog
```

For native CUDA builds, follow [runtime requirements](single-node.md) and use
the CUDA feature matching the host. Run Cargo/native builds sequentially in one
checkout: they restage Mooncake libraries that running processes may have mapped.

Use the [Python gate matrix](https://github.com/feichai0017/orbitkv/blob/main/python/tests/README.md)
to choose units, integration, E2E, stress or installed-artifact checks. Default
Python tests require no GPU, engine service or native build. Heavy gates have
explicit triggers and remain unqualified when hardware is unavailable.
Local hooks use `prek run` on a topic branch.

## Submit a change

Create a `feat/`, `fix/`, `chore/`, `refactor/`, `style/` or `ci/` branch. Use
Commitizen messages and `feichai0017 <songguocheng348@gmail.com>` for repository
commits. Keep the PR focused on the final behavior and relevant validation.

Update affected technical documentation under `docs/`; the website renders it
directly. Preserve website components and visual identity. When shortening a
historical report, verify its archive and keep immutable evidence/reproduction
links. Generated experiment output belongs outside Git; see [evidence policy](benchmark-evidence.md).

Record the commit, changed contract, tests, external evidence and remaining
limits. Distinguish implemented, independently accepted and released. An upstream
issue or PR submission is not a released fix, and a missing hardware gate is
not a pass.
