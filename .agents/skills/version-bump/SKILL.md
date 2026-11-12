---
name: version-bump
description: Update OrbitKV package versions or prepare a release candidate while keeping Rust, Python, CUDA runtime and release tags consistent.
---

# OrbitKV versions and release candidates

Read `AGENTS.md` and `docs/releases.md` from the Git root. Follow the user's
requested version and release scope; candidate preparation and publishing are
different operations. Existing session authorization still applies.

1. Inspect the current version, working tree and branch. Update
   `Cargo.toml` `[workspace.package]`, `python/pyproject.toml` `[project]` and
   `[tool.commitizen]` together. Refresh affected workspace entries in `Cargo.lock`
   without upgrading unrelated dependencies.
   Rust and Commitizen use SemVer (`0.1.0-beta.1`); Python uses canonical
   PEP 440 (`0.1.0b1`). They denote the same candidate, not identical strings.
2. Run `python3 scripts/check-versions.py --tag vVERSION`. The single
   `orbitkv` distribution targets CUDA 13 and must retain its documented
   engine/runtime constraints. Check the ten-wheel release matrix in `docs/releases.md` and
   `docs/completion-plan.md`; an engine target becomes supported only after its
   upgrade gate passes. Keep license and bundled upstream provenance with artifacts.
3. Build the candidate with `scripts/build-wheel.sh` or the manual Release
   workflow. Check the complete installed package, Manager and loaded CUDA/RDMA paths,
   then execute the relevant installed-engine and two-host gates.
4. Keep artifacts and qualification results outside Git. Supply their version,
   source commit, hashes, reproduction commands and known limits for review.
   Before publication, `check-versions.py --publication` must also pass: OrbiTE
   must be an exact crates.io dependency with a matching lock checksum. A Git
   candidate can rehearse builds but cannot publish.

A matching `v*` tag triggers GitHub/PyPI publication in this repository; do not
push one merely to test version consistency. Publish only within the user's
release authorization and after candidate acceptance. Do not embed credentials
or replace the project's release workflow with skill-specific scripts.
