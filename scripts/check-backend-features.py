#!/usr/bin/env python3
"""Inventory backend cfg predicates and keep production gates in native adapters."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = {
    "crates/orbitkv-core/src/memory/pool/backend.rs",
    "crates/orbitkv-core/src/peer/mod.rs",  # Native adapter module declarations.
    "crates/orbitkv-core/src/peer/orbite.rs",
    "crates/orbitkv-core/src/peer/transport.rs",
    "crates/orbitkv-core/src/transfer/local/backend.rs",
    "crates/orbitkv-core/src/transfer/local/orbite.rs",
    "crates/orbitkv-core/src/transfer_engine/backend.rs",
    "python/src/lib.rs",  # Optional native Python module declaration/registration.
}
PREDICATE = re.compile(r"\bcfg(?:_attr)?!?\s*\(")
BACKEND = re.compile(r'feature\s*=\s*"(?:mooncake|orbite[^\"]*|orbitte[^\"]*)"')


def main() -> int:
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "crates",
            "python/src",
            "benches",
        ],
        cwd=ROOT,
        text=True,
    ).split("\0")
    total = 0
    violations = []
    for path in sorted(
        path for path in paths if path.endswith(".rs") and (ROOT / path).is_file()
    ):
        source = (ROOT / path).read_text()
        count = 0
        for match in PREDICATE.finditer(source):
            depth, end = 1, match.end()
            while end < len(source) and depth:
                depth += (source[end] == "(") - (source[end] == ")")
                end += 1
            count += bool(BACKEND.search(source[match.start() : end]))
        if count:
            print(f"{path}: {count}")
            total += count
            if "/src/" in path and path not in ADAPTERS:
                violations.append(path)
    print(f"Backend cfg predicates: {total}")
    if violations:
        print("Backend feature gates outside adapters: " + ", ".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
