"""Collect actual dynamic loading dependencies without creating a transfer engine."""

import ctypes
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def main() -> None:
    library = Path(sys.argv[1])
    if not library.is_absolute():
        raise RuntimeError("Dynamic library path must be absolute")
    ctypes.CDLL(str(library))
    files = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if str(path) in files:
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4 * 1024**3:
            raise RuntimeError("Loaded mapping has no bounded regular backing file")
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        files[str(path)] = value.hexdigest()
        if len(files) > 4096:
            raise RuntimeError("Loaded mapping census exceeded its bound")
    if str(library.resolve()) not in files:
        raise RuntimeError("Requested library has no actual mapping")
    print(
        json.dumps(
            {
                "schema": "orbitkv.runtime-dlopen.v1",
                "pid": os.getpid(),
                "library": str(library),
                "files": files,
                "engine_created": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
