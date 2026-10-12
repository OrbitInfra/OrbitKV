"""Bounded diagnostic traces through the released SGLang HTTP profiler."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import requests


def inspect_trace(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Profiler trace must be a regular file: {path}")
    with gzip.open(path, "rt") as source:
        trace = json.load(source)
    events = trace["traceEvents"]
    kernels = sum(event.get("cat") == "kernel" for event in events)
    copies = sum(event.get("cat") == "gpu_memcpy" for event in events)
    if not kernels:
        raise ValueError(f"No GPU kernels recorded in profiler trace: {path}")
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "events": len(events),
        "GPU_kernels": kernels,
        "GPU_memcopies": copies,
    }


@contextmanager
def sglang_profile(url: str, output: Path):
    """Retain start/stop receipts and require a complete CPU/CUDA trace."""
    output.mkdir(parents=True, exist_ok=False)
    control = {
        "mode": "sglang-public-cpu-cuda",
        "output_directory": str(output),
        "scope": "Diagnostic only; profiling can perturb TTFT and CUDA submission",
        "requests": [],
        "traces": [],
    }
    error = None
    stop_error = None

    def request(endpoint: str, payload: dict | None = None) -> None:
        receipt = {"endpoint": endpoint, "start_mono_ns": time.monotonic_ns()}
        control["requests"].append(receipt)
        try:
            response = requests.post(url + endpoint, json=payload, timeout=(10, 120))
            receipt.update(status=response.status_code, response=response.text[:4096])
            response.raise_for_status()
        except BaseException as failure:
            receipt["error"] = repr(failure)
            raise
        finally:
            receipt["end_mono_ns"] = time.monotonic_ns()

    try:
        request(
            "/start_profile",
            {
                "output_dir": str(output),
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
                "profile_id": "restore",
                "detailed_annotations": True,
            },
        )
        yield control
    except BaseException as failure:
        error = failure
        control["workload_error"] = repr(failure)
        raise
    finally:
        try:
            request("/stop_profile")
            if error is None:
                paths = sorted(output.glob("*.trace.json.gz"))
                if not paths:
                    raise ValueError("SGLang did not export a profiler trace")
                control["traces"] = [inspect_trace(path) for path in paths]
        except BaseException as failure:
            stop_error = failure
            control["stop_or_trace_error"] = repr(failure)
        (output / "CONTROL.json").write_text(json.dumps(control, indent=2) + "\n")
        if stop_error is not None and error is None:
            raise stop_error
