"""Cache-source evidence and latency summaries shared by benchmark reports."""

from __future__ import annotations

import math
import re
import statistics
import threading
import time
from contextlib import contextmanager

import requests

REMOTE_STAGES = ("authorization", "allocation", "read", "rebuild", "release")
ENGINE_ITL = {engine: f"{engine}:inter_token_latency_seconds" for engine in ("vllm", "sglang")}
OBSERVATION_MODES = ("sampled", "boundary")
MANAGER_BACKENDS = ("orbitkv", "lmcache")
REQUIRED_OWNERSHIP_GAUGES = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_inflight_bytes",
    "orbitkv_transfer_lock_active",
    "orbitkv_transfer_reserved_bytes",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_ssd_write_inflight",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_read_pinned_bytes",
)
DRAIN_GAUGES = (
    *REQUIRED_OWNERSHIP_GAUGES,
    "orbitkv_transfer_completion_outstanding",
    "orbitkv_ssd_cufile_inflight_batches",
    "orbitkv_storage_codec_reserved_bytes",
)


def delta(before: dict, after: dict) -> dict:
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if value != before.get(key, 0)
        or any(key.startswith(f"{name}_bucket{{le=") for name in ENGINE_ITL.values())
    }


def cached_tokens(engine: str, sample: dict) -> int:
    usage = sample["usage"]
    if engine == "sglang":
        return usage.get("cached_tokens", 0)
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


@contextmanager
def measure(
    base_url: str,
    manager_url: str | None,
    settle_seconds: float,
    *,
    backend: str,
    observation_mode: str = "sampled",
):
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"Invalid observation mode: {observation_mode}")
    if (manager_url is not None) != (backend in MANAGER_BACKENDS):
        raise ValueError("Backend and Manager presence do not match")
    manager_backend = backend if manager_url is not None else None
    before, manager_before = metrics(base_url), metrics(manager_url)
    if manager_backend == "orbitkv":
        validate_ownership(manager_before, drained=False)
    stop = threading.Event()
    peaks = {} if observation_mode == "sampled" and manager_url is not None else None
    errors = []
    result = {
        "observation_mode": observation_mode,
        "manager_present": manager_url is not None,
        "manager_backend": manager_backend,
        "memory_samples": 0,
    }

    def observe():
        while not stop.is_set():
            try:
                observed = metrics(manager_url)
                if manager_backend == "orbitkv":
                    validate_ownership(observed, drained=False)
                result["memory_samples"] += 1
                for key, value in observed.items():
                    if key in (
                        "orbitkv_pool_used_bytes",
                        "orbitkv_inflight_bytes",
                        "orbitkv_cache_protected_bytes",
                        "orbitkv_warmup_pending_bytes",
                        "orbitkv_query_speculative_reserved_bytes",
                        "orbitkv_ssd_read_pinned_bytes",
                        "orbitkv_ssd_gpu_staging_bytes",
                        "orbitkv_storage_codec_reserved_bytes",
                        "orbitkv_storage_codec_workspace_bytes",
                    ) or key.startswith("orbitkv_query_reserved_bytes"):
                        peaks[key] = max(value, peaks.get(key, 0))
            except Exception as error:
                errors.append(str(error))
                return
            stop.wait(0.025)

    monitor = None
    if peaks is not None:
        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
    started = time.perf_counter()
    try:
        yield result
        result["wall_seconds"] = time.perf_counter() - started
        drain_started = time.perf_counter()
        time.sleep(settle_seconds)
        deadline = time.monotonic() + 30
        quiet = 0
        while True:
            manager_after = metrics(manager_url)
            if manager_backend == "orbitkv":
                validate_ownership(manager_after, drained=False)
            busy = manager_backend == "orbitkv" and any(
                manager_after[name] != 0 for name in DRAIN_GAUGES if name in manager_after
            )
            quiet = 0 if busy else quiet + 1
            if quiet >= 3:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Cache work survived completed requests: {manager_after}")
            time.sleep(0.1)
        result.update(
            drain_seconds=time.perf_counter() - drain_started,
            metrics_delta=delta(before, metrics(base_url)),
            manager_delta=delta(manager_before, manager_after),
            sampled_peak_bytes=peaks,
            manager_before=manager_before,
            manager_after=manager_after,
        )
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=12)
    if (monitor is not None and monitor.is_alive()) or errors:
        raise RuntimeError(f"Memory observation failed: {errors}")


def validate_ownership(snapshot: dict, *, drained: bool) -> None:
    if not isinstance(snapshot, dict) or any(
        name not in snapshot for name in REQUIRED_OWNERSHIP_GAUGES
    ):
        raise ValueError("Missing required OrbitKV ownership snapshot")
    for name in DRAIN_GAUGES:
        if name not in snapshot:
            continue
        value = snapshot[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid OrbitKV ownership metric: {name}")
        if drained and value != 0:
            raise ValueError(f"OrbitKV ownership did not drain: {name}={value}")


def validate_observation(
    mode: str, measurements: list[dict], *, manager_backend: str | None
) -> None:
    if mode not in OBSERVATION_MODES or not measurements:
        raise ValueError("Missing or invalid observation mode")
    if manager_backend is not None and manager_backend not in MANAGER_BACKENDS:
        raise ValueError("Invalid Manager backend")
    for row in measurements:
        if (
            row.get("observation_mode") != mode
            or row.get("manager_present") is not (manager_backend is not None)
            or "manager_backend" not in row
            or row["manager_backend"] != manager_backend
        ):
            raise ValueError("Mixed or missing observation metadata")
        for field in ("manager_before", "manager_after"):
            if not isinstance(row.get(field), dict):
                raise ValueError("Missing Manager boundary snapshot")
            if manager_backend == "orbitkv":
                validate_ownership(row[field], drained=field == "manager_after")
        count, peaks = row.get("memory_samples"), row.get("sampled_peak_bytes")
        if type(count) is not int or count < 0 or "sampled_peak_bytes" not in row:
            raise ValueError("Missing or invalid memory observation")
        if mode == "sampled" and manager_backend is not None:
            if count == 0 or not isinstance(peaks, dict):
                raise ValueError("Missing sampled memory observations")
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in peaks.values()
            ):
                raise ValueError("Invalid sampled memory peak")
            if manager_backend == "orbitkv" and any(
                name not in peaks
                for name in ("orbitkv_query_reserved_bytes", "orbitkv_inflight_bytes")
            ):
                raise ValueError("Missing sampled OrbitKV ownership peaks")
        elif count != 0 or peaks is not None:
            raise ValueError("Unsampled memory peaks must remain unknown")


def sampled_peak(measurements: list[dict], metric: str) -> float | None:
    values = [(row.get("sampled_peak_bytes") or {}).get(metric) for row in measurements]
    return max(values) if values and all(value is not None for value in values) else None


def workload_phases(ssd: bool = False) -> tuple[str, ...]:
    phases = ("cold", "hbm_hit", "after_pressure")
    return (*phases, "after_host_eviction") if ssd else phases


def metrics(url: str | None) -> dict[str, float]:
    if url is None:
        return {}
    response = requests.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, value = line.rsplit(" ", 1)
        name = series.split("{", 1)[0]
        if name.endswith("_created"):
            continue
        number = float(value)
        if name in DRAIN_GAUGES and (not math.isfinite(number) or number < 0):
            raise ValueError(f"Invalid OrbitKV ownership metric: {name}")
        if math.isfinite(number):
            if "bucket" in name:
                if name in {f"{prefix}_bucket" for prefix in ENGINE_ITL.values()}:
                    labels = dict(re.findall(r'(\w+)="([^"\\]*)"', series))
                    key = f"{name}{{le={labels['le']}}}"
                    values[key] = values.get(key, 0) + number
                continue
            values[name] = values.get(name, 0) + number
            if name.startswith("orbitkv_storage_codec_"):
                labels = dict(re.findall(r'(\w+)="([^"\\]*)"', series))
                dimensions = [
                    labels[label]
                    for label in ("representation", "direction", "operation", "reason")
                    if label in labels
                ]
                if dimensions:
                    key = f"{name}_{'_'.join(dimensions)}"
                    values[key] = values.get(key, 0) + number
            if name.startswith("orbitkv_cost_"):
                labels = dict(re.findall(r'(\w+)="([^"\\]*)"', series))
                dimensions = [
                    f"{label}={labels[label]}"
                    for label in (
                        "path",
                        "stage",
                        "outcome",
                        "evidence",
                        "decision",
                        "scope",
                        "reason",
                    )
                    if label in labels
                ]
                if dimensions:
                    key = f"{name}{{{','.join(dimensions)}}}"
                    values[key] = values.get(key, 0) + number
            if name.startswith("orbitkv_remote_stage_duration_seconds_"):
                for stage in REMOTE_STAGES:
                    if f'stage="{stage}"' in series:
                        key = f"{name}_{stage}"
                        values[key] = values.get(key, 0) + number
            if name == "orbitkv_query_reserved_bytes_by_phase" and 'phase="warming"' in series:
                key = "orbitkv_query_reserved_bytes_warming"
                values[key] = values.get(key, 0) + number
            if name == "orbitkv_warmup_wait_byte_seconds_total":
                for outcome in ("restored", "unused"):
                    if f'outcome="{outcome}"' in series:
                        key = f"{name}_{outcome}"
                        values[key] = values.get(key, 0) + number
            if name == "orbitkv_ssd_write_admission_skips_total":
                for reason in ("cold", "resident", "pending", "duplicate"):
                    if f'reason="{reason}"' in series:
                        key = f"{name}_{reason}"
                        values[key] = values.get(key, 0) + number
    return values


def engine_itl_summary(engine: str, counters: dict, slo_ms: float) -> dict:
    """Estimate quantiles from official engine histograms, preserving bucket uncertainty."""
    name = ENGINE_ITL[engine]
    prefix = f"{name}_bucket{{le="
    buckets = sorted(
        (float(key[len(prefix) : -1]), value)
        for key, value in counters.items()
        if key.startswith(prefix)
    )
    count = counters.get(f"{name}_count", 0)
    result = {
        "metric": name,
        "status": "missing",
        "count": count,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "slo_ms": slo_ms,
        "within_slo_fraction": None,
        "buckets": [
            {"upper_ms": bound * 1000 if math.isfinite(bound) else None, "count": value}
            for bound, value in buckets
        ],
        "scope": "Official engine histogram delta over the measured window; bucket-linear quantile estimates, not exact samples or SSE packet intervals. No request-level goodput attribution.",
        "engine_boundary": "Engine-core adjacent token timestamps"
        if engine == "vllm"
        else "Tokenizer output receipt interval divided by new-token count and weighted by that count; coalesced output intervals are averaged",
    }
    if not buckets or count <= 0:
        return result
    previous = 0
    for bound, value in buckets:
        if math.isnan(bound) or bound <= 0 or not math.isfinite(value) or value < previous:
            result["status"] = "invalid_buckets"
            return result
        previous = value
    if not math.isinf(buckets[-1][0]) or buckets[-1][1] != count:
        result["status"] = "incomplete_buckets"
        return result
    result["status"] = "observed"
    duration = counters.get(f"{name}_sum")
    result["mean_ms"] = duration / count * 1000 if duration is not None else None
    for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        low, before = 0.0, 0.0
        for high, cumulative in buckets:
            if cumulative >= count * fraction:
                result[f"{label}_bounds_ms"] = [
                    low * 1000,
                    high * 1000 if math.isfinite(high) else None,
                ]
                if math.isfinite(high) and cumulative > before:
                    result[f"{label}_ms"] = (
                        low + (high - low) * (count * fraction - before) / (cumulative - before)
                    ) * 1000
                break
            low, before = high, cumulative
    for bound, cumulative in buckets:
        if math.isclose(bound * 1000, slo_ms):
            result["within_slo_fraction"] = cumulative / count
            break
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def codec_summary(measurements: list[dict]) -> dict:
    """Aggregate publication/transfer counters once per measured interval, never per request."""
    keys = {
        key
        for row in measurements
        for snapshot in ("manager_delta", "manager_before", "manager_after")
        for key in row.get(snapshot, {})
        if key.startswith("orbitkv_storage_codec_")
        and ("_total" in key or "_sum" in key or "_count" in key)
        and key != "orbitkv_storage_codec_bytes_total"
    }
    counters = {
        key: sum(row.get("manager_delta", {}).get(key, 0) for row in measurements)
        for key in sorted(keys)
    }
    logical = counters.get("orbitkv_storage_codec_bytes_total_logical", 0)
    stored = counters.get("orbitkv_storage_codec_bytes_total_stored")
    summary = {
        **counters,
        "encoded_publication_stored_fraction": stored / logical
        if logical > 0 and stored is not None
        else None,
    }
    for name in ("reserved", "workspace"):
        metric = f"orbitkv_storage_codec_{name}_bytes"
        summary[f"sampled_peak_codec_{name}_bytes"] = sampled_peak(measurements, metric)
        summary[f"boundary_max_codec_{name}_bytes"] = max(
            (
                row[snapshot][metric]
                for row in measurements
                for snapshot in ("manager_before", "manager_after")
                if metric in row.get(snapshot, {})
            ),
            default=None,
        )
        summary[f"max_codec_{name}_bytes_after"] = max(
            (
                row["manager_after"][metric]
                for row in measurements
                if metric in row.get("manager_after", {})
            ),
            default=None,
        )
    summary["pool_used_bytes_after"] = max(
        (
            row["manager_after"]["orbitkv_pool_used_bytes"]
            for row in measurements
            if "orbitkv_pool_used_bytes" in row.get("manager_after", {})
        ),
        default=None,
    )
    return summary


def cache_source(engine: str, result: dict) -> str:
    if engine == "sglang":
        details = result["usage"].get("cached_tokens_details") or {}
        device = details.get("device", 0)
        external = details.get("host", 0) + details.get("storage", 0)
    else:
        delta = result["metrics_delta"]
        device = delta.get("vllm:prefix_cache_hits_total", 0)
        external = delta.get("vllm:external_prefix_cache_hits_total", 0)
    external = max(external, result["manager_delta"].get("orbitkv_load_bytes_total", 0))
    if device > 0:
        return "mixed" if external > 0 else "hbm"
    if external > 0:
        return "external"
    cached = (
        result["usage"].get("cached_tokens", 0)
        if engine == "sglang"
        else (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    )
    return "unverified" if cached else "miss"


def summarize(samples: list[dict], lengths: list[int]) -> list[dict]:
    summary = []
    phases = tuple(dict.fromkeys(sample["phase"] for sample in samples))
    for length in lengths:
        for phase in phases:
            group = [
                sample
                for sample in samples
                if sample["length"] == length and sample["phase"] == phase
            ]
            times = [sample["ttft_ms"] for sample in group]
            profiling_modes = {sample.get("profiling", {}).get("mode", "off") for sample in group}
            if len(profiling_modes) != 1 or not profiling_modes <= {
                "off",
                "sglang-public-cpu-cuda",
            }:
                raise ValueError("Cannot summarize mixed or invalid profiling modes")
            summary.append(
                {
                    "length": length,
                    "phase": phase,
                    "observation_mode": group[0].get("observation_mode"),
                    "profiling_mode": profiling_modes.pop(),
                    "n": len(group),
                    "ttft_p50_ms": statistics.median(times),
                    "ttft_p95_ms": percentile(times, 0.95),
                    "e2e_p50_ms": statistics.median(sample["e2e_ms"] for sample in group),
                    "output_mismatches": sum(not sample["matches_cold_output"] for sample in group),
                    "cache_sources": {
                        source: sum(sample["cache_source"] == source for sample in group)
                        for source in sorted({sample["cache_source"] for sample in group})
                    },
                    "orbitkv_load_bytes": sum(
                        sample["manager_delta"].get("orbitkv_load_bytes_total", 0)
                        for sample in group
                    ),
                    "orbitkv_ssd_read_bytes": sum(
                        sample["manager_delta"].get("orbitkv_ssd_prefetch_bytes_total", 0)
                        + sample["manager_delta"].get("orbitkv_ssd_cufile_read_bytes_total", 0)
                        for sample in group
                    ),
                    **{
                        key: sum(sample["manager_delta"].get(key, 0) for sample in group)
                        for key in (
                            "orbitkv_ssd_prefetch_bytes_total",
                            "orbitkv_ssd_cufile_read_bytes_total",
                        )
                    },
                    "ssd_reads_without_gpu_restore": sum(
                        (
                            sample["manager_delta"].get("orbitkv_ssd_prefetch_bytes_total", 0)
                            + sample["manager_delta"].get("orbitkv_ssd_cufile_read_bytes_total", 0)
                        )
                        > 0
                        and sample["manager_delta"].get("orbitkv_load_bytes_total", 0) == 0
                        for sample in group
                    ),
                    "load_task_p50_ms": statistics.median(
                        sample["manager_delta"].get("orbitkv_load_duration_seconds_sum", 0) * 1000
                        for sample in group
                    ),
                    "ssd_prefetch_p50_ms": statistics.median(
                        sample["manager_delta"].get("orbitkv_ssd_prefetch_duration_seconds_sum", 0)
                        * 1000
                        for sample in group
                    ),
                    **codec_summary(group),
                }
            )
    return summary
