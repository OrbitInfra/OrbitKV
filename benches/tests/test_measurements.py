"""Protect cache-source evidence and incomplete-run rejection in comparisons."""

import csv
import json
from types import SimpleNamespace

import pytest

from benches.metrics import cache_source, metrics, summarize
from benches.report import collect_run


@pytest.mark.parametrize(
    ("configured", "expected"), [(None, "0"), ("0", "0"), ("1", "1"), ("invalid", "0")]
)
def test_manifest_records_effective_cost_observations_switch(
    tmp_path, monkeypatch, configured, expected
):
    from benches.runtime import manifest

    monkeypatch.setattr("benches.runtime.subprocess.check_output", lambda *a, **kw: "fixture")
    monkeypatch.setattr("benches.runtime.importlib.metadata.version", lambda name: "fixture")
    manager = tmp_path / "manager"
    manager.write_bytes(b"fixed-manager-artifact")
    env = {"PYTHONPATH": "python"}
    if configured is not None:
        env["ORBITKV_COST_OBSERVATIONS"] = configured
    args = SimpleNamespace(
        engine="vllm",
        backend="orbitkv",
        workload="serial",
        model=tmp_path,
        gpu_tokens=8192,
        host_gib=1,
        ssd_gib=0,
        storage_codec_budget=64 * 1024**2,
        observation_mode="sampled",
    )
    launch = SimpleNamespace(
        command=["vllm"],
        manager_command=[str(manager)],
        manager_url="http://manager",
        env=env,
        backend_configuration={},
    )
    for backend in ("native", "cpu", "orbitkv", "lmcache"):
        args.backend = backend
        launch.manager_url = "http://manager" if backend in ("orbitkv", "lmcache") else None
        recorded = manifest(args, launch, 147456)
        assert recorded["cost_observations"] == expected
        assert recorded["capacity"]["host_pool_bytes"] == (0 if backend == "native" else 1024**3)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    ("device", "external", "bytes_loaded", "expected"),
    [
        (0, 0, 0, "miss"),
        (1, 0, 0, "hbm"),
        (0, 0, 512, "external"),
        (1, 0, 512, "mixed"),
        (0, 1, 0, "external"),
        (1, 1, 0, "mixed"),
    ],
)
def test_source_requires_evidence(engine, device, external, bytes_loaded, expected):
    sample = {
        "usage": {"cached_tokens_details": {"device": device, "host": external}},
        "metrics_delta": {
            "vllm:prefix_cache_hits_total": device,
            "vllm:external_prefix_cache_hits_total": external,
        },
        "manager_delta": {"orbitkv_load_bytes_total": bytes_loaded},
    }
    assert cache_source(engine, sample) == expected
    if expected == "miss":
        sample["usage"].update(cached_tokens=1, prompt_tokens_details={"cached_tokens": 1})
        assert cache_source(engine, sample) == "unverified"


def test_unused_nan_metrics_do_not_poison_json(monkeypatch):
    monkeypatch.setattr(
        "benches.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text=(
                'unused NaN\ncounter{worker="a"} 10\ncounter{worker="b"} 5\nidle +Inf\n'
                "orbitkv_query_reserved_bytes 400\n"
                "orbitkv_query_speculative_reserved_bytes 200\n"
                "orbitkv_cache_protected_bytes 300\n"
                'orbitkv_ssd_write_admission_skips_total{reason="cold"} 2\n'
                'orbitkv_ssd_write_admission_skips_total{reason="pending"} 3\n'
                'orbitkv_query_reserved_bytes_by_phase{phase="warming"} 100\n'
                'orbitkv_query_reserved_bytes_by_phase{phase="ready"} 200\n'
                'orbitkv_query_reserved_bytes_by_phase{phase="preloading"} 100\n'
                'orbitkv_query_reserved_bytes_by_phase{phase="prepared"} 60\n'
                'orbitkv_warmup_wait_byte_seconds_total{outcome="restored"} 300\n'
                'orbitkv_warmup_wait_byte_seconds_total{outcome="unused"} 40\n'
                'orbitkv_remote_stage_duration_seconds_count{stage="read",status="ok"} 3\n'
                'orbitkv_remote_stage_duration_seconds_count{stage="release",status="ok"} 3\n'
                'orbitkv_remote_stage_duration_seconds_sum{stage="release",status="ok"} 0.012\n'
            ),
            raise_for_status=lambda: None,
        ),
    )
    observed = metrics("http://localhost")
    assert observed == {
        "counter": 15,
        "orbitkv_remote_stage_duration_seconds_count": 6,
        "orbitkv_remote_stage_duration_seconds_count_read": 3,
        "orbitkv_remote_stage_duration_seconds_count_release": 3,
        "orbitkv_remote_stage_duration_seconds_sum": 0.012,
        "orbitkv_remote_stage_duration_seconds_sum_release": 0.012,
        "orbitkv_query_reserved_bytes": 400,
        "orbitkv_query_reserved_bytes_by_phase": 460,
        "orbitkv_query_reserved_bytes_warming": 100,
        "orbitkv_query_speculative_reserved_bytes": 200,
        "orbitkv_cache_protected_bytes": 300,
        "orbitkv_ssd_write_admission_skips_total": 5,
        "orbitkv_ssd_write_admission_skips_total_cold": 2,
        "orbitkv_ssd_write_admission_skips_total_pending": 3,
        "orbitkv_warmup_wait_byte_seconds_total": 340,
        "orbitkv_warmup_wait_byte_seconds_total_restored": 300,
        "orbitkv_warmup_wait_byte_seconds_total_unused": 40,
    }
    json.dumps(observed, allow_nan=False)


@pytest.mark.parametrize(
    "flag",
    [
        ["--queue-warmup", "on"],
        ["--trace-transfers"],
        ["--cache-protected-percent", "80"],
        ["--ssd-write-policy", "reuse"],
        ["--ssd-read-path", "uring"],
        ["--orbitkv-transfer-backend", "kernel"],
    ],
)
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_non_orbitkv_runs_reject_inapplicable_controls(monkeypatch, tmp_path, flag, engine):
    from benches.single_node import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--engine",
            engine,
            "--backend",
            "native",
            "--output",
            str(tmp_path / "run"),
            "--model",
            "/missing",
            *flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_fixed_backend_cli_reaches_configuration_without_runtime_start(
    tmp_path, monkeypatch, engine
):
    from benches.single_node import main

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 1}
        )
    )

    def inspect_configuration(args, bytes_per_token):
        assert args.engine == engine
        assert args.orbitkv_transfer_backend == "kernel"
        raise RuntimeError("configuration reached")

    monkeypatch.setattr("benches.single_node.configure", inspect_configuration)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--engine",
            engine,
            "--backend",
            "orbitkv",
            "--model",
            str(model),
            "--output",
            str(tmp_path / "run"),
            "--orbitkv-transfer-backend",
            "kernel",
        ],
    )
    with pytest.raises(RuntimeError, match="configuration reached"):
        main()


@pytest.mark.parametrize("backend", [None, "direct", "kernel"])
def test_report_preserves_requested_transfer_backend(tmp_path, monkeypatch, backend):
    from benches.report import main

    args = {
        "engine": "sglang",
        "backend": "orbitkv",
        "gpu_tokens": 8192,
        "host_gib": 1,
        "orbitkv_transfer_backend": backend,
        "observation_mode": "sampled",
    }
    run = {
        "directory": "fixture",
        "manifest": {"arguments": args},
        "summary": [{"cache_sources": {}}],
    }
    monkeypatch.setattr("benches.report.collect_run", lambda path: run)
    output = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["report", "fixture", "--output", str(output)])
    main()
    with (output / "summary.csv").open() as report:
        row = next(csv.DictReader(report))
    assert row["orbitkv_transfer_backend"] == (backend or "")
    saved = json.loads((output / "summary.json").read_text())[0]
    assert saved["manifest"]["arguments"]["orbitkv_transfer_backend"] == backend


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "read_counter", ["orbitkv_ssd_prefetch_bytes_total", "orbitkv_ssd_cufile_read_bytes_total"]
)
def test_disk_prefetch_without_gpu_load_is_not_a_cache_hit(engine, read_counter):
    sample = {
        "usage": {},
        "metrics_delta": {},
        "manager_delta": {read_counter: 1024},
        "length": 64,
        "phase": "after_host_eviction",
        "ttft_ms": 10,
        "e2e_ms": 20,
        "matches_cold_output": True,
    }
    sample["cache_source"] = cache_source(engine, sample)
    result = summarize([sample], [64])[0]
    assert result["cache_sources"] == {"miss": 1}
    assert result["ssd_reads_without_gpu_restore"] == 1
    assert result["orbitkv_ssd_read_bytes"] == 1024
    assert result[read_counter] == 1024
    assert (
        result["orbitkv_ssd_prefetch_bytes_total"] + result["orbitkv_ssd_cufile_read_bytes_total"]
        == 1024
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--ssd-read-path", "uring"],
        ["--ssd-gib", "4", "--ssd-read-path", "cufile"],
        [
            "--ssd-gib",
            "4",
            "--ssd-backend",
            "cufile",
            "--ssd-read-path",
            "uring",
            "--gds-stats",
            "/gds_stats",
        ],
    ],
)
def test_read_path_rejects_missing_ssd_capability_and_native_claims_for_host_reads(
    monkeypatch, tmp_path, flags
):
    from benches.single_node import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--engine",
            "vllm",
            "--backend",
            "orbitkv",
            "--output",
            str(tmp_path / "run"),
            "--model",
            "/missing",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("read_path", ["uring", "cufile"])
@pytest.mark.parametrize("preparation", ["--queue-warmup", "--prepare-requests"])
def test_explicit_read_path_rejects_preparation_before_model_or_runtime_access(
    monkeypatch, capsys, tmp_path, read_path, preparation
):
    from benches.single_node import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--engine",
            "vllm",
            "--backend",
            "orbitkv",
            "--output",
            str(tmp_path / "run"),
            "--model",
            "/missing",
            "--ssd-gib",
            "4",
            "--ssd-backend",
            "cufile",
            "--ssd-read-path",
            read_path,
            preparation,
            "on",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "cannot separate demand and preparation reads" in capsys.readouterr().err


def test_report_keeps_output_mismatches_and_rejects_partial_runs(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": {
                    "engine": "vllm",
                    "backend": "native",
                    "lengths": [64],
                    "repeats": 1,
                    "observation_mode": "boundary",
                },
                "observation_mode": "boundary",
                "manager_present": False,
                "manager_backend": None,
            }
        )
    )
    samples = [
        {
            "length": 64,
            "repeat": 0,
            "phase": phase,
            "ttft_ms": 5,
            "e2e_ms": 10,
            "matches_cold_output": phase == "cold",
            "usage": {},
            "metrics_delta": {},
            "manager_delta": {},
            "observation_mode": "boundary",
            "manager_present": False,
            "manager_backend": None,
            "manager_before": {},
            "manager_after": {},
            "memory_samples": 0,
            "sampled_peak_bytes": None,
        }
        for phase in ("cold", "hbm_hit", "after_pressure")
    ]
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text("\n".join(map(json.dumps, samples)))
    assert sum(s["output_mismatches"] for s in collect_run(tmp_path)["summary"]) == 2
    for invalid in (samples[:2], [*samples, samples[0]]):
        sample_file.write_text("\n".join(map(json.dumps, invalid)))
        with pytest.raises(ValueError, match="Incomplete or duplicated"):
            collect_run(tmp_path)
    sample_file.write_text("\n".join(map(json.dumps, samples)))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arguments"]["ssd_gib"] = 16
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        collect_run(tmp_path)
    samples.append({**samples[-1], "phase": "after_host_eviction"})
    sample_file.write_text("\n".join(map(json.dumps, samples)))
    (tmp_path / "storage.json").write_text('{"direct_io":true}')
    assert len(collect_run(tmp_path)["summary"]) == 4


def test_concurrent_report_counts_each_batch_once_and_rejects_missing_requests():
    from benches.concurrent import PATTERNS, summarize, validate

    args = {"concurrencies": [4], "repeats": 1, "ssd_gib": 0}
    samples, batches = [], []
    for pattern in PATTERNS:
        for phase in ("cold", "after_pressure"):
            identity = {"concurrency": 4, "pattern": pattern, "repeat": 0, "phase": phase}
            for index in range(4):
                samples.append(
                    {
                        **identity,
                        "index": index,
                        "ttft_ms": 10,
                        "e2e_ms": 20,
                        "usage": {"completion_tokens": 16},
                        "cached_tokens": 64,
                        "matches_cold_output": index != 0,
                    }
                )
            batches.append(
                {
                    **identity,
                    "wall_seconds": 0.1,
                    "sampled_peak_bytes": {},
                    "manager_delta": {"orbitkv_ssd_prefetch_bytes_total": 1024},
                }
            )
    validate(args, samples, batches)
    summary = summarize(samples, batches)
    assert all(row["orbitkv_ssd_prefetch_bytes_total"] == 1024 for row in summary)
    assert all(row["n"] == 4 and row["output_mismatches"] == 1 for row in summary)
    for broken in (samples[:-1], [*samples, samples[0]]):
        with pytest.raises(ValueError, match="Incomplete or duplicated concurrent requests"):
            validate(args, broken, batches)
    batches[0]["wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="Invalid concurrent wall time"):
        validate(args, samples, batches)
