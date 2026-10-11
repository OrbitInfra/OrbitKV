"""Codec controls and evidence must stay usable without native/GPU imports."""

import json
import os
import subprocess
import sys
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benches.launch import STORAGE_CODECS, configure, storage_codec_budget
from benches.metrics import codec_summary, delta, measure, metrics
from benches.report import compare_outputs


def launch_arguments(tmp_path, engine="vllm", codec="none", ssd_gib=4):
    return Namespace(
        engine=engine,
        backend="orbitkv",
        installed_artifact=False,
        enforce_eager=False,
        model=tmp_path / "model",
        output=tmp_path,
        host_gib=1,
        ssd_gib=ssd_gib,
        ssd_path=tmp_path / "private/cache.bin",
        ssd_backend="uring",
        ssd_read_path=None,
        ssd_write_policy="all",
        storage_codec=codec,
        storage_codec_budget=64 * 1024**2,
        cache_protected_percent=0,
        query_budget_gib=None,
        queue_warmup="off",
        prepare_requests="off",
        trace_transfers=False,
        deterministic_inference=False,
        read_batch_mib=0,
        read_timeout_ms=0,
        read_max_batches=0,
        workload="sustained",
        lengths=[4096],
        gpu_tokens=8192,
        prefill_tokens=8192,
        orbitkv_transfer_backend=None,
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_absolute_engine_python_resolves_its_jit_tools(tmp_path, monkeypatch, engine):
    engine_bin = tmp_path / "engine/bin"
    inherited_bin = tmp_path / "inherited-bin"
    for directory, label in ((engine_bin, "selected"), (inherited_bin, "inherited")):
        directory.mkdir(parents=True)
        tool = directory / "ninja"
        tool.write_text(f"#!{sys.executable}\nprint({label!r})\n")
        tool.chmod(0o755)
    python = engine_bin / "python"
    python.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(inherited_bin))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/unrelated/shared-cache")
    monkeypatch.setattr("benches.launch.sys.executable", str(python))
    args = launch_arguments(tmp_path, engine, ssd_gib=0)
    args.backend = "native"
    launch = configure(args, 147456)
    result = subprocess.run(
        ["ninja", "--version"], env=launch.env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "selected"
    assert launch.env["TORCH_EXTENSIONS_DIR"] == str(tmp_path / "torch-extensions")
    assert os.environ["PATH"] == str(inherited_bin)
    assert os.environ["TORCH_EXTENSIONS_DIR"] == "/unrelated/shared-cache"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_installed_benchmark_uses_wheel_without_source_or_duplicate_plugin(
    tmp_path, monkeypatch, engine
):
    site = tmp_path / "site-packages"
    package = site / "orbitkv"
    package.mkdir(parents=True)
    (package / "orbitkv-cache-manager-py").touch()
    monkeypatch.setattr(
        "benches.launch.sys.path",
        [
            str(site),
            *(
                path
                for path in sys.path
                if Path(path).name not in {"site-packages", "dist-packages"}
            ),
        ],
    )
    monkeypatch.setattr(
        "benches.launch.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    monkeypatch.delenv("ORBITKV_CACHE_MANAGER_BINARY", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/source/that/must/not/shadow/the/wheel")
    args = launch_arguments(tmp_path, engine)
    args.installed_artifact = True
    launch = configure(args, 147456)
    assert launch.manager_command[0] == str(package / "orbitkv-cache-manager-py")
    assert launch.env["PYTHONPATH"].split(":") == [str(site), str(tmp_path)]
    assert launch.backend_configuration["artifact_mode"] == "installed"
    assert launch.backend_configuration["orbitkv_package_dir"] == str(package)
    assert not (tmp_path / "orbitkv_benchmark-0.0.dist-info").exists()


@pytest.mark.parametrize("origin", [None, "/checkout/python/orbitkv/__init__.py"])
def test_installed_benchmark_rejects_missing_or_shadowed_wheel(tmp_path, monkeypatch, origin):
    monkeypatch.setattr(
        "benches.launch.importlib.util.find_spec",
        lambda name: None if origin is None else SimpleNamespace(origin=origin),
    )
    args = launch_arguments(tmp_path)
    args.installed_artifact = True
    with pytest.raises(RuntimeError, match="complete OrbitKV wheel|outside site-packages"):
        configure(args, 147456)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("eager", [False, True])
def test_matched_execution_mode_uses_released_engine_controls(tmp_path, engine, eager):
    args = launch_arguments(tmp_path, engine)
    args.backend = "native"
    args.enforce_eager = eager
    launch = configure(args, 147456)
    assert launch.backend_configuration["execution_mode"] == (
        "eager" if eager else "engine-default"
    )
    if engine == "vllm":
        assert ("--enforce-eager" in launch.command) == eager
    else:
        for flag in ("--cuda-graph-backend-decode", "--cuda-graph-backend-prefill"):
            assert (flag in launch.command) == eager
            if eager:
                assert launch.command[launch.command.index(flag) + 1] == "disabled"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("codec", STORAGE_CODECS)
@pytest.mark.parametrize("ssd_gib", [0, 4])
def test_codec_launch_uses_same_engine_capacity_and_explicit_manager_policy(
    tmp_path, monkeypatch, engine, codec, ssd_gib
):
    monkeypatch.setattr("benches.launch.free_port", lambda: 23456)
    monkeypatch.setenv("ORBITKV_CACHE_MANAGER_BINARY", "/prebuilt/manager")
    monkeypatch.setenv("ORBITKV_NVCOMP_LIBRARY", "/runtime/libnvcomp.so.5")
    args = launch_arguments(tmp_path, engine, codec, ssd_gib)
    launch = configure(args, 147456)
    command = launch.manager_command
    assert command[0] == "/prebuilt/manager"
    assert command[command.index("--storage-codec") + 1] == codec
    assert command[command.index("--storage-codec-budget") + 1] == str(64 * 1024**2)
    assert launch.env["ORBITKV_NVCOMP_LIBRARY"] == "/runtime/libnvcomp.so.5"
    assert launch.backend_configuration["storage_codec"] == codec
    assert ("--ssd-cache-path" in command) == bool(ssd_gib)
    if ssd_gib:
        assert command[command.index("--ssd-cache-path") + 1] == str(args.ssd_path)
        assert launch.backend_configuration["ssd_backend"] == "uring"
        assert launch.backend_configuration["ssd_read_path"] is None
    assert "--ssd-read-path" not in command
    capacity_flag = "--kv-cache-memory-bytes" if engine == "vllm" else "--max-total-tokens"
    capacity = 147456 * 8192 if engine == "vllm" else 8192
    assert launch.command[launch.command.index(capacity_flag) + 1] == str(capacity)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("read_path", [None, "uring", "cufile"])
def test_ssd_read_path_is_separate_from_cufile_initialization_and_writes(
    tmp_path, monkeypatch, engine, read_path
):
    monkeypatch.setattr("benches.launch.free_port", lambda: 23456)
    args = launch_arguments(tmp_path, engine)
    args.ssd_backend = "cufile"
    args.ssd_read_path = read_path
    launch = configure(args, 147456)
    command = launch.manager_command
    assert command[command.index("--ssd-backend") + 1] == "cufile"
    assert ("--ssd-read-path" in command) == (read_path is not None)
    if read_path is not None:
        assert command[command.index("--ssd-read-path") + 1] == read_path
    assert launch.backend_configuration["io"] == "O_DIRECT"
    assert launch.backend_configuration["ssd_backend"] == "cufile"
    assert launch.backend_configuration["ssd_read_path"] == read_path
    assert "--ssd-read-path" not in launch.command


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("backend", [None, "direct", "kernel"])
def test_fixed_transfer_backend_is_explicit_and_ignores_inherited_environment(
    tmp_path, monkeypatch, engine, backend
):
    monkeypatch.setattr("benches.launch.free_port", lambda: 23456)
    monkeypatch.setenv("ORBITKV_TRANSFER_BACKEND", "inherited-invalid")
    args = launch_arguments(tmp_path, engine)
    args.orbitkv_transfer_backend = backend
    launch = configure(args, 147456)
    assert launch.backend_configuration["orbitkv_transfer_backend"] == backend
    if engine == "sglang":
        assert launch.env.get("ORBITKV_TRANSFER_BACKEND") == backend
    else:
        assert "ORBITKV_TRANSFER_BACKEND" not in launch.env
        connector = json.loads(launch.command[launch.command.index("--kv-transfer-config") + 1])
        extra = connector.get("kv_connector_extra_config", {})
        assert extra.get("orbitkv.transfer_backend") == backend
        assert ("orbitkv.transfer_backend" in extra) == (backend is not None)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("enabled", [False, True])
def test_deterministic_qualification_is_explicit_and_ignores_inherited_environment(
    tmp_path, monkeypatch, engine, enabled
):
    monkeypatch.setattr("benches.launch.free_port", lambda: 23456)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "inherited-invalid")
    args = launch_arguments(tmp_path, engine)
    args.deterministic_inference = enabled
    launch = configure(args, 147456)
    if engine == "vllm":
        assert launch.env.get("VLLM_BATCH_INVARIANT") == ("1" if enabled else None)
        assert ("--attention-backend" in launch.command) == enabled
        if enabled:
            assert launch.command[launch.command.index("--attention-backend") + 1] == "FLASH_ATTN"
    else:
        assert "VLLM_BATCH_INVARIANT" not in launch.env
        assert ("--enable-deterministic-inference" in launch.command) == enabled


def test_codec_budget_accepts_binary_units_and_rejects_out_of_range():
    for value, expected in (("64mb", 67108864), ("4096", 4096), ("1.5 GB", 1610612736)):
        assert storage_codec_budget(value) == expected
    for value in ("0", "4095", "4gb", "-1", "nan", "inf", "64mib"):
        with pytest.raises(ArgumentTypeError):
            storage_codec_budget(value)


def test_codec_labels_batch_histograms_and_retained_workspace_are_not_combined(monkeypatch):
    monkeypatch.setattr(
        "benches.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text="\n".join(
                [
                    'orbitkv_storage_codec_bytes_total{representation="logical"} 4096',
                    'orbitkv_storage_codec_bytes_total{representation="stored"} 1024',
                    'orbitkv_storage_codec_transfer_bytes_total{direction="d2h"} 2048',
                    'orbitkv_storage_codec_transfer_bytes_total{direction="h2d"} 1024',
                    'orbitkv_storage_codec_duration_seconds_sum{operation="encode"} 0.2',
                    'orbitkv_storage_codec_duration_seconds_sum{operation="decode"} 0.1',
                    'orbitkv_storage_codec_batches_total{operation="encode"} 2',
                    'orbitkv_storage_codec_batches_total{operation="decode"} 3',
                    'orbitkv_storage_codec_batch_segments_sum{operation="encode"} 64',
                    'orbitkv_storage_codec_batch_segments_count{operation="encode"} 2',
                    'orbitkv_storage_codec_batch_segments_bucket{operation="encode",le="64"} 2',
                    'orbitkv_storage_codec_skips_total{reason="range_ratio_or_budget"} 4',
                    "orbitkv_storage_codec_workspace_bytes 67108864",
                    "orbitkv_storage_codec_workspace_allocations_total 1",
                    "orbitkv_storage_codec_reserved_bytes 0",
                ]
            ),
            raise_for_status=lambda: None,
        ),
    )
    after = metrics("http://manager")
    before = {"orbitkv_storage_codec_workspace_allocations_total": 1}
    row = codec_summary(
        [
            {
                "manager_before": before,
                "manager_after": after,
                "manager_delta": delta(before, after),
                "sampled_peak_bytes": {"orbitkv_storage_codec_reserved_bytes": 134217728},
            }
        ]
    )
    assert row["encoded_publication_stored_fraction"] == 0.25
    assert "orbitkv_storage_codec_bytes_total" not in row
    assert row["orbitkv_storage_codec_duration_seconds_sum_encode"] == 0.2
    assert row["orbitkv_storage_codec_duration_seconds_sum_decode"] == 0.1
    assert row["orbitkv_storage_codec_batches_total_encode"] == 2
    assert row["orbitkv_storage_codec_batches_total_decode"] == 3
    assert row["orbitkv_storage_codec_batch_segments_sum_encode"] == 64
    assert row["orbitkv_storage_codec_batch_segments_count_encode"] == 2
    assert row["orbitkv_storage_codec_workspace_allocations_total"] == 0
    assert row["sampled_peak_codec_reserved_bytes"] == 134217728
    assert row["max_codec_reserved_bytes_after"] == 0
    assert row["max_codec_workspace_bytes_after"] == 67108864
    assert not any("bucket" in key for key in row)
    old = codec_summary([{"manager_delta": {"orbitkv_storage_codec_bytes_total_logical": 4096}}])
    assert old["encoded_publication_stored_fraction"] is None
    assert old["sampled_peak_codec_workspace_bytes"] is None
    assert "orbitkv_storage_codec_workspace_allocations_total" not in old
    assert "orbitkv_storage_codec_batches_total_encode" not in old


def test_none_reference_retains_output_drift_and_requires_verified_inputs(tmp_path):
    args = {
        "engine": "vllm",
        "workload": "sustained",
        "seed": 42,
        "storage_codec": "none",
        "observation_mode": "sampled",
    }
    control = tmp_path / "none"
    control.mkdir()
    samples = [
        {"concurrency": 4, "index": i, "length": 64, "prompt_sha256": str(i), "text": "same"}
        for i in range(2)
    ]
    (control / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    environment = {
        "gpu": "H20",
        "python": "3.11.10",
        "cpu_affinity": [8, 10, 12, 14],
        "packages": {"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.10.4"},
    }
    reference = {"directory": str(control), "manifest": {"arguments": args, **environment}}
    run = {
        "directory": str(tmp_path),
        "manifest": {"arguments": {**args, "storage_codec": "fp8"}, **environment},
        "summary": [{"concurrency": 4}],
    }
    samples[1]["text"] = "changed"
    samples.append({**samples[0], "index": 2})
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    comparison = compare_outputs(run, reference)
    assert comparison["compared_requests"] == 2
    assert comparison["output_mismatches"] == 1
    assert comparison["run_uncompared_requests"] == 1
    assert run["summary"][0]["reference_output_mismatches"] == 1
    del samples[0]["prompt_sha256"]
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    assert compare_outputs(run, reference)["unverified_input_requests"] == 1
    samples[1]["prompt_sha256"] = "different"
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    with pytest.raises(ValueError, match="prompt differs"):
        compare_outputs(run, reference)
    run["manifest"]["arguments"]["seed"] = 43
    with pytest.raises(ValueError, match="different seed"):
        compare_outputs(run, reference)


def test_cross_backend_reference_requires_the_same_engine_runtime(tmp_path):
    args = {
        "engine": "vllm",
        "backend": "native",
        "storage_codec": "none",
        "seed": 42,
        "observation_mode": "sampled",
    }
    sample = {"length": 1024, "phase": "after_pressure", "prompt_sha256": "same", "text": "ok"}
    (tmp_path / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    manifest = {
        "arguments": args,
        "gpu": "H20",
        "python": "3.11.10",
        "cpu_affinity": [8, 10, 12, 14],
        "packages": {"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.10.4"},
        "model_revision": "fixed-model-revision",
        "kv_bytes_per_token": 147456,
    }
    reference = {"directory": str(tmp_path), "manifest": manifest}
    for backend in ("orbitkv", "lmcache", "cpu"):
        run = {
            "directory": str(tmp_path),
            "manifest": {**manifest, "arguments": {**args, "backend": backend}},
            "summary": [{"length": 1024, "phase": "after_pressure"}],
        }
        result = compare_outputs(run, reference)
        assert result["compared_requests"] == 1
        assert result["output_mismatches"] == 0
        for setting, value in (
            ("deterministic_inference", True),
            ("enforce_eager", True),
            ("observation_mode", "boundary"),
        ):
            changed = {
                **run,
                "manifest": {
                    **run["manifest"],
                    "arguments": {**run["manifest"]["arguments"], setting: value},
                },
            }
            with pytest.raises(ValueError, match=f"different {setting}"):
                compare_outputs(changed, reference)
        for field in ("gpu", "python", "cpu_affinity", "model_revision", "kv_bytes_per_token"):
            changed = {**run, "manifest": {**run["manifest"], field: "different"}}
            with pytest.raises(ValueError, match=f"different {field}"):
                compare_outputs(changed, reference)
        for package in ("vllm", "torch", "transformers"):
            changed = {
                **run,
                "manifest": {
                    **run["manifest"],
                    "packages": {**manifest["packages"], package: "different"},
                },
            }
            with pytest.raises(ValueError, match=f"different {package} version"):
                compare_outputs(changed, reference)


def test_idle_retained_codec_workspace_does_not_prevent_measurement_drain(monkeypatch):
    observed = {
        "orbitkv_storage_codec_workspace_bytes": 67108864,
        "orbitkv_storage_codec_reserved_bytes": 0,
    }
    monkeypatch.setattr("benches.metrics.metrics", lambda url: observed)
    monkeypatch.setattr("benches.metrics.time.sleep", lambda seconds: None)
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr("benches.metrics.time.monotonic", lambda: next(ticks))
    with measure("engine", "manager", 0) as measurement:
        pass
    assert measurement["manager_after"] == observed
    summary = codec_summary([measurement])
    assert summary["max_codec_reserved_bytes_after"] == 0
    assert summary["max_codec_workspace_bytes_after"] == 67108864
