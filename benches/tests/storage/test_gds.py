import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benches.common.launch import STORAGE_CODECS
from benches.storage.gds import (
    benchmark_commands,
    correctness_result,
    native_io_stats,
    qualify_benchmark,
    require_bare_metal,
)


def stats(read=2, write=3, posix=0, extra=""):
    return (
        f"GPU 0 Read: bw=1 n={read} posix={posix} unalign=0 r_sparse=0 err=0 MiB=8 {extra} "
        f"Write: bw=1 n={write} posix=0 unalign=0 err=0 MiB=8 BufRegister: n=1 err=0\n"
        "GPU 1 Read: bw=0 n=0 posix=0 unalign=0 err=0 MiB=0 "
        "Write: bw=0 n=0 posix=0 unalign=0 err=0 MiB=0 BufRegister: n=0 err=0\n"
    )


def test_native_evidence_requires_both_directions_and_rejects_unknown_or_fallback_counters():
    assert native_io_stats(stats())["operations"] == {"read": 2, "write": 3}
    for output in (
        "no statistics",
        stats(read=0),
        stats(write=0),
        stats(posix=1),
        stats(extra="err=1"),
        stats(extra="r_sparse=1"),
        stats(extra="r_inline=1"),
        stats().replace("posix=0", "unknown=0"),
    ):
        with pytest.raises(ValueError):
            native_io_stats(output)


def test_native_codec_matrix_preserves_workloads_and_only_requests_gds_stats_for_ssd(tmp_path):
    args = SimpleNamespace(
        vllm_python=Path("/venv/vllm/python"),
        sglang_python=Path("/venv/sglang/python"),
        model=Path("/models/qwen3"),
        host_gib=1,
        ssd_gib=8,
        storage_codecs=STORAGE_CODECS,
        storage_codec_budget=64 * 1024**2,
        workloads=["serial", "sustained"],
        working_set=12,
        length=4096,
        duration_seconds=20,
        max_requests=512,
        gpu_tokens=8192,
        prefill_tokens=4096,
        output_tokens=16,
        seed=20260920,
        concurrencies=[4],
        gds_tools=Path("/gds/tools"),
    )
    cases = list(benchmark_commands(args, tmp_path))
    assert len(cases) == len({name for name, _ in cases}) == 80
    groups = {}
    for name, command in cases:

        def value(flag, command=command):
            return command[command.index(flag) + 1]

        assert command[:3] == [
            str(getattr(args, f"{value('--engine')}_python")),
            "-m",
            "benches.serving.single_node",
        ]
        assert value("--output") == str(tmp_path / name)
        for flag, expected in {
            "--host-gib": 1,
            "--lengths": 4096,
            "--working-set": 12,
            "--duration-seconds": 20,
            "--max-requests": 512,
            "--gpu-tokens": 8192,
            "--prefill-tokens": 4096,
            "--output-tokens": 16,
            "--seed": 20260920,
            "--storage-codec-budget": 67108864,
        }.items():
            assert value(flag) == str(expected)
        assert ("--gds-stats" in command) == (
            value("--ssd-gib") != "0" and value("--ssd-backend") != "uring"
        )
        assert "--ssd-read-path" not in command
        key = tuple(
            value(flag) for flag in ("--engine", "--workload", "--ssd-gib", "--ssd-backend")
        )
        groups.setdefault(key, []).append(value("--storage-codec"))
    assert all(codecs == list(STORAGE_CODECS) for codecs in groups.values())


@pytest.mark.parametrize("read_path", [None, "cufile"])
def test_qualification_requires_gpu_restore_encoding_and_native_evidence(tmp_path, read_path):
    run = {
        "directory": str(tmp_path),
        "manifest": {
            "arguments": {
                "workload": "sustained",
                "ssd_gib": 8,
                "ssd_backend": "cufile",
                "ssd_read_path": read_path,
                "storage_codec": "ans",
            }
        },
        "summary": [
            {"orbitkv_ssd_cufile_read_bytes_total": 4096, "orbitkv_load_bytes_total": 8192}
        ],
    }
    usage = {
        "manager_delta": {
            "orbitkv_storage_codec_bytes_total_logical": 8192,
            "orbitkv_storage_codec_bytes_total_stored": 4096,
        }
    }
    native = {**native_io_stats(stats()), "backend_fallbacks": 0}
    (tmp_path / "native-io.json").write_text(json.dumps(native))
    evidence = qualify_benchmark(run, usage)
    assert evidence["measured_gpu_load_bytes"] == 8192
    assert evidence["native_read_qualified"]
    with pytest.raises(ValueError, match="encoded-publication"):
        qualify_benchmark(run, {})
    usage["manager_delta"]["orbitkv_storage_codec_decode_failures_total"] = 1
    with pytest.raises(ValueError, match="decode failures"):
        qualify_benchmark(run, usage)
    del usage["manager_delta"]["orbitkv_storage_codec_decode_failures_total"]
    native["backend_fallbacks"] = 1
    (tmp_path / "native-io.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="fallback observed"):
        qualify_benchmark(run, usage)
    run["summary"][0]["orbitkv_load_bytes_total"] = 0
    with pytest.raises(ValueError, match="GPU restore evidence"):
        qualify_benchmark(run, usage)


@pytest.mark.parametrize("workload", ["serial", "sustained"])
def test_cufile_with_uring_reads_is_a_host_control_even_with_native_stats(tmp_path, workload):
    load_key = "orbitkv_load_bytes" if workload == "serial" else "orbitkv_load_bytes_total"
    row = {
        load_key: 8192,
        "orbitkv_ssd_read_bytes": 4096,
        "orbitkv_ssd_prefetch_bytes_total": 4096,
    }
    run = {
        "directory": str(tmp_path),
        "manifest": {
            "arguments": {
                "workload": workload,
                "ssd_gib": 8,
                "ssd_backend": "cufile",
                "ssd_read_path": "uring",
                "storage_codec": "none",
            }
        },
        "summary": [row],
    }
    usage = {"manager_delta": {"orbitkv_ssd_cufile_write_bytes_total": 8192}}
    native = {**native_io_stats(stats()), "backend_fallbacks": 0}
    (tmp_path / "native-io.json").write_text(json.dumps(native))
    evidence = qualify_benchmark(run, usage)
    assert evidence["measured_ssd_read_bytes"] == 4096
    assert evidence["ssd_read_counter"] == "orbitkv_ssd_prefetch_bytes_total"
    assert not evidence["native_read_qualified"]
    assert "native_io" not in evidence
    with pytest.raises(ValueError, match="write evidence"):
        qualify_benchmark(run, {})
    row["orbitkv_ssd_cufile_read_bytes_total"] = 4096
    with pytest.raises(ValueError, match="different path"):
        qualify_benchmark(run, usage)
    row["orbitkv_ssd_prefetch_bytes_total"] = 0
    with pytest.raises(ValueError, match="GPU restore evidence"):
        qualify_benchmark(run, usage)


@pytest.mark.parametrize("read_path", ["uring", "cufile"])
@pytest.mark.parametrize("preparation", ["queue_warmup", "prepare_requests"])
def test_explicit_read_qualification_rejects_preparation_but_preserves_default_controls(
    tmp_path, read_path, preparation
):
    arguments = {
        "workload": "sustained",
        "ssd_gib": 8,
        "ssd_backend": "cufile",
        "ssd_read_path": read_path,
        "storage_codec": "none",
        preparation: "on",
    }
    run = {
        "directory": str(tmp_path),
        "manifest": {"arguments": arguments},
        "summary": [
            {
                "orbitkv_load_bytes_total": 8192,
                "orbitkv_ssd_cufile_read_bytes_total": 4096,
                "orbitkv_ssd_prefetch_bytes_total": 4096,
            }
        ],
    }
    usage = {"manager_delta": {"orbitkv_ssd_cufile_write_bytes_total": 8192}}
    native = {**native_io_stats(stats()), "backend_fallbacks": 0}
    (tmp_path / "native-io.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="cannot separate demand and preparation reads"):
        qualify_benchmark(run, usage)
    arguments["ssd_read_path"] = None
    assert qualify_benchmark(run, usage)["native_read_qualified"]


def test_container_is_rejected_before_any_gpu_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: "0::/container")
    monkeypatch.setattr("benches.storage.gds.subprocess.check_output", lambda *a, **kw: "overlay\n")
    monkeypatch.setattr(
        "benches.storage.gds.subprocess.run",
        lambda *a, **kw: pytest.fail("probe ran before prerequisite check"),
    )
    with pytest.raises(RuntimeError, match="bare-metal"):
        require_bare_metal(tmp_path)


@pytest.mark.parametrize(
    ("cases", "exit_code", "qualified"),
    [
        ("<testcase/>", 0, True),
        ("<testcase/>", 1, False),
        ("<testcase><failure/></testcase><testcase/>", 1, False),
        ("<testcase><error/></testcase>", 1, False),
        ("<testcase><skipped/></testcase>", 0, False),
        ("", 0, False),
    ],
)
def test_skipped_or_failed_correctness_cannot_qualify(tmp_path, cases, exit_code, qualified):
    report = tmp_path / "quality.xml"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert correctness_result(report, exit_code)["qualified"] == qualified
