"""Two official engine installations share one Manager without sharing page layouts."""

import contextlib
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tests.support.cache_manager import evict_dram_after_ssd_writes, find_available_port
from tests.support.installed_artifacts import isolated_environment
from tests.support.installed_serving import (
    cached_tokens,
    compare_output,
    complete,
    engine_command,
    manager_command,
    probe_installation,
    service,
    wait_for_drain,
)
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]
ENGINES = ("vllm", "sglang")


@pytest.mark.parametrize("tier", ["dram", "ssd"])
def test_official_engines_share_manager(tier, model, tmp_path, request, orbitkv_transfer_backend):
    from transformers import AutoTokenizer

    assert Path(model).is_dir(), "shared-Manager gate requires a local dense --model"
    interpreters = {}
    environments = {}
    directories = {}
    before = {}
    for engine in ENGINES:
        configured = request.config.getoption(f"--{engine}-release-python")
        assert configured, f"shared-Manager gate requires --{engine}-release-python"
        # Preserve the venv interpreter path rather than resolving its symlink.
        interpreters[engine] = Path(configured).absolute()
        assert interpreters[engine].is_file(), interpreters[engine]
        directories[engine] = tmp_path / engine
        directories[engine].mkdir()
        env = isolated_environment(dict(os.environ), directories[engine])
        env["ORBITKV_CACHE_SCOPE"] = tmp_path.name
        env["ORBITKV_TRANSFER_BACKEND"] = orbitkv_transfer_backend
        env["VLLM_SERVER_DEV_MODE"] = "1"
        environments[engine] = env
        before[engine] = probe_installation(
            interpreters[engine], engine, env, directories[engine], "installed-before", native=True
        )
    assert (
        before["vllm"]["distributions"]["orbitkv"]["version"]
        == before["sglang"]["distributions"]["orbitkv"]["version"]
    )
    vllm_files = before["vllm"]["distributions"]["orbitkv"]
    sglang_files = before["sglang"]["distributions"]["orbitkv"]
    for relative in (
        "orbitkv-cache-manager-py",
    ):
        assert (
            vllm_files["files"][str(Path(vllm_files["package_root"]) / relative)]
            == sglang_files["files"][str(Path(sglang_files["package_root"]) / relative)]
        ), relative

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    fragment = tokenizer.encode("Two independent engines retain reusable context in one cache. ")
    tokens = (fragment * (769 // len(fragment) + 1))[:769]
    port, http_port = (find_available_port() for _ in range(2))
    manager_url = f"http://127.0.0.1:{http_port}"
    urls = {}
    commands = {}
    options = {}
    for engine in ENGINES:
        engine_port = find_available_port()
        urls[engine] = f"http://127.0.0.1:{engine_port}"
        commands[engine], options[engine] = engine_command(
            engine, interpreters[engine], model, engine_port, orbitkv_transfer_backend
        )
    snapshots = {}
    responses = {}
    concurrent = {}
    baseline = {}

    def snapshot(name):
        observed = wait_for_drain(http_port)
        snapshots[name] = observed
        (tmp_path / f"{name}-metrics.json").write_text(json.dumps(observed, indent=2) + "\n")
        return observed

    def generate(engine, phase):
        started = time.monotonic_ns()
        response = complete(engine, urls[engine], model, tokens)
        ended = time.monotonic_ns()
        record = {"started_ns": started, "ended_ns": ended, "response": response}
        (directories[engine] / f"{phase}.json").write_text(json.dumps(record, indent=2) + "\n")
        return response, record

    def generate_pair(phase):
        barrier = Barrier(2)

        def run(engine):
            barrier.wait(timeout=10)
            return generate(engine, phase)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {engine: executor.submit(run, engine) for engine in ENGINES}
            records = {engine: future.result() for engine, future in futures.items()}
        intervals = {engine: record for engine, (_, record) in records.items()}
        assert max(r["started_ns"] for r in intervals.values()) < min(
            r["ended_ns"] for r in intervals.values()
        ), intervals
        concurrent[phase] = intervals
        return {engine: response for engine, (response, _) in records.items()}

    try:
        for engine in ENGINES:
            with service(
                commands[engine], urls[engine], environments[engine], tmp_path, f"{engine}-native"
            ):
                cold, _ = generate(engine, "native-cold")
                warm, _ = generate(engine, "native-hbm")
                compare_output(engine, warm, cold)
                baseline[engine] = warm

        for engine in ENGINES:
            environments[engine]["ORBITKV_PORT"] = str(port)
            environments[engine]["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
        manager_args = manager_command(interpreters["vllm"], port, http_port, tier, tmp_path)
        with service(
            manager_args, manager_url, environments["vllm"], tmp_path, "manager"
        ) as manager:
            with contextlib.ExitStack() as stack:
                owners = {}
                processes = {}
                for engine in ENGINES:
                    owners[engine] = stack.enter_context(contextlib.ExitStack())
                    processes[engine] = owners[engine].enter_context(
                        service(
                            commands[engine] + options[engine],
                            urls[engine],
                            environments[engine],
                            tmp_path,
                            f"{engine}-cold",
                        )
                    )
                responses["parallel_cold"] = generate_pair("parallel-cold")
                for engine in ENGINES:
                    compare_output(engine, responses["parallel_cold"][engine], baseline[engine])
                    assert cached_tokens(engine, responses["parallel_cold"][engine]) == 0
                snapshot("after_cold")
                assert snapshots["after_cold"].get("orbitkv_save_bytes_total", 0) > 0
                responses["parallel_hbm"] = generate_pair("parallel-hbm")
                for engine in ENGINES:
                    compare_output(engine, responses["parallel_hbm"][engine], baseline[engine])
                    assert cached_tokens(engine, responses["parallel_hbm"][engine]) >= 704
                snapshot("after_hbm")
                for counter in ("orbitkv_load_bytes_total", "orbitkv_hll_total_requests"):
                    assert snapshots["after_cold"].get(counter, 0) == snapshots["after_hbm"].get(
                        counter, 0
                    ), (counter, snapshots)

                for engine in ENGINES:
                    other = next(name for name in ENGINES if name != engine)
                    owners[engine].close()
                    assert processes[engine].poll() is not None
                    assert manager.poll() is None and processes[other].poll() is None
                    snapshot(f"{engine}_stopped")
                    survivor, _ = generate(other, f"while-{engine}-stopped")
                    compare_output(other, survivor, baseline[other])
                    assert cached_tokens(other, survivor) >= 704
                    snapshot(f"{engine}_survivor")
                    for counter in ("orbitkv_load_bytes_total", "orbitkv_hll_total_requests"):
                        assert snapshots[f"{engine}_stopped"].get(counter, 0) == snapshots[
                            f"{engine}_survivor"
                        ].get(counter, 0), (counter, snapshots)
                    if tier == "ssd":
                        resident = snapshot(f"{engine}_before_eviction")[
                            "orbitkv_cache_resident_bytes"
                        ]
                        if resident:
                            evict_dram_after_ssd_writes(http_port)
                        assert fetch_orbitkv_metrics(http_port)["orbitkv_cache_resident_bytes"] == 0
                    snapshot(f"{engine}_before_restart")
                    owners[engine] = stack.enter_context(contextlib.ExitStack())
                    processes[engine] = owners[engine].enter_context(
                        service(
                            commands[engine] + options[engine],
                            urls[engine],
                            environments[engine],
                            tmp_path,
                            f"{engine}-restart",
                        )
                    )
                    restored, _ = generate(engine, "after-restart")
                    compare_output(engine, restored, baseline[engine])
                    assert cached_tokens(engine, restored) >= 704
                    snapshot(f"{engine}_after_restart")
                    start, end = (
                        snapshots[f"{engine}_before_restart"],
                        snapshots[f"{engine}_after_restart"],
                    )
                    assert end.get("orbitkv_load_bytes_total", 0) > start.get(
                        "orbitkv_load_bytes_total", 0
                    ), snapshots
                    if tier == "ssd":
                        assert end.get("orbitkv_ssd_prefetch_bytes_total", 0) > start.get(
                            "orbitkv_ssd_prefetch_bytes_total", 0
                        ), snapshots
                        assert end.get("orbitkv_ssd_cufile_read_bytes_total", 0) == start.get(
                            "orbitkv_ssd_cufile_read_bytes_total", 0
                        ), snapshots
                    assert manager.poll() is None and processes[other].poll() is None

                responses["parallel_after_restart"] = generate_pair("parallel-after-restart")
                for engine in ENGINES:
                    compare_output(
                        engine, responses["parallel_after_restart"][engine], baseline[engine]
                    )
                    assert cached_tokens(engine, responses["parallel_after_restart"][engine]) >= 704
                for engine in ENGINES:
                    owners[engine].close()
                    assert processes[engine].poll() is not None
                snapshot("parallel_stopped")
                if tier == "ssd":
                    if snapshots["parallel_stopped"]["orbitkv_cache_resident_bytes"]:
                        evict_dram_after_ssd_writes(http_port)
                    assert fetch_orbitkv_metrics(http_port)["orbitkv_cache_resident_bytes"] == 0
                snapshot("parallel_before_restore")
                for engine in ENGINES:
                    owners[engine] = stack.enter_context(contextlib.ExitStack())
                    processes[engine] = owners[engine].enter_context(
                        service(
                            commands[engine] + options[engine],
                            urls[engine],
                            environments[engine],
                            tmp_path,
                            f"{engine}-parallel-restore",
                        )
                    )
                restored = generate_pair("parallel-external-restore")
                for engine in ENGINES:
                    compare_output(engine, restored[engine], baseline[engine])
                    assert cached_tokens(engine, restored[engine]) >= 704
                end = snapshot("parallel_after_restore")
                start = snapshots["parallel_before_restore"]
                expected_bytes = sum(
                    snapshots[f"{engine}_after_restart"]["orbitkv_load_bytes_total"]
                    - snapshots[f"{engine}_before_restart"].get("orbitkv_load_bytes_total", 0)
                    for engine in ENGINES
                )
                assert (
                    end["orbitkv_load_bytes_total"] - start.get("orbitkv_load_bytes_total", 0)
                    == expected_bytes
                    > 0
                ), snapshots
                if tier == "ssd":
                    assert (
                        end["orbitkv_ssd_prefetch_bytes_total"]
                        - start.get("orbitkv_ssd_prefetch_bytes_total", 0)
                        == expected_bytes
                    ), snapshots
                    assert end.get("orbitkv_ssd_cufile_read_bytes_total", 0) == start.get(
                        "orbitkv_ssd_cufile_read_bytes_total", 0
                    ), snapshots
            snapshot("final")
            assert snapshots["final"].get("orbitkv_load_failures_total", 0) == 0, snapshots
        assert not Path(f"/tmp/orbitkv-{port}.sock").exists(), "Manager left its UDS behind"
        old_pid = manager.pid
        # Stop engine owners before restarting their Manager; a process restart
        # does not establish native drain for an outstanding transfer.
        with service(
            manager_args, manager_url, environments["vllm"], tmp_path, "manager-restart"
        ) as replacement:
            assert replacement.pid != old_pid
            snapshot("manager_restarted")
            with contextlib.ExitStack() as stack:
                for engine in ENGINES:
                    stack.enter_context(
                        service(
                            commands[engine] + options[engine],
                            urls[engine],
                            environments[engine],
                            tmp_path,
                            f"{engine}-manager-restart",
                        )
                    )
                cold = generate_pair("after-manager-restart")
                for engine in ENGINES:
                    compare_output(engine, cold[engine], baseline[engine])
                    assert cached_tokens(engine, cold[engine]) == 0
                snapshot("manager_restart_cold")
                assert snapshots["manager_restart_cold"].get("orbitkv_load_bytes_total", 0) == 0
                assert snapshots["manager_restart_cold"].get("orbitkv_save_bytes_total", 0) > 0
                warm = generate_pair("manager-restart-hbm")
                for engine in ENGINES:
                    compare_output(engine, warm[engine], baseline[engine])
                    assert cached_tokens(engine, warm[engine]) >= 704
            snapshot("manager_restart_final")
            assert snapshots["manager_restart_final"].get("orbitkv_load_failures_total", 0) == 0
        assert not Path(f"/tmp/orbitkv-{port}.sock").exists(), "replacement left its UDS behind"
        for name in ("manager", "manager-restart"):
            assert f"backend={orbitkv_transfer_backend}" in (tmp_path / f"{name}.log").read_text()
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "tier": tier,
                    "transfer_backend": orbitkv_transfer_backend,
                    "model": model,
                    "profile": "one A100; dense TP=1 PP=1 eager; two official engines; one installed Manager",
                    "model_config_sha256": hashlib.sha256(
                        (Path(model) / "config.json").read_bytes()
                    ).hexdigest(),
                    "manager_pids": [old_pid, replacement.pid],
                    "concurrent": concurrent,
                    "snapshots": snapshots,
                    "manager_restart_policy": "drain engines, cold rebuild, no durable SSD-index claim",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for engine in ENGINES:
            after = probe_installation(
                interpreters[engine],
                engine,
                environments[engine],
                directories[engine],
                "installed-after",
            )
            assert before[engine]["distributions"] == after["distributions"], (
                f"{engine} installation changed"
            )
