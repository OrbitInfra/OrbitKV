"""Written-only NVMe serving gate: opt-in persistence across Manager restart.

Requires installed artifacts, a matching test-hooks Manager and a dedicated
external --basetemp on a direct NVMe/ext4 mount. This is not a power-loss test.
"""

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

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

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_nvme_manager_restart_under_serving_load(engine, model, tmp_path):
    from transformers import AutoTokenizer

    assert os.environ.get("ORBITKV_REQUIRE_ENGINE_TESTS") == "1"
    assert os.environ.get("ORBITE_REQUIRE_IO_URING") == "1"
    manager_binary = Path(os.environ["ORBITKV_CACHE_MANAGER_BINARY"]).resolve(strict=True)
    filesystem, source = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE,SOURCE", "-T", str(tmp_path)], text=True
    ).split()
    assert filesystem == "ext4" and Path(source).name.startswith("nvme"), (filesystem, source)
    assert json.loads((Path(model) / "config.json").read_text())["model_type"] == "qwen3"
    env = isolated_environment(dict(os.environ), tmp_path)
    env["ORBITKV_CACHE_SCOPE"] = tmp_path.name
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    probe_installation(sys.executable, engine, env, tmp_path, "installed-before", native=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    fragment = tokenizer.encode("Committed cache state survives a Manager incarnation. ")
    prompt = (fragment * (1025 // len(fragment) + 1))[:1025]
    port, http_port, engine_port = (find_available_port() for _ in range(3))
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    env["ORBITKV_TEST_FAULTS"] = str(barriers)
    args = manager_command(sys.executable, port, http_port, "ssd", tmp_path)
    args[0] = str(manager_binary)
    args.append("--ssd-persistent")
    command, options = engine_command(engine, sys.executable, model, engine_port)
    url, manager_url = f"http://127.0.0.1:{engine_port}", f"http://127.0.0.1:{http_port}"
    manager_stack, engine_stack = contextlib.ExitStack(), contextlib.ExitStack()
    results = {
        "power_loss": "unqualified",
        "filesystem": filesystem,
        "source": source,
        "manager": str(manager_binary),
        "manager_sha256": hashlib.file_digest(manager_binary.open("rb"), "sha256").hexdigest(),
    }

    def start_manager(name):
        return manager_stack.enter_context(service(args, manager_url, env, tmp_path, name))

    def start_engine(name):
        return engine_stack.enter_context(service(command + options, url, env, tmp_path, name))

    def until(predicate):
        deadline = time.monotonic() + 180
        while not (value := predicate()):
            assert time.monotonic() < deadline, "required persistence gate timed out"
            time.sleep(0.05)
        return value

    try:
        with service(command, url, env, tmp_path, f"{engine}-native"):
            baseline = complete(engine, url, model, prompt)
        start_manager("manager-original")
        start_engine(f"{engine}-seed")
        compare_output(engine, complete(engine, url, model, prompt), baseline)
        wait_for_drain(http_port)
        assert fetch_orbitkv_metrics(http_port)["orbitkv_ssd_write_bytes_total"] > 0
        engine_stack.close()
        evict_dram_after_ssd_writes(http_port)
        start_engine(f"{engine}-before-manager-restart")
        # A real model load is pending while graceful Manager shutdown starts.
        # The test-hooks barrier withholds completion, not a fabricated receipt.
        (barriers / "ssd.pause").touch()
        with ThreadPoolExecutor(max_workers=2) as executor:
            request = executor.submit(complete, engine, url, model, prompt)
            try:
                until(lambda: (barriers / "ssd.reached").exists())
                shutdown = executor.submit(manager_stack.close)
            finally:
                (barriers / "ssd.pause").unlink()
            shutdown.result(timeout=100)
            try:
                response = request.result(timeout=100)
                compare_output(engine, response, baseline)
                results["interrupted_request"] = "completed correctly"
            except requests.RequestException as error:
                results["interrupted_request"] = str(error)
        engine_stack.close()
        start_manager("manager-restarted")
        until(lambda: "SSD recovery complete:" in (tmp_path / "manager-restarted.log").read_text())
        start_engine(f"{engine}-after-manager-restart")
        before = fetch_orbitkv_metrics(http_port)
        recovered = complete(engine, url, model, prompt)
        compare_output(engine, recovered, baseline)
        assert cached_tokens(engine, recovered) > 0, recovered
        after = wait_for_drain(http_port)
        assert after["orbitkv_ssd_prefetch_bytes_total"] > before.get(
            "orbitkv_ssd_prefetch_bytes_total", 0
        )
        assert after["orbitkv_load_bytes_total"] > before.get("orbitkv_load_bytes_total", 0)
        results.update(recovered=recovered, before=before, after=after)
        engine_stack.close()
        env["ORBITKV_CACHE_SCOPE"] += "-cold-control"
        start_engine(f"{engine}-identity-control")
        before = fetch_orbitkv_metrics(http_port)
        cold = complete(engine, url, model, prompt)
        compare_output(engine, cold, baseline)
        assert cached_tokens(engine, cold) == 0
        after = wait_for_drain(http_port)
        assert after.get("orbitkv_load_bytes_total", 0) == before.get("orbitkv_load_bytes_total", 0)
        results["cold_identity"] = cold
    finally:
        (barriers / "ssd.pause").unlink(missing_ok=True)
        engine_stack.close()
        manager_stack.close()
        (tmp_path / "persistence.json").write_text(json.dumps(results, indent=2) + "\n")
