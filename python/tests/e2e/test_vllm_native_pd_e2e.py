"""Official vLLM 0.33.0 NIXL P/D + MultiConnector + upstream Router.

Two TP=1 processes share one GPU using NIXL/UCX. Cache-first and P/D-first modes
exercise exclusive destination writes through native MultiConnector selection.
This gate does not qualify RDMA or parallelism changes.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.vllm_helpers import VLLMServer

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.parametrize("channel_server", [{"tier": "dram", "pool_size": "1gb"}], indirect=True)
@pytest.mark.parametrize(
    "cache_order",
    [None, "save_only", "pd_first", "cache_first", "pd_first_pressure"],
    ids=["pd", "save-only", "pd-first", "cache-first", "preemption"],
)
def test_native_pd_nixl_matches_monolithic(model, tmp_path, channel_server, cache_order):
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not Path(model).is_dir():
        pytest.skip("requires one GPU and --model pointing to a local dense model")
    router_binary = shutil.which("vllm-router")
    assert router_binary, "install the S5.4 pinned vllm-router in PATH"

    plugin_dir = tmp_path / "orbitkv_source_plugin-0.0.dist-info"
    plugin_dir.mkdir()
    (plugin_dir / "METADATA").write_text("Name: orbitkv-source-plugin\nVersion: 0.0\n")
    (plugin_dir / "entry_points.txt").write_text(
        "[vllm.general_plugins]\norbitkv = orbitkv.vllm.plugin:register\n"
    )

    pressure = cache_order == "pd_first_pressure"
    common = {
        "model": model,
        "max_model_len": 1536 if pressure else 2048,
        "gpu_memory_utilization": 0.4,
        "prefix_caching": False,
        "env_overrides": {
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_LOG_STATS_INTERVAL": "1",
            "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]),
        },
        "extra_args": [
            "--host",
            "127.0.0.1",
            "--dtype",
            "bfloat16",
            "--block-size",
            "64",
            "--kv-cache-memory-bytes",
            str(576 * 1024**2),
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "512",
            "--seed",
            "42",
            "--attention-config.flash_attn_version",
            "2",
            "--enforce-eager",
            "--enable-log-requests",
        ],
    }
    if pressure:
        common["extra_args"].extend(["--num-gpu-blocks-override", "32"])
    bodies = [
        {
            "model": model,
            "prompt": [100 + i % 17 + index for i in range(length)],
            "max_tokens": 400 if pressure else 16,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 42,
            "logprobs": 1,
            "return_token_ids": True,
        }
        for index, length in enumerate((769, 769) if pressure else (129, 257, 769))
    ]

    def launch(stack, name, config=None, side_channel=None):
        options = dict(common)
        if side_channel is not None:
            options["env_overrides"] = {
                **common["env_overrides"],
                "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel),
            }
        server = VLLMServer(
            port=find_available_port(),
            log_file=tmp_path / f"{name}.log",
            kv_transfer_config=config,
            server_label=name,
            **options,
        )
        stack.callback(server.__exit__, None, None, None)
        server.__enter__()
        return f"http://127.0.0.1:{server.port}"

    def generate(url, body):
        response = requests.post(url + "/v1/completions", json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    with contextlib.ExitStack() as stack:
        native = launch(stack, "monolithic")
        controls = [generate(native, body) for body in bodies]
        if pressure:
            continuations = [
                {
                    **body,
                    "prompt": body["prompt"] + output["choices"][0]["token_ids"],
                    "max_tokens": 8,
                }
                for body, output in zip(bodies, controls, strict=True)
            ]
            continuation_controls = [generate(native, body) for body in continuations]
            (tmp_path / "monolithic-continuations.json").write_text(
                json.dumps(continuation_controls, indent=2)
            )
        (tmp_path / "monolithic.json").write_text(json.dumps(controls, indent=2))

    evidence_root = tmp_path
    for restart in range(2 if cache_order else 1):
        if pressure and restart:
            bodies, controls = continuations, continuation_controls
        tmp_path = evidence_root / f"pd-processes-{restart}"
        tmp_path.mkdir()
        with contextlib.ExitStack() as stack:
            urls = {}
            for role in ("producer", "consumer"):
                native_config = {
                    "kv_connector": "NixlConnector",
                    "kv_role": f"kv_{role}",
                    "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {"backends": ["UCX"]},
                }
                config = native_config
                if cache_order:
                    cache_config = {
                        "kv_connector": "OrbitKVConnector",
                        "kv_connector_module_path": "orbitkv.vllm",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "orbitkv.bootstrap_socket": channel_server.bootstrap_socket,
                            "orbitkv.transfer_backend": "direct",
                            "orbitkv.mode": (
                                "save_only"
                                if role == "consumer" and cache_order == "save_only"
                                else "read_write"
                            ),
                        },
                    }
                    children = [cache_config, native_config]
                    if role == "consumer" and cache_order in {
                        "save_only",
                        "pd_first",
                        "pd_first_pressure",
                    }:
                        children.reverse()
                    config = {
                        "kv_connector": "MultiConnector",
                        "kv_role": "kv_both",
                        "kv_load_failure_policy": "fail",
                        "kv_connector_extra_config": {"connectors": children},
                    }
                urls[role] = launch(stack, role, config, find_available_port())

            router_port = find_available_port()
            command = [
                router_binary,
                "--vllm-pd-disaggregation",
                "--kv-connector",
                "nixl",
                "--prefill",
                urls["producer"],
                "--decode",
                urls["consumer"],
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
            ]
            (tmp_path / "router-command.json").write_text(json.dumps(command, indent=2))
            with (tmp_path / "router.log").open("w") as log:
                router = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
                )

            def stop_router(process=router, directory=tmp_path):
                forced = False
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        forced = True
                        process.kill()
                        process.wait(timeout=5)
                cleanup = {
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "forced_kill": forced,
                }
                (directory / "router-cleanup.json").write_text(json.dumps(cleanup, indent=2))
                assert not forced and process.returncode in {0, -15}, cleanup

            stack.callback(stop_router)
            router_url = f"http://127.0.0.1:{router_port}"
            deadline = time.monotonic() + 60
            while True:
                assert router.poll() is None, (tmp_path / "router.log").read_text()[-6000:]
                try:
                    if requests.get(router_url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, "native Router did not become ready"
                time.sleep(0.2)

            for repetition in range(2 if cache_order and not pressure else 1):
                before = fetch_orbitkv_metrics(channel_server.http_port)
                if pressure:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
                        outputs = list(
                            callers.map(lambda body, url=router_url: generate(url, body), bodies)
                        )
                else:
                    outputs = [generate(router_url, body) for body in bodies]
                after = fetch_orbitkv_metrics(channel_server.http_port)
                (tmp_path / f"pd-{repetition}.json").write_text(json.dumps(outputs, indent=2))
                (tmp_path / f"cache-{repetition}.json").write_text(
                    json.dumps({"before": before, "after": after}, indent=2)
                )
                for control, output in zip(controls, outputs, strict=True):
                    actual, expected = output["choices"][0], control["choices"][0]
                    assert actual["token_ids"] == expected["token_ids"]
                    assert actual["text"] == expected["text"]
                    assert actual["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
                    assert actual["finish_reason"] == expected["finish_reason"]
                if cache_order:
                    assert after.get("orbitkv_save_bytes_total", 0) > 0
                    if repetition or restart:
                        assert after.get("orbitkv_load_bytes_total", 0) > before.get(
                            "orbitkv_load_bytes_total", 0
                        )
                    assert after.get("orbitkv_load_failures_total", 0) == 0

            if cache_order == "save_only":
                cancellation = {**bodies[-1], "max_tokens": 1024, "stream": True}
                cancellation["prompt"] = [token + 31 for token in cancellation["prompt"]]
                with requests.post(
                    router_url + "/v1/completions",
                    json=cancellation,
                    stream=True,
                    timeout=120,
                ) as stream:
                    stream.raise_for_status()
                    chunks = []
                    token_count = 0
                    for line in stream.iter_lines(chunk_size=1):
                        if not line.startswith(b"data:"):
                            continue
                        assert b"[DONE]" not in line, "request finished before cancellation"
                        chunk = json.loads(line.removeprefix(b"data:"))
                        chunks.append(chunk)
                        token_count += len(chunk["choices"][0].get("token_ids") or [])
                        if token_count >= 3:
                            break
                    assert token_count >= 3
                    cancel_id = chunks[-1]["id"]
                    (tmp_path / "cancel-chunks.json").write_text(json.dumps(chunks, indent=2))
                deadline = time.monotonic() + 30
                while True:
                    response = requests.get(urls["consumer"] + "/metrics", timeout=5)
                    response.raise_for_status()
                    gauges = re.findall(
                        r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([\d.eE+-]+)$",
                        response.text,
                        re.MULTILINE,
                    )
                    (tmp_path / "cancel-last-metrics.txt").write_text(response.text)
                    aborted = any(
                        "Aborted request(s)" in line and cancel_id in line
                        for line in (tmp_path / "consumer.log").read_text().splitlines()
                    )
                    if aborted and gauges and all(float(value) == 0 for value in gauges):
                        (tmp_path / "cancel-drained-metrics.txt").write_text(response.text)
                        break
                    assert time.monotonic() < deadline, "client cancellation did not drain"
                    time.sleep(0.2)
                recovered = generate(router_url, bodies[0])
                assert (
                    recovered["choices"][0]["token_ids"] == controls[0]["choices"][0]["token_ids"]
                )
                (tmp_path / "after-cancel.json").write_text(json.dumps(recovered, indent=2))

            if pressure and not restart:
                response = requests.get(urls["consumer"] + "/metrics", timeout=5)
                response.raise_for_status()
                (tmp_path / "decode-metrics.txt").write_text(response.text)
                preemptions = sum(
                    float(value)
                    for value in re.findall(
                        r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$",
                        response.text,
                        re.MULTILINE,
                    )
                )
                assert preemptions > 0, "the pressure gate did not preempt a native request"

            deadline = time.monotonic() + 30
            while True:
                response = requests.get(urls["consumer"] + "/metrics", timeout=5)
                response.raise_for_status()
                (tmp_path / "nixl-metrics.txt").write_text(response.text)
                completed = sum(
                    float(value)
                    for value in re.findall(
                        r"^vllm:nixl_bytes_transferred_count(?:\{[^}]*\})?\s+([\d.eE+-]+)$",
                        response.text,
                        re.MULTILINE,
                    )
                )
                if completed >= (0 if cache_order == "cache_first" else len(bodies)):
                    break
                assert time.monotonic() < deadline, response.text
                time.sleep(0.2)
            if cache_order != "cache_first":
                transferred = re.findall(
                    r"^vllm:nixl_bytes_transferred_sum(?:\{[^}]*\})?\s+([\d.eE+-]+)$",
                    response.text,
                    re.MULTILINE,
                )
                assert transferred and sum(map(float, transferred)) > 0
            for metric in ("failed_transfers", "failed_notifications"):
                values = re.findall(
                    rf"^vllm:nixl_num_{metric}_total(?:\{{[^}}]*\}})?\s+([\d.eE+-]+)$",
                    response.text,
                    re.MULTILINE,
                )
                assert values and all(float(value) == 0 for value in values), response.text
            if pressure and restart:
                producer_log = (tmp_path / "producer.log").read_text()
                hits = [
                    int(value)
                    for value in re.findall(
                        r"cache_lookup(?:_reuse)?:.*?hit_tokens=(\d+)", producer_log
                    )
                ]
                assert len(hits) >= len(bodies) and min(hits) >= 1152, producer_log[-6000:]
            for role in ("producer", "consumer"):
                role_log = (tmp_path / f"{role}.log").read_text()
                assert "Nixl" in role_log
