"""Two-host GPU byte recovery, independent of model/kernel numerical differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from .artifacts import external_path
from .metrics import metrics
from .shared_cache import drain, synchronize
from .workload import evict_host_cache

LAYERS = 4
BLOCKS = 64
SEGMENT_BYTES = 65536
SOURCE_PAGES = list(range(0, 32, 2))
DESTINATION_PAGES = list(range(31, 0, -2))
PAYLOAD_BYTES = LAYERS * 2 * len(SOURCE_PAGES) * SEGMENT_BYTES


def worker(args: argparse.Namespace) -> None:
    import torch

    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading

    segment_bytes = args.segment_bytes
    payload_bytes = LAYERS * 2 * len(SOURCE_PAGES) * segment_bytes
    names = [f"layer_{index}" for index in range(LAYERS)]
    source_hashes = [
        hashlib.sha256(f"byte-fixture-{index}".encode()).digest()
        for index in range(len(SOURCE_PAGES))
    ]
    hashes = BlockHashes(source_hashes)
    actor = (
        {
            "pid": os.getpid(),
            "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "instance": args.instance,
            "namespace": args.namespace,
        }
        if args.profile
        else None
    )
    pattern = (
        torch.arange(LAYERS * 2 * BLOCKS * segment_bytes, dtype=torch.int32)
        .remainder_(251)
        .to(torch.uint8)
        .reshape(LAYERS, 2, BLOCKS, segment_bytes)
    )
    tensors = [
        torch.full((2, BLOCKS, segment_bytes), 165, dtype=torch.uint8, device="cuda") for _ in names
    ]
    client = CacheManagerClient(args.manager_socket)
    registered = False
    try:
        registration = client.register_context_batch(
            args.instance,
            args.namespace,
            0,
            0,
            1,
            1,
            0,
            names,
            [BLOCKS] * LAYERS,
            [segment_bytes] * LAYERS,
            [BLOCKS * segment_bytes] * LAYERS,
            [2] * LAYERS,
            "direct",
            False,
            tensors=tensors,
        )
        registered = True
        client.start_session_watcher(args.instance, args.namespace, 1, 1)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("fixture command must contain 1..1024 bytes")
                    operation = json.loads(self.rfile.read(length))["operation"]
                    if operation == "save":
                        for tensor, value in zip(tensors, pattern, strict=True):
                            tensor.copy_(value)
                        torch.cuda.synchronize()
                        ok, message = client.save(
                            registration,
                            [(name, SOURCE_PAGES, source_hashes) for name in names],
                        )
                        if not ok:
                            raise RuntimeError(message)
                        result = {"saved_bytes": payload_bytes}
                    elif operation == "load":
                        clear_started = time.monotonic_ns()
                        for tensor in tensors:
                            tensor.fill_(165)
                        if args.profile:
                            torch.cuda.synchronize()
                        query_started = time.monotonic_ns()
                        native_query_ns = poll_sleep_ns = polls = 0
                        deadline = time.monotonic() + 40
                        request = f"byte-load-{time.time_ns()}"
                        while True:
                            call_started = time.monotonic_ns()
                            ready = client.query_prefetch(args.instance, hashes, req_id=request)
                            native_query_ns += time.monotonic_ns() - call_started
                            polls += 1
                            if not isinstance(ready, QueryLoading):
                                break
                            if time.monotonic() >= deadline:
                                raise TimeoutError("fixture query did not complete")
                            sleep_started = time.monotonic_ns()
                            time.sleep(0.01)
                            poll_sleep_ns += time.monotonic_ns() - sleep_started
                        restore_started = time.monotonic_ns()
                        if ready.num_hit_blocks != len(SOURCE_PAGES):
                            if ready.lease:
                                client.release(ready.lease)
                            raise AssertionError(f"Expected all pages, got {ready.num_hit_blocks}")
                        handle = client.start_restore(
                            args.instance,
                            0,
                            0,
                            [names],
                            [(ready.lease, [DESTINATION_PAGES])],
                            ready_stream=torch.cuda.current_stream().cuda_stream,
                        )
                        submitted = time.monotonic_ns()
                        status = client.wait_restore(handle, timeout=30)
                        completed = time.monotonic_ns()
                        if not status.done or not status.success:
                            raise RuntimeError(status.message or "GPU Restore did not complete")
                        digests = []
                        for layer, tensor in enumerate(tensors):
                            actual = tensor.cpu()
                            expected = torch.full_like(actual, 165)
                            expected[:, DESTINATION_PAGES] = pattern[layer, :, SOURCE_PAGES]
                            if not torch.equal(actual, expected):
                                raise AssertionError(
                                    f"Layer {layer}: data or sentinel gaps changed"
                                )
                            digests.append(hashlib.sha256(actual.numpy().tobytes()).hexdigest())
                        result = {"checked_payload_bytes": payload_bytes, "sha256": digests}
                        if args.profile:
                            result.update(
                                query_id=request,
                                query_window={
                                    **actor,
                                    "clock": "CLOCK_MONOTONIC",
                                    "begin_mono_ns": query_started,
                                    "ready_mono_ns": restore_started,
                                },
                                timing={
                                    "clear_ms": (query_started - clear_started) / 1e6,
                                    "query_ms": (restore_started - query_started) / 1e6,
                                    "native_query_calls_ms": native_query_ns / 1e6,
                                    "poll_sleep_ms": poll_sleep_ns / 1e6,
                                    "query_calls": polls,
                                    "restore_submit_ms": (submitted - restore_started) / 1e6,
                                    "restore_wait_ms": (completed - submitted) / 1e6,
                                    "restore_ms": (completed - restore_started) / 1e6,
                                    "query_restore_ms": (completed - query_started) / 1e6,
                                    "oracle_ms": (time.monotonic_ns() - completed) / 1e6,
                                },
                            )
                    else:
                        raise ValueError(f"Unknown fixture operation: {operation}")
                    status_code, body = 200, json.dumps(result)
                except Exception:
                    status_code, body = 500, traceback.format_exc()
                self.send_response(status_code)
                self.end_headers()
                self.wfile.write(body.encode())

        class Server(HTTPServer):
            address_family = socket.AF_INET6 if ":" in args.host else socket.AF_INET

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        with Server((args.host, args.port), Handler) as server:
            server.serve_forever()
    finally:
        try:
            if registered:
                client.unregister_context(args.instance)
        finally:
            client.close()
            registration = None


def qualify(args: argparse.Namespace) -> dict:
    if args.source_url == args.target_url or args.source_manager == args.target_manager:
        raise ValueError("Byte recovery requires independent workers and Managers")

    def command(url: str, operation: str) -> dict:
        response = requests.post(url, json={"operation": operation}, timeout=60)
        if not response.ok:
            raise RuntimeError(response.text)
        return response.json()

    command(args.source_url, "save")
    synchronize(args.source_manager, args.target_manager)
    rows = []
    for source, target, worker_url in [
        (args.source_manager, args.target_manager, args.target_url),
        (args.target_manager, args.source_manager, args.source_url),
    ]:
        before = metrics(target)
        restored = command(worker_url, "load")
        _, after = drain(source, target)
        for label, counter in [
            ("remote_bytes", "orbitkv_remote_fetch_bytes_total"),
            ("h2d_bytes", "orbitkv_load_bytes_total"),
        ]:
            restored[label] = int(after.get(counter, 0) - before.get(counter, 0))
            if restored[label] != PAYLOAD_BYTES:
                raise AssertionError(f"Expected {PAYLOAD_BYTES} {label}: {restored}")
        release_counter = "orbitkv_remote_stage_duration_seconds_count_release"
        restored["acknowledged_releases"] = int(
            after.get(release_counter, 0) - before.get(release_counter, 0)
        )
        if restored["acknowledged_releases"] <= 0:
            raise AssertionError("Missing acknowledged source release")
        rows.append(restored)
        if len(rows) == 1:
            synchronize(args.target_manager, args.source_manager)
            evict_host_cache(args.source_manager)
            synchronize(args.source_manager, args.target_manager)
    if rows[0]["sha256"] != rows[1]["sha256"]:
        raise AssertionError("Re-served replica changed GPU bytes")
    return {"forward": rows[0], "reverse_from_received_replica": rows[1], "resources_drained": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    local = modes.add_parser("worker", help="start a fixed GPU fixture beside its Manager")
    local.add_argument("--host", default="127.0.0.1")
    local.add_argument("--port", type=int, required=True)
    local.add_argument("--manager-socket", required=True)
    local.add_argument("--instance", required=True)
    local.add_argument("--namespace", required=True)
    local.add_argument(
        "--segment-bytes", type=int, choices=(4096, 65536, 262144), default=SEGMENT_BYTES
    )
    local.add_argument(
        "--profile",
        action="store_true",
        help="time query/restore outside clearing and byte validation",
    )
    run = modes.add_parser("run", help="check forward Restore and re-serving after source eviction")
    for name in ("source-url", "target-url", "source-manager", "target-manager"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    else:
        result = qualify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
