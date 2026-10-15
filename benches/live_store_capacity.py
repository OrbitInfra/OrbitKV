"""Same-host multi-Manager real-Publish inventory capacity qualification."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import requests

from tests.integration.test_distributed_cache import (
    _await_fence,
    _await_fence_for_scope,
    _cleanup_dram,
    _discover_storage_namespaces,
    _etcd_keys,
    _metadata,
    _restore,
    _sync,
    _until,
    _wait_for_remote_drain,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

from .artifacts import external_path
from .live_store_measurements import (
    MEASUREMENT_CONTRACT,
    _clock_domain,
    _installation_sample,
    _payload,
    _process_sample,
    _summary,
)
from .scoped_metadata import _etcd_revision


def _binding(manager):
    status = _metadata(manager)
    stream = status["stream"]
    return {
        "node": stream["source_node"],
        "epoch": stream["source_node_epoch"],
        "incarnation": stream["source_incarnation"],
        "scope_digest": stream["scope_digest"],
    }


def _assert_binding(manager, expected):
    actual = _binding(manager)
    assert actual == expected, (expected, actual)
    return actual


def _owner_statuses(manager):
    response = requests.get(
        f"http://127.0.0.1:{manager.http_port}/cache/metadata/owners",
        params={"limit": 128},
        timeout=5,
    )
    response.raise_for_status()
    return {row["owner"]: row for row in response.json()}


def _wait_for_installs(manager, targets, managers, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        rows = _owner_statuses(manager)
        installed = {}
        for node, target in targets.items():
            row = rows.get(target["incarnation"])
            if row and row["applied_sequence"] > target["sequence"]:
                raise AssertionError(f"sample target was superseded: {target}, {row}")
            if (
                row
                and row["fresh"]
                and row["applied_sequence"] == target["sequence"]
                and row["installed_mono_ns"] >= target["published_mono_ns"]
            ):
                installed[node] = row
        if len(installed) == len(targets):
            return installed
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"owner installs did not reach targets: targets={targets}, rows={rows}, "
                f"logs={[manager.read_logs() for manager in managers]}"
            )
        time.sleep(0.001)


def _await_concurrently(observer, fences, scope_digest, managers, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _await_fence_for_scope,
                observer,
                fence,
                scope_digest,
                managers,
                timeout=30,
            )
            for fence in fences.values()
        ]
        for future in futures:
            future.result()


def run(
    output: Path,
    owners: int,
    duration_seconds: int,
    index_budget: str,
    expect_degraded: bool,
    seed: str,
    pages: int,
    skip_restore: bool,
    visibility_mode: str,
    barrier_concurrency: int,
    enforce_thresholds: bool,
    samples: int | None,
    warmup_cycles: int,
):
    import torch

    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    block_bytes = 4096
    payload_bytes = pages * block_bytes
    identity = f"s2.10:capacity:{owners}:{seed}"
    (storage_namespace,) = _discover_storage_namespaces(output, [identity], pages, block_bytes)
    cluster = f"s210-capacity-{owners}-{uuid.uuid4().hex[:8]}"
    result = {
        "measurement_contract": MEASUREMENT_CONTRACT,
        "payload_schema": "u64-generation-u64-block-le-v1",
        "owners": owners,
        "seed": seed,
        "duration_seconds": duration_seconds,
        "requested_samples": samples,
        "warmup_cycles": warmup_cycles,
        "index_budget": index_budget,
        "expect_degraded": expect_degraded,
        "pages": pages,
        "skip_restore": skip_restore,
        "visibility_mode": visibility_mode,
        "barrier_concurrency": barrier_concurrency,
        "thresholds_enforced": enforce_thresholds,
        "storage_namespace": storage_namespace,
        "cluster": cluster,
        "latency_endpoints": {
            "mutation_submit_loop_ms": "all owners: DRAM cleanup, payload copy/CUDA sync and save submission; sequence diagnostic reads occur after all submissions",
            "save_start_to_install_ms": "individual owner CacheManagerClient.save call start through exact terminal-sequence installation",
            "publication_to_install_ms": "terminal burst publication commit to same-sequence owner-view commit; next mutation starts only after observation",
        },
        "artifacts": {
            name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for name, path in {
                "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
                "etcd": os.environ["ETCD_BIN"],
                "extension": native.__file__,
                "tent": str(Path(os.environ["ORBITKV_MOONCAKE_LIB_DIR"]) / "libtent_shared.so"),
            }.items()
        },
    }

    raw = (output / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        managers = {}
        clients = {}
        tensors = {}
        gpu_registrations = {}

        def start_manager(node):
            port = find_available_port()
            journal_bytes = 16 * 1024 * 1024 if pages > 128 else 256 * 1024
            manager = CacheManagerProcess(
                port,
                pool_size="32mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s210-capacity-{port}.sock",
                query_budget="16mb",
                log_path=output / f"{node}-manager.log",
                extra_args=(
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--cluster-name",
                    cluster,
                    "--membership-ttl-secs",
                    "120",
                    "--inventory-journal-bytes",
                    str(journal_bytes),
                    "--inventory-stream-coalesce-ms",
                    "2",
                    "--index-budget",
                    index_budget,
                    "--metadata-namespace",
                    storage_namespace,
                    "--enable-prometheus",
                ),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers[node] = manager
            client = CacheManagerClient(manager.bootstrap_socket)
            stack.callback(client.close)
            clients[node] = client
            tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
            tensors[node] = tensor
            client.start_session_watcher(node, identity, 1, 1)
            gpu_registrations[node] = client.register_context_batch(
                node,
                identity,
                0,
                0,
                1,
                1,
                resolve_device_id(),
                ["kv:0"],
                [serialize_gpu_buffer(tensor)],
                [pages],
                [block_bytes],
                [0],
                [1],
                "direct",
                False,
                tensors=[tensor],
            )

        start_manager("observer")
        for owner in range(owners):
            start_manager(f"source-{owner}")
        manager_list = list(managers.values())
        source_nodes = [f"source-{owner}" for owner in range(owners)]
        result["clock_domain"] = _clock_domain(manager.process.pid for manager in manager_list)
        hashes = {
            node: [
                hashlib.sha256(f"s2.10:capacity:{node}:{block}".encode()).digest()
                for block in range(pages)
            ]
            for node in source_nodes
        }

        for owner, node in enumerate(source_nodes):
            payload = _payload(torch, pages, block_bytes, owner)
            tensors[node].copy_(payload)
            torch.cuda.synchronize()
            ok, message = clients[node].save(
                gpu_registrations[node], [("kv:0", list(range(pages)), hashes[node])]
            )
            assert ok, message

        bootstrap_started = time.monotonic()
        fences = {node: _sync(managers[node]) for node in source_nodes}
        if not expect_degraded:
            for node in source_nodes:
                _await_fence(managers["observer"], fences[node], manager_list, timeout=30)
            bootstrap_ms = (time.monotonic() - bootstrap_started) * 1000
            assert _metadata(managers["observer"])["index"]["coverage"] == (
                "complete_at_watermarks"
            )
        else:
            _until(
                lambda: (
                    _metadata(managers["observer"])["index"]["coverage"] != "complete_at_watermarks"
                ),
                manager_list,
                timeout=30,
            )
            bootstrap_ms = None

        before_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        process_before = _process_sample(manager.process.pid for manager in manager_list)
        observer_binding = _binding(managers["observer"])
        source_bindings = {node: _binding(managers[node]) for node in source_nodes}
        assert all(
            binding["scope_digest"] == observer_binding["scope_digest"]
            for binding in source_bindings.values()
        )
        installed_views = _owner_statuses(managers["observer"])
        source_sequences = {node: fences[node]["inventory_sequence"] for node in source_nodes}
        publication_to_install = []
        save_start_to_install = []
        install_to_harness = []
        ordinary_harness_wait = []
        serial_barrier_verification = []
        concurrent_barrier_verification = []
        save_latency = []
        started = time.monotonic()
        cycle = 0
        remote_bytes = 0
        if not expect_degraded:
            while (
                cycle < warmup_cycles + samples
                if samples is not None
                else time.monotonic() - started < duration_seconds
            ):
                measured = cycle >= warmup_cycles
                cycle_started = time.monotonic()
                save_started_ns = time.monotonic_ns()
                pending_targets = {}
                for owner, node in enumerate(source_nodes):
                    before_sequence = source_sequences[node]
                    cleaned = _cleanup_dram(managers[node])
                    assert cleaned["evicted_blocks"] == pages
                    payload = _payload(torch, pages, block_bytes, cycle * owners + owner + 1)
                    tensors[node].copy_(payload)
                    torch.cuda.synchronize()
                    owner_save_started_ns = time.monotonic_ns()
                    ok, message = clients[node].save(
                        gpu_registrations[node],
                        [("kv:0", list(range(pages)), hashes[node])],
                    )
                    assert ok, message
                    pending_targets[node] = {
                        "minimum_sequence": before_sequence + pages * 2,
                        "save_start_mono_ns": owner_save_started_ns,
                    }
                mutation_submit_loop_ms = (time.monotonic_ns() - save_started_ns) / 1_000_000
                if measured:
                    save_latency.append(mutation_submit_loop_ms)
                targets = {}
                for node, pending in pending_targets.items():
                    source = _until(
                        lambda node=node, minimum=pending["minimum_sequence"]: (
                            status
                            if (status := _metadata(managers[node]))["inventory_sequence"]
                            >= minimum
                            else None
                        ),
                        manager_list,
                    )
                    stream = source["stream"]
                    binding = {
                        "node": stream["source_node"],
                        "epoch": stream["source_node_epoch"],
                        "incarnation": stream["source_incarnation"],
                        "scope_digest": stream["scope_digest"],
                    }
                    assert binding == source_bindings[node], (binding, source_bindings[node])
                    assert source["inventory_sequence"] == pending["minimum_sequence"], source
                    published_ns = source["inventory_last_change_mono_ns"]
                    assert published_ns >= pending["save_start_mono_ns"]
                    targets[node] = {
                        "node": node,
                        "epoch": binding["epoch"],
                        "incarnation": binding["incarnation"],
                        "scope_digest": binding["scope_digest"],
                        "sequence": source["inventory_sequence"],
                        "view_id": installed_views[binding["incarnation"]]["view_id"],
                        "published_mono_ns": published_ns,
                        "save_start_mono_ns": pending["save_start_mono_ns"],
                    }

                _assert_binding(managers["observer"], observer_binding)
                verification_ms = None
                wait_started_ns = time.monotonic_ns()
                if visibility_mode != "ordinary":
                    fences = {node: _sync(managers[node]) for node in source_nodes}
                    for node, fence in fences.items():
                        target = targets[node]
                        assert fence["source_incarnation"] == target["incarnation"]
                        assert fence["source_node_epoch"] == target["epoch"]
                        assert fence["inventory_sequence"] == target["sequence"]
                    verification_started_ns = time.monotonic_ns()
                    if visibility_mode == "serial-barrier":
                        for node in source_nodes:
                            _await_fence(
                                managers["observer"], fences[node], manager_list, timeout=30
                            )
                    else:
                        _await_concurrently(
                            managers["observer"],
                            fences,
                            observer_binding["scope_digest"],
                            manager_list,
                            barrier_concurrency,
                        )
                    verification_ms = (time.monotonic_ns() - verification_started_ns) / 1_000_000
                installed = _wait_for_installs(
                    managers["observer"], targets, manager_list, timeout=30
                )
                harness_completed_ns = time.monotonic_ns()
                _assert_binding(managers["observer"], observer_binding)
                installation = _installation_sample(targets, installed, harness_completed_ns)
                source_sequences = {node: target["sequence"] for node, target in targets.items()}
                if measured:
                    publication_to_install.append(installation["publication_to_install_ms"])
                    save_start_to_install.append(installation["save_start_to_install_ms"])
                    install_to_harness.append(installation["install_to_harness_ms"])
                    if visibility_mode == "ordinary":
                        ordinary_harness_wait.append(
                            (harness_completed_ns - wait_started_ns) / 1_000_000
                        )
                    elif visibility_mode == "serial-barrier":
                        serial_barrier_verification.append(verification_ms)
                    else:
                        concurrent_barrier_verification.append(verification_ms)
                if not skip_restore:
                    selected = source_nodes[cycle % owners]
                    expected = _payload(
                        torch, pages, block_bytes, cycle * owners + (cycle % owners) + 1
                    )
                    _cleanup_dram(managers["observer"])
                    before = fetch_orbitkv_metrics(managers["observer"].http_port).get(
                        "orbitkv_remote_fetch_bytes_total", 0
                    )
                    _restore(
                        clients["observer"],
                        "observer",
                        tensors["observer"],
                        hashes[selected],
                        f"capacity-{cycle}",
                        expected,
                        manager_list,
                    )
                    _wait_for_remote_drain(managers[selected], managers["observer"], manager_list)
                    after = fetch_orbitkv_metrics(managers["observer"].http_port)[
                        "orbitkv_remote_fetch_bytes_total"
                    ]
                    assert after >= before + payload_bytes
                    remote_bytes += after - before
                observer = _metadata(managers["observer"])
                raw.write(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "elapsed": time.monotonic() - started,
                            "measurement_contract": MEASUREMENT_CONTRACT,
                            "visibility_mode": visibility_mode,
                            "installation": installation,
                            "measured": measured,
                            "harness_completed_mono_ns": harness_completed_ns,
                            "barrier_verification_ms": verification_ms,
                            "mutation_submit_loop_ms": mutation_submit_loop_ms,
                            "index": observer["index"],
                            "stream": observer["stream"],
                        }
                    )
                    + "\n"
                )
                cycle += 1
                remaining = 1 - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
            if samples is not None:
                assert len(publication_to_install) == samples
            else:
                assert time.monotonic() - started >= duration_seconds
        else:
            while time.monotonic() - started < duration_seconds:
                cycle_started = time.monotonic()
                save_started = time.monotonic()
                for owner, node in enumerate(source_nodes):
                    cleaned = _cleanup_dram(managers[node])
                    assert cleaned["evicted_blocks"] == pages
                    payload = _payload(torch, pages, block_bytes, cycle * owners + owner + 1)
                    tensors[node].copy_(payload)
                    torch.cuda.synchronize()
                    owner_save_started_ns = time.monotonic_ns()
                    ok, message = clients[node].save(
                        gpu_registrations[node],
                        [("kv:0", list(range(pages)), hashes[node])],
                    )
                    assert ok, message
                save_latency.append((time.monotonic() - save_started) * 1000)
                observer = _metadata(managers["observer"])
                assert observer["index"]["coverage"] != "complete_at_watermarks"
                raw.write(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "elapsed": time.monotonic() - started,
                            "measurement_contract": MEASUREMENT_CONTRACT,
                            "visibility_mode": "degraded-no-complete-view",
                            "save_ms": save_latency[-1],
                            "index": observer["index"],
                            "stream": observer["stream"],
                        }
                    )
                    + "\n"
                )
                cycle += 1
                remaining = 1 - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
            assert time.monotonic() - started >= duration_seconds

        observer_status = _metadata(managers["observer"])
        owner_rows = requests.get(
            f"http://127.0.0.1:{managers['observer'].http_port}/cache/metadata/owners",
            params={"limit": 128},
            timeout=5,
        ).json()
        if expect_degraded:
            assert observer_status["index"]["coverage"] != "complete_at_watermarks"
            assert observer_status["index"]["accounted_bytes"] <= 1024 * 1024
            assert observer_status["index"]["expected_owner_views"] >= owners
            assert (
                observer_status["index"]["installed_owner_views"]
                < observer_status["index"]["expected_owner_views"]
            )
        else:
            final_rows = {row["owner"]: row for row in owner_rows}
            final_oracle = {}
            for owner, node in enumerate(source_nodes):
                binding = source_bindings[node]
                row = final_rows[binding["incarnation"]]
                assert row["fresh"] and row["records"] == pages, row
                assert row["view_id"] == installed_views[binding["incarnation"]]["view_id"]
                assert row["applied_sequence"] == source_sequences[node]
                if not skip_restore:
                    _cleanup_dram(managers["observer"])
                    expected = _payload(torch, pages, block_bytes, (cycle - 1) * owners + owner + 1)
                    before = fetch_orbitkv_metrics(managers["observer"].http_port).get(
                        "orbitkv_remote_fetch_bytes_total", 0
                    )
                    _restore(
                        clients["observer"],
                        "observer",
                        tensors["observer"],
                        hashes[node],
                        f"capacity-final-{node}",
                        expected,
                        manager_list,
                    )
                    _wait_for_remote_drain(managers[node], managers["observer"], manager_list)
                    after = fetch_orbitkv_metrics(managers["observer"].http_port)[
                        "orbitkv_remote_fetch_bytes_total"
                    ]
                    assert after >= before + payload_bytes
                    _assert_binding(managers[node], binding)
                    assert _metadata(managers[node])["inventory_sequence"] == source_sequences[node]
                    final_oracle[node] = {
                        **binding,
                        "sequence": source_sequences[node],
                        "view_id": row["view_id"],
                        "medium": "dram",
                        "namespace": storage_namespace,
                        "keys": [key.hex() for key in hashes[node]],
                        "payload_sha256": hashlib.sha256(
                            tensors["observer"].cpu().numpy().tobytes()
                        ).hexdigest(),
                        "remote_bytes": after - before,
                    }
            result["final_oracle"] = final_oracle
        after_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert after_revision == before_revision
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        process_after = _process_sample(manager.process.pid for manager in manager_list)
        result.update(
            {
                "status": "bounded_degradation" if expect_degraded else "passed",
                "cycles": cycle,
                "measured_samples": len(publication_to_install),
                "wall_seconds": time.monotonic() - started,
                "bootstrap_ms": bootstrap_ms,
                "ordinary_publication_to_install_ms": _summary(
                    publication_to_install if visibility_mode == "ordinary" else []
                ),
                "forced_publication_to_install_ms": _summary(
                    publication_to_install if visibility_mode != "ordinary" else []
                ),
                "save_start_to_install_ms": _summary(save_start_to_install),
                "install_to_harness_ms": _summary(install_to_harness),
                "ordinary_harness_wait_ms": _summary(ordinary_harness_wait),
                "historical_serial_barrier_verification_ms": _summary(serial_barrier_verification),
                "bounded_concurrent_barrier_verification_ms": _summary(
                    concurrent_barrier_verification
                ),
                "mutation_submit_loop_ms": _summary(save_latency),
                "remote_fetch_bytes": remote_bytes,
                "observer": observer_status,
                "owner_rows": owner_rows,
                "process_before": process_before,
                "process_after": process_after,
                "etcd_revision_delta": after_revision - before_revision,
                "etcd_keys": keys,
            }
        )
        result["collection_status"] = result["status"]
        result["qualification_status"] = "not_evaluated"
        if enforce_thresholds:
            result["qualification_status"] = (
                "passed" if result["ordinary_publication_to_install_ms"]["p99"] <= 50 else "failed"
            )
            if result["qualification_status"] == "failed":
                result["status"] = "failed"

        for node, client in clients.items():
            with contextlib.suppress(Exception):
                client.unregister_context(node)
            client.close()
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        result["cleanup"] = {}
        for node, manager in managers.items():
            exit_code, seconds = manager.terminate_gracefully(timeout=10)
            result["cleanup"][node] = {"exit_code": exit_code, "seconds": seconds}
            assert exit_code == 0 and seconds <= 10
        (output / "capacity-result.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["qualification_status"] != "failed", result[
            "ordinary_publication_to_install_ms"
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners", type=int, choices=(1, 4, 16), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--skip-restore", action="store_true")
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--warmup-cycles", type=int, default=0)
    parser.add_argument("--index-budget", default="16mb")
    parser.add_argument("--expect-degraded", action="store_true")
    parser.add_argument(
        "--visibility-mode",
        choices=("ordinary", "serial-barrier", "concurrent-barrier"),
        default="ordinary",
    )
    parser.add_argument("--barrier-concurrency", type=int, default=4)
    parser.add_argument("--enforce-thresholds", action="store_true")
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be positive")
    if args.warmup_cycles < 0 or (args.warmup_cycles and args.samples is None):
        parser.error("nonnegative warm-up cycles require explicit --samples")
    if args.expect_degraded and (args.samples is not None or args.warmup_cycles):
        parser.error("degraded collection uses duration, not visibility sample qualification")
    if args.enforce_thresholds and (
        args.samples is None or args.samples < 1000 or args.warmup_cycles < 50 or args.skip_restore
    ):
        parser.error(
            "qualification requires >=1000 measured samples, >=50 warm-up cycles and real restores"
        )
    if args.duration_seconds <= 0:
        parser.error("duration must be positive")
    if args.pages <= 0 or args.pages > 1024:
        parser.error("--pages must be in 1..=1024")
    if args.barrier_concurrency <= 0 or args.barrier_concurrency > 16:
        parser.error("--barrier-concurrency must be in 1..=16")
    if args.enforce_thresholds and (args.owners != 16 or args.visibility_mode != "ordinary"):
        parser.error("visibility qualification is the ordinary 16-owner cell")
    if args.expect_degraded and args.enforce_thresholds:
        parser.error("degraded-budget runs do not enforce supported visibility thresholds")
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY", "ORBITKV_MOONCAKE_LIB_DIR"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(
            args.output,
            args.owners,
            args.duration_seconds,
            args.index_budget,
            args.expect_degraded,
            args.seed,
            args.pages,
            args.skip_restore,
            args.visibility_mode,
            args.barrier_concurrency,
            args.enforce_thresholds,
            args.samples,
            args.warmup_cycles,
        )
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
