"""Check the installed native restore on the released Qwen3-32B SGLang pool.

Every registered byte is checked, including unselected pages. This workload is
separate from model timings and does not prove CUDA IPC retirement or fault drain.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from uuid import UUID, uuid4

from .artifacts import external_path
from .metrics import metrics, validate_ownership

PAGE_SIZE = 64
GPU_TOKENS = 16384
LAYERS = 64
PLANES = LAYERS * 2
PAGES = GPU_TOKENS // PAGE_SIZE + 1
PAGE_BYTES = PAGE_SIZE * 8 * 128 * 2
SENTINEL = 0xA7


def validate_model(config: dict) -> None:
    expected = {"num_hidden_layers": LAYERS, "num_key_value_heads": 8, "head_dim": 128}
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get(
        "use_sliding_window", False
    ):
        raise ValueError("Byte qualification requires the declared dense Qwen3-32B profile")
    for field, value in expected.items():
        if type(config.get(field)) is not int or config[field] != value:
            raise ValueError(f"Unexpected model geometry: {field}")


def page_maps(length: int, generation: int) -> tuple[list[int], list[int | None]]:
    if type(length) is not int or length not in {4096, 8192}:
        raise ValueError("Byte qualification requires length 4096 or 8192")
    if type(generation) is not int or generation not in {0, 1, 2}:
        raise ValueError("Expected exactly three restore generations")
    sources = list(range(1, length // PAGE_SIZE))
    targets = [
        None if generation == 2 and index % 4 == 2 else PAGES - 2 - index
        for index in range(len(sources))
    ]
    return sources, targets


def plane_bytes(
    plane: int,
    seed: int,
    sources: list[int],
    targets: list[int | None],
    *,
    pages: int = PAGES,
    page_bytes: int = PAGE_BYTES,
) -> bytearray:
    if (
        type(plane) is not int
        or not 0 <= plane < PLANES
        or type(seed) is not int
        or not 0 <= seed < 2**64
        or type(pages) is not int
        or pages < 2
        or type(page_bytes) is not int
        or page_bytes <= 0
        or len(sources) != len(targets)
        or len(set(sources)) != len(sources)
        or any(type(page) is not int or not 0 < page < pages for page in sources)
    ):
        raise ValueError("Invalid full-pool byte oracle geometry")
    written = [page for page in targets if page is not None]
    if len(set(written)) != len(written) or any(
        type(page) is not int or not 0 < page < pages for page in written
    ):
        raise ValueError("Invalid or overlapping GPU destination pages")
    expected = bytearray([SENTINEL]) * (pages * page_bytes)
    for source, target in zip(sources, targets, strict=True):
        if target is not None:
            domain = f"sglang-kv-byte-v1:{seed}:{plane}:{source}".encode()
            expected[target * page_bytes : (target + 1) * page_bytes] = hashlib.shake_256(
                domain
            ).digest(page_bytes)
    return expected


def verify_plane(actual: bytes, expected: bytearray, plane: int) -> str:
    if len(actual) != len(expected):
        raise AssertionError(f"Plane {plane}: registered byte length differs")
    if actual != expected:
        first = next(
            index for index, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b
        )
        raise AssertionError(f"Plane {plane}: byte mismatch at {first}")
    return hashlib.sha256(actual).hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def _drain(manager_url: str) -> dict:
    deadline = time.monotonic() + 30
    while True:
        observed = metrics(manager_url)
        try:
            validate_ownership(observed, drained=True)
            return observed
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _hold_unknown(output: Path, error: BaseException) -> None:
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, signal.SIG_IGN)
    try:
        _write(
            output / "UNKNOWN-NATIVE-OWNERSHIP.json",
            {"pid": os.getpid(), "error": repr(error), "allocations_and_locks_retained": True},
        )
    except OSError as logging_error:
        with contextlib.suppress(OSError):
            print(
                f"Cannot record unknown native ownership; retaining allocations: {logging_error!r}",
                file=sys.stderr,
            )
    while True:
        time.sleep(1)


def run(args: argparse.Namespace) -> dict:
    import torch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    import orbitkv
    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.layout import GpuLayout

    origin = Path(orbitkv.__file__).resolve()
    if "site-packages" not in origin.parts or importlib.metadata.version("sglang") != "0.5.21":
        raise RuntimeError("Byte qualification requires the installed wheel and SGLang 0.5.21")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly the reviewed physical GPU by UUID")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = f"GPU-{UUID(str(properties.uuid).removeprefix('GPU-'))}"
    if gpu_uuid != args.gpu_uuid:
        raise RuntimeError("Byte oracle GPU UUID differs from the frozen contract")
    config_bytes = args.model_config.read_bytes()
    validate_model(json.loads(config_bytes))
    pool = MHATokenToKVPool(
        size=GPU_TOKENS,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=8,
        head_dim=128,
        layer_num=LAYERS,
        device="cuda:0",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS,
        enable_alt_stream=False,
        kv_cache_layout="nhd",
    )
    layout = GpuLayout.from_pool(PAGE_SIZE, pool, None, {ComponentType.FULL})
    group = next(iter(layout.pools.values()))
    tensors = list(group.entry.kv_buffer)
    if (
        layout.num_layers != LAYERS
        or len(tensors) != PLANES
        or group.num_blocks != [PAGES] * PLANES
        or group.block_bytes != [PAGE_BYTES] * PLANES
        or any(tuple(tensor.shape) != (GPU_TOKENS + PAGE_SIZE, 8, 128) for tensor in tensors)
    ):
        raise RuntimeError("Released GPU pool does not match the consumed registration geometry")
    counter = _LayerDoneCounter(layout)
    instance = f"sglang-byte-{uuid4().hex}"
    namespace = f"sglang-byte-v1:{instance}"
    device = resolve_device_id()
    client = CacheManagerClient(args.manager_socket)
    registered = False
    inflight = False
    handle = None
    unsubmitted_lease = None
    rounds = []
    primary_error = None
    try:
        client.start_session_watcher(instance, namespace, 1, 1)
        ok, message = client.register_context_batch(
            instance,
            namespace,
            0,
            0,
            1,
            1,
            device,
            group.layer_names,
            [serialize_gpu_buffer(tensor) for tensor in tensors],
            group.num_blocks,
            group.block_bytes,
            [0] * PLANES,
            [1] * PLANES,
            args.backend,
            False,
            tensors=tensors,
            layer_group_ids=[0] * PLANES,
            layer_formats=["bf16"] * PLANES,
            layer_attention=group.attention_layouts(LAYERS, 0),
        )
        if not ok:
            raise RuntimeError(f"GPU registration failed: {message}")
        registered = True
        observed = args.samples is not None
        operations = args.warmup + args.samples if observed else 3
        for operation in range(operations):
            generation = 0 if observed else operation
            sources, targets = page_maps(args.length, generation)
            seed = args.seed + generation
            hashes = [
                hashlib.sha256(f"{namespace}:{seed}:{page}".encode()).digest() for page in sources
            ]
            native_hashes = BlockHashes(hashes)
            for plane, tensor in enumerate(tensors):
                reference = plane_bytes(plane, seed, sources, sources)
                tensor.copy_(
                    torch.frombuffer(reference, dtype=torch.uint8)
                    .view(torch.bfloat16)
                    .reshape(tensor.shape)
                )
            torch.cuda.synchronize()
            ok, message = client.save(
                instance, 0, 0, device, [(name, sources, hashes) for name in group.layer_names]
            )
            if not ok:
                raise RuntimeError(f"Native save failed: {message}")
            request = f"operation-{operation}" if observed else f"generation-{generation}"
            deadline = time.monotonic() + 120
            while True:
                ready = client.query_prefetch(instance, native_hashes, request)
                if not isinstance(ready, QueryLoading):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Saved full prefix did not become queryable")
                time.sleep(0.001)
            if not isinstance(ready, QueryReady):
                raise RuntimeError("Native query did not return a restore lease")
            unsubmitted_lease = ready.lease
            if ready.num_hit_blocks != len(sources) or not unsubmitted_lease:
                raise RuntimeError("Native query does not cover every saved source page")
            before = metrics(args.manager_url)
            for tensor in tensors:
                tensor.view(torch.uint8).fill_(SENTINEL)
            if observed:
                torch.cuda.synchronize()
                started_ns = time.monotonic_ns()
            inflight = True
            handle = client.start_restore(
                instance,
                0,
                device,
                counter.layer_groups,
                [(unsubmitted_lease, [targets])],
                ready_stream=torch.cuda.current_stream().cuda_stream,
                layer_events=list(counter.layer_events.items()),
            )
            unsubmitted_lease = None
            client.wait_restore_enqueued(handle, timeout=120)
            if observed:
                enqueued_ns = time.monotonic_ns()
            for layer in range(LAYERS):
                counter.wait_until(layer)
            status = client.wait_restore(handle, timeout=120)
            if observed:
                drained_ns = time.monotonic_ns()
            if not status.done:
                raise RuntimeError("Native terminal drain was not returned")
            inflight = False
            if not status.success:
                raise RuntimeError(f"Native restore failed: {status.message}")
            digests = []
            for plane, tensor in enumerate(tensors):
                expected = plane_bytes(plane, seed, sources, targets)
                actual = tensor.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
                digests.append(verify_plane(actual, expected, plane))
            controls = []
            if generation == 0:
                plane = PLANES - 1
                expected = plane_bytes(plane, seed, sources, targets)
                raw = tensors[plane].view(torch.uint8).reshape(-1)
                last_target = targets[-1]
                for region, position in (
                    ("last-payload-byte", (last_target + 1) * PAGE_BYTES - 1),
                    ("last-sentinel-byte", len(expected) - 1),
                ):
                    raw[position].bitwise_xor_(1)
                    try:
                        verify_plane(raw.cpu().numpy().tobytes(), expected, plane)
                    except AssertionError as error:
                        if str(error) != f"Plane {plane}: byte mismatch at {position}":
                            raise
                        controls.append({"region": region, "position": position, "rejected": True})
                    else:
                        raise AssertionError(f"GPU corruption control was accepted: {region}")
                    finally:
                        raw[position].bitwise_xor_(1)
                    verify_plane(raw.cpu().numpy().tobytes(), expected, plane)
            after = _drain(args.manager_url)
            payload = sum(page is not None for page in targets) * PAGE_BYTES * PLANES
            restored = after.get("orbitkv_load_bytes_total", 0) - before.get(
                "orbitkv_load_bytes_total", 0
            )
            if restored != payload:
                raise AssertionError(
                    f"Native restore byte counter differs: {restored} != {payload}"
                )
            row = {
                "generation": generation,
                "seed": seed,
                "source_pages": sources,
                "destination_pages": targets,
                "verified_pool_bytes": PAGES * PAGE_BYTES * PLANES,
                "verified_payload_bytes": payload,
                "verified_sentinel_bytes": PAGES * PAGE_BYTES * PLANES - payload,
                "plane_sha256": digests,
                "negative_controls": controls,
                "manager_restored_bytes": restored,
                "manager_after": after,
            }
            if observed:
                row["observation"] = {
                    "operation": operation,
                    "phase": "warmup" if operation < args.warmup else "measured",
                    "clock": "CLOCK_MONOTONIC",
                    "started_ns": started_ns,
                    "enqueued_ns": enqueued_ns,
                    "drained_ns": drained_ns,
                    "terminal_done": status.done,
                    "terminal_success": status.success,
                    "session_epoch": handle.session_epoch,
                    "operation_id": handle.operation_id,
                    "operation_key": handle.key,
                }
            filename = (
                f"operation-{operation}.json" if observed else f"generation-{generation}.json"
            )
            _write(args.output / filename, row)
            rounds.append(row)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if inflight:
            try:
                if handle is None or not client.wait_restore(handle, timeout=120).done:
                    raise RuntimeError("Native submission/terminal ownership is unknown")
            except BaseException as error:
                _hold_unknown(args.output, error)
        try:
            if unsubmitted_lease:
                client.release(unsubmitted_lease)
            if registered:
                ok, message = client.unregister_context(instance)
                if not ok:
                    raise RuntimeError(f"GPU unregistration failed: {message}")
        except BaseException as error:
            if primary_error is None:
                primary_error = error
                raise
            with contextlib.suppress(OSError):
                print(
                    f"GPU cleanup also failed: {error!r}; original error: {primary_error!r}",
                    file=sys.stderr,
                )
        finally:
            try:
                client.close()
            except BaseException as error:
                if primary_error is None:
                    raise
                with contextlib.suppress(OSError):
                    print(
                        f"Native client close also failed: {error!r}; original error: {primary_error!r}",
                        file=sys.stderr,
                    )
    return {
        "status": "VALID_FULL_LAYOUT_OBSERVATION" if observed else "VALID_FULL_LAYOUT_BYTES",
        "backend": args.backend,
        "length": args.length,
        "gpu_uuid": gpu_uuid,
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "installed_orbitkv": str(origin),
        "sglang": importlib.metadata.version("sglang"),
        "torch": torch.__version__,
        "logical_layers": LAYERS,
        "physical_planes": PLANES,
        "pages_per_plane": PAGES,
        "bytes_per_page_per_plane": PAGE_BYTES,
        "rounds": rounds,
        "observation_design": {"warmup": args.warmup, "samples": args.samples}
        if observed
        else None,
        "unregistered": True,
        "client_closed": True,
        "scope": (
            "Full-byte native restore and complete-operation host wall observation, including Python/native return boundaries; not device-only time, model serving or physical fault reclamation"
            if observed
            else "Actual released pool/native transfer full-byte control, not computed model KV, timing, native IPC retirement or S3 fault proof"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--backend", choices=("direct", "kernel"), required=True)
    parser.add_argument("--manager-socket", required=True)
    parser.add_argument("--manager-url", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized oracle execution is forbidden")
    if (args.warmup is None) != (args.samples is None) or (
        args.samples is not None and (args.warmup < 1 or not 1 <= args.samples <= 16)
    ):
        parser.error("Observation requires both --warmup >= 1 and --samples between 1 and 16")
    if not 0 <= args.seed < 2**64 - 2:
        raise ValueError("Oracle seed must fit all three u64 generations")
    validate_model(json.loads(args.model_config.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
    except BaseException as error:
        try:
            _write(
                args.output / "FAILURE.json",
                {"error": repr(error), "traceback": traceback.format_exc()},
            )
        except OSError as report_error:
            with contextlib.suppress(OSError):
                print(
                    f"Failure report also failed: {report_error!r}; original error: {error!r}",
                    file=sys.stderr,
                )
        raise
    _write(args.output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "backend": args.backend, "length": args.length}))


if __name__ == "__main__":
    main()
