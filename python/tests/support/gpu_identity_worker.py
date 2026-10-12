"""Own real CUDA tensors in a separately masked integration-test process."""

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from uuid import UUID


def main():
    import torch

    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady
    from orbitkv.client.gpu import serialize_gpu_buffer
    from tests.support.cache_manager import evict_dram_after_ssd_writes
    from tests.support.gpu_identity import UnissuedCudaIPCWrapper
    from tests.support.metrics import fetch_orbitkv_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--http-port", required=True, type=int)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--medium", choices=("dram", "ssd"), required=True)
    parser.add_argument("--rejection", default="")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    actual_uuid = str(torch.cuda.get_device_properties(args.device).uuid)
    assert UUID(actual_uuid.removeprefix("GPU-")) == UUID(args.uuid.removeprefix("GPU-")), (
        actual_uuid,
        args.uuid,
    )
    expected = (
        torch.arange(8 * 4096, dtype=torch.int64).remainder(251).to(torch.uint8).view(8, 4096)
    )
    tensor = expected.to(f"cuda:{args.device}")
    tensors = [tensor]
    if args.rejection:
        uuids = [f"GPU-{UUID(actual_uuid.removeprefix('GPU-'))}"]
        if args.rejection == "UUID differs":
            wrong = tensor.to(f"cuda:{1 - args.device}")
            tensors.append(tensor)
            wrong_uuid = str(torch.cuda.get_device_properties(wrong.device).uuid)
            uuids.insert(0, f"GPU-{UUID(wrong_uuid.removeprefix('GPU-'))}")
        wrappers = [pickle.dumps(UnissuedCudaIPCWrapper(uuid)) for uuid in uuids]
    else:
        wrappers = [serialize_gpu_buffer(tensor)]
    count = len(tensors)
    client = CacheManagerClient(args.socket)
    registered = False
    result = {
        "pid": os.getpid(),
        "device": args.device,
        "uuid": actual_uuid,
        "medium": args.medium,
        "ipc_exported": not bool(args.rejection),
        "rejection_scope": "preimport_metadata_guard" if args.rejection else None,
    }
    try:
        try:
            ok, message = client.register_context_batch(
                "identity",
                "gpu-identity",
                0,
                0,
                1,
                1,
                7 if args.rejection == "client-local" else args.device,
                [f"kv:{index}" for index in range(count)],
                wrappers,
                [8] * count,
                [4096] * count,
                [0] * count,
                [1] * count,
                "direct",
                False,
                tensors=tensors,
            )
            assert ok, message
            registered = True
        except Exception as error:
            if not args.rejection or args.rejection not in str(error):
                raise
            assert client.health()[0]
            result["rejection"] = str(error)
        else:
            assert not args.rejection, "Registration unexpectedly accepted a negative control"
            hashes = [hashlib.sha256(f"identity-{index}".encode()).digest() for index in range(8)]
            assert client.save("identity", 0, 0, args.device, [("kv:0", list(range(8)), hashes)])[0]
            if args.medium == "ssd":
                evict_dram_after_ssd_writes(args.http_port)
            before = fetch_orbitkv_metrics(args.http_port)
            deadline = time.monotonic() + 30
            request_id = 0
            while True:
                query = client.query_prefetch(
                    "identity", BlockHashes(hashes), req_id=f"identity-{request_id}"
                )
                if isinstance(query, QueryReady):
                    break
                if not isinstance(query, QueryLoading):
                    request_id += 1
                assert time.monotonic() < deadline, query
                time.sleep(0.01)
            assert query.num_hit_blocks == 8
            tensor.zero_()
            torch.cuda.synchronize()
            handle = client.start_restore(
                "identity",
                0,
                args.device,
                [["kv:0"]],
                [(query.lease, [list(range(8))])],
                ready_stream=torch.cuda.current_stream(args.device).cuda_stream,
            )
            status = client.wait_restore(handle, timeout=30)
            assert status.done and status.success, status.message
            torch.cuda.synchronize()
            assert tensor.cpu().equal(expected)
            after = fetch_orbitkv_metrics(args.http_port)
            result.update(
                restored_bytes=tensor.numel(),
                before_metrics=before,
                after_metrics=after,
            )
            args.output.with_suffix(".observation.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            if args.medium == "ssd":
                assert (
                    after["orbitkv_ssd_prefetch_bytes_total"]
                    - before.get("orbitkv_ssd_prefetch_bytes_total", 0)
                    == tensor.numel()
                ), result
                assert (
                    after["orbitkv_ssd_uring_read_operations_total"]
                    - before.get("orbitkv_ssd_uring_read_operations_total", 0)
                    == len(hashes)
                ), result
        args.output.with_suffix(".ready.json").write_text(json.dumps(result, indent=2) + "\n")
        deadline = time.monotonic() + 60
        while not args.output.with_suffix(".release").exists():
            assert time.monotonic() < deadline, "Controller did not release the registered owner"
            time.sleep(0.02)
        if registered:
            ok, message = client.unregister_context("identity")
            assert ok, message
            registered = False
    finally:
        try:
            if registered:
                ok, message = client.unregister_context("identity")
                assert ok, message
        finally:
            client.close()
            torch.cuda.synchronize()
    result["normal_unregister"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
