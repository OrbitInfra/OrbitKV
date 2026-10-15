"""Standalone worker used by test_session_watcher::test_crashed_client_releases_ipc.

Allocates a GPU tensor, registers it as a CUDA IPC context with
Cache Manager, opens the liveness Session stream, signals the parent
that it is ready, then blocks forever waiting to be SIGKILLed.

Run as: python session_crash_helper.py <endpoint> <instance_id> <ready_file>
"""

from __future__ import annotations

import importlib
import sys
import time

import torch

from orbitkv.client.gpu import serialize_gpu_buffer


def main() -> int:
    endpoint, instance_id, ready_file = sys.argv[1], sys.argv[2], sys.argv[3]

    orbitkv_module = importlib.import_module("orbitkv.orbitkv")
    client = orbitkv_module.CacheManagerClient(endpoint)

    device = torch.device("cuda:0")
    kv = torch.rand((2, 16, 16, 8, 128), dtype=torch.bfloat16, device=device).contiguous()
    wrapper_bytes = serialize_gpu_buffer(kv)

    shape = tuple(kv.shape)
    stride = tuple(kv.stride())
    element_size = kv.element_size()
    num_blocks = shape[1]
    bytes_per_block = stride[1] * element_size
    kv_stride_bytes = stride[0] * element_size
    segments = 2

    client.register_context_batch(
        instance_id,
        "test-ns",
        0,  # tp_rank
        0,  # pp_rank
        1,  # tp_size
        1,  # world_size
        0,  # device_id
        ["layer_0"],
        [wrapper_bytes],
        [num_blocks],
        [bytes_per_block],
        [kv_stride_bytes],
        [segments],
        "direct",
        False,
        tensors=[kv],
    )

    client.start_session_watcher(instance_id, "test-ns", 1, 1)

    with open(ready_file, "w") as f:
        f.write("ready")

    # Block until SIGKILLed. We must not drop the client gracefully —
    # the test verifies cleanup fires on abrupt process death, not on
    # orderly shutdown.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
