"""Native TENT completion bytes must remain exact after descriptor coalescing."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

_RETAINED_GPU: list[tuple] = []


@pytest.mark.parametrize("operation", ["write", "read"])
def test_merged_gpu_descriptors_report_exact_completed_bytes(monkeypatch, operation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the CUDA TENT artifact needs a working CUDA driver")
    from orbitkv.orbitkv import MooncakeTransferEngine

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    sender = MooncakeTransferEngine(bind_host="127.0.0.1")
    receiver = MooncakeTransferEngine(bind_host="127.0.0.1")
    source = torch.arange(4096, dtype=torch.int32, device="cuda").to(torch.uint8)
    destination = torch.zeros_like(source)
    torch.cuda.synchronize()
    try:
        for engine, tensor in ((sender, source), (receiver, destination)):
            engine.register_memory_borrowed(
                [{"addr": tensor.data_ptr(), "len": tensor.numel(), "location": "*"}]
            )
        if operation == "write":
            engine, endpoint, local, remote = (sender, receiver.endpoint, source, destination)
        else:
            engine, endpoint, local, remote = (receiver, sender.endpoint, destination, source)
        slices = [
            (local.data_ptr() + offset, remote.data_ptr() + offset, 1024)
            for offset in range(0, 4096, 1024)
        ]
        completed = getattr(engine, operation)(endpoint, slices, timeout_s=10)
        torch.cuda.synchronize()
        assert torch.equal(source, destination), "TENT completed before payload equality"
        assert completed == source.numel(), "merged descriptors counted the same bytes twice"
        sender.unregister_memory([source.data_ptr()])
        receiver.unregister_memory([destination.data_ptr()])
    except BaseException:
        _RETAINED_GPU.append((sender, receiver, source, destination))
        raise
