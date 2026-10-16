"""Real CUDA export-credit gates; test proxies never replace runtime modules."""

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from tests.support.trace_sglang_ipc import read_counter

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


class StorageExportControl:
    def __init__(self, storage, exports, fail=False):
        self.storage = storage
        self.exports = exports
        self.fail = fail

    def data_ptr(self):
        return self.storage.data_ptr()

    def nbytes(self):
        return self.storage.nbytes()

    def _share_cuda_(self):
        if self.fail:
            raise RuntimeError("controlled second-export failure")
        shared = self.storage._share_cuda_()
        self.exports.append((shared[4], shared[5]))
        return shared


class TensorExportControl:
    def __init__(self, tensor, exports, fail=False):
        self.tensor = tensor
        self.device = tensor.device
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.storage = StorageExportControl(tensor.untyped_storage(), exports, fail)

    def data_ptr(self):
        return self.tensor.data_ptr()

    def untyped_storage(self):
        return self.storage

    def storage_offset(self):
        return self.tensor.storage_offset()

    def element_size(self):
        return self.tensor.element_size()

    def stride(self):
        return self.tensor.stride()


def counter(shared):
    name, offset = shared
    name = name.decode() if isinstance(name, bytes) else name
    return read_counter(Path("/dev/shm") / name.lstrip("/"), offset)


def register(client, instance, tensors, names):
    return client.register_context_batch(
        instance,
        instance,
        0,
        0,
        1,
        1,
        tensors[0].device.index,
        names,
        [8] * len(names),
        [4096] * len(names),
        [0] * len(names),
        [1] * len(names),
        "direct",
        False,
        tensors=tensors,
    )


def test_partial_export_repayments_are_real_and_no_credit_is_forgotten(channel_server, tmp_path):
    torch = pytest.importorskip("torch")
    from orbitkv import CacheManagerClient

    instance = f"partial-export-{uuid.uuid4().hex}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    exports = []
    tensors = [
        TensorExportControl(torch.zeros((8, 4096), dtype=torch.uint8, device="cuda"), exports),
        TensorExportControl(
            torch.zeros((8, 4096), dtype=torch.uint8, device="cuda"), exports, fail=True
        ),
    ]
    torch.cuda.synchronize()
    before = client.registration_retention()
    try:
        with pytest.raises(RuntimeError, match="controlled second-export failure"):
            register(client, instance, tensors, ["first", "second"])
        assert len(exports) == 1
        assert counter(exports[0]) == 0, "real never-sent CUDA credit was not repaid"
        after = client.registration_retention()
        # The throwing framework call has no return proving that it created no
        # credit. Its native attempt therefore remains charged conservatively.
        assert after[0] == before[0] + 1 and after[1] > before[1]
        retained = client.retained_registration_attempts()
        assert any(row[1] == instance and row[4] for row in retained)
        with pytest.raises(Exception, match="pending or retained"):
            client.unregister_context(instance)
        torch.cuda.synchronize()
        (tmp_path / "partial-export.credit-state.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "instance_id": instance,
                    "start_ticks": int(
                        Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                    ),
                    "credit_counter_exists_and_zero": counter(exports[0]) == 0,
                    "native_retained_attempts": after[0],
                    "native_retained_bytes": after[1],
                    "framework_throw_retained": True,
                    "unregister_refused": True,
                    "native_registration_or_payload_submitted": False,
                    "physical_fault_drain_claimed": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        client.close()


def test_duplicate_registration_creates_no_additional_real_export_credit(channel_server, tmp_path):
    torch = pytest.importorskip("torch")
    from orbitkv import CacheManagerClient

    instance = f"duplicate-export-{uuid.uuid4().hex}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    exports = []
    tensor = TensorExportControl(
        torch.arange(8 * 4096, device="cuda").remainder(251).to(torch.uint8).reshape(8, 4096),
        exports,
    )
    registered = False
    torch.cuda.synchronize()
    try:
        authority = register(client, instance, [tensor], ["kv"])
        registered = True
        assert len(exports) == 1
        assert authority.ipc_counters == exports
        with pytest.raises(ValueError, match="already registered"):
            register(client, instance, [tensor], ["kv"])
        assert len(exports) == 1
        ok, message = client.unregister_context(instance)
        assert ok, message
        registered = False
        deadline = time.monotonic() + 10
        while counter(exports[0]) != 0:
            assert time.monotonic() < deadline, "Manager importer did not retire its real credit"
            time.sleep(0.01)
        (tmp_path / "duplicate-export.credit-state.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "instance_id": instance,
                    "start_ticks": int(
                        Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                    ),
                    "registration_generation": authority.generation.hex(),
                    "export_calls": len(exports),
                    "duplicate_refused_before_export": True,
                    "credit_counter_exists_and_zero": counter(exports[0]) == 0,
                    "unregister_succeeded": True,
                    "physical_fault_drain_claimed": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if registered:
            ok, message = client.unregister_context(instance)
            assert ok, message
        client.close()
