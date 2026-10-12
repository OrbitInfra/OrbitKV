from types import SimpleNamespace

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import gpu  # noqa: E402

CudaIPCWrapper = gpu.CudaIPCWrapper


class FakeTensor:
    def __init__(self):
        self.set_args = None

    def set_(self, *args):
        self.set_args = args
        return self


class FakeUntypedStorage:
    @staticmethod
    def _new_shared_cuda(device, *handle_args):
        return ("storage", device, handle_args)


class FakeTorch:
    UntypedStorage = FakeUntypedStorage

    def __init__(self):
        self.created_tensor = FakeTensor()

    def tensor(self, *args, **kwargs):
        return self.created_tensor


def _wrapper_without_init(stride, storage_offset):
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = ("ignored_device", "ipc_handle", "size")
    wrapper.dtype = "fake-dtype"
    wrapper.shape = (2, 3)
    wrapper.device_uuid = "GPU-fake"
    wrapper.stride = stride
    wrapper.storage_offset = storage_offset
    return wrapper


def test_to_tensor_preserves_strided_storage(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gpu, "torch", fake_torch)
    monkeypatch.setattr(CudaIPCWrapper, "_get_device_index_from_uuid", staticmethod(lambda _: 7))

    tensor = _wrapper_without_init(stride=(4, 1), storage_offset=5).to_tensor()

    assert tensor.set_args == (("storage", 7, ("ipc_handle", "size")), 5, (2, 3), (4, 1))


@pytest.mark.parametrize("visible", [None, "7", "7,0", "GPU-seven", "GPU-seven,GPU-zero"])
def test_registration_device_is_the_client_local_ordinal(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        gpu, "torch", SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0))
    )
    assert gpu.resolve_device_id() == 0


@pytest.mark.parametrize(
    "value",
    [
        "c4930667-4d65-aae5-8205-fe785c3654be",
        "GPU-c4930667-4d65-aae5-8205-fe785c3654be",
        "C4930667-4D65-AAE5-8205-FE785C3654BE",
    ],
)
def test_registration_and_ipc_share_canonical_gpu_uuid(monkeypatch, value):
    seen = []

    def properties(device):
        seen.append(device)
        return SimpleNamespace(uuid=value)

    monkeypatch.setattr(
        gpu, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties))
    )
    assert CudaIPCWrapper._get_device_uuid(1) == "GPU-c4930667-4d65-aae5-8205-fe785c3654be"
    assert seen == [1]
