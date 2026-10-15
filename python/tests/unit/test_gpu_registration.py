from types import SimpleNamespace

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import gpu  # noqa: E402


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
    assert gpu.device_uuid(1) == "GPU-c4930667-4d65-aae5-8205-fe785c3654be"
    assert seen == [1]


@pytest.mark.parametrize("accepted", [False, True], ids=["failed-unregister", "normal-unregister"])
def test_test_context_retains_registration_after_failed_unregistration(accepted):
    from types import SimpleNamespace

    from tests.support.cache_manager import ClientContext

    context = object.__new__(ClientContext)
    registration = object()
    context._registered = True
    context.registration = registration
    context.instance_id = "cleanup"
    context.client = SimpleNamespace(unregister_context=lambda _: (accepted, "controlled cleanup"))

    context.unregister_context()

    assert context._registered is (not accepted)
    assert context.registration is (None if accepted else registration)
