"""Framework CUDA identity used during native registration handoff."""

import threading
from uuid import UUID

import torch

_discovered_devices: dict[str, int] = {}
_device_mapping_lock = threading.Lock()


def device_uuid(device_index: int) -> str:
    value = str(torch.cuda.get_device_properties(device_index).uuid)
    return f"GPU-{UUID(value.removeprefix('GPU-'))}"


def device_index_from_uuid(value: str) -> int:
    with _device_mapping_lock:
        if not _discovered_devices:
            for index in range(torch.cuda.device_count()):
                _discovered_devices[device_uuid(index)] = index
        index = _discovered_devices.get(value)
    if index is None:
        raise RuntimeError(f"GPU UUID {value} is not visible to this process")
    return index


def resolve_device_id() -> int:
    return torch.cuda.current_device()
