"""OrbitKV - High-performance key-value storage engine with Python bindings.

This package provides:
1. CacheManagerClient: node-local Cache Manager client
2. OrbitKVConnector: vLLM KV connector for distributed inference
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_NATIVE_EXPORTS = {
    "ChannelProbeClient",
    "CacheManagerClient",
    "GpuRegistration",
    "ChannelCallObservation",
    "PublishProgress",
    "BlockHashes",
    "MooncakeTransferEngine",
    "OrbitKVError",
    "OrbitKVInternal",
    "QueryLoading",
    "QueryReady",
    "ShardedQueryReady",
    "QueryCandidates",
    "RecoveryContract",
    "rekey_hashes",
    "RestoreHandle",
    "RestoreStatus",
}

_native = None
__version__ = "0.0.0"
for _distribution in ("orbitkv-llm", "orbitkv-llm-cu13"):
    try:
        __version__ = version(_distribution)
        break
    except PackageNotFoundError:
        continue


def __getattr__(name: str) -> Any:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(name)
    global _native
    if _native is None:
        try:
            _native = import_module(".orbitkv", __name__)
        except ImportError:
            raise ImportError(
                "orbitkv rust extension is not available, check orbitkv-xxx.so file exists"
            ) from None
    return getattr(_native, name)


__all__ = [
    "__version__",
    "ChannelProbeClient",
    "CacheManagerClient",
    "GpuRegistration",
    "ChannelCallObservation",
    "BlockHashes",
    "MooncakeTransferEngine",
    "OrbitKVError",
    "OrbitKVInternal",
    "QueryLoading",
    "QueryReady",
    "ShardedQueryReady",
    "QueryCandidates",
    "RecoveryContract",
    "rekey_hashes",
    "RestoreHandle",
    "RestoreStatus",
]
