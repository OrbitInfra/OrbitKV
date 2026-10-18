"""Lightweight import stubs for connector unit tests.

The default Python test gate should exercise scheduler/worker helper contracts
without requiring a full vLLM + torch runtime. These stubs are intentionally
small and are only installed by tests that mock the connector boundary.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from importlib import util as importlib_util
from unittest.mock import MagicMock


def install_connector_unit_stubs() -> None:
    """Install minimal torch/vLLM/native-extension modules for unit tests."""

    _install_torch_stub()
    _install_vllm_stubs()
    _install_native_extension_stub()


def _install_torch_stub() -> None:
    if _module_available("torch"):
        return

    torch = types.ModuleType("torch")

    class _Cuda:
        Event = MagicMock

        @staticmethod
        def current_stream(_device=None):
            return "producer-stream"

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def get_device_properties(_device_index):
            raise RuntimeError("torch stub has no CUDA devices")

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def synchronize(_device=None) -> None:
            return None

    class _Random:
        @staticmethod
        def manual_seed(_seed: int) -> None:
            return None

    torch.cuda = _Cuda()  # type: ignore[attr-defined]
    torch.random = _Random()  # type: ignore[attr-defined]
    torch.Tensor = object  # type: ignore[attr-defined]
    torch.dtype = object  # type: ignore[attr-defined]
    torch.device = lambda value: value  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    sys.modules["torch"] = torch


def _install_vllm_stubs() -> None:
    if _module_available("vllm"):
        return

    _ensure_module("vllm")
    _ensure_module("vllm.distributed")
    _ensure_module("vllm.distributed.kv_transfer")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1")

    base = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorRole(Enum):
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorBase_V1:
        def bind_connector_metadata(self, metadata):
            self._connector_metadata = metadata

        def clear_connector_metadata(self):
            self._connector_metadata = None

        def __init__(self, *_args, **_kwargs) -> None:
            return None

    class KVConnectorMetadata:
        return_none: bool = False

    class KVConnectorHandshakeMetadata:
        pass

    class KVConnectorWorkerMetadata:
        pass

    class SupportsHMA:
        pass

    @dataclass
    class KVConnectorTransferResults:
        finished_sending: set[str] = field(default_factory=set)
        finished_recving: set[str] = field(default_factory=set)
        failed_recving: set[str] = field(default_factory=set)

    base.KVConnectorRole = KVConnectorRole
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorHandshakeMetadata = KVConnectorHandshakeMetadata
    base.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata
    base.SupportsHMA = SupportsHMA
    base.KVConnectorTransferResults = KVConnectorTransferResults

    metrics = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

    @dataclass
    class KVConnectorStats:
        data: dict | None = field(default_factory=dict)

    class KVConnectorPromMetrics:
        def __init__(
            self,
            _vllm_config=None,
            _metric_types=None,
            _labelnames=None,
            per_engine_labelvalues=None,
        ) -> None:
            self.per_engine_labelvalues = per_engine_labelvalues or {0: []}

    class PromMetric:
        pass

    metrics.KVConnectorStats = KVConnectorStats
    metrics.KVConnectorPromMetrics = KVConnectorPromMetrics
    metrics.PromMetric = PromMetric
    metrics.PromMetricT = PromMetric

    parallel_state = _ensure_module("vllm.distributed.parallel_state")

    def _not_in_distributed_context():
        raise RuntimeError("unit test stub: not in distributed context")

    parallel_state.get_tensor_model_parallel_rank = _not_in_distributed_context
    parallel_state.get_tensor_model_parallel_world_size = _not_in_distributed_context
    parallel_state.get_decode_context_model_parallel_rank = lambda: 0

    class _PPGroup:
        rank_in_group = 0

    parallel_state.get_pp_group = lambda: _PPGroup()

    _ensure_module("vllm.config").VllmConfig = object

    class CUDAGraphMode(Enum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2

    _ensure_module("vllm.config").CUDAGraphMode = CUDAGraphMode
    models_utils = _ensure_module("vllm.model_executor.models.utils")

    def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
        int_vals: list[int] = []
        for subname in layer_name.split("."):
            try:
                int_vals.append(int(subname))
            except ValueError:
                continue
        if num_attn_module == 1 or "attn" not in layer_name:
            assert len(int_vals) == 1, f"layer name {layer_name} should only contain one integer"
            return int_vals[0]
        assert len(int_vals) <= 2, f"layer name {layer_name} should contain most two integers"
        return int_vals[0] * num_attn_module + int_vals[1] if len(int_vals) == 2 else int_vals[0]

    models_utils.extract_layer_index = extract_layer_index
    _ensure_module("vllm.v1")
    kv_cache_interface = _ensure_module("vllm.v1.kv_cache_interface")

    class FullAttentionSpec:
        pass

    class MLAAttentionSpec(FullAttentionSpec):
        pass

    class MambaSpec:
        pass

    class SlidingWindowSpec:
        pass

    class UniformTypeKVCacheSpecs:
        def __init__(self, block_size, kv_cache_specs):
            self.block_size = block_size
            self.kv_cache_specs = kv_cache_specs

    kv_cache_interface.FullAttentionSpec = FullAttentionSpec
    kv_cache_interface.MLAAttentionSpec = MLAAttentionSpec
    kv_cache_interface.MambaSpec = MambaSpec
    kv_cache_interface.SlidingWindowSpec = SlidingWindowSpec
    kv_cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    _ensure_module("vllm.v1.metrics")
    _ensure_module("vllm.v1.metrics.utils").create_metric_per_engine = lambda *_args, **_kwargs: {}


def _install_native_extension_stub() -> None:
    module_name = "orbitkv.orbitkv"
    if _module_available(module_name):
        return

    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    class _QueryLoading:
        def __init__(self, admitted: bool = True) -> None:
            self.admitted = admitted

    class _QueryReady:
        def __init__(self, num_hit_blocks: int = 0, lease: bytes = b"", hit_positions=None) -> None:
            self.num_hit_blocks = num_hit_blocks
            self.lease = lease
            self.hit_positions = [] if hit_positions is None else hit_positions

    @dataclass(frozen=True)
    class _ShardedQueryReady:
        num_hit_blocks: int
        leases: tuple[bytes, ...]
        control_id: bytes

    module.ShardedQueryReady = getattr(module, "ShardedQueryReady", _ShardedQueryReady)

    class _QueryCandidates:
        def __init__(self, hit_positions):
            self.hit_positions = hit_positions
            self.num_hit_blocks = len(hit_positions)

    module.QueryCandidates = getattr(module, "QueryCandidates", _QueryCandidates)

    class _BlockHashes(list):
        def __getitem__(self, view):
            return type(self)(super().__getitem__(view))

    module.BlockHashes = getattr(module, "BlockHashes", _BlockHashes)
    module.ChannelProbeClient = getattr(module, "ChannelProbeClient", MagicMock)
    module.CacheManagerClient = getattr(module, "CacheManagerClient", MagicMock)
    module.RestoreHandle = getattr(module, "RestoreHandle", MagicMock)

    @dataclass(frozen=True)
    class _RestoreStatus:
        done: bool
        success: bool
        message: str = ""

    module.RestoreStatus = getattr(module, "RestoreStatus", _RestoreStatus)
    module.MooncakeTransferEngine = getattr(module, "MooncakeTransferEngine", MagicMock)
    module.OrbitKVError = getattr(module, "OrbitKVError", type("OrbitKVError", (Exception,), {}))
    module.OrbitKVInternal = getattr(
        module, "OrbitKVInternal", type("OrbitKVInternal", (Exception,), {})
    )
    module.SaveNotSubmittedError = getattr(
        module,
        "SaveNotSubmittedError",
        type("SaveNotSubmittedError", (module.OrbitKVError,), {}),
    )
    module.QueryLoading = getattr(module, "QueryLoading", _QueryLoading)
    module.QueryReady = getattr(module, "QueryReady", _QueryReady)
    module.RecoveryContract = getattr(module, "RecoveryContract", MagicMock)
    module.__version__ = getattr(module, "__version__", "test")


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib_util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False
