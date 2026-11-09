"""Released vLLM consumers of OrbitKV completion and delivery contracts.

Run when upgrading vLLM or changing connector completion/composition callbacks.
Requires official vLLM 0.33.0. Model composition uses test_vllm_native_pd_e2e.py.
"""

import importlib.metadata
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def test_native_multi_connector_keeps_cache_best_effort_and_handoff_reliable():
    from tests.support.installed_artifacts import ENGINE_VERSIONS

    assert importlib.metadata.version("vllm") == ENGINE_VERSIONS["vllm"]
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    from orbitkv.vllm.connector import OrbitKVConnector

    cache = OrbitKVConnector.__new__(OrbitKVConnector)
    cache._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    prefill = NixlConnector.__new__(NixlConnector)
    prefill._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [cache]
    assert not multi.requires_kv_delivery
    multi._connectors.append(prefill)
    assert multi.requires_kv_delivery
    assert not multi.supports_divergent_local_hybrid_hits


@pytest.fixture
def released_config():
    import torch
    from vllm.config import KVTransferConfig
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OrbitKVConnector",
            kv_connector_module_path="orbitkv.vllm",
            kv_role="kv_both",
            engine_id="contract-control",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(), enable_sleep_mode=False),
        lora_config=None,
        use_v2_model_runner=False,
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
    cache = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer.0"], spec)])
    return config, cache


@pytest.mark.parametrize("profile", ["v2", "multi-group", "recurrent", "sleep"])
def test_unsupported_released_profiles_fail_before_connecting(
    released_config, monkeypatch, profile
):
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import MambaSpec

    from orbitkv.vllm.connector import OrbitKVConnector

    def unexpected_connection(**kwargs):
        pytest.fail("unsupported profile opened a Manager connection")

    monkeypatch.setattr("orbitkv.vllm.connector.connect_cache", unexpected_connection)
    config, cache = released_config
    if profile == "sleep":
        config.model_config.enable_sleep_mode = True
        with pytest.raises(ValueError, match="does not support sleep mode"):
            VllmConfig._verify_kv_transfer_compat(config)
        return
    if profile == "v2":
        config.use_v2_model_runner = True
    elif profile == "multi-group":
        cache.kv_cache_groups *= 2
    else:
        cache.kv_cache_groups[0].kv_cache_spec = MambaSpec(
            block_size=16, shapes=((2, 4),), dtypes=(torch.float32,), mamba_cache_mode="align"
        )
    with pytest.raises(RuntimeError, match="VLLM_USE_V2|require window|single cache group"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)


def test_released_single_group_and_hit_source_defaults(released_config):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.metrics.cache_hit_source import CacheHitSource

    from orbitkv.vllm.connector import OrbitKVConnector
    from orbitkv.vllm.layout import CacheGroupLayout

    config, cache = released_config
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config = SimpleNamespace(block_size=16, prefix_match_unit=None)
    assert resolve_kv_cache_block_sizes(cache, config) == (16, 16)
    assert CacheGroupLayout.from_config(cache).recovery_groups == ((0, "attention", 0),)
    connector = object.__new__(OrbitKVConnector)
    KVConnectorBase_V1.__init__(connector, config, KVConnectorRole.SCHEDULER, cache)
    assert connector.get_loaded_kv_cache_group_ids(None) == (0,)
    assert connector.get_external_cache_hit_sources(None, 32) == {
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32
    }
    config.cache_config.prefix_match_unit = 8
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix_match_unit"):
        resolve_kv_cache_block_sizes(cache, config)


@pytest.mark.parametrize("opt_in", [False, "true", True])
def test_v2_admission_requires_boolean_opt_in(released_config, monkeypatch, opt_in):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    config.use_v2_model_runner = True
    config.kv_transfer_config.kv_connector_extra_config["orbitkv.v2"] = opt_in
    config.cache_config = SimpleNamespace(block_size=16, prefix_match_unit=None)
    config.parallel_config.decode_context_parallel_size = 1

    class Admitted(Exception):
        pass

    # Stop after admission, before identity resolution or any Manager connection.
    def admitted(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr("orbitkv.vllm.connector.derive_namespace", admitted)
    if opt_in is True:
        with pytest.raises(Admitted):
            OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
    else:
        with pytest.raises(RuntimeError, match="opt-in"):
            OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize("profile", ["missing-group", "mla", "speculation", "producer-only"])
def test_v2_opt_in_does_not_widen_the_audited_profile(released_config, profile):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    config.use_v2_model_runner = True
    config.kv_transfer_config.kv_connector_extra_config["orbitkv.v2"] = True
    if profile == "missing-group":
        cache.kv_cache_groups = []
    elif profile == "mla":
        cache.kv_cache_groups[0].kv_cache_spec = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        )
    elif profile == "speculation":
        config.speculative_config = object()
    else:
        config.kv_transfer_config.kv_role = "kv_producer"
    with pytest.raises(RuntimeError, match="one uniform FullAttention"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
