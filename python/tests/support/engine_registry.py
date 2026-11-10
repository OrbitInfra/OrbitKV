"""Reviewed evidence levels, independent of a test's directory or filename.

Source suites use lightweight/local contracts. Explicit CPU replays run the same
assertions with official packages and the real native extension, never stubs.
GPU entries describe required execution, not a passing qualification.
"""

VLLM = ("vllm",)
SGLANG = ("sglang",)
BOTH = VLLM + SGLANG

# Default evidence for every test in a suite; exceptions below are exact nodes.
SOURCE = {
    "unit/test_adapter_plugins.py": BOTH,
    "unit/test_cache_group_layout.py": VLLM,
    "unit/test_combine_hashes.py": VLLM,
    "unit/test_connector_fault_tolerance.py": VLLM,
    "unit/test_connector_save_lifecycle.py": VLLM,
    "unit/test_cuda_ipc_counters.py": BOTH,
    "unit/test_derive_namespace.py": VLLM,
    "unit/test_engine_conformance.py": BOTH,
    "unit/test_gpu_identity_ownership.py": BOTH,
    "unit/test_hash_block_size_scale.py": VLLM,
    "unit/test_hma_boundary_offloads.py": VLLM,
    "unit/test_identity.py": BOTH,
    "unit/test_installed_artifacts.py": BOTH,
    "unit/test_installed_serving.py": BOTH,
    "unit/test_manager_runtime_paths.py": BOTH,
    "unit/test_release_pins.py": BOTH,
    "unit/test_restore_admission.py": VLLM,
    "unit/test_runtime_checks.py": BOTH,
    "unit/test_save_intent_preempt_resume.py": VLLM,
    "unit/test_sglang_config.py": SGLANG,
    "unit/test_sglang_identity.py": SGLANG,
    "unit/test_sglang_shutdown.py": SGLANG,
    "unit/test_shared_cache_entrypoint.py": BOTH,
    "unit/test_strict_native_cleanup.py": BOTH,
    "unit/test_tp_shards.py": VLLM,
    "unit/test_unit_stubs.py": VLLM,
    "unit/test_vllm_launcher_imports.py": VLLM,
    "unit/test_vllm_recovery.py": VLLM,
    "unit/test_worker_kv_layout.py": VLLM,
}
CPU = {
    "integration/test_vllm_connector_contract.py": VLLM,
    "integration/test_vllm_v2_contract.py": VLLM,
    "integration/test_vllm_hma_contract.py": VLLM,
    "integration/test_sglang_admission.py": SGLANG,
    "integration/test_sglang_components.py": SGLANG,
    "integration/test_engine_fault_contracts.py": BOTH,
    "integration/test_vllm_recovery.py::test_native_contract_gates_vllm_hits": VLLM,
    "integration/test_vllm_recovery.py::test_native_contract_rejects_duplicate_or_unordered_evidence": VLLM,
    "integration/test_sglang_direct_transfer.py::test_linker_failure_never_acknowledges_gpu_destinations": SGLANG,
    "integration/test_sglang_direct_transfer.py::test_layer_publication_does_not_acknowledge_a_partially_drained_batch": SGLANG,
    "integration/test_sglang_direct_transfer.py::test_cancel_before_forward_releases_only_unsubmitted_leases": SGLANG,
}
GPU = {
    "integration/test_sglang_components.py::test_native_component_bytes_and_faults": SGLANG,
    "integration/test_vllm_hma_contract.py::test_native_hma_page_reuse_bytes": VLLM,
    "integration/test_vllm_v2_contract.py::test_native_v2_page_reuse_bytes": VLLM,
    "integration/test_cache_faults.py": BOTH,
    "e2e/test_vllm_e2e_correctness.py": VLLM,
    "e2e/test_vllm_native_pd_e2e.py": VLLM,
    "e2e/test_vllm_p2p_pp4_e2e.py": VLLM,
    "e2e/test_vllm_save_only_e2e.py": VLLM,
    "e2e/test_vllm_tp_shards_e2e.py": VLLM,
    "e2e/test_sglang_direct_e2e.py": SGLANG,
    "e2e/test_sglang_pd_e2e.py": SGLANG,
    "e2e/test_shared_cache.py": BOTH,
    "e2e/test_ssd_persistence.py": BOTH,
    "integration/test_explicit_registration.py": BOTH,
    "integration/test_gpu_identity.py": BOTH,
    "integration/test_mla_replica_registration.py": VLLM,
    "integration/test_session_watcher.py": BOTH,
    "integration/test_sglang_combined_recovery.py": SGLANG,
    "integration/test_sglang_direct_transfer.py": SGLANG,
    "integration/test_sglang_recovery.py": SGLANG,
    "integration/test_sglang_registration_lifetime.py": SGLANG,
    "integration/test_vllm_recovery.py": VLLM,
    "release/test_installed_wheel.py": BOTH,
    "release/test_shared_manager.py": BOTH,
    "release/test_static_lora.py": BOTH,
    "stress/test_recovery_faults.py": BOTH,
    "stress/test_vllm_warm_hit_stress.py": VLLM,
}
CPU_REPLAYS = {
    "unit/test_connector_fault_tolerance.py": VLLM,
    "unit/test_connector_save_lifecycle.py": VLLM,
    "unit/test_restore_admission.py": VLLM,
    "unit/test_save_intent_preempt_resume.py": VLLM,
    "unit/test_identity.py": BOTH,
    "unit/test_derive_namespace.py": VLLM,
}
REGISTRY = {"source": SOURCE, "installed-cpu": CPU, "installed-gpu": GPU}

# No exceptions are approved. Future entries must name an exact node, profile,
# reason and review link here; a free-form pytest.skip reason is not approval.
REVIEWED_SKIPS: dict[tuple[str, str], str] = {}


def classify(node: str, profile: str | None = None) -> tuple[str, tuple[str, ...]] | None:
    node = node.split("[", 1)[0]
    module = node.split("::", 1)[0]
    if profile == "installed-cpu" and module in CPU_REPLAYS:
        return profile, CPU_REPLAYS[module]
    for key in (node, module):
        for evidence, entries in REGISTRY.items():
            if key in entries:
                return evidence, entries[key]
    return None


def modules(profile: str, engine: str) -> set[str]:
    entries = dict(REGISTRY[profile])
    if profile == "installed-cpu":
        entries.update(CPU_REPLAYS)
    return {node.split("::", 1)[0] for node, engines in entries.items() if engine in engines}
