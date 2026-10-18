"""Remote recovery bytes, drain and cold/warm profile validity."""

import pytest

from benches.common.metrics import REMOTE_STAGES, REQUIRED_OWNERSHIP_GAUGES
from benches.distributed.shared_cache import drain, qualify, verify_restore, verify_source_ssd
from benches.distributed.shared_cache_profile import sample_evidence, summarize


@pytest.mark.parametrize("remote,restored,text", [(0, 64, "ok"), (64, 0, "ok"), (64, 64, "wrong")])
def test_shared_cache_gate_requires_payload_gpu_copy_and_output(remote, restored, text):
    with pytest.raises(AssertionError):
        verify_restore(
            {},
            {
                "orbitkv_remote_fetch_bytes_total": remote,
                "orbitkv_load_bytes_total": restored,
                "orbitkv_remote_stage_duration_seconds_count_release": 1,
            },
            "ok",
            {"text": text},
        )


def test_shared_cache_gate_uses_counter_deltas():
    before = {"orbitkv_remote_fetch_bytes_total": 1024, "orbitkv_load_bytes_total": 2048}
    after = {
        "orbitkv_remote_fetch_bytes_total": 1088,
        "orbitkv_load_bytes_total": 2112,
        "orbitkv_remote_stage_duration_seconds_count_release": 1,
    }
    result = verify_restore(before, after, "ok", {"text": "ok", "ttft_ms": 1, "e2e_ms": 2})
    assert result["remote_bytes"] == result["h2d_bytes"] == 64
    without_ack = {**after, "orbitkv_remote_stage_duration_seconds_count_release": 0}
    with pytest.raises(AssertionError, match="completion evidence"):
        verify_restore(before, without_ack, "ok", {"text": "ok"})
    with pytest.raises(AssertionError):
        verify_restore(before, before, "ok", {"text": "ok"})


def test_source_ssd_gate_requires_successful_physical_reads():
    before = {
        "orbitkv_ssd_prefetch_bytes_total": 1024,
        "orbitkv_ssd_prefetch_success_total": 3,
    }
    after = {
        "orbitkv_ssd_prefetch_bytes_total": 5120,
        "orbitkv_ssd_prefetch_success_total": 5,
    }
    assert verify_source_ssd(before, after) == {
        "source_ssd_read_bytes": 4096,
        "source_ssd_reads": 2,
    }
    for missing in [
        {**after, "orbitkv_ssd_prefetch_bytes_total": 1024},
        {**after, "orbitkv_ssd_prefetch_success_total": 3},
    ]:
        with pytest.raises(AssertionError, match="SSD materialization"):
            verify_source_ssd(before, missing)


def test_forced_source_ssd_qualification_evicts_dram_and_checks_both_nodes(monkeypatch):
    source_empty = {"orbitkv_save_bytes_total": 0, "orbitkv_ssd_write_bytes_total": 0}
    source_saved = {"orbitkv_save_bytes_total": 4096, "orbitkv_ssd_write_bytes_total": 4096}
    source_before_read = {
        **source_saved,
        "orbitkv_ssd_prefetch_bytes_total": 0,
        "orbitkv_ssd_prefetch_success_total": 0,
    }
    source_after_read = {
        **source_saved,
        "orbitkv_ssd_prefetch_bytes_total": 4096,
        "orbitkv_ssd_prefetch_success_total": 1,
    }
    target_before = {
        "orbitkv_remote_fetch_bytes_total": 0,
        "orbitkv_load_bytes_total": 0,
        "orbitkv_remote_stage_duration_seconds_count_release": 0,
    }
    target_after = {
        "orbitkv_remote_fetch_bytes_total": 4096,
        "orbitkv_load_bytes_total": 4096,
        "orbitkv_remote_stage_duration_seconds_count_release": 1,
    }
    snapshots = iter(
        [
            [source_empty, {}],
            [source_saved, target_before],
            [source_before_read, target_before],
            [source_after_read, target_after],
        ]
    )
    monkeypatch.setattr("benches.distributed.shared_cache.drain", lambda *_: next(snapshots))
    generated = iter(
        [
            {"text": "same", "ttft_ms": 2, "e2e_ms": 3},
            {"text": "same", "ttft_ms": 1, "e2e_ms": 2},
        ]
    )
    monkeypatch.setattr("benches.distributed.shared_cache.generate", lambda *_: next(generated))
    syncs = []
    monkeypatch.setattr(
        "benches.distributed.shared_cache.synchronize", lambda *urls: syncs.append(urls)
    )
    cleanup = {"cleanup": {"evicted_blocks": 1, "still_referenced_blocks": 0}}
    monkeypatch.setattr("benches.distributed.shared_cache.evict_host_cache", lambda _: cleanup)

    result = qualify(
        engine="vllm",
        model="model",
        source_url="http://source-engine",
        target_url="http://target-engine",
        source_manager="http://source-manager",
        target_manager="http://target-manager",
        prompts=[[1] * 128],
        source_medium="ssd",
    )

    assert syncs == [("http://source-manager", "http://target-manager")] * 2
    assert result[0]["source_medium"] == "ssd"
    assert result[0]["source_preparation"] == cleanup
    assert result[0]["source_ssd_read_bytes"] == 4096
    assert result[0]["remote_bytes"] == result[0]["h2d_bytes"] == 4096


@pytest.mark.parametrize(
    "metric",
    [
        "orbitkv_transfer_reserved_bytes",
        "orbitkv_transfer_completion_outstanding",
        "orbitkv_ssd_read_pinned_bytes",
        "orbitkv_ssd_cufile_inflight_batches",
        "orbitkv_storage_codec_reserved_bytes",
    ],
)
def test_finished_requests_do_not_hide_unreleased_transfer_resources(monkeypatch, metric):
    snapshot = dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0)
    snapshot.update(orbitkv_transfer_completion_outstanding=0)
    snapshot[metric] = 1
    monkeypatch.setattr("benches.distributed.shared_cache.metrics", lambda _: snapshot)
    with pytest.raises(TimeoutError, match=metric):
        drain("source", "consumer", timeout=0)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("orbitkv_query_reserved_bytes", None),
        ("orbitkv_transfer_completion_outstanding", None),
        ("orbitkv_inflight_bytes", True),
        ("orbitkv_transfer_completion_outstanding", float("nan")),
        ("orbitkv_storage_codec_reserved_bytes", -1),
    ],
)
def test_shared_drain_rejects_missing_or_invalid_ownership(monkeypatch, metric, value):
    snapshot = dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0)
    snapshot.update(orbitkv_transfer_completion_outstanding=0)
    if value is None:
        del snapshot[metric]
    else:
        snapshot[metric] = value
    monkeypatch.setattr("benches.distributed.shared_cache.metrics", lambda _: snapshot)
    with pytest.raises(ValueError, match="ownership|metric"):
        drain("consumer", timeout=0)


def test_shared_drain_waits_for_present_ownership_to_reach_zero(monkeypatch):
    idle = dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0)
    idle.update(orbitkv_transfer_completion_outstanding=0)
    snapshots = iter([{**idle, "orbitkv_transfer_completion_outstanding": 1}, idle, idle, idle])
    monkeypatch.setattr("benches.distributed.shared_cache.metrics", lambda _: next(snapshots))
    monkeypatch.setattr("benches.distributed.shared_cache.time.sleep", lambda _: None)
    assert drain("consumer") == [idle]


def boundaries():
    before = {
        "orbitkv_remote_fetch_bytes_total": 64,
        "orbitkv_load_bytes_total": 128,
        "orbitkv_remote_fetch_duration_seconds_count": 2,
        "orbitkv_remote_fetch_duration_seconds_sum": 0.02,
    }
    after = {
        **before,
        "orbitkv_remote_fetch_bytes_total": 128,
        "orbitkv_load_bytes_total": 192,
        "orbitkv_remote_fetch_duration_seconds_count": 3,
        "orbitkv_remote_fetch_duration_seconds_sum": 0.021,
    }
    for stage in REMOTE_STAGES:
        count = f"orbitkv_remote_stage_duration_seconds_count_{stage}"
        total = f"orbitkv_remote_stage_duration_seconds_sum_{stage}"
        before.update({count: 2, total: 0.02})
        after.update({count: 3, total: 0.0202})
    restored = {
        "checked_payload_bytes": 64,
        "sha256": ["a"] * 4,
        "query_id": "query-0",
        "timing": {"query_ms": 2},
        "query_window": {
            "clock": "CLOCK_MONOTONIC",
            "pid": 43,
            "start_ticks": 222,
            "boot_id": "0f7f8389-3d2e-424f-84b8-c52d59f20908",
            "instance": "consumer",
            "namespace": "profile",
            "begin_mono_ns": 100,
            "ready_mono_ns": 2000100,
        },
    }
    return before, after, restored


@pytest.mark.parametrize(
    "field,value",
    [
        ("orbitkv_remote_fetch_bytes_total", 64),
        ("orbitkv_load_bytes_total", 128),
        ("orbitkv_remote_stage_duration_seconds_count_release", 2),
        ("orbitkv_remote_stage_duration_seconds_count_read", 4),
        ("orbitkv_remote_stage_duration_seconds_sum_read", float("nan")),
        ("orbitkv_remote_fetch_duration_seconds_sum", float("inf")),
    ],
    ids=["local-hit", "no-gpu-copy", "no-release", "multiple-reads", "nan-stage", "infinite-fetch"],
)
def test_profile_rejects_invalid_remote_sample(field, value):
    before, after, restored = boundaries()
    after[field] = value
    with pytest.raises(AssertionError):
        sample_evidence(before, after, restored, 64)


def test_profile_uses_deltas_and_separates_cold_and_warmup():
    before, after, restored = boundaries()
    evidence = sample_evidence(before, after, restored, 64)
    assert evidence["remote_bytes"] == evidence["h2d_bytes"] == 64
    assert evidence["remote_stages_ms"]["read"] == pytest.approx(0.2)
    assert evidence["query_window"] == restored["query_window"]
    rows = []
    for phase, duration in [("cold", 100), ("warmup", 50), ("measured", 1), ("measured", 3)]:
        rows.append({**evidence, "phase": phase, "manager_fetch_ms": duration})
    summary = summarize(rows)
    assert summary["measured_samples"] == 2
    assert summary["statistics"]["manager_fetch_ms"] == {
        "p50": 2,
        "p95_nearest_rank": 3,
        "min": 1,
        "max": 3,
    }
    assert "begin_mono_ns" not in summary["statistics"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("clock", "CLOCK_REALTIME"),
        ("pid", True),
        ("start_ticks", 0),
        ("boot_id", "unknown"),
        ("instance", ""),
        ("namespace", None),
        ("begin_mono_ns", 2000101),
        ("ready_mono_ns", 2000100.0),
        ("ready_mono_ns", 3000100),
    ],
)
def test_profile_rejects_ambiguous_client_query_window(field, value):
    before, after, restored = boundaries()
    restored["query_window"][field] = value
    with pytest.raises(ValueError):
        sample_evidence(before, after, restored, 64)
