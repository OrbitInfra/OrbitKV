"""Guard complete remote evidence and released HBM-only reset semantics."""

from types import SimpleNamespace

import pytest

from benches.distributed.shared_cache_serving import clear_dram, reset_hbm, restore_evidence


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_repeated_restore_preserves_shared_prefix_identity(monkeypatch, engine):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"success": True}, text="success"
        )

    monkeypatch.setattr("benches.distributed.shared_cache_serving.requests.post", post)
    result = reset_hbm("http://engine", engine)
    assert calls == [
        "http://engine/reset_prefix_cache"
        if engine == "vllm"
        else "http://engine/flush_cache?timeout=30"
    ]
    assert "reset_external" not in result["path"]


@pytest.mark.parametrize("resident,referenced", [(0, 0), (1, 0), (0, 1)])
def test_cleanup_proves_empty_dram_even_without_new_evictions(monkeypatch, resident, referenced):
    snapshots = iter([[{}], [{"orbitkv_cache_resident_bytes": resident}]])
    monkeypatch.setattr("benches.distributed.shared_cache_serving.drain", lambda _: next(snapshots))
    monkeypatch.setattr(
        "benches.distributed.shared_cache_serving.requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"evicted_blocks": 0, "still_referenced_blocks": referenced},
        ),
    )
    if resident or referenced:
        with pytest.raises(AssertionError):
            clear_dram("http://manager")
    else:
        assert clear_dram("http://manager")["cleanup"]["evicted_blocks"] == 0


@pytest.mark.parametrize("failure", [None, "partial", "counts", "ownership"])
def test_serving_evidence_requires_complete_remote_bytes_counts_and_ownership(failure):
    source = {"orbitkv_transfer_lock_active": 0, "orbitkv_transfer_reserved_bytes": 0}
    target = {
        "orbitkv_remote_fetch_bytes_total": 4096,
        "orbitkv_load_bytes_total": 4096,
        "orbitkv_remote_stage_duration_seconds_count_release": 1,
        "orbitkv_query_reserved_bytes": 0,
        "orbitkv_transfer_completion_outstanding": 0,
    }
    expected = {"text": "same", "usage": {"prompt_tokens": 129, "completion_tokens": 8}}
    actual = {**expected, "ttft_ms": 1, "e2e_ms": 2}
    if failure == "partial":
        target["orbitkv_remote_fetch_bytes_total"] = 2048
    elif failure == "counts":
        actual["usage"] = {"prompt_tokens": 128, "completion_tokens": 8}
    elif failure == "ownership":
        source.pop("orbitkv_transfer_lock_active")
    if failure:
        with pytest.raises(AssertionError):
            restore_evidence([{}, {}], [source, target], expected, actual, 4096)
    else:
        result = restore_evidence([{}, {}], [source, target], expected, actual, 4096)
        assert result["remote_bytes"] == result["h2d_bytes"] == 4096
        assert result["source_response"] == expected and result["consumer_response"] == actual


def test_failed_restore_preserves_response_and_both_metrics_boundaries(tmp_path, monkeypatch):
    import json

    from benches.distributed.shared_cache_serving import profile

    cold = {"text": "same", "usage": {"prompt_tokens": 64, "completion_tokens": 8}}
    snapshots = iter(
        [
            [{}, {}],
            [{"orbitkv_save_bytes_total": 4096}, {}],
            [{}, {}],
            [
                {"orbitkv_transfer_lock_active": 0, "orbitkv_transfer_reserved_bytes": 0},
                {
                    "orbitkv_remote_fetch_bytes_total": 2048,
                    "orbitkv_load_bytes_total": 4096,
                    "orbitkv_remote_stage_duration_seconds_count_release": 1,
                    "orbitkv_query_reserved_bytes": 0,
                    "orbitkv_transfer_completion_outstanding": 0,
                },
            ],
        ]
    )
    monkeypatch.setattr(
        "benches.distributed.shared_cache_serving.drain", lambda *args: next(snapshots)
    )
    monkeypatch.setattr("benches.distributed.shared_cache_serving.synchronize", lambda *args: {})
    monkeypatch.setattr(
        "benches.distributed.shared_cache_serving.generate",
        lambda *args: {**cold, "ttft_ms": 1, "e2e_ms": 2},
    )
    output = tmp_path / "failed.json"
    args = SimpleNamespace(
        engine="vllm",
        model="model",
        source_url="http://source",
        target_url="http://target",
        source_manager="http://source-manager",
        target_manager="http://target-manager",
        source_medium="dram",
        bytes_per_token=64,
        block_tokens=64,
        remote_repeats=1,
        output_tokens=8,
        output=output,
    )
    with pytest.raises(AssertionError, match="Expected 4096 remote/H2D"):
        profile(args, [[1] * 64])
    result = json.loads(output.read_text())
    assert result["status"] == "INVALID_FAIL_STOP"
    assert result["samples"][0]["consumer_response"]["text"] == "same"
    assert result["samples"][0]["manager_before"] == [{}, {}]
    assert result["samples"][0]["manager_after"][1]["orbitkv_remote_fetch_bytes_total"] == 2048
    assert result["samples"][0]["status"] == "CHECKING"


def test_declared_warmup_is_retained_and_excluded_by_label(tmp_path, monkeypatch):
    from benches.distributed.shared_cache_serving import profile

    monkeypatch.setattr(
        "benches.distributed.shared_cache_serving.drain",
        lambda *args: [{"orbitkv_save_bytes_total": 4096}, {}],
    )
    # The source pre-publication snapshot must precede its fresh save.
    import benches.distributed.shared_cache_serving as serving

    snapshots = serving.drain
    first = True

    def drain(*args):
        nonlocal first
        if first:
            first = False
            return [{}, {}]
        return snapshots(*args)

    monkeypatch.setattr(serving, "drain", drain)
    monkeypatch.setattr(serving, "synchronize", lambda *args: {})
    monkeypatch.setattr(serving, "reset_hbm", lambda *args: {})
    monkeypatch.setattr(serving, "clear_dram", lambda *args: {})
    monkeypatch.setattr(serving, "restore_evidence", lambda *args: {})
    monkeypatch.setattr(serving, "generate", lambda *args: {"text": "same"})
    args = SimpleNamespace(
        engine="vllm",
        model="model",
        source_url="http://source",
        target_url="http://target",
        source_manager="http://source-manager",
        target_manager="http://target-manager",
        source_medium="dram",
        bytes_per_token=64,
        block_tokens=64,
        remote_repeats=2,
        warmup_repeats=1,
        output_tokens=8,
        output=tmp_path / "profile.json",
    )
    result = profile(args, [[1] * 64])
    assert [row["sample_kind"] for row in result["samples"]] == ["warmup", "measured", "measured"]
    assert result["samples"][0]["phase"] == "first_peer_read"
    assert all(
        row["request_end_mono_ns"] >= row["request_start_mono_ns"] for row in result["samples"]
    )
