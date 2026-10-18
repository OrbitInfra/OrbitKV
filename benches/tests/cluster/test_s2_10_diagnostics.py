"""Bounded same-host isolation diagnostic summaries."""

import json

from benches.cluster.summarize_live_store_diagnostics import summarize


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_diagnostic_summary_partitions_channel_manager_storage_and_ssd(tmp_path):
    save = {
        "request_id": 7,
        "session_epoch": 2,
        "session_token": 3,
        "submitted_mono_ns": 1_000_000,
        "returned_mono_ns": 7_000_000,
    }
    query = {
        "request_id": 8,
        "session_epoch": 2,
        "session_token": 4,
        "submitted_mono_ns": 10_000_000,
        "returned_mono_ns": 14_000_000,
    }
    _write(
        tmp_path / "samples.jsonl",
        [
            {
                "round": 50,
                "measured": True,
                "save_channel": save,
                "query_channels": [query],
            }
        ],
    )

    def event(stage, operation, now, **fields):
        return {
            "stage": stage,
            "request_id": operation["request_id"],
            "session_epoch": operation["session_epoch"],
            "session_token": operation["session_token"],
            "monotonic_ns": now,
            **fields,
        }

    events = [
        event("publish_manager_receive", save, 2_000_000),
        event("publish_manager_process_start", save, 3_000_000),
        event("publish_storage_enqueue", save, 4_000_000),
        event("publish_storage_dequeue", save, 4_500_000),
        event("publish_ssd_enqueue", save, 5_000_000),
        event(
            "publish_ssd_dequeue",
            save,
            5_500_000,
            pending_blocks=0,
            inflight_writes=1,
        ),
        event("publish_manager_process_complete", save, 6_000_000),
        event("publish_manager_response_publish", save, 6_500_000),
        event("publish_storage_complete", save, 7_500_000),
        event("publish_ssd_complete", save, 9_000_000, inflight_writes=0),
        event("query_manager_receive", query, 11_000_000),
        event("query_manager_process_complete", query, 13_000_000),
    ]
    _write(tmp_path / "diagnostic-events.jsonl", events)
    result = summarize(tmp_path)
    assert result["measured_save_operations"] == 1
    assert result["measured_query_operations"] == 1
    assert result["save_intervals"]["client_total_ms"]["p99"] == 6
    assert result["save_intervals"]["manager_runtime_queue_ms"]["p99"] == 1
    assert result["save_intervals"]["manager_execute_ms"]["p99"] == 3
    assert result["save_intervals"]["manager_complete_to_response_publish_ms"]["p99"] == 0.5
    assert result["save_intervals"]["response_publish_to_client_ms"]["p99"] == 0.5
    assert result["save_intervals"]["storage_queue_ms"]["p99"] == 0.5
    assert result["save_intervals"]["client_return_to_storage_complete_ms"]["p99"] == 0.5
    assert result["save_intervals"]["client_return_to_ssd_complete_ms"]["p99"] == 2
    assert result["query_intervals"]["manager_execute_ms"]["p99"] == 2
