"""Summarize bounded same-host client/Manager/storage diagnostic stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benches.cluster.live_store_measurements import _summary
from benches.common.artifacts import external_path


def _rows(path):
    with path.open() as file:
        return [json.loads(line) for line in file]


def _key(row):
    return row["session_epoch"], row["session_token"], row["request_id"]


def _elapsed(later, earlier):
    assert later >= earlier, (earlier, later)
    return (later - earlier) / 1_000_000


def summarize(run: Path):
    samples = _rows(run / "samples.jsonl")
    events = _rows(run / "diagnostic-events.jsonl")
    grouped = {}
    for event in events:
        grouped.setdefault(_key(event), {}).setdefault(event["stage"], []).append(event)

    stage_samples = []
    for sample in samples:
        save = sample["save_channel"]
        stages = grouped[_key(save)]
        receive = stages["publish_manager_receive"][0]["monotonic_ns"]
        process_start = stages["publish_manager_process_start"][0]["monotonic_ns"]
        process_complete = stages["publish_manager_process_complete"][0]["monotonic_ns"]
        response_publish = stages["publish_manager_response_publish"][0]["monotonic_ns"]
        storage_enqueue = stages["publish_storage_enqueue"][0]["monotonic_ns"]
        storage_dequeue = stages["publish_storage_dequeue"][0]["monotonic_ns"]
        storage_complete = stages["publish_storage_complete"][0]["monotonic_ns"]
        row = {
            "round": sample["round"],
            "measured": sample["measured"],
            "operation": "save",
            "request_id": save["request_id"],
            "client_total_ms": _elapsed(save["returned_mono_ns"], save["submitted_mono_ns"]),
            "client_to_manager_receive_ms": _elapsed(receive, save["submitted_mono_ns"]),
            "manager_runtime_queue_ms": _elapsed(process_start, receive),
            "manager_execute_ms": _elapsed(process_complete, process_start),
            "manager_complete_to_response_publish_ms": _elapsed(response_publish, process_complete),
            "response_publish_to_client_ms": _elapsed(save["returned_mono_ns"], response_publish),
            "storage_queue_ms": _elapsed(storage_dequeue, storage_enqueue),
            "storage_execute_ms": _elapsed(storage_complete, storage_dequeue),
            "client_return_to_storage_complete_ms": (storage_complete - save["returned_mono_ns"])
            / 1_000_000,
        }
        if "publish_ssd_enqueue" in stages:
            ssd_enqueue = stages["publish_ssd_enqueue"][0]["monotonic_ns"]
            ssd_dequeue = stages["publish_ssd_dequeue"][0]["monotonic_ns"]
            ssd_complete = stages["publish_ssd_complete"][0]["monotonic_ns"]
            row.update(
                {
                    "ssd_queue_ms": _elapsed(ssd_dequeue, ssd_enqueue),
                    "ssd_execute_ms": _elapsed(ssd_complete, ssd_dequeue),
                    "client_return_to_ssd_complete_ms": (ssd_complete - save["returned_mono_ns"])
                    / 1_000_000,
                    "ssd_queue_at_dequeue": stages["publish_ssd_dequeue"][0]["pending_blocks"],
                    "ssd_inflight_at_dequeue": stages["publish_ssd_dequeue"][0]["inflight_writes"],
                    "ssd_inflight_at_complete": stages["publish_ssd_complete"][0].get(
                        "inflight_writes"
                    ),
                }
            )
        stage_samples.append(row)

        for observation in sample["query_channels"]:
            stages = grouped[_key(observation)]
            receive = stages["query_manager_receive"][0]["monotonic_ns"]
            complete = stages["query_manager_process_complete"][0]["monotonic_ns"]
            row = {
                "round": sample["round"],
                "measured": sample["measured"],
                "operation": "query",
                "request_id": observation["request_id"],
                "client_total_ms": _elapsed(
                    observation["returned_mono_ns"], observation["submitted_mono_ns"]
                ),
                "client_to_manager_receive_ms": _elapsed(receive, observation["submitted_mono_ns"]),
                "manager_execute_ms": _elapsed(complete, receive),
                "manager_complete_to_client_ms": _elapsed(
                    observation["returned_mono_ns"], complete
                ),
            }
            stage_samples.append(row)

    measured = [row for row in stage_samples if row["measured"]]
    save_measured = [row for row in measured if row["operation"] == "save"]
    query_measured = [row for row in measured if row["operation"] == "query"]
    save_interval_names = sorted(
        {name for row in save_measured for name in row if name.endswith("_ms")}
    )
    query_interval_names = sorted(
        {name for row in query_measured for name in row if name.endswith("_ms")}
    )
    return {
        "contract": "s2.10-isolation-diagnosis-v1",
        "samples": len(samples),
        "measured_save_operations": sum(row["operation"] == "save" for row in measured),
        "measured_query_operations": sum(row["operation"] == "query" for row in measured),
        "save_intervals": {
            name: _summary([row[name] for row in save_measured]) for name in save_interval_names
        },
        "query_intervals": {
            name: _summary([row[name] for row in query_measured]) for name in query_interval_names
        },
        "stage_samples": stage_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=external_path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "stage_samples"}))


if __name__ == "__main__":
    main()
