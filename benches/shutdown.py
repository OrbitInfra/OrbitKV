"""Check native shutdown evidence before accepting serving measurements."""

from __future__ import annotations


def validate_engine_shutdown(log: str, engine: str) -> None:
    if engine not in {"vllm", "sglang"}:
        raise ValueError(f"Unsupported shutdown protocol: {engine}")
    if not isinstance(log, str) or not log.strip():
        raise ValueError("Missing native engine shutdown log")
    for marker in (
        "Traceback (most recent call last):",
        "EngineDeadError",
        "bootstrap socket operation failed",
        "Exception ignored in:",
        "SIGQUIT received",
    ):
        if marker in log:
            raise ValueError(f"Native engine shutdown failed: {marker}")
    required = (
        ("API server: engine client stopped", "Application shutdown complete.")
        if engine == "vllm"
        else (
            "SIGTERM received.",
            "Remaining number of requests 0.",
            "Finished server process [",
            "include_parent=False",
        )
    )
    for marker in required:
        if marker not in log:
            raise ValueError(f"Incomplete {engine} shutdown protocol: {marker}")


def validate_component_stops(cell: dict, stops: list[dict]) -> None:
    source, consumer = cell.get("source_host"), cell.get("consumer_host")
    if (
        type(source) is not int
        or type(consumer) is not int
        or min(source, consumer) < 0
        or source == consumer
    ):
        raise ValueError("Source and consumer must be distinct frozen host indices")
    expected = [
        ("engine", consumer),
        ("engine", source),
        ("manager", consumer),
        ("manager", source),
    ]
    if not isinstance(stops, list) or len(stops) != len(expected):
        raise ValueError(f"Missing complete engine/Manager stops: {cell['name']}")
    for stop, (kind, host) in zip(stops, expected, strict=True):
        name = f"{cell['name']}-{kind}-h{host}-r0"
        if (
            not isinstance(stop, dict)
            or stop.get("name") != name
            or type(stop.get("host")) is not int
            or stop.get("host") != host
            or type(stop.get("pid")) is not int
            or stop["pid"] <= 0
            or stop.get("op") != "stopped"
            or stop.get("kind") != kind
            or stop.get("was_running") is not True
            or "pre_stop_exit_code" not in stop
            or stop["pre_stop_exit_code"] is not None
            or stop.get("requested_signal") != "SIGTERM"
            or stop.get("stop_requested") is not True
            or type(stop.get("exit_code")) is not int
            or stop["exit_code"] != 0
            or stop.get("forced_cleanup") is not False
            or stop.get("remaining_group") != []
            or stop.get("ownership_retained") is not False
        ):
            raise ValueError(f"Invalid stop or engines stopped after Managers: {name}")
        if kind == "engine":
            if stop.get("shutdown_log_truncated") is not False:
                raise ValueError(f"Missing or truncated native shutdown log: {name}")
            validate_engine_shutdown(stop.get("shutdown_log"), cell["engine"])
