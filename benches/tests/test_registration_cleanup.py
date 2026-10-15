"""Harness-owned authorities retire only after normal native cleanup."""

import ast
import contextlib
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


class Exporter:
    pass


class Registration:
    def __init__(self, exporter):
        self.exporter = exporter


class Client:
    def __init__(self, failure):
        self.failure = failure
        self.calls = []

    def unregister_context(self, _instance):
        self.calls.append("unregister")
        return self.failure != "unregister", "controlled unregistration failure"

    def close(self):
        self.calls.append("close")
        if self.failure == "close":
            raise RuntimeError("controlled close failure")


def cleanup(path, namespace):
    module = ast.parse(path.read_text())
    if path.stem == "live_store_soak":
        body = [
            next(
                node
                for node in ast.walk(module)
                if isinstance(node, ast.FunctionDef) and node.name == "close_node_clients"
            )
        ]
    else:
        body = [
            next(
                node
                for node in ast.walk(module)
                if isinstance(node, ast.For)
                and {"unregister_context", "close"}
                <= {
                    call.func.attr
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                }
            )
        ]
    # Execute the consumed workload cleanup body, including its failure order.
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    if path.stem == "live_store_soak":
        namespace["close_node_clients"]("source")


@pytest.mark.parametrize("workload", ["live_store_soak", "live_store_capacity", "scoped_metadata"])
@pytest.mark.parametrize("queued_hold", [False, True], ids=["sole-owner", "queued-owner"])
@pytest.mark.parametrize("failure", [None, "unregister", "close"])
def test_retired_harness_registration_releases_only_its_own_exporter_hold(
    workload, queued_hold, failure
):
    key = "source" if workload == "live_store_capacity" else ("source", "dram")
    exporter = Exporter()
    observed = weakref.ref(exporter)
    registrations = {key: Registration(exporter)}
    tensors = {key: exporter}
    queued = registrations[key] if queued_hold else None
    del exporter
    client = Client(failure)
    namespace = {
        "contextlib": contextlib,
        "clients": {key: client},
        "gpu_registrations": registrations,
        "tensors": tensors,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None)
        ),
    }
    source = Path(os.environ.get("ORBITKV_CLEANUP_SOURCE_ROOT", Path(__file__).parents[1]))
    path = source / f"{workload}.py"
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            cleanup(path, namespace)
        assert key in registrations and key in namespace["clients"] and key in tensors
        assert observed() is not None
        return
    cleanup(path, namespace)
    assert client.calls == ["unregister", "close"]
    assert not registrations
    tensors.clear()
    assert (observed() is not None) == queued_hold
    del queued
    assert observed() is None


@pytest.mark.parametrize(
    ("failure", "queued_owner"),
    [
        pytest.param(None, None, id="normal"),
        pytest.param(None, "source", id="queued-source"),
        pytest.param(None, "consumer", id="queued-consumer"),
        pytest.param("source-unregister", None, id="source-unregister-fails"),
        pytest.param("consumer-unregister", None, id="consumer-unregister-fails"),
        pytest.param("source-close", None, id="source-close-fails"),
        pytest.param("consumer-close", None, id="consumer-close-fails"),
        pytest.param("consumer-close", "source", id="partial-close-with-queued-source"),
    ],
)
def test_distributed_fault_cleanup_retires_matching_registration(failure, queued_owner):
    source = Path(os.environ.get("ORBITKV_CLEANUP_SOURCE_ROOT", Path(__file__).parents[1]))
    path = source.parent / "python/tests/integration/test_distributed_cache.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_manager_inventory_stream_faults_preserve_exact_dram_and_ssd"
    )
    loops = sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.For)),
        key=lambda node: node.lineno,
    )
    unregister = next(
        node
        for node in loops
        if isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "zip"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "unregister_context"
            for call in ast.walk(node)
        )
    )
    close = next(node for node in loops if node.lineno == unregister.end_lineno + 1)
    registrations = {node: Registration(Exporter()) for node in ("source", "consumer")}
    exporters = {node: weakref.ref(owner.exporter) for node, owner in registrations.items()}
    queued = registrations.get(queued_owner)
    owner, _, stage = (failure or "").partition("-")
    clients = [Client(stage if node == owner else None) for node in ("source", "consumer")]
    namespace = {"clients": clients, "gpu_registrations": registrations}
    body = compile(ast.Module(body=[unregister, close], type_ignores=[]), str(path), "exec")
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            exec(body, namespace)
        remaining = {"consumer"} if failure == "consumer-close" else {"source", "consumer"}
    else:
        exec(body, namespace)
        remaining = set()
        assert all(client.calls == ["unregister", "close"] for client in clients)
    assert set(registrations) == remaining
    assert all(
        (ref() is not None) == (node in remaining or node == queued_owner)
        for node, ref in exporters.items()
    )
    del queued
    assert all((ref() is not None) == (node in remaining) for node, ref in exporters.items())
