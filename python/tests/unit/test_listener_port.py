"""Test listener allocation on hosts with interface-specific services."""

import importlib
import socket
from unittest.mock import Mock

import pytest

from tests.support.paths import REPO_ROOT


@pytest.mark.parametrize(
    "module_name,function_name",
    [
        ("tests.support.cache_manager", "find_available_port"),
        ("benches.common.runtime", "free_port"),
    ],
)
def test_listener_port_rejects_reserved_ranges_and_other_interfaces(
    monkeypatch, module_name, function_name
):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    module = importlib.import_module(module_name)
    find_port = getattr(module, function_name)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied_port = find_port()
        occupied.bind(("127.0.0.2", occupied_port))
        occupied.listen(1)
        free_port = find_port()
        monkeypatch.setattr(
            module.secrets,
            "randbelow",
            Mock(side_effect=[15000 - 1024, 16999 - 1024, occupied_port - 1024, free_port - 1024]),
        )

        selected = find_port()

        assert selected == free_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("0.0.0.0", selected))
            listener.listen(1)
