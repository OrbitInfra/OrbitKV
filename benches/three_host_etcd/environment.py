"""Fail-closed process-local network environment for the three-host campaign."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_ENVIRONMENT_VARIABLES = ("NO_PROXY", "no_proxy")
UNIX_SOCKET_PATH_MAX_BYTES = 107
MANAGER_SOCKET_NAMES = ("a100-consumer", "query-source", "cold-observer")


def validate_network_environment_policy(policy: dict) -> None:
    if set(policy) != {"unset", "no_proxy"}:
        raise RuntimeError("direct network environment policy fields differ")
    if policy["unset"] != list(PROXY_ENVIRONMENT_VARIABLES):
        raise RuntimeError("direct network proxy unset list differs")
    if policy["no_proxy"] != "*":
        raise RuntimeError("direct network no-proxy policy differs")


def direct_network_environment(
    policy: dict,
    *,
    source: Mapping[str, str] | None = None,
    updates: Mapping[str, str] | None = None,
) -> dict[str, str]:
    validate_network_environment_policy(policy)
    environment = dict(os.environ if source is None else source)
    for name in (*PROXY_ENVIRONMENT_VARIABLES, *NO_PROXY_ENVIRONMENT_VARIABLES):
        environment.pop(name, None)
    environment.update({"NO_PROXY": policy["no_proxy"], "no_proxy": policy["no_proxy"]})
    if updates:
        controlled = set(updates) & {
            *PROXY_ENVIRONMENT_VARIABLES,
            *NO_PROXY_ENVIRONMENT_VARIABLES,
        }
        if controlled:
            raise RuntimeError(
                f"network environment overrides controlled variables: {sorted(controlled)}"
            )
        environment.update(updates)
    direct_network_environment_evidence(environment, policy)
    return environment


def direct_network_environment_evidence(environment: Mapping[str, str], policy: dict) -> dict:
    validate_network_environment_policy(policy)
    present = [name for name in PROXY_ENVIRONMENT_VARIABLES if name in environment]
    no_proxy = {name: environment.get(name) for name in NO_PROXY_ENVIRONMENT_VARIABLES}
    if present or set(no_proxy.values()) != {policy["no_proxy"]}:
        raise RuntimeError(
            f"process-local direct network environment differs: "
            f"proxy_variables_present={present}, no_proxy={no_proxy}"
        )
    return {
        "proxy_variables_absent": list(PROXY_ENVIRONMENT_VARIABLES),
        "no_proxy": no_proxy,
    }


def validated_manager_socket_paths(runtime: Mapping[str, object]) -> dict[str, Path]:
    configured = runtime.get("uds_root")
    if not isinstance(configured, str) or not configured:
        raise RuntimeError("Manager UDS root is not configured")
    root = Path(configured)
    if not root.is_absolute() or root.parent != Path("/tmp"):
        raise RuntimeError(f"Manager UDS root must be a direct child of /tmp: {root}")
    if not root.name.startswith("orbitkv-s2-etcd-"):
        raise RuntimeError(f"Manager UDS root is not task-specific: {root}")
    paths = {name: root / f"{name}.sock" for name in MANAGER_SOCKET_NAMES}
    too_long = {
        name: len(os.fsencode(path))
        for name, path in paths.items()
        if len(os.fsencode(path)) > UNIX_SOCKET_PATH_MAX_BYTES
    }
    if too_long:
        raise RuntimeError(
            f"Manager UDS path exceeds {UNIX_SOCKET_PATH_MAX_BYTES} bytes: {too_long}"
        )
    return paths
