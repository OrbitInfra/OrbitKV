"""Real Cache Manager and GPU registration fixtures."""

import hashlib
import json
import os
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.support.cache_manager import (
    CacheManagerProcess,
    ClientContext,
    _torch,
    find_available_port,
)


@pytest.fixture
def instance_id() -> str:
    """Generate a unique instance ID for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def namespace() -> str:
    """Generate a test namespace."""
    return "test_namespace"


@pytest.fixture
def block_hashes() -> list[bytes]:
    """Generate deterministic block hashes for testing."""
    hashes = []
    for i in range(20):
        content = f"test_block_{i}".encode()
        hash_bytes = hashlib.sha256(content).digest()
        hashes.append(hash_bytes)
    return hashes


@pytest.fixture(scope="session")
def orbitkv_server() -> Generator[CacheManagerProcess, None, None]:
    """Session-scoped fixture that starts a Cache Manager for integration tests."""
    port = find_available_port()
    server = CacheManagerProcess(port=port)

    if not server.start() or not server._binary_path:
        pytest.skip("Cache Manager binary not found or failed to start")

    yield server
    server.stop()


@pytest.fixture
def channel_client_context(
    channel_server: CacheManagerProcess, instance_id: str, namespace: str, tmp_path
) -> Generator[ClientContext, None, None]:
    """Register a minimal GPU context on the isolated Cache Manager."""
    import importlib

    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    ctx = ClientContext(
        client=orbitkv_native.CacheManagerClient(channel_server.bootstrap_socket),
        instance_id=instance_id,
        namespace=namespace,
        device_id=0,
        num_blocks=4,
        num_layers=1,
    )
    try:
        ctx.register_kv_caches()
    finally:
        if os.environ.get("ORBITKV_TEST_STRICT_SHUTDOWN") == "1":
            channel_server.record_runtime()
    yield ctx
    if os.environ.get("ORBITKV_TEST_STRICT_SHUTDOWN") == "1":
        if not channel_server.is_running():
            raise RuntimeError("Manager ended before explicit native client release")
        session_epoch = ctx.client.session_epoch
        ctx.unregister_context(strict=True)
        ctx.client.close()
        with pytest.raises(orbitkv_native.OrbitKVError, match="reconnect"):
            ctx.client.health()
        (tmp_path / "registration.native-release.json").write_text(
            json.dumps(
                {
                    "stage": "registration-close",
                    "pid": os.getpid(),
                    "start_ticks": int(
                        Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                    ),
                    "instance_id": instance_id,
                    "session_epoch": session_epoch,
                    "unregistered": True,
                    "closed": True,
                    "completed_mono_ns": time.monotonic_ns(),
                },
                indent=2,
            )
            + "\n"
        )
    elif channel_server.is_running():
        ctx.unregister_context()


@pytest.fixture
def client(orbitkv_server: CacheManagerProcess):
    """Create a local Cache Manager client for integration tests."""
    from orbitkv import CacheManagerClient

    client = CacheManagerClient(orbitkv_server.bootstrap_socket)
    yield client
    client.close()


@pytest.fixture
def client_context(
    client, instance_id: str, namespace: str
) -> Generator[ClientContext, None, None]:
    """Fixture that provides a ClientContext representing a vLLM instance.

    Args:
        client: CacheManagerClient connected to the Cache Manager
        instance_id: Unique instance identifier
        namespace: Namespace for the instance

    Returns:
        ClientContext instance (automatically registered)
    """
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ctx = ClientContext(
        client=client,
        instance_id=instance_id,
        namespace=namespace,
        device_id=0,
    )

    # Auto-register on creation
    ctx.register_kv_caches()

    yield ctx

    # Cleanup
    ctx.unregister_context()
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture
def registered_instance(client_context: ClientContext) -> Generator[str, None, None]:
    """Fixture that returns the instance_id of a registered ClientContext.

    The client_context fixture already handles registration/unregistration.
    This fixture just provides the instance_id for convenience.
    """
    yield client_context.instance_id
