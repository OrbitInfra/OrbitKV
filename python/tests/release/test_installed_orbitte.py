"""Installed CUDA IPC and same-host OrbitTE peer bytes, without an inference engine.

Trigger: optional wheel features or OrbitTE Manager integration. Requires a complete
wheel, Torch, ETCD_BIN, a leased GPU and a private ORBITKV_CHANNEL_CONFIG domain.
The RDMA case additionally requires a leased RNIC and explicit device/port/GID.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from tests.support.cache_manager import CacheManagerProcess, ClientContext, find_available_port
from tests.support.cluster import etcd_server
from tests.support.installed_artifacts import distribution_snapshot
from tests.support.ipc_credit import hold_counters, release_receipt
from tests.support.runtime_checks import identity

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]


@pytest.mark.parametrize("provider", ["orbitte-tcp", "orbitte-rdma"])
def test_installed_orbitte_peer_restores_cuda_ipc_bytes(provider, tmp_path, monkeypatch):
    if os.environ.get("ORBITKV_TEST_INSTALLED_ORBITTE") != "1":
        pytest.skip("select ORBITKV_TEST_INSTALLED_ORBITTE=1 with leased runtime resources")
    import torch

    import orbitkv
    from benches.distributed.shared_cache import drain, synchronize

    native = orbitkv
    package = Path(orbitkv.__file__).resolve().parent
    before = distribution_snapshot(importlib.metadata.distribution("orbitkv-llm"), "orbitkv")
    assert Path(before["package_root"]) == package
    assert not (package.parent / "pyproject.toml").exists(), "non-editable installation required"
    assert torch.version.cuda and torch.version.cuda.split(".")[0] == "13", torch.version.cuda
    assert torch.__version__.split("+")[0].startswith("2.13."), torch.__version__
    channel = Path(os.environ["ORBITKV_CHANNEL_CONFIG"])
    assert channel.is_absolute() and channel.resolve(strict=True) == channel
    assert os.environ.get("ORBITKV_TEST_STRICT_SHUTDOWN") == "1"
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    device = os.environ.get("ORBITKV_TEST_ORBITTE_RDMA_DEVICE")
    if provider == "orbitte-rdma" and not device:
        pytest.skip("RDMA requires an explicitly selected and leased RNIC")
    monkeypatch.chdir(tmp_path)
    cluster = f"installed-orbitte-{uuid.uuid4().hex}"
    managers, contexts, credits = [], [], {}
    report = {"provider": provider, "client": identity(os.getpid()), "package": str(package)}

    def retire(index, context):
        if context.registration is None:
            evidence = context.client.retained_credit_evidence(context.instance_id)
            (tmp_path / str(index) / "failed.credit-owner.json").write_text(str(evidence) + "\n")
        context.unregister_context(strict=True)
        if index in credits:
            (tmp_path / str(index) / "credit-release.json").write_text(
                json.dumps(release_receipt(credits[index]), indent=2) + "\n"
            )

    with contextlib.ExitStack() as counters, contextlib.ExitStack() as processes:
        endpoint, _ = processes.enter_context(etcd_server(tmp_path))
        for index in range(2):
            directory = tmp_path / str(index)
            directory.mkdir()
            args = [
                "--etcd-endpoints",
                endpoint,
                "--node-id",
                f"node-{index}",
                "--cluster-name",
                cluster,
                "--peer-transfer-backend",
                provider,
                "--enable-prometheus",
            ]
            if provider == "orbitte-rdma":
                args += [
                    "--orbitte-rdma-device",
                    device,
                    "--orbitte-rdma-port",
                    os.environ["ORBITKV_TEST_ORBITTE_RDMA_PORT"],
                    "--orbitte-rdma-gid-index",
                    os.environ["ORBITKV_TEST_ORBITTE_RDMA_GID_INDEX"],
                ]
            port = find_available_port()
            server = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=find_available_port(),
                channel_service=f"orbitkv/installed/{uuid.uuid4().hex}",
                bootstrap_socket=str(tmp_path / f"manager-{index}.sock"),
                log_path=directory / "manager.log",
                runtime_python_paths=(str(package.parent),),
                extra_args=tuple(args),
            )
            managers.append(server)
            processes.callback(server.stop)
            assert server.start(), server.read_logs()
            client = native.CacheManagerClient(server.bootstrap_socket)
            processes.callback(client.close)
            context = ClientContext(
                client,
                f"client-{index}",
                cluster,
                num_blocks=8,
                num_layers=2,
                num_heads=2,
                head_size=32,
            )
            contexts.append(context)
            processes.callback(retire, index, context)
            context.register_kv_caches()
            server.record_runtime()
            handoff = directory / "original.credit-owner.json"
            handoff.write_text(
                json.dumps(
                    {
                        "owner": report["client"],
                        "instance_id": context.instance_id,
                        "evidence": json.loads(context.registration.ipc_credit_evidence),
                    },
                    indent=2,
                )
                + "\n"
            )
            credits[index] = hold_counters(counters, handoff)
        urls = [f"http://127.0.0.1:{server.http_port}" for server in managers]
        block_hashes = [hashlib.sha256(f"{cluster}-{i}".encode()).digest() for i in range(2)]
        hashes = native.BlockHashes(block_hashes)
        source, target = contexts
        expected = []
        for layer, tensor in enumerate(source.gpu_kv_caches):
            tensor.copy_(
                (
                    torch.arange(tensor.numel(), device="cuda").reshape(tensor.shape) % 127
                    + layer * 128
                ).to(tensor.dtype)
            )
            expected.append(tensor[:, :2].cpu().clone())
        for tensor in target.gpu_kv_caches:
            tensor.fill_(-1)
        torch.cuda.synchronize()
        before_source, before_target = drain(*urls)
        ok, message = source.client.save(
            source.registration, [(name, [0, 1], block_hashes) for name in source._layer_names]
        )
        assert ok, message
        drain(*urls)
        fence = synchronize(urls[0], urls[1])
        result = target.client.query_prefetch(target.instance_id, hashes, req_id="remote")
        deadline = time.monotonic() + 30
        while isinstance(result, native.QueryLoading):
            assert time.monotonic() < deadline, "remote peer query never completed"
            time.sleep(0.01)
            result = target.client.query_prefetch(target.instance_id, hashes, req_id="remote")
        assert isinstance(result, native.QueryReady) and result.num_hit_blocks == 2, result
        operation = target.client.start_restore(
            target.instance_id,
            0,
            0,
            [target._layer_names],
            [(result.lease, [[4, 5]])],
            ready_stream=torch.cuda.current_stream().cuda_stream,
        )
        status = target.client.wait_restore(operation, timeout=30)
        assert status.done and status.success, status.message
        for tensor, original in zip(target.gpu_kv_caches, expected, strict=True):
            actual = tensor.cpu()
            assert actual[:, 4:6].equal(original), "restored GPU bytes differ"
            assert actual[:, :4].eq(-1).all() and actual[:, 6:].eq(-1).all(), (
                "unrequested GPU pages changed"
            )
        after_source, after_target = drain(*urls)
        payload_bytes = sum(t.numel() * t.element_size() for t in expected)
        assert (
            after_source["orbitkv_save_bytes_total"]
            - before_source.get("orbitkv_save_bytes_total", 0)
            == payload_bytes
        )
        assert (
            after_target["orbitkv_remote_fetch_bytes_total"]
            - before_target.get("orbitkv_remote_fetch_bytes_total", 0)
            == payload_bytes
        )
        assert (
            after_target["orbitkv_load_bytes_total"]
            - before_target.get("orbitkv_load_bytes_total", 0)
            == payload_bytes
        )
        assert after_target[
            "orbitkv_remote_stage_duration_seconds_count_release"
        ] > before_target.get("orbitkv_remote_stage_duration_seconds_count_release", 0)
        for index, server in enumerate(managers):
            server.record_runtime()
            maps = Path(f"/proc/{server.process.pid}/maps").read_text()
            (tmp_path / str(index) / "manager-maps.txt").write_text(maps)
            assert str(package / "orbitkv-cache-manager-py") in maps
            assert "libcudart.so.13" in maps and "libcuda.so" in maps
            assert "libtent" not in maps and "libmooncake" not in maps
            if provider == "orbitte-rdma":
                assert "libibverbs" in maps and "libmlx5" in maps
        report.update(
            payload_bytes=payload_bytes,
            inventory_fence=fence,
            before=[before_source, before_target],
            after=[after_source, after_target],
            requested_bytes_equal=True,
            unrequested_pages_unchanged=True,
            registration_and_publish="cross-process CUDA IPC",
            restore="client-local DRAM H2D executor",
        )
    after = distribution_snapshot(importlib.metadata.distribution("orbitkv-llm"), "orbitkv")
    assert before == after, "installed package changed during qualification"
    report.update(
        installed_files_unchanged=True,
        scope="same-host installed client/Manager CUDA IPC and staged peer READ; not serving, GDR or fault reclamation",
    )
    (tmp_path / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
