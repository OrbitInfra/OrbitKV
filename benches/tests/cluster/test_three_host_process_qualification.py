"""Narrow process-only qualification oracles and fail-stop guards."""

import base64

import pytest

from benches.cluster.three_host_etcd.process_qualification import ProcessQualification


def process():
    owner = ProcessQualification.__new__(ProcessQualification)
    owner.settings = {"process_qualification": {"prefix": "/orbitkv-s2-process-test/"}}
    owner.cluster_id = "7"
    owner.last_write_revision = 10
    return owner


def test_complete_oracle_rejects_wrong_cluster_revision_and_final_values():
    owner = process()
    rows = {
        "header": {"cluster_id": "7", "revision": "10"},
        "kvs": [
            {
                "key": base64.b64encode(b"/orbitkv-s2-process-test/key").decode(),
                "value": base64.b64encode(b"exact").decode(),
            }
        ],
    }
    owner.range_prefix = lambda *_args, **_kwargs: rows
    expected = {"/orbitkv-s2-process-test/key": "exact"}
    assert owner.read_exact("endpoint", expected) == rows
    for invalid in [{"cluster_id": "8", "revision": "10"}, {"cluster_id": "7", "revision": "9"}]:
        rows["header"] = invalid
        with pytest.raises(RuntimeError):
            owner.read_exact("endpoint", expected)
    rows["header"] = {"cluster_id": "7", "revision": "11"}
    with pytest.raises(RuntimeError, match="complete process oracle"):
        owner.read_exact("endpoint", {})


def test_formal_process_run_refuses_before_creating_output_without_review(tmp_path):
    owner = process()
    owner.root = tmp_path

    def rejected():
        raise PermissionError("independent launch acceptance required")

    owner.validate_launch_authorization = rejected
    with pytest.raises(PermissionError):
        owner.run_process()
    assert not (tmp_path / "process-qualification-1").exists()


def test_failed_process_cleanup_never_releases_resource_lock():
    owner = process()
    owner.state = {"state": "invalid_no_retry", "node_agents": {"node": {}}}
    owner.launched_nodes = {"node"}
    owner.nodes = {"node": {}}
    owner.network_probes = {}
    owner.resource_locks = {"node": {"pid": 7}}
    owner.remaining = lambda **_: 180

    def failed(*_args, **_kwargs):
        raise TimeoutError("owned child remains alive")

    owner.node_request = failed
    owner.release_resource_locks = lambda: pytest.fail("unsafe resource release")
    result = owner.cleanup_processes()
    assert result["safe"] is False
    assert result["resource_locks"]["node"]["status"] == "retained"
