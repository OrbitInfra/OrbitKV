"""Real interpreter controls for the process-only frozen runtime consumer."""

import hashlib
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from benches.three_host_etcd import integrity


def execute(root, source, *, prefix=""):
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-E",
            "-s",
            "-X",
            f"pycache_prefix={root / '.runtime-bytecode'}",
            "-c",
            prefix + source,
            str(root),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.fixture
def frozen(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    snapshot = execute(
        tmp_path,
        "import json; from benches.three_host_etcd import campaign,node_agent,launch_agent,network_probe_agent,resource_lock_agent; from benches.three_host_etcd.integrity import process_runtime_snapshot; print(json.dumps(process_runtime_snapshot()))",
    )
    assert snapshot.returncode == 0, snapshot.stderr
    (inputs / "node.runtime.json").write_text(snapshot.stdout)
    return tmp_path


VERIFY = (
    "import json,pathlib,sys; from benches.three_host_etcd.integrity import verify_process_runtime; "
    "r=pathlib.Path(sys.argv[1]); print(json.dumps(verify_process_runtime(r,{'process_qualification':{}},'node')))"
)


def test_actual_process_runtime_rejects_changed_digest_and_shadowed_import(frozen):
    valid = execute(frozen, VERIFY)
    assert valid.returncode == 0, valid.stderr
    path = frozen / "inputs/node.runtime.json"
    snapshot = json.loads(path.read_text())
    original = dict(snapshot["files"])
    snapshot["files"][snapshot["python"]] = "0" * 64
    path.write_text(json.dumps(snapshot))
    changed = execute(frozen, VERIFY)
    assert changed.returncode != 0
    assert "file inventory or digest differs" in changed.stderr
    snapshot["files"] = original
    path.write_text(json.dumps(snapshot))
    shadow = frozen / "shadow/requests"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("SHADOW = True\n")
    rejected = execute(
        frozen,
        VERIFY,
        prefix=f"import sys; sys.path.insert(0,{str(shadow.parent)!r}); ",
    )
    assert rejected.returncode != 0
    assert "import origin differs: requests" in rejected.stderr


def test_runtime_manifest_requires_every_node_snapshot_before_launch(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in (
        "campaign.py",
        "consumer_agent.py",
        "environment.py",
        "integrity.py",
        "launch_agent.py",
        "network_probe_agent.py",
        "node_agent.py",
        "proxy_readiness.py",
        "resource_lock_agent.py",
        "validate_inputs.py",
        "process_qualification.py",
    ):
        (inputs / name).write_text("\n")
    (inputs / "settings.json").write_text(
        json.dumps({"process_qualification": {}, "nodes": [{"name": "node"}]})
    )
    manifest = inputs / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{integrity.sha256(p)}  {p.name}\n" for p in sorted(inputs.iterdir()))
    )
    with pytest.raises(RuntimeError, match="node.runtime.json"):
        integrity.verify_inputs(tmp_path, hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_launcher_runtime_mismatch_creates_no_child_or_ownership(frozen):
    inputs = frozen / "inputs"
    for name in (
        "campaign.py",
        "consumer_agent.py",
        "environment.py",
        "integrity.py",
        "launch_agent.py",
        "network_probe_agent.py",
        "node_agent.py",
        "proxy_readiness.py",
        "resource_lock_agent.py",
        "validate_inputs.py",
        "process_qualification.py",
    ):
        (inputs / name).write_bytes((Path(integrity.__file__).parent / name).read_bytes())
    snapshot_path = inputs / "node.runtime.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["version"] = "wrong-interpreter"
    snapshot_path.write_text(json.dumps(snapshot))
    (inputs / "settings.json").write_text(
        json.dumps({"process_qualification": {}, "nodes": [{"name": "node"}]})
    )
    manifest = inputs / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{integrity.sha256(p)}  {p.name}\n" for p in sorted(inputs.iterdir()))
    )
    result = subprocess.run(
        [
            *integrity.process_python(frozen, "node"),
            str(inputs / "launch_agent.py"),
            "--root",
            str(frozen),
            "--kind",
            "resource-lock",
            "--node",
            "node",
            "--approved-input-manifest-sha256",
            integrity.sha256(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "interpreter/runtime contract differs" in result.stderr
    assert not (frozen / "node.resource-lock.launcher.json").exists()
    assert not (frozen / "node.resource-lock.log").exists()


def test_inherited_bytecode_is_bypassed_and_task_cache_is_rejected(frozen):
    module = frozen / "frozen_module.py"
    module.write_text('VALUE="expected"\n')
    mutation = frozen / "mutation.py"
    mutation.write_text('VALUE="unexpected"\n')
    bytecode = Path(py_compile.compile(str(module)))
    wrong = Path(py_compile.compile(str(mutation))).read_bytes()
    bytecode.write_bytes(bytecode.read_bytes()[:16] + wrong[16:])
    source = "import sys; sys.path.insert(0,sys.argv[1]); import frozen_module; print(frozen_module.VALUE)"
    inherited = subprocess.run(
        [sys.executable, "-B", "-E", "-s", "-c", source, str(frozen)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert inherited.returncode == 0 and inherited.stdout.strip() == "unexpected"
    isolated = execute(frozen, source)
    assert isolated.returncode == 0 and isolated.stdout.strip() == "expected"
    assert not (frozen / ".runtime-bytecode").exists()
    (frozen / ".runtime-bytecode").mkdir()
    rejected = execute(frozen, VERIFY)
    assert rejected.returncode != 0
    assert "bytecode/environment isolation differs" in rejected.stderr
