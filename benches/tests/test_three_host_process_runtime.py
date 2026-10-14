"""Real interpreter controls for the process-only frozen runtime consumer."""

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from benches.tests.three_host_fixtures import freeze_inputs, stage_helpers
from benches.three_host_etcd import integrity


def execute(root, source, *, prefix="", executable=sys.executable):
    # Runtime inventory controls need a small deterministic stdlib root;
    # real three-host readiness separately checks each complete host runtime.
    stdlib = root / "control-stdlib"
    stdlib.mkdir(exist_ok=True)
    (stdlib / "control.py").write_text("VALUE = 1\n")
    site_root = Path(__import__("requests").__file__).parent.parent
    setup = (
        f"import sys; sys.path[:0]=[{str(Path.cwd())!r},{str(site_root)!r}]; "
        "import sysconfig; original_path=sysconfig.get_path; "
        f"sysconfig.get_path=lambda name,*a,**k: {str(stdlib)!r} if name=='stdlib' else original_path(name,*a,**k); "
    )
    return subprocess.run(
        [
            executable,
            "-B",
            "-E",
            "-s",
            "-S",
            "-P",
            "-X",
            f"pycache_prefix={root / '.runtime-bytecode'}",
            "-c",
            setup + prefix + source,
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
    stage_helpers(inputs, process_only=True, stub=True)
    (inputs / "settings.json").write_text(
        json.dumps({"process_qualification": {}, "nodes": [{"name": "node"}]})
    )
    approved = freeze_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="node.runtime.json"):
        integrity.verify_inputs(tmp_path, approved)


def test_launcher_runtime_mismatch_creates_no_child_or_ownership(frozen):
    inputs = frozen / "inputs"
    stage_helpers(inputs, process_only=True)
    snapshot_path = inputs / "node.runtime.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["python"] = "/unavailable/frozen-python"
    snapshot_path.write_text(json.dumps(snapshot))
    (inputs / "settings.json").write_text(
        json.dumps({"process_qualification": {}, "nodes": [{"name": "node"}]})
    )
    approved = freeze_inputs(frozen)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-E",
            "-s",
            "-S",
            "-P",
            "-X",
            f"pycache_prefix={frozen / '.runtime-bytecode'}",
            "-c",
            f"import sys,runpy; sys.path[:0]=[{str(Path(__import__('requests').__file__).parent.parent)!r},{str(inputs)!r}]; runpy.run_path(sys.argv.pop(1),run_name='__main__')",
            str(inputs / "launch_agent.py"),
            "--root",
            str(frozen),
            "--kind",
            "resource-lock",
            "--node",
            "node",
            "--approved-input-manifest-sha256",
            approved,
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


def test_unreviewed_site_startup_is_disabled_before_verification(frozen):
    virtual = frozen / "unreviewed-venv"
    made = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(virtual)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert made.returncode == 0, made.stderr
    site = next((virtual / "lib").glob("python*/site-packages"))
    marker = frozen / "unreviewed-startup"
    (site / "unreviewed.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('unfrozen code ran')\n"
    )
    command = [str(virtual / "bin/python"), "-B", "-E", "-s", "-c", "print('started')"]
    inherited = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert inherited.returncode == 0 and marker.exists()
    marker.unlink()
    isolated = execute(frozen, VERIFY, executable=str(virtual / "bin/python"))
    assert isolated.returncode == 0, isolated.stderr
    assert not marker.exists()


def test_optional_transitive_import_is_blocked_before_execution(frozen):
    optional = frozen / "optional"
    optional.mkdir()
    marker = frozen / "unfrozen-import"
    (optional / "simplejson.py").write_text(
        f"from json import *\nopen({str(marker)!r},'w').write('executed')\n"
    )
    (frozen / "inputs/SHA256SUMS").write_text("")
    script = (
        "import sys,json,pathlib; from benches.three_host_etcd.integrity import install_import_guard; "
        "r=pathlib.Path(sys.argv[1]); s=json.loads((r/'inputs/node.runtime.json').read_text()); "
        "install_import_guard(r,s); sys.path.insert(0,str(r/'optional')); "
        "import simplejson"
    )
    rejected = execute(frozen, script)
    assert rejected.returncode != 0
    assert "unfrozen runtime import: simplejson" in rejected.stderr
    assert not marker.exists()


def test_actual_bootstrap_rejects_extra_stdlib_shadow_before_its_execution(frozen):
    inputs = frozen / "inputs"
    stage_helpers(inputs)
    (inputs / "settings.json").write_text(json.dumps({"nodes": [{"name": "node"}]}))
    approved = freeze_inputs(frozen)
    marker = frozen / "stdlib-shadow-executed"
    (inputs / "hashlib.py").write_text(f"open({str(marker)!r},'w').write('executed')\n")
    with pytest.raises(RuntimeError, match="file inventory differs"):
        integrity.process_python(frozen, "node", approved)
    assert not marker.exists()
    (inputs / "hashlib.py").unlink()
    helper = inputs / "integrity.py"
    helper.write_text(helper.read_text() + "\n# unreviewed modified helper\n")
    freeze_inputs(frozen)
    with pytest.raises(RuntimeError, match="independently approved digest"):
        integrity.process_python(frozen, "node", approved)
