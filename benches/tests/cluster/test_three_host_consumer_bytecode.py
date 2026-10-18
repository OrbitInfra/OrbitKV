"""Consumed helper bytecode controls for the full Manager fault harness."""

import hashlib
import json
import os
import py_compile
import subprocess
import sys


def test_actual_consumer_reads_frozen_source_and_rejects_unisolated_cache(tmp_path):
    runtime = tmp_path / "runtime"
    site = runtime / "runtime-venv/lib/python3.11/site-packages"
    contents = {
        "orbitkv/__init__.py": 'MARKER="trusted!"\n',
        "orbitkv/client/__init__.py": "",
        "orbitkv/client/gpu.py": 'MARKER="trusted!"\n',
        "tests/__init__.py": "",
        "tests/integration/__init__.py": "",
        "tests/integration/test_distributed_cache.py": "",
        "tests/support/__init__.py": "",
        "tests/support/cache_manager.py": "",
        "tests/support/metrics.py": "",
    }
    payload = {}
    for relative, source in contents.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        payload[path.relative_to(runtime).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    gpu = site / "orbitkv/client/gpu.py"
    gpu.write_text('MARKER="altered!"\n')
    os.utime(gpu, (1700000000, 1700000000))
    py_compile.compile(str(gpu), doraise=True)
    gpu.write_text(contents["orbitkv/client/gpu.py"])
    os.utime(gpu, (1700000000, 1700000000))
    origins = {
        "orbitkv": "orbitkv/__init__.py",
        "orbitkv.client.gpu": "orbitkv/client/gpu.py",
        "tests.integration.test_distributed_cache": "tests/integration/test_distributed_cache.py",
        "tests.support.cache_manager": "tests/support/cache_manager.py",
        "tests.support.metrics": "tests/support/metrics.py",
    }
    config = {
        "installed_payload_roots": [site.relative_to(runtime).as_posix()],
        "import_origins": {
            name: (site / path).relative_to(runtime).as_posix() for name, path in origins.items()
        },
    }
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"config": config, "payload": payload}))
    script = (
        "import json,sys,importlib; from pathlib import Path; "
        "from benches.cluster.three_host_etcd.consumer_agent import Agent; "
        "[sys.modules.pop(n,None) for n in list(sys.modules) if n=='tests' or n.startswith('tests.') or n=='orbitkv' or n.startswith('orbitkv.')]; "
        "sys.path.insert(0,sys.argv[2]); d=json.loads(Path(sys.argv[3]).read_text()); "
        "a=Agent.__new__(Agent); a.root=Path(sys.argv[1]); a.runtime=a.root/'runtime'; "
        "r=a.verify_imported_helpers(d['config'],d['payload'],{}); "
        "print(importlib.import_module('orbitkv.client.gpu').MARKER,len(r))"
    )
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path / ".consumer-bytecode")}
    command = [sys.executable, "-B", "-E", "-s"]
    rejected = subprocess.run(
        [*command, "-c", script, str(tmp_path), str(site), str(data)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert rejected.returncode != 0
    assert "bytecode isolation differs" in rejected.stderr
    isolated = subprocess.run(
        [
            *command,
            "-X",
            f"pycache_prefix={tmp_path / '.consumer-bytecode'}",
            "-c",
            script,
            str(tmp_path),
            str(site),
            str(data),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert isolated.returncode == 0, isolated.stderr
    assert isolated.stdout.strip() == "trusted! 5"
    assert not (tmp_path / ".consumer-bytecode").exists()
