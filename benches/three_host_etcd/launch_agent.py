"""Launch one persistent reviewed agent with a durable PID/start-time record."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

try:
    from .environment import direct_network_environment, direct_network_environment_evidence
    from .integrity import process_python, verify_inputs, verify_process_runtime
except ImportError:
    from environment import direct_network_environment, direct_network_environment_evidence
    from integrity import process_python, verify_inputs, verify_process_runtime


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("node", "consumer", "resource-lock", "network-probe"),
        required=True,
    )
    parser.add_argument("--node")
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    verify_inputs(args.root, args.approved_input_manifest_sha256)
    settings = json.loads((args.root / "inputs/settings.json").read_text())
    runtime_evidence = verify_process_runtime(args.root, settings, args.node)
    if args.kind in ("node", "resource-lock", "network-probe"):
        if not args.node:
            parser.error("--node is required for node, lock and network agents")
        scripts = {
            "node": "node_agent.py",
            "resource-lock": "resource_lock_agent.py",
            "network-probe": "network_probe_agent.py",
        }
        script = scripts[args.kind]
        command = [
            "python3",
            str(args.root / "inputs" / script),
            "--root",
            str(args.root),
            "--node",
            args.node,
            "--approved-input-manifest-sha256",
            args.approved_input_manifest_sha256,
        ]
        suffixes = {
            "node": "agent",
            "resource-lock": "resource-lock",
            "network-probe": "network-probe",
        }
        suffix = suffixes[args.kind]
        log_path = args.root / f"{args.node}.{suffix}.log"
        record_path = args.root / f"{args.node}.{suffix}.launcher.json"
        environment = direct_network_environment(settings["network_environment"])
        if "process_qualification" in settings:
            command = [*process_python(args.root, args.node), *command[1:]]
    else:
        runtime = Path(settings["orbitkv_runtime"]["source_root"])
        command = [
            str(runtime / "runtime-venv/bin/python"),
            "-B",
            "-E",
            "-s",
            "-X",
            f"pycache_prefix={args.root / '.consumer-bytecode'}",
            "-c",
            "import runpy,sys; script=sys.argv.pop(1); sys.path[:0]=sys.argv.pop(1).split(':'); runpy.run_path(script,run_name='__main__')",
            str(args.root / "inputs/consumer_agent.py"),
            f"{runtime / 'runtime-venv/lib/python3.11/site-packages'}:{runtime / 'projection'}:{args.root / 'inputs'}",
            "--root",
            str(args.root),
            "--approved-input-manifest-sha256",
            args.approved_input_manifest_sha256,
        ]
        log_path = args.root / "consumer.agent.log"
        record_path = args.root / "consumer.launcher.json"
        site = runtime / "runtime-venv/lib/python3.11/site-packages"
        environment = direct_network_environment(
            settings["network_environment"],
            updates={
                "PYTHONPATH": f"{site}:{runtime / 'projection'}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPYCACHEPREFIX": str(args.root / ".consumer-bytecode"),
                "ETCD_BIN": str(runtime / "frozen/bin/etcd"),
                "ORBITKV_CACHE_MANAGER_BINARY": str(runtime / "frozen/bin/orbitkv-cache-manager"),
                "ORBITKV_SERVER_TEST_BINARY": str(runtime / "frozen/bin/server-tests"),
                "ORBITKV_MOONCAKE_LIB_DIR": str(runtime / "frozen/native"),
                "LD_LIBRARY_PATH": str(runtime / "frozen/native"),
                "MC_FORCE_TCP": "1",
            },
        )
        if (args.root / ".consumer-bytecode").exists() or (
            args.root / ".consumer-bytecode"
        ).is_symlink():
            raise RuntimeError("consumer bytecode prefix already exists")
    if record_path.exists() or log_path.exists():
        raise FileExistsError(record_path if record_path.exists() else log_path)
    log = log_path.open("xb")
    process = subprocess.Popen(
        command,
        cwd=args.root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    record = {
        "kind": args.kind,
        "node": args.node,
        "pid": process.pid,
        "start_ticks": start_ticks(process.pid),
        "process_group": process.pid,
        "command": command,
        "process_runtime": runtime_evidence,
        "started_unix_ns": time.time_ns(),
        "network_environment": direct_network_environment_evidence(
            environment, settings["network_environment"]
        ),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
