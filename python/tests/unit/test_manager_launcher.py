"""The console entry must give the Manager its PID, signals and exit status."""

import json
import os
import signal
import subprocess
import sys
import sysconfig
import textwrap
import time

from tests.support.paths import PYTHON_ROOT


def test_manager_launcher_executes_without_a_wrapper_process(tmp_path):
    legacy_site = tmp_path / "legacy-site"
    legacy_site.mkdir()
    (legacy_site / "enum.py").write_text("raise RuntimeError('site package shadows stdlib enum')")
    binary = tmp_path / "manager"
    pid_file = tmp_path / "pid"
    arguments = tmp_path / "arguments"
    binary.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""
            import enum
            import json
            import os
            import signal
            import sys
            from pathlib import Path

            signal.signal(signal.SIGTERM, lambda *args: sys.exit(23))
            Path(sys.argv[1]).write_text(str(os.getpid()))
            Path(sys.argv[2] + ".env").write_text(json.dumps({
                "librarypath": os.environ["LD_LIBRARY_PATH"],
                "enum_file": enum.__file__,
            }))
            Path(sys.argv[2]).write_text(sys.argv[3])
            while True:
                signal.pause()
            """)
    )
    binary.chmod(0o755)
    program = textwrap.dedent("""
        import sys
        from orbitkv import _bin_utils, _cache_manager

        legacy_site = sys.argv.pop(1)
        _bin_utils.site.getsitepackages = lambda: [legacy_site]
        binary = sys.argv.pop(1)
        _cache_manager.get_cache_manager_binary = lambda: binary
        _cache_manager.main()
        """)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(legacy_site),
            str(binary),
            str(pid_file),
            str(arguments),
            "literal $()",
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(PYTHON_ROOT),
            LD_LIBRARY_PATH="/selected/cuda/compat:/selected/cuda/lib",
        ),
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not arguments.exists():
            assert process.poll() is None, process.returncode
            assert time.monotonic() < deadline, "Manager executable did not start"
            time.sleep(0.01)
        assert int(pid_file.read_text()) == process.pid
        assert arguments.read_text() == "literal $()"
        observed = json.loads((tmp_path / "arguments.env").read_text())
        assert observed["enum_file"] == os.path.join(sysconfig.get_path("stdlib"), "enum.py")
        assert observed["librarypath"].split(os.pathsep)[:2] == [
            "/selected/cuda/compat",
            "/selected/cuda/lib",
        ]
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 23
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
