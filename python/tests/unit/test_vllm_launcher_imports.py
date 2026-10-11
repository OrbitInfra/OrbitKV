"""The real Python module launcher must preserve installed-package precedence."""

import json
import os
from pathlib import Path

import pytest

from tests.support import vllm_helpers


@pytest.mark.parametrize("write_log", [False, True], ids=["discard-log", "file-log"])
@pytest.mark.parametrize(
    "path_mode",
    ["absolute", "relative", "override", "symlink-parent", "empty-caller", "source-only"],
)
def test_module_child_preserves_explicit_import_order(monkeypatch, tmp_path, write_log, path_mode):
    installed = tmp_path / "installed"
    source = tmp_path / "harness" / "python"
    cli_root = tmp_path / "cpu-modules"
    report = tmp_path / "child-import.json"
    if path_mode == "symlink-parent":
        linked = tmp_path / "linked"
        (linked / "entry").mkdir(parents=True)
        installed = source / "installed"
    for root in (installed, source):
        (root / "orbitkv").mkdir(parents=True)
    (installed / "orbitkv" / "__init__.py").write_text("ORIGIN = 'installed'\n")
    source_first = path_mode in {"empty-caller", "source-only"}
    (source / "orbitkv" / "__init__.py").write_text(
        "ORIGIN = 'source'\n"
        if source_first
        else "raise RuntimeError('source-only package shadowed the installed wheel')\n"
    )
    cli = cli_root / "vllm" / "entrypoints" / "cli"
    cli.mkdir(parents=True)
    for directory in (cli, cli.parent, cli.parent.parent):
        (directory / "__init__.py").write_text("")
    (cli / "main.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "import orbitkv\n"
        "Path(os.environ['ORBITKV_IMPORT_PROBE']).write_text(json.dumps({\n"
        "    'origin': orbitkv.ORIGIN, 'file': orbitkv.__file__, 'cwd': os.getcwd(),\n"
        "    'pythonpath': os.environ['PYTHONPATH']}))\n"
    )
    preferred = str(installed)
    if path_mode in {"relative", "override"}:
        preferred = os.path.relpath(installed, source)
    elif path_mode == "symlink-parent":
        (source / "entry-link").symlink_to(linked / "entry", target_is_directory=True)
        preferred = "entry-link/../installed"
    entries = [preferred, str(cli_root)]
    if path_mode == "empty-caller":
        entries.insert(0, "")
    elif path_mode == "source-only":
        entries = [str(cli_root)]
    pythonpath = os.pathsep.join(entries)
    monkeypatch.chdir(source)
    monkeypatch.setattr(vllm_helpers, "PYTHON_ROOT", source)
    monkeypatch.setenv("PYTHONPATH", str(source) if path_mode == "override" else pythonpath)
    monkeypatch.setenv("ORBITKV_IMPORT_PROBE", str(report))
    monkeypatch.delenv("ORBITKV_TEST_GPU_LOCK_FDS", raising=False)

    def wait_for_cpu_probe(server, timeout):
        assert server.process.wait(timeout=10) == 0, "module child imported source-only package"

    monkeypatch.setattr(vllm_helpers.VLLMServer, "_wait_for_ready", wait_for_cpu_probe)
    server = vllm_helpers.VLLMServer(
        model=str(tmp_path / "unused-model"),
        port=1,
        use_orbitkv=True,
        log_file=tmp_path / "child.log" if write_log else None,
        env_overrides={"PYTHONPATH": pythonpath} if path_mode == "override" else None,
    )
    try:
        server.__enter__()
        observed = json.loads(report.read_text())
        expected_entries = entries if path_mode == "override" else [*entries, str(source)]
        assert observed["pythonpath"] == os.pathsep.join(
            entry if os.path.isabs(entry) else os.path.join(str(source), entry)
            for entry in expected_entries
        )
        assert observed["origin"] == ("source" if source_first else "installed")
        assert (
            Path(observed["file"]).resolve()
            == ((source if source_first else installed) / "orbitkv" / "__init__.py").resolve()
        )
        assert observed["cwd"] == str(source.parent)
    finally:
        if server.process is not None:
            server.process.wait(timeout=10)
        if server.log_handle is not None:
            server.log_handle.close()
