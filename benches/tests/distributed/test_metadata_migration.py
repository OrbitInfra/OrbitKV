"""The coordinated cutover must fence the old protocol and preserve its archive."""

import base64
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("fault", [None, "live", "wrong_format"])
def test_integrity_cutover_requires_the_exact_stopped_cluster(
    monkeypatch, tmp_path, capsys, version, fault
):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "migration", root / "scripts/migrate-metadata-format.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = uuid.uuid4()
    protocol = f"orbitkv/inventory-stream/v{version}"
    value = module.stream_format(protocol if fault != "wrong_format" else "wrong", owner)
    rows = [
        {
            "key": module.encode(b"/orbitkv/v2/test/format"),
            "value": module.encode(value),
            "mod_revision": "7",
        }
    ]
    if fault == "live":
        rows.append(
            {
                "key": module.encode(b"/orbitkv/v2/test/members/live"),
                "value": module.encode(b"live"),
            }
        )
    monkeypatch.setattr(module, "range_prefix", lambda *_: rows)
    archive = tmp_path / "archive.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migration",
            "--endpoint",
            "http://unused",
            "--cluster",
            "test",
            "--from-protocol",
            protocol,
            "--cluster-uuid",
            str(owner),
            "--archive",
            str(archive),
        ],
    )
    if fault:
        with pytest.raises(SystemExit, match="live members|current format"):
            module.main()
        assert not archive.exists()
    else:
        module.main()
        recorded = json.loads(capsys.readouterr().out)
        assert not archive.exists(), "dry run must not mutate the archive or cluster"
        assert module.INTEGRITY_PROTOCOL == "orbitkv/inventory-stream/v5"
        assert (
            module.INTEGRITY_PROTOCOL
            in (root / "crates/orbitkv-state/src/inventory.rs").read_text()
        )
        assert base64.b64decode(recorded["current_format_base64"]) == value
