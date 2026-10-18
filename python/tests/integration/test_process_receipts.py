"""Explicit CPU integration gate for installed-campaign process ownership tracing."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support import registration_campaign as campaign

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("control", ["normal", "child-error", "missing-trace"])
def test_real_service_child_wait_is_required_even_when_leader_exits_zero(
    tmp_path, request, control
):
    tracer = request.config.getoption("--process-trace-binary")
    if not tracer:
        pytest.fail("CPU receipt gate requires --process-trace-binary with the frozen strace")
    fixture = Path(__file__).parents[1] / "support" / "process_receipt_fixture.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (tmp_path / "pytest.log").open("w") as log:
        tracer_process = subprocess.Popen(
            [
                tracer,
                "-ff",
                "-ttt",
                "-s",
                "256",
                "-e",
                "trace=process",
                "-o",
                str(tmp_path / "process.trace"),
                sys.executable,
                "-B",
                str(fixture),
                "root",
                str(tmp_path),
                "7" if control == "child-error" else "0",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert tracer_process.wait(timeout=30) == 0
    deadline = time.monotonic() + 5
    owner = json.loads((tmp_path / "root.json").read_text())
    assert owner["parent"] == tracer_process.pid
    observed = [
        owner,
        *(json.loads((tmp_path / f"{role}.json").read_text()) for role in ("service", "child")),
    ]
    (tmp_path / "junit.xml").write_text("<testsuite><testcase name='cpu'/></testsuite>")
    cell = {
        "kind": "vllm-worker",
        "passed": 1,
        "cleanup_receipts": 1,
        "normal_exit_codes": [0],
        "adopted_exit_codes": [0],
        "log_sets": [{"pattern": "pytest.log", "count": 1}, {"pattern": "manager.log", "count": 1}],
    }
    with pytest.raises(RuntimeError, match="no terminal receipt"):
        campaign.validate_cell(cell, tmp_path, owner, observed, deadline)
    cell["process_trace_binary"] = tracer
    if control == "missing-trace":
        (tmp_path / f"process.trace.{observed[-1]['pid']}").unlink()
    if control == "normal":
        campaign.validate_cell(cell, tmp_path, owner, observed, deadline)
        tasks = campaign.process_trace_inventory(tmp_path, owner["pid"], deadline)
        assert tasks[observed[-1]["pid"]]["waited_by"] is not None
    else:
        with pytest.raises(RuntimeError):
            campaign.validate_cell(cell, tmp_path, owner, observed, deadline)
