"""CPU process tree with a service-owned short child and an actual threaded wait."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from tests.support.registration_campaign import identity


def main() -> None:
    role, output, child_exit = sys.argv[1:]
    root = Path(output)
    owner = identity(os.getpid())
    (root / f"{role}.json").write_text(json.dumps(owner))
    if role == "child":
        raise SystemExit(int(child_exit))
    if role == "service":

        def wait_child() -> None:
            child = subprocess.Popen([sys.executable, __file__, "child", output, child_exit])
            code = child.wait(timeout=10)
            (root / "child-wait.json").write_text(json.dumps({"exit_code": code}))

        thread = threading.Thread(target=wait_child)
        thread.start()
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("CPU fixture child wait did not finish")
        return
    service = subprocess.Popen([sys.executable, __file__, "service", output, child_exit])
    code = service.wait(timeout=20)
    service_owner = json.loads((root / "service.json").read_text())
    (root / "manager.log").write_text("CPU service fixture; no GPU or native transfer")
    (root / "manager.log.cleanup.json").write_text(
        json.dumps(
            dict(
                service_owner,
                exit_code=code,
                forced_kill=False,
                reaped=True,
                remaining_processes=[],
                inspection_errors=[],
            )
        )
    )


if __name__ == "__main__":
    main()
