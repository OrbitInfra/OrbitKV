"""Exercise fatal bounded dependency retries; no packages are downloaded."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallRetry(unittest.TestCase):
    def test_success_transient_and_exhausted(self):
        for failures, expected_attempts, succeeds in [
            (0, 1, True),
            (2, 3, True),
            (3, 3, False),
        ]:
            with (
                self.subTest(failures=failures),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                command = path / "install"
                command.write_text(
                    '#!/bin/sh\nn=$(cat "$COUNT" 2>/dev/null || echo 0)\n'
                    'n=$((n + 1))\necho "$n" > "$COUNT"\n'
                    f'[ "$n" -gt {failures} ]\n'
                )
                # Avoid sleeps, and make the timeout contract inspectable on macOS too.
                (path / "sleep").write_text("#!/bin/sh\nexit 0\n")
                (path / "timeout").write_text(
                    '#!/bin/sh\n[ "$1" = 900 ] || exit 90\nshift\nexec "$@"\n'
                )
                for binary in path.iterdir():
                    binary.chmod(0o755)
                result = subprocess.run(
                    ["bash", str(ROOT / "scripts/retry-install.sh"), str(command)],
                    env=dict(
                        os.environ,
                        PATH=f"{path}:{os.environ['PATH']}",
                        COUNT=str(path / "count"),
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                self.assertEqual(int((path / "count").read_text()), expected_attempts)
                if not succeeds:
                    self.assertIn("exhausted its retry budget", result.stderr)

    def test_cuda_retries_are_pinned_bounded_and_final_failure_is_fatal(self):
        action = (ROOT / "actions/install-cuda/action.yml").read_text()
        attempts = re.findall(r"    - id: (\w+)\n(.*?)(?=    - |\Z)", action, re.S)
        self.assertEqual([name for name, _ in attempts], ["first", "second", "last"])
        for name, body in attempts:
            self.assertIn("uses: Jimver/cuda-toolkit@v0.2.35", body)
            self.assertIn("cuda: '13.0.2'", body)
            if name == "first":
                self.assertNotIn("if:", body)
            else:
                prior = "first" if name == "second" else "second"
                self.assertIn(f"if: steps.{prior}.outcome == 'failure'", body)
            self.assertEqual("continue-on-error: true" in body, name != "last")
        self.assertIn("/usr/local/cuda/bin/nvcc --version", action)
        for workflow in ["ci.yml", "release.yml"]:
            source = (ROOT / "workflows" / workflow).read_text()
            self.assertNotIn("uses: Jimver/", source)
            for line in source.splitlines():
                if "sudo apt-get" in line:
                    self.assertIn("bash .github/scripts/retry-install.sh", line)
