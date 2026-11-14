"""Exercise bounded installs and mirror selection without downloading packages."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mirrors", ROOT / "scripts/apt-mirrors.py"
)
MIRRORS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIRRORS)


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
                (path / "install").write_text(
                    '#!/bin/sh\necho "$ORBITKV_INSTALL_MIRROR" >> "$COUNT"\n'
                    'n=$(wc -l < "$COUNT")\n'
                    f'[ "$n" -gt {failures} ]\n'
                )
                (path / "sleep").write_text("#!/bin/sh\nexit 0\n")
                (path / "timeout").write_text(
                    '#!/bin/sh\n[ "$1" = 900 ] || exit 90\nshift\nexec "$@"\n'
                )
                for binary in path.iterdir():
                    binary.chmod(0o755)
                result = subprocess.run(
                    [
                        "bash",
                        str(ROOT / "scripts/retry-install.sh"),
                        str(path / "install"),
                    ],
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
                self.assertEqual(
                    (path / "count").read_text().splitlines(),
                    ["primary", "alternate", "primary"][:expected_attempts],
                )
                if not succeeds:
                    self.assertIn("exhausted its retry budget", result.stderr)

    def test_failed_update_never_installs_from_stale_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
            (path / "sleep").write_text("#!/bin/sh\nexit 0\n")
            (path / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
            (path / "apt-get").write_text(
                '#!/bin/sh\necho "$*" >> "$COUNT"\nexit 100\n'
            )
            for binary in path.iterdir():
                binary.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/retry-install.sh"),
                    "apt-get",
                    "install",
                    "-y",
                    "pkg",
                ],
                env=dict(
                    os.environ,
                    PATH=f"{path}:{os.environ['PATH']}",
                    COUNT=str(path / "count"),
                ),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            calls = (path / "count").read_text().splitlines()
            self.assertEqual(len(calls), 3)
            for call in calls:
                self.assertTrue(call.endswith(" update"))
                self.assertIn("APT::Update::Error-Mode=any", call)
            self.assertIn("mirror=alternate", result.stderr)
            self.assertIn("exhausted its retry budget", result.stderr)

    def test_sources_change_only_mirrors_and_keep_signature_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "apt"
            (source / "sources.list.d").mkdir(parents=True)
            legacy = "deb [signed-by=/usr/share/keyrings/ubuntu.gpg] http://archive.ubuntu.com/ubuntu jammy main\n"
            modern = "Types: deb\nURIs: http://ports.ubuntu.com/ubuntu-ports\nSuites: noble noble-updates\nComponents: main\nSigned-By: /usr/share/keyrings/ubuntu.gpg\n"
            cuda = "deb [signed-by=/usr/share/keyrings/cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /\n"
            (source / "sources.list").write_text(legacy)
            (source / "sources.list.d/ubuntu.sources").write_text(modern)
            (source / "sources.list.d/cuda.list").write_text(cuda)
            (source / "sources.list.d/disabled.list.save").write_text("do not enable")
            for mode in ["primary", "alternate"]:
                destination = root / mode
                MIRRORS.sources(source, destination, mode)
                self.assertEqual(
                    (destination / "sources.list").read_text(),
                    legacy.replace(
                        "http://archive.ubuntu.com",
                        "https://azure.archive.ubuntu.com"
                        if mode == "alternate"
                        else "https://archive.ubuntu.com",
                    ),
                )
                self.assertEqual(
                    (destination / "sources.list.d/ubuntu.sources").read_text(),
                    modern.replace(
                        "http://ports.ubuntu.com",
                        "https://mirrors.tuna.tsinghua.edu.cn"
                        if mode == "alternate"
                        else "https://ports.ubuntu.com",
                    ),
                )
                self.assertEqual(
                    (destination / "sources.list.d/cuda.list").read_text(),
                    cuda.replace("nvidia.com", "nvidia.cn")
                    if mode == "alternate"
                    else cuda,
                )
                self.assertFalse(
                    (destination / "sources.list.d/disabled.list.save").exists()
                )
            self.assertEqual((source / "sources.list").read_text(), legacy)

    def test_cuda_and_all_native_jobs_share_the_bounded_installer(self):
        installer = (ROOT / "scripts/install-cuda.sh").read_text()
        self.assertLess(
            installer.index("sha256sum --check --strict"), installer.index("sudo dpkg")
        )
        for pin in [
            "cuda-nvcc-13-0=13.0.88-1",
            "cuda-cudart-dev-13-0=13.0.96-1",
            "cuda-nvrtc-dev-13-0=13.0.88-1",
        ]:
            self.assertIn(pin, installer)
        self.assertIn(
            'bash "$scripts/retry-install.sh" bash "$scripts/install-cuda.sh" keyring',
            installer,
        )
        self.assertIn('bash "$scripts/retry-install.sh" apt-get install', installer)
        self.assertIn("developer.download.nvidia.cn", installer)
        for workflow, count in [("ci.yml", 3), ("release.yml", 2)]:
            source = (ROOT / "workflows" / workflow).read_text()
            self.assertEqual(
                source.count("uses: ./.github/actions/install-cuda"), count
            )
            self.assertEqual(
                source.count("bash .github/scripts/retry-install.sh apt-get install"),
                count,
            )
            self.assertNotIn("sudo apt-get", source)
            self.assertNotIn("wget", source)
        release = (
            (ROOT / "workflows/release.yml").read_text().split("  publish-pypi:\n")[1]
        )
        self.assertNotIn("retry-install", release)
