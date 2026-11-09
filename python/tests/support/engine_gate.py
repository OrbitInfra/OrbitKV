"""One evidence/requirement gate for source, installed-CPU and installed-GPU tests."""

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from tests.support import engine_registry as registry
from tests.support.installed_artifacts import ENGINE_VERSIONS

TEST_ROOT = Path(__file__).resolve().parents[1]


def installed_package(name: str, version: str | None = None) -> Path:
    distribution = importlib.metadata.distribution(name if name != "orbitkv" else "orbitkv-llm")
    if version is not None and distribution.version != version:
        raise RuntimeError(f"{name}: expected official {version}, got {distribution.version}")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable"):
        raise RuntimeError(f"{name}: an installed, non-editable package is required")
    root = Path(distribution.locate_file(name)).resolve()
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != root:
        raise RuntimeError(f"{name}: source/stub import does not match its installed distribution")
    return root


def prerequisites(engine: str, evidence: str, model: str) -> dict[str, Path]:
    if evidence == "source":
        return {}
    if sys.platform != "linux":
        raise RuntimeError("installed engine gates require Linux")
    roots = {
        engine: installed_package(engine, ENGINE_VERSIONS[engine]),
        "torch": installed_package("torch"),
        "orbitkv": installed_package("orbitkv"),
    }
    native = importlib.import_module("orbitkv.orbitkv")
    if not any(str(native.__file__).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError("a real installed OrbitKV extension is required")
    if Path(native.__file__).resolve().parent != roots["orbitkv"]:
        raise RuntimeError("native extension is outside the installed OrbitKV package")
    # Import the released boundary itself, before any test can install a stub.
    importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
        if engine == "vllm"
        else "sglang.srt.mem_cache.unified_cache.unified_cache_linker"
    )
    if evidence == "installed-gpu":
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("installed-GPU gate requires a visible CUDA GPU")
        if not Path(model).is_dir() or not (Path(model) / "config.json").is_file():
            raise RuntimeError("installed-GPU gate requires --model with local config and weights")
        if not any(Path(model).glob("*.safetensors")):
            raise RuntimeError("installed-GPU gate requires local model weights")
        manager = Path(
            os.environ.get(
                "ORBITKV_CACHE_MANAGER_BINARY", str(roots["orbitkv"] / "orbitkv-cache-manager-py")
            )
        )
        if not manager.is_file() or not os.access(manager, os.X_OK):
            raise RuntimeError("installed-GPU gate requires an executable Cache Manager")
        if engine == "vllm" and os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
            raise RuntimeError("this gate requires explicit VLLM_USE_V2_MODEL_RUNNER=0")
    return roots


def pytest_addoption(parser):
    group = parser.getgroup("engine evidence")
    group.addoption("--engine", choices=tuple(ENGINE_VERSIONS))
    group.addoption("--engine-evidence", choices=tuple(registry.REGISTRY))
    group.addoption("--engine-report", type=Path, help="external JSON evidence inventory")


def pytest_configure(config):
    required = os.environ.get("ORBITKV_REQUIRE_ENGINE_TESTS") == "1"
    engine, evidence = config.getoption("engine"), config.getoption("engine_evidence")
    if required or engine or evidence:
        if not (engine and evidence):
            raise pytest.UsageError("engine gate requires both --engine and --engine-evidence")
        config.pluginmanager.register(EngineGate(config, engine, evidence, required))


class EngineGate:
    def __init__(self, config, engine, evidence, required):
        self.config, self.engine, self.evidence, self.required = config, engine, evidence, required
        self.modules = registry.modules(evidence, engine)
        self.roots = {}
        self.selected = []
        self.reports = []
        self.unapproved_skips = []

    def pytest_sessionstart(self):
        if sys.flags.optimize:
            raise pytest.UsageError("engine controls require assertions enabled (no -O)")
        try:
            self.roots = prerequisites(
                self.engine, self.evidence, self.config.getoption("model", default="")
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise pytest.UsageError(f"engine gate prerequisite failed: {error}") from error

    def pytest_ignore_collect(self, collection_path):
        if collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
            if not collection_path.is_relative_to(TEST_ROOT):
                return True
            return collection_path.relative_to(TEST_ROOT).as_posix() not in self.modules
        return None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        selected, deselected = [], []
        for item in items:
            node = item.path.relative_to(TEST_ROOT).as_posix() + "::" + item.originalname
            entry = registry.classify(node, self.evidence)
            parameter = getattr(item, "callspec", None)
            parameter_engine = parameter.params.get("engine") if parameter else None
            if (
                entry is None
                or entry[0] != self.evidence
                or self.engine not in entry[1]
                or parameter_engine not in (None, self.engine)
            ):
                deselected.append(item)
            else:
                item.user_properties.extend((("engine", self.engine), ("evidence", self.evidence)))
                selected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    def pytest_collection_finish(self, session):
        # Record the final selection after pytest applies -k/-m filters too.
        self.selected = [item.nodeid for item in session.items]

    def pytest_runtest_setup(self, item):
        self.check_imports()

    def check_imports(self):
        for package in (self.engine, "orbitkv"):
            if package in self.roots and sys.modules.get(package) is None:
                raise RuntimeError(f"installed engine gate lost its installed package: {package}")
        # Official packages may add namespace modules, but not synthetic engine
        # modules with no origin. Inspect every imported engine/native submodule.
        for name, module in tuple(sys.modules.items()):
            root_name = name.split(".", 1)[0]
            if (
                root_name not in (self.engine, "orbitkv")
                or root_name not in self.roots
                or module is None
            ):
                continue
            origin = getattr(module, "__file__", None)
            paths = (
                [Path(origin).resolve()]
                if origin
                else [Path(p).resolve() for p in getattr(module, "__path__", ())]
            )
            if not paths or any(not p.is_relative_to(self.roots[root_name]) for p in paths):
                raise RuntimeError(
                    f"installed engine gate encountered a stub/source module: {name}"
                )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = (yield).get_result()
        if report.when == "call":
            # Fixtures run after setup hooks. Check again before their teardown
            # can undo a stub installed during the test.
            try:
                self.check_imports()
            except RuntimeError as error:
                report.outcome, report.longrepr = "failed", str(error)
        if report.skipped and self.required:
            node = item.nodeid.removeprefix("tests/")
            reason = registry.REVIEWED_SKIPS.get((self.evidence, node))
            if reason is None or reason not in str(report.longrepr):
                self.unapproved_skips.append(item.nodeid)
                report.outcome = "failed"
                report.longrepr = f"Required engine gate forbids unreviewed skip: {report.longrepr}"
        if report.when == "call" or not report.passed:
            self.reports.append(
                {"node": item.nodeid, "phase": report.when, "outcome": report.outcome}
            )

    def pytest_collectreport(self, report):
        if report.skipped and self.required:
            self.unapproved_skips.append(report.nodeid)

    def pytest_sessionfinish(self, session):
        if self.required and (self.unapproved_skips or session.testscollected == 0):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        path = self.config.getoption("engine_report")
        if path is not None:
            with path.open("x") as output:
                json.dump(
                    {
                        "engine": self.engine,
                        "version": ENGINE_VERSIONS[self.engine],
                        "evidence": self.evidence,
                        "required": self.required,
                        "selected": self.selected,
                        "results": self.reports,
                        "unapproved_skips": self.unapproved_skips,
                        "exitstatus": int(session.exitstatus),
                        "hardware_qualification": False,
                    },
                    output,
                    indent=2,
                )
                output.write("\n")

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.write_sep(
            "=",
            f"{self.engine}: {self.evidence}; required={self.required}; not hardware qualification",
        )
        if self.unapproved_skips:
            terminalreporter.write_line(f"Unreviewed skips: {self.unapproved_skips}")
