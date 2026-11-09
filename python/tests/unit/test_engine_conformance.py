"""Negative controls for the harness, not evidence about an inference engine."""

import ast
import importlib.metadata
from types import SimpleNamespace

import pytest

from tests.support import engine_gate as gate
from tests.support import engine_registry as registry

pytest_plugins = ["pytester"]


def declared_nodes(tree, prefix=()):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield from declared_nodes(node, (*prefix, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield "::".join((*prefix, node.name))


def test_declared_nodes_include_async_methods_and_class_identity():
    tree = ast.parse("""
async def test_async(): pass
class TestFirst:
    async def test_same(): pass
class TestSecond:
    def test_same(): pass
""")
    assert set(declared_nodes(tree)) == {
        "test_async",
        "TestFirst::test_same",
        "TestSecond::test_same",
    }


def test_registry_covers_engine_imports_and_resolves_every_declared_node():
    roots = ("vllm", "sglang", "orbitkv.vllm", "orbitkv.sglang")
    entries = {node for rows in registry.REGISTRY.values() for node in rows}
    entries.update(registry.CPU_REPLAYS)
    modules = {node.split("::", 1)[0] for node in entries}
    for node in entries:
        relative, _, function = node.partition("::")
        path = gate.TEST_ROOT / relative
        assert path.is_file(), f"missing registry suite: {node}"
        if function:
            definitions = set(declared_nodes(ast.parse(path.read_text())))
            assert function in definitions, f"stale registry selector: {node}"
    for path in gate.TEST_ROOT.rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        if any(name == root or name.startswith(root + ".") for name in imports for root in roots):
            assert path.relative_to(gate.TEST_ROOT).as_posix() in modules, path


def test_exact_class_selector_does_not_admit_another_class_method(pytester, monkeypatch):
    monkeypatch.setenv("ORBITKV_REQUIRE_ENGINE_TESTS", "1")
    pytester.makeconftest(f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(gate.TEST_ROOT.parent)!r})
from tests.support import engine_gate as gate, engine_registry as registry
pytest_plugins = ["tests.support.engine_gate"]
gate.TEST_ROOT = Path(__file__).parent
registry.REGISTRY = {{
    "source": {{"test_control.py": ("vllm",)}},
    "installed-cpu": {{"test_control.py::TestAdmitted::test_same": ("vllm",)}},
}}
registry.CPU_REPLAYS = {{}}
gate.prerequisites = lambda *args: {{}}
""")
    pytester.makepyfile(
        test_control="""
class TestAdmitted:
    def test_same(self): pass
class TestOther:
    def test_same(self):
        raise AssertionError("a different class was incorrectly admitted")
"""
    )
    result = pytester.runpytest_subprocess(
        "--engine=vllm",
        "--engine-evidence=installed-cpu",
        timeout=30,
    )
    result.assert_outcomes(passed=1, deselected=1)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("evidence", ["source", "installed-cpu", "installed-gpu"])
@pytest.mark.parametrize("failure", ["skip", "collection-skip", "empty", "prerequisite"])
def test_required_gate_cannot_report_success_after_missing_evidence(
    pytester, monkeypatch, engine, evidence, failure
):
    monkeypatch.setenv("ORBITKV_REQUIRE_ENGINE_TESTS", "1")
    # This child exercises pytest enforcement only. It never claims installed
    # evidence: real-package prerequisite validation has separate controls below.
    pytester.makeconftest(f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(gate.TEST_ROOT.parent)!r})
from tests.support import engine_gate as gate, engine_registry as registry
pytest_plugins = ["tests.support.engine_gate"]
gate.TEST_ROOT = Path(__file__).parent
registry.REGISTRY = {{{evidence!r}: {{"test_control.py": ({engine!r},)}}}}
registry.CPU_REPLAYS = {{}}
def preflight(*args):
    if {failure!r} == "prerequisite":
        raise RuntimeError("required prerequisite removed")
    return {{}}
gate.prerequisites = preflight
""")
    pytester.makepyfile(
        test_control=f"""
import pytest
if {failure!r} == "collection-skip":
    pytest.skip("forced collection skip", allow_module_level=True)
def test_control():
    if {failure!r} == "skip":
        pytest.skip("forced test skip")
"""
    )
    args = ["--engine=" + engine, "--engine-evidence=" + evidence, "-m", ""]
    if failure == "empty":
        args += ["-k", "absent"]
    result = pytester.runpytest_subprocess(*args, timeout=30)
    assert result.ret != 0, result.stdout.str()
    if failure == "skip":
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*forbids unreviewed skip*"])
    if failure == "prerequisite":
        assert "prerequisite removed" in result.stderr.str()


@pytest.mark.parametrize(
    "engine,missing",
    [
        (engine, missing)
        for engine in ("vllm", "sglang")
        for missing in ("engine", "torch", "extension", "cuda", "model", "manager")
    ]
    + [("vllm", "v1-runner"), ("vllm", "v2-runner")],
)
def test_installed_preflight_rejects_each_missing_prerequisite(
    monkeypatch, tmp_path, engine, missing
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"fixture")
    manager = tmp_path / "manager"
    manager.touch(mode=0o700)
    monkeypatch.setenv("ORBITKV_CACHE_MANAGER_BINARY", str(manager))
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if missing == "v1-runner" else "0")
    monkeypatch.setattr(gate.sys, "platform", "linux")

    def package(name, version=None):
        if (missing == "engine" and name == engine) or (missing == "torch" and name == "torch"):
            raise importlib.metadata.PackageNotFoundError(name)
        return tmp_path

    monkeypatch.setattr(gate, "installed_package", package)
    native = SimpleNamespace(
        __file__=str(tmp_path / ("native" + gate.importlib.machinery.EXTENSION_SUFFIXES[0]))
    )
    original_import = gate.importlib.import_module

    def import_module(name):
        if name == "orbitkv.orbitkv":
            if missing == "extension":
                raise ImportError("native extension removed")
            return native
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: missing != "cuda"))
        if name.startswith(("vllm.", "sglang.")):
            return SimpleNamespace()  # Not inserted into sys.modules; harness unit only.
        return original_import(name)

    monkeypatch.setattr(gate.importlib, "import_module", import_module)
    if missing == "model":
        (model / "config.json").unlink()
    if missing == "manager":
        manager.unlink()
    messages = {
        "engine": engine,
        "torch": "torch",
        "extension": "native extension removed",
        "cuda": "visible CUDA GPU",
        "model": "local config and weights",
        "manager": "executable Cache Manager",
        "v1-runner": "VLLM_USE_V2_MODEL_RUNNER=0",
        "v2-runner": "VLLM_USE_V2_MODEL_RUNNER=1",
    }
    with pytest.raises((ImportError, RuntimeError), match=messages[missing]):
        gate.prerequisites(
            engine, "installed-gpu", str(model), "v2" if missing == "v2-runner" else "v1"
        )


@pytest.mark.parametrize("runner,marked", [("v2", True), ("v2", False), ("v1", True)])
def test_gpu_selection_honors_the_runner_scope(pytester, monkeypatch, runner, marked):
    monkeypatch.setenv("ORBITKV_REQUIRE_ENGINE_TESTS", "1")
    pytester.makeconftest(f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(gate.TEST_ROOT.parent)!r})
from tests.support import engine_gate as gate, engine_registry as registry
pytest_plugins = ["tests.support.engine_gate"]
gate.TEST_ROOT = Path(__file__).parent
registry.REGISTRY = {{"installed-gpu": {{"test_control.py": ("vllm",)}}}}
registry.CPU_REPLAYS = {{}}
gate.prerequisites = lambda *args: {{}}
""")
    pytester.makepyfile(
        test_control=f"""
import pytest
{"@pytest.mark.vllm_v2(only=True)" if marked else ""}
def test_control(): assert {runner!r} == "v2"
def test_ordinary(): assert {runner!r} == "v1"
"""
    )
    result = pytester.runpytest_subprocess(
        "--engine=vllm",
        "--engine-evidence=installed-gpu",
        f"--vllm-runner={runner}",
        timeout=30,
    )
    if marked:
        result.assert_outcomes(passed=1, deselected=1)
    else:
        result.assert_outcomes(deselected=2)
        assert result.ret != 0


@pytest.mark.parametrize("invalid", ["version", "editable", "source", "stub"])
def test_installed_engine_provenance_rejects_non_release_inputs(monkeypatch, tmp_path, invalid):
    package = tmp_path / "vllm"
    package.mkdir()
    distribution = SimpleNamespace(
        version="wrong" if invalid == "version" else "0.33.0",
        locate_file=lambda _: package,
        read_text=lambda _: '{"dir_info":{"editable":true}}' if invalid == "editable" else "{}",
    )
    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda _: distribution)
    root = tmp_path / "source" if invalid == "source" else package
    spec = None if invalid == "stub" else SimpleNamespace(origin=str(root / "__init__.py"))
    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda _: spec)
    message = {
        "version": "expected official",
        "editable": "non-editable package",
        "source": "source/stub import",
        "stub": "source/stub import",
    }[invalid]
    with pytest.raises(RuntimeError, match=message):
        gate.installed_package("vllm", "0.33.0")


def test_mixed_suite_and_cpu_replays_have_explicit_evidence():
    assert (
        registry.classify(
            "unit/test_connector_save_lifecycle.py::test_preemption_waits_for_every_save_task"
        )[0]
        == "source"
    )
    assert (
        registry.classify(
            "unit/test_connector_save_lifecycle.py::test_preemption_waits_for_every_save_task",
            "installed-cpu",
        )[0]
        == "installed-cpu"
    )
    assert (
        registry.classify(
            "integration/test_sglang_direct_transfer.py::test_cancel_before_forward_releases_only_unsubmitted_leases"
        )[0]
        == "installed-cpu"
    )
    assert (
        registry.classify(
            "integration/test_sglang_direct_transfer.py::test_direct_page_transfer_overwrites_poisoned_gpu_slots"
        )[0]
        == "installed-gpu"
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("evidence", ["installed-cpu", "installed-gpu"])
def test_last_test_cannot_hide_an_engine_stub_in_fixture_teardown(
    pytester, monkeypatch, engine, evidence
):
    monkeypatch.setenv("ORBITKV_REQUIRE_ENGINE_TESTS", "1")
    # Harness self-test only: no installed-engine evidence is claimed here.
    pytester.makeconftest(f"""
import sys
from pathlib import Path
from types import ModuleType
sys.path.insert(0, {str(gate.TEST_ROOT.parent)!r})
from tests.support import engine_gate as gate, engine_registry as registry
pytest_plugins = ["tests.support.engine_gate"]
gate.TEST_ROOT = Path(__file__).parent
registry.REGISTRY = {{{evidence!r}: {{"test_control.py": ({engine!r},)}}}}
registry.CPU_REPLAYS = {{}}
def preflight(*args):
    root = gate.TEST_ROOT / {engine!r}
    module = ModuleType({engine!r})
    module.__file__ = str(root / "__init__.py")
    sys.modules[{engine!r}] = module
    return {{{engine!r}: root}}
gate.prerequisites = preflight
""")
    pytester.makepyfile(
        test_control=f"""
import sys
from types import ModuleType
def test_control(monkeypatch):
    monkeypatch.setitem(sys.modules, {engine + ".injected"!r}, ModuleType("injected"))
"""
    )
    result = pytester.runpytest_subprocess(
        "--engine=" + engine, "--engine-evidence=" + evidence, timeout=30
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*stub/source module*"])
