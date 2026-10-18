"""The installed serving gate must isolate its children from the checkout."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("binary_mode", ["relative", "path"])
def test_shared_gate_anchors_inputs_before_private_working_directory(
    tmp_path, monkeypatch, engine, binary_mode
):
    from tests.e2e.test_shared_cache import test_shared_cache_serving_and_restart as shared_gate

    source = tmp_path / "checkout/python"
    model = source / "models/dense"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
        )
    )
    tools = source / "tools"
    tools.mkdir()
    for name in ("etcd", "manager"):
        binary = tools / name
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    run = tmp_path / "private-run"
    run.mkdir()
    monkeypatch.chdir(source)
    prefix = "./tools/" if binary_mode == "relative" else ""
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("ETCD_BIN", prefix + "etcd")
    monkeypatch.setenv("ORBITKV_CACHE_MANAGER_BINARY", prefix + "manager")
    monkeypatch.setattr(pytest, "importorskip", lambda _: None)

    class InputsReady(Exception):
        pass

    def check_inputs(path, *, local_files_only):
        assert Path.cwd() == run, "Engine children would inherit the source working directory"
        assert path == model.resolve() and path.is_absolute()
        assert local_files_only is True
        for variable, name in (("ETCD_BIN", "etcd"), ("ORBITKV_CACHE_MANAGER_BINARY", "manager")):
            assert Path(os.environ[variable]) == tools / name
        raise InputsReady

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=check_inputs)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "models/dense"))
    with pytest.raises(InputsReady):
        shared_gate(engine, request, run, monkeypatch)
