"""CPU controls for byte coverage and retaining uncertain native ownership."""

import errno
import fcntl
import hashlib
import select
import subprocess
import sys
from pathlib import Path

import pytest

from benches.transfer.sglang_kv_oracle import (
    PAGE_BYTES,
    PAGES,
    PLANES,
    SENTINEL,
    page_maps,
    plane_bytes,
    validate_model,
    verify_plane,
)


def test_failure_report_enospc_preserves_original_foreground_error(tmp_path, monkeypatch):
    import benches.transfer.sglang_kv_oracle as oracle

    config = tmp_path / "model.json"
    config.write_text(
        '{"architectures":["Qwen3ForCausalLM"],"num_hidden_layers":64,'
        '"num_key_value_heads":8,"head_dim":128}'
    )
    original = RuntimeError("CPU-only foreground failure")

    def fail_run(args):
        raise original

    def fail_report(path, value):
        raise OSError(errno.ENOSPC, "CPU-only report failure")

    monkeypatch.setattr(oracle, "run", fail_run)
    monkeypatch.setattr(oracle, "_write", fail_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sglang_kv_oracle",
            "--model-config",
            str(config),
            "--length",
            "4096",
            "--backend",
            "kernel",
            "--manager-socket",
            "unused",
            "--manager-url",
            "unused",
            "--gpu-uuid",
            "unused",
            "--output",
            str(tmp_path / "result"),
        ],
    )
    with pytest.raises(RuntimeError) as caught:
        oracle.main()
    assert caught.value is original


def test_actual_geometry_and_partial_generations_have_disjoint_destinations():
    for length, count, payload in ((4096, 63, 1056964608), (8192, 127, 2130706432)):
        for generation in range(3):
            sources, targets = page_maps(length, generation)
            assert len(sources) == count
            assert not set(sources) & {target for target in targets if target is not None}
            assert all(target not in {0, PAGES - 1} for target in targets)
            if generation < 2:
                assert len(targets) * PAGE_BYTES * PLANES == payload
            else:
                assert sum(target is None for target in targets) == count // 4 + 1


@pytest.mark.parametrize(
    "corruption",
    [
        "payload-first",
        "payload-last",
        "sentinel-last",
        "short",
        "wrong-plane",
        "wrong-generation",
        "wrong-page-order",
    ],
)
def test_whole_pool_verifier_rejects_truncation_aliases_and_sentinel_corruption(corruption):
    sources, targets = [1, 2], [6, 4]
    expected = plane_bytes(127, 41, sources, targets, pages=8, page_bytes=32)
    assert expected[4 * 32 : 5 * 32] == hashlib.shake_256(b"sglang-kv-byte-v1:41:127:2").digest(32)
    assert expected[:32] == bytes([SENTINEL]) * 32
    assert verify_plane(bytes(expected), expected, 127) == hashlib.sha256(expected).hexdigest()
    actual = expected.copy()
    if corruption == "short":
        actual.pop()
    elif corruption in {"wrong-plane", "wrong-generation", "wrong-page-order"}:
        actual = plane_bytes(
            126 if corruption == "wrong-plane" else 127,
            42 if corruption == "wrong-generation" else 41,
            sources,
            list(reversed(targets)) if corruption == "wrong-page-order" else targets,
            pages=8,
            page_bytes=32,
        )
    else:
        position = {
            "payload-first": 6 * 32,
            "payload-last": 5 * 32 - 1,
            "sentinel-last": len(actual) - 1,
        }[corruption]
        actual[position] ^= 1
    with pytest.raises(AssertionError, match="byte mismatch|byte length"):
        verify_plane(bytes(actual), expected, 127)


def test_partial_none_targets_stay_poisoned_and_duplicates_are_rejected():
    expected = plane_bytes(3, 41, [1, 2], [6, None], pages=8, page_bytes=32)
    assert expected[4 * 32 : 5 * 32] == bytes([SENTINEL]) * 32
    with pytest.raises(ValueError, match="overlapping"):
        plane_bytes(3, 41, [1, 2], [6, 6], pages=8, page_bytes=32)
    with pytest.raises(ValueError):
        page_maps(True, 0)
    with pytest.raises(ValueError):
        page_maps(8192, 3)


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": True},
        {"num_hidden_layers": 32},
        {"use_sliding_window": True},
        {"architectures": ["OtherModel"]},
    ],
)
def test_wrong_model_layout_cannot_enter_gpu_allocation(change):
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    validate_model(config)
    with pytest.raises(ValueError):
        validate_model({**config, **change})


@pytest.mark.parametrize("write_failure", [False, True], ids=["receipt", "enospc"])
def test_unknown_owner_keeps_real_cpu_lock_even_when_receipt_write_fails(tmp_path, write_failure):
    lock = (tmp_path / "cpu-owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = """import errno,os,sys
from pathlib import Path
import benches.transfer.sglang_kv_oracle as oracle
original=oracle._write
def write(path,value):
    if sys.argv[2]=='fail':
        os.write(1,b'holding\\n')
        raise OSError(errno.ENOSPC,'CPU-only injected receipt failure')
    original(path,value)
    os.write(1,b'holding\\n')
oracle._write=write
oracle._hold_unknown(Path(sys.argv[1]),RuntimeError('CPU-only unknown-owner control'))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), "fail" if write_failure else "ok"],
        pass_fds=(lock.fileno(),),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lock.close()
    try:
        readable, _, _ = select.select([process.stdout], [], [], 10)
        assert readable, "CPU holder never reached the ownership boundary"
        assert process.stdout.readline() == b"holding\n"
        process.terminate()
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
        with (
            (tmp_path / "cpu-owner.lock").open("a") as probe,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()  # CPU-only control has no native/GPU operation.
        assert process.wait(timeout=5) == -9
        process.stdout.close()
        process.stderr.close()
    assert not Path(f"/proc/{process.pid}").exists()
    with (tmp_path / "cpu-owner.lock").open("a") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (tmp_path / "UNKNOWN-NATIVE-OWNERSHIP.json").exists() != write_failure
