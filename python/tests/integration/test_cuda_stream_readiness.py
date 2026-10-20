"""GPU readiness primitive gate, before any Manager or adapter qualification."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.support.registration_campaign import identity, write

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _stream_gate(library, flags):
    gate = ctypes.CDLL(str(library))
    if gate.orbitkv_test_stream_gate_version() != 1:
        raise RuntimeError("Stream gate helper version differs")
    for name in ("orbitkv_test_stream_wait", "orbitkv_test_stream_release"):
        function = getattr(gate, name)
        function.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        function.restype = ctypes.c_int

    def call(name, stream, index):
        status = getattr(gate, name)(stream.cuda_stream, flags.data_ptr() + index * 4)
        if status != 0:
            raise RuntimeError(f"Stream gate primitive failed: {name} status={status}")

    return call


def test_explicit_cuda_stream_gate_has_independent_release_and_exact_bytes(tmp_path):
    import torch

    library = Path(os.environ["ORBITKV_TEST_CUDA_GATE_LIB"]).resolve(strict=True)
    flags = torch.zeros(2, device="cuda", dtype=torch.int32)
    source = torch.arange(1024, device="cuda", dtype=torch.int32)
    target = torch.full((1024,), -167, device="cuda", dtype=torch.int32)
    other = torch.zeros(1, device="cuda", dtype=torch.int32)
    producer, unrelated, release = (torch.cuda.Stream() for _ in range(3))
    producer_done, unrelated_done = torch.cuda.Event(), torch.cuda.Event()
    torch.cuda.synchronize()

    call = _stream_gate(library, flags)

    completed = False
    try:
        with torch.cuda.stream(producer):
            call("orbitkv_test_stream_wait", producer, 0)
            target.copy_(source)
            producer_done.record()
        with torch.cuda.stream(unrelated):
            call("orbitkv_test_stream_wait", unrelated, 1)
            other.fill_(1)
            unrelated_done.record()
        if producer_done.query() or unrelated_done.query():
            raise RuntimeError("CUDA wait primitive did not preserve pending readiness")
        call("orbitkv_test_stream_release", release, 0)
        release.synchronize()
        producer_done.synchronize()
        if unrelated_done.query():
            raise RuntimeError("Producer release accidentally completed the other stream")
        call("orbitkv_test_stream_release", release, 1)
        release.synchronize()
        unrelated_done.synchronize()
        actual = target.cpu()
        if not torch.equal(actual, torch.arange(1024, dtype=torch.int32)):
            raise RuntimeError("CUDA readiness full payload differs")
        raw = actual.numpy().tobytes()
        actual[0] ^= 1
        if torch.equal(actual, torch.arange(1024, dtype=torch.int32)):
            raise RuntimeError("CUDA readiness mutation escaped the byte oracle")
        if other.item() != 1:
            raise RuntimeError("Independent stream did not complete after its release")
        completed = True
    finally:
        errors = []
        for index in (0, 1):
            try:
                call("orbitkv_test_stream_release", release, index)
            except BaseException as error:
                errors.append({"index": index, "error": repr(error)})
        if errors:
            (tmp_path / "STREAM-READINESS-ERRORS.json").write_text(json.dumps(errors) + "\n")
            raise RuntimeError("CUDA stream release remains unknown")
        release.synchronize()
        producer.synchronize()
        unrelated.synchronize()
    if not completed:
        raise RuntimeError("CUDA readiness gate did not complete")
    (tmp_path / "STREAM-READINESS.payload.bin").write_bytes(raw)
    (tmp_path / "STREAM-READINESS.json").write_text(
        json.dumps(
            {
                "owner": identity(os.getpid()),
                "helper_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "pending_before_release": [True, True],
                "independent_release": True,
                "observed_bytes": len(raw),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
                "mutation_rejected": True,
                "streams_synchronized": True,
                "Manager_created": False,
                "physical_fault_qualification": False,
            },
            indent=2,
        )
        + "\n"
    )


def test_prewarmed_strided_cuda_copy_is_enqueueable_behind_gate(tmp_path):
    import torch

    library = Path(os.environ["ORBITKV_TEST_CUDA_GATE_LIB"]).resolve(strict=True)
    flags = torch.zeros(1, device="cuda", dtype=torch.int32)
    producer, release = torch.cuda.Stream(), torch.cuda.Stream()
    call = _stream_gate(library, flags)
    payload = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 1, 32)
    device_payload = payload.to("cuda")
    target = torch.full((2, 8, 16, 1, 32), -167, device="cuda", dtype=torch.float32)
    expected = torch.full_like(target, -167, device="cpu")
    expected[:, 1] = payload
    done = torch.cuda.Event()
    stage_index = 0

    def stage(name):
        nonlocal stage_index
        write(
            tmp_path / f"strided-stage-{stage_index:02d}.json",
            {
                "owner": identity(os.getpid()),
                "stage": name,
                "monotonic_ns": __import__("time").monotonic_ns(),
                "physical_termination_claimed": False,
            },
        )
        stage_index += 1

    completed = False
    try:
        stage("copy_kernel_prewarm_started")
        warmup = torch.full_like(target, -167)
        warmup[:, 1].copy_(device_payload)
        torch.cuda.synchronize()
        if not torch.equal(warmup[:, 1].cpu(), payload):
            raise RuntimeError("Strided CUDA copy prewarm differs")
        del warmup
        stage("copy_kernel_prewarmed")
        with torch.cuda.stream(producer):
            call("orbitkv_test_stream_wait", producer, 0)
            stage("producer_wait_enqueued")
            target[:, 1].copy_(device_payload)
            stage("producer_copy_enqueued")
            done.record()
            stage("producer_event_recorded")
        if done.query():
            raise RuntimeError("Strided copy completed before gate release")
        stage("pending_observed")
        call("orbitkv_test_stream_release", release, 0)
        release.synchronize()
        stage("producer_released")
        done.synchronize()
        stage("producer_synchronized")
        actual = target.cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError("Strided CUDA readiness full payload differs")
        raw = actual.numpy().tobytes()
        actual[0, 1, 0, 0, 0] += 1
        if torch.equal(actual, expected):
            raise RuntimeError("Strided CUDA readiness mutation escaped the oracle")
        completed = True
    finally:
        call("orbitkv_test_stream_release", release, 0)
        release.synchronize()
        producer.synchronize()
    if not completed:
        raise RuntimeError("Strided CUDA readiness did not complete")
    (tmp_path / "STRIDED-READINESS.payload.bin").write_bytes(raw)
    (tmp_path / "STRIDED-READINESS.json").write_text(
        json.dumps(
            {
                "owner": identity(os.getpid()),
                "helper_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "copy_kernel_prewarmed": True,
                "pending_before_release": True,
                "observed_bytes": len(raw),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
                "mutation_rejected": True,
                "streams_synchronized": True,
                "Manager_created": False,
                "physical_fault_qualification": False,
            },
            indent=2,
        )
        + "\n"
    )
