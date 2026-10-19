"""Actual engine unregister failure/retry control with an injected CPU transport.

Run against a complete native build with matching headers. Fixed-runtime strict
controls must pass. --expect-defect preserves historical failures as explicit
negative controls, independently of hardware qualification.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--control", choices=("engine", "rdma"), default="engine")
    parser.add_argument("--expect-defect", action="store_true")
    parser.add_argument("--build-rust-control", action="store_true")
    args = parser.parse_args()
    if args.build_rust_control and args.control != "engine":
        parser.error("--build-rust-control requires --control engine")
    build, runtime = (
        args.native_build.resolve(strict=True),
        args.runtime.resolve(strict=True),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve(strict=True)
    flags_path = build / "tent/src/CMakeFiles/tent_shared.dir/flags.make"
    flags = dict(
        line.split(" = ", 1)
        for line in flags_path.read_text().splitlines()
        if " = " in line
    )
    includes = shlex.split(flags["CXX_INCLUDES"])
    headers = []
    for flag in includes:
        if flag.startswith("-I"):
            candidate = Path(flag[2:]) / "tent/runtime/transfer_engine_impl.h"
            if candidate.is_file():
                headers.append(candidate.resolve())
    if not headers:
        raise RuntimeError("Matching native source headers are absent")
    native_source = headers[0].parents[3] / "src/platform"
    helper = (
        Path(__file__)
        .with_name(
            "unregister_error_owner.cpp"
            if args.control == "engine"
            else "rdma_unregister_owner.cpp"
        )
        .resolve(strict=True)
    )
    sources = [helper]
    if args.control == "engine":
        sources.extend(
            native_source / name for name in ("cpu_allocator.cpp", "cpu_probe.cpp")
        )
    binary = output / "unregister-error-owner"
    libraries = [
        runtime / name
        for name in ("libtent_shared.so", "libmooncake_common.so", "libasio.so")
    ]
    frozen = {str(path): sha(path) for path in libraries}
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise RuntimeError("Requested compiler is unavailable")
    compiler = str(Path(compiler).absolute())
    command = [
        compiler,
        "-std=c++20",
        "-O2",
        "-fno-access-control",
        *shlex.split(flags["CXX_DEFINES"]),
        *includes,
        *(str(source) for source in sources),
        "-L",
        str(runtime),
        f"-Wl,-rpath,{runtime}",
        "-Wl,--export-dynamic",
        "-ltent_shared",
        "-lmooncake_common",
        "-lasio",
        "-lglog",
        "-lgflags",
        "-lnuma",
        "-libverbs",
        "-pthread",
        "-o",
        str(binary),
    ]
    if not args.expect_defect:
        command.insert(1, "-DORBITKV_UNREGISTER_PUBLICATION_REVISION")
    result = {
        "compile_argv": command,
        "compiler_sha256": sha(Path(compiler)),
        "runner_sha256": sha(Path(__file__).resolve(strict=True)),
        "library_sha256": frozen,
        "helper_sha256": sha(helper),
        "native_flags_path": str(flags_path),
        "native_flags_sha256": sha(flags_path),
        "native_header_sha256": sha(headers[0]),
        "CPU_allocator_sha256": sha(native_source / "cpu_allocator.cpp"),
        "CPU_probe_sha256": sha(native_source / "cpu_probe.cpp"),
        "strict_unregister_qualified": False,
        "physical_termination_qualified": False,
        "payload_DMA": False,
        "test_transport_injection": True,
        "control": args.control,
        "expect_defect": args.expect_defect,
        "test_only_private_access_flag": "-fno-access-control",
        "scenes": [],
    }
    with (output / "compiler-version.log").open("x") as log:
        version = subprocess.run(
            [compiler, "--version"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    result["compiler_version_exit"] = version.returncode
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "compile.log").open("x") as log:
        try:
            compiled = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False
            )
        except subprocess.TimeoutExpired:
            result["compile_timeout_forced_cleanup"] = True
            (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
            raise
    result["compile_exit"] = compiled.returncode
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    if compiled.returncode:
        raise RuntimeError(
            f"Unregister control compilation failed: {compiled.returncode}"
        )
    result["binary_sha256"] = sha(binary)
    dependencies = set()
    result["dependency_commands"] = []
    for index, source in enumerate(sources):
        dependency = output / f"input-{index}.d"
        argv = [
            compiler,
            "-std=c++20",
            *shlex.split(flags["CXX_DEFINES"]),
            *includes,
            "-M",
            str(source),
        ]
        with (
            dependency.open("x") as stdout,
            (output / f"input-{index}.log").open("x") as stderr,
        ):
            preprocessed = subprocess.run(
                argv, stdout=stdout, stderr=stderr, timeout=10, check=False
            )
        result["dependency_commands"].append(
            {"argv": argv, "exit_code": preprocessed.returncode}
        )
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        if preprocessed.returncode:
            raise RuntimeError("Actual compiler dependency discovery failed")
        words = shlex.split(dependency.read_text().replace("\\\n", " "))
        for word in words[1:]:
            dependencies.add(Path(word).resolve(strict=True))
    if not dependencies:
        raise RuntimeError("Actual compiler input dependencies are absent")
    result["compiled_input_sha256"] = {
        str(path): sha(path) for path in sorted(dependencies)
    }
    if args.build_rust_control:
        library = output / "unregister-control.so"
        rust_command = [
            arg for arg in command if arg not in {str(source) for source in sources[1:]}
        ]
        rust_command[rust_command.index("-o") + 1] = str(library)
        rust_command[1:1] = [
            "-fPIC",
            "-shared",
            "-DORBITKV_UNREGISTER_RUST_CONTROL",
        ]
        with (output / "rust-control-compile.log").open("x") as log:
            compiled = subprocess.run(
                rust_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        result["rust_control"] = {
            "compile_argv": rust_command,
            "compile_exit": compiled.returncode,
        }
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        if compiled.returncode:
            raise RuntimeError("Rust native-control library compilation failed")
        result["rust_control"]["library"] = str(library)
        result["rust_control"]["sha256"] = sha(library)
    if args.control == "rdma":
        scenes = [
            (scene, "strict", int(args.expect_defect and scene != "normal"))
            for scene in ("normal", "context-refusal", "buffer-refusal")
        ]
    else:
        scenes = [
            ("normal", "strict", 0),
            ("refused", "strict", int(args.expect_defect)),
            ("refused", "reproduce", int(not args.expect_defect)),
        ]
        if not args.expect_defect:
            scenes.extend(
                (scene, "strict", 0)
                for scene in (
                    "retry-partial",
                    "retry-publication",
                    "retry-references",
                    "retry-batch-publication",
                    "retry-batch-normal",
                    "retry-batch-prefix",
                    "retry-batch-wildcard",
                    "retry-publication-revision",
                )
            )
    for scene, expectation, expected_exit in scenes:
        mappings = output / f"{scene}-{expectation}.maps"
        argv = [str(binary), scene]
        if args.control == "engine":
            argv.append(expectation)
        argv.append(str(mappings))
        evidence = {
            "argv": argv,
            "expected_exit": expected_exit,
            "actual_wait": False,
            "control_timeout_forced_cleanup": False,
        }
        result["scenes"].append(evidence)
        started = time.monotonic()
        with (output / f"{scene}-{expectation}.log").open("x") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            evidence["pid"] = process.pid
            stat = Path(f"/proc/{process.pid}/stat")
            try:
                evidence["start_ticks"] = int(
                    stat.read_text().rsplit(")", 1)[1].split()[19]
                )
            except FileNotFoundError:
                evidence["start_ticks"] = None
            try:
                evidence["exit_code"] = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                evidence["control_timeout_forced_cleanup"] = True
                process.kill()
                evidence["exit_code"] = process.wait(timeout=5)
            evidence["actual_wait"] = True
        evidence["elapsed_seconds"] = time.monotonic() - started
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        if evidence["control_timeout_forced_cleanup"]:
            raise RuntimeError(
                "CPU control exceeded its budget; forced cleanup is not drain"
            )
        actual_paths = set()
        for line in mappings.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                actual_paths.add(str(Path(fields[5]).resolve()))
        evidence["actual_mappings_sha256"] = sha(mappings)
        evidence["actual_mapped_file_sha256"] = {
            path: sha(Path(path)) for path in sorted(actual_paths)
        }
        evidence["actual_library_paths"] = sorted(
            set(frozen).intersection(actual_paths)
        )
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        if not set(frozen) <= actual_paths:
            raise RuntimeError("Actual native mappings differ from the frozen runtime")
    if any(sha(Path(path)) != value for path, value in frozen.items()):
        raise RuntimeError("Frozen native libraries changed during control")
    if any(
        sha(path) != result["compiled_input_sha256"][str(path)] for path in dependencies
    ):
        raise RuntimeError("Compiler inputs changed during control")
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    if any(scene["exit_code"] != scene["expected_exit"] for scene in result["scenes"]):
        raise RuntimeError(
            "Native unregister control differs from its declared expectation"
        )
    result["unregister_retry_control_passed"] = not args.expect_defect
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": str(output / "RESULT.json"),
                "compile_exit": result["compile_exit"],
                "scene_exits": [scene["exit_code"] for scene in result["scenes"]],
            }
        )
    )


if __name__ == "__main__":
    main()
