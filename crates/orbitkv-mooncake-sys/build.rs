use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const RUNTIME_LIBRARIES: [(&str, &str); 3] = [
    ("tent/src/libtent_shared.so", "libtent_shared.so"),
    (
        "mooncake-common-src/libmooncake_common.so",
        "libmooncake_common.so",
    ),
    ("mooncake-common/libasio.so", "libasio.so"),
];
const OBSOLETE_RUNTIME_LIBRARIES: [&str; 1] = ["libtransfer_engine.so"];
fn main() {
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_CMAKE");
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_LIB_DIR");

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("sys crate lives under <workspace>/crates");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("native");
    let variant = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        "cuda"
    } else {
        "cpu"
    };
    let runtime_dir = workspace
        .join(".orbitkv/mooncake")
        .join(variant)
        .join("lib");
    fs::create_dir_all(&runtime_dir).expect("create Mooncake runtime directory");
    remove_obsolete_libraries(&runtime_dir);

    if let Some(prebuilt_dir) = env::var_os("ORBITKV_MOONCAKE_LIB_DIR") {
        println!(
            "cargo:warning=prebuilt TENT override: verify copy/context, worker ownership and source-admission candidate patches in artifact provenance; matching TCP wire and tent_close_source are required"
        );
        stage_prebuilt_libraries(Path::new(&prebuilt_dir), &runtime_dir);
        set_origin_runpaths(&runtime_dir);
        return;
    }

    let source = workspace.join("third-party/mooncake/mooncake-transfer-engine");
    let pybind = workspace.join("third-party/mooncake/extern/pybind11/CMakeLists.txt");
    assert!(
        source.join("CMakeLists.txt").is_file() && pybind.is_file(),
        "Mooncake TENT source is incomplete; run `git submodule update --init --recursive \
         third-party/mooncake`"
    );
    println!(
        "cargo:rerun-if-changed={}",
        workspace.join("third-party/mooncake").display()
    );

    // Keep the pinned checkout and libraries used by other processes immutable.
    let native_source = out_dir.join("mooncake-source");
    if native_source.exists() {
        fs::remove_dir_all(&native_source).expect("remove old private Mooncake source");
    }
    verify_backport_source(&workspace.join("third-party/mooncake"));
    copy_mooncake_source(&workspace.join("third-party/mooncake"), &native_source);
    for name in [
        "tcp-copy-status-and-cuda-context.patch",
        "tcp-worker-terminal-owner.patch",
        "tcp-source-admission-drain.patch",
    ] {
        let patch = manifest_dir.join("patches").join(name);
        println!("cargo:rerun-if-changed={}", patch.display());
        let status = Command::new("patch")
            .args(["--batch", "--forward", "--fuzz=0", "--dry-run", "-p1", "-i"])
            .arg(&patch)
            .current_dir(&native_source)
            .status()
            .expect("check merged upstream Mooncake backport");
        assert!(
            status.success(),
            "Mooncake backport does not match its pinned source"
        );
        let status = Command::new("patch")
            .args(["--batch", "--forward", "--fuzz=0", "-p1", "-i"])
            .arg(&patch)
            .current_dir(&native_source)
            .status()
            .expect("apply merged upstream Mooncake backport");
        assert!(status.success(), "Mooncake backport application failed");
    }
    configure(&native_source.join("mooncake-transfer-engine"), &build_dir);
    build(&build_dir);
    stage_libraries(&build_dir, &runtime_dir);
    set_origin_runpaths(&runtime_dir);
}

fn remove_obsolete_libraries(runtime_dir: &Path) {
    for name in OBSOLETE_RUNTIME_LIBRARIES {
        let path = runtime_dir.join(name);
        if let Err(error) = fs::remove_file(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            panic!("failed to remove obsolete {}: {error}", path.display());
        }
    }
}

fn configure(source: &Path, build_dir: &Path) {
    let cmake = env::var("ORBITKV_MOONCAKE_CMAKE").unwrap_or_else(|_| "cmake".to_string());
    let cuda = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        "ON"
    } else {
        "OFF"
    };
    let status = Command::new(&cmake)
        .args(["-S"])
        .arg(source)
        .args(["-B"])
        .arg(build_dir)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_BUILD_RPATH=$ORIGIN",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DBUILD_UNIT_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DWITH_RUST_EXAMPLE=OFF",
            "-DUSE_HTTP=OFF",
            "-DUSE_ETCD=OFF",
            "-DUSE_REDIS=OFF",
            "-DUSE_TENT=ON",
            "-DWITH_METRICS=OFF",
        ])
        .arg(format!("-DUSE_CUDA={cuda}"))
        .status()
        .unwrap_or_else(|error| panic!("failed to run {cmake}: {error}"));
    assert!(
        status.success(),
        "Mooncake TENT CMake configure failed; install its documented build dependencies"
    );
}

fn build(build_dir: &Path) {
    let cmake = env::var("ORBITKV_MOONCAKE_CMAKE").unwrap_or_else(|_| "cmake".to_string());
    let jobs = env::var("ORBITKV_MOONCAKE_BUILD_JOBS").unwrap_or_else(|_| "8".to_string());
    let status = Command::new(&cmake)
        .args(["--build"])
        .arg(build_dir)
        .args(["--target", "tent_shared", "--parallel"])
        .arg(&jobs)
        .status()
        .unwrap_or_else(|error| panic!("failed to build Mooncake TENT: {error}"));
    assert!(status.success(), "Mooncake TENT build failed");
}

fn stage_libraries(build_dir: &Path, link_dir: &Path) {
    for (relative, name) in RUNTIME_LIBRARIES {
        let source = build_dir.join(relative);
        assert!(
            source.is_file(),
            "Mooncake TENT build did not produce {}",
            source.display()
        );
        fs::copy(&source, link_dir.join(name))
            .unwrap_or_else(|error| panic!("failed to stage {}: {error}", source.display()));
    }
}

fn stage_prebuilt_libraries(source_dir: &Path, runtime_dir: &Path) {
    for (_, name) in RUNTIME_LIBRARIES {
        let source = source_dir.join(name);
        assert!(
            source.is_file(),
            "ORBITKV_MOONCAKE_LIB_DIR is missing TENT runtime {}",
            source.display()
        );
        let destination = runtime_dir.join(name);
        if source != destination {
            fs::copy(&source, &destination)
                .unwrap_or_else(|error| panic!("failed to stage {}: {error}", source.display()));
        }
    }
}

fn set_origin_runpaths(runtime_dir: &Path) {
    let patchelf = Command::new("patchelf").arg("--version").status();
    if !patchelf.is_ok_and(|status| status.success()) {
        println!(
            "cargo:warning=patchelf not found; staged Mooncake libraries may retain build-directory RUNPATH entries"
        );
        return;
    }
    for (_, name) in RUNTIME_LIBRARIES {
        let library = runtime_dir.join(name);
        let status = Command::new("patchelf")
            .args(["--set-rpath", "$ORIGIN"])
            .arg(&library)
            .status()
            .unwrap_or_else(|error| {
                panic!("failed to run patchelf for {}: {error}", library.display())
            });
        assert!(
            status.success(),
            "failed to make {} relocatable",
            library.display()
        );
    }
}

fn copy_mooncake_source(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create private Mooncake source directory");
    for entry in fs::read_dir(source).expect("read pinned Mooncake source") {
        let entry = entry.expect("read Mooncake source entry");
        if entry.file_name() == ".git" {
            continue;
        }
        let target = destination.join(entry.file_name());
        let kind = entry.file_type().expect("read Mooncake source type");
        if kind.is_dir() {
            copy_mooncake_source(&entry.path(), &target);
        } else if kind.is_symlink() {
            let link = fs::read_link(entry.path()).expect("read Mooncake source symlink");
            std::os::unix::fs::symlink(link, target).expect("copy Mooncake source symlink");
        } else {
            fs::copy(entry.path(), target).expect("copy pinned Mooncake source file");
        }
    }
}

fn verify_backport_source(source: &Path) {
    for (path, expected) in [
        (
            "mooncake-transfer-engine/tent/include/tent/platform/cuda.h",
            "a808a2b6f0729c25512e5e3d44ebbd9b18383357",
        ),
        (
            "mooncake-transfer-engine/tent/src/platform/cuda/cuda_allocator.cpp",
            "d60ddffa2f6b220e6294162b422a5c25b963164b",
        ),
        (
            "mooncake-transfer-engine/tent/src/platform/cuda/cuda_probe.cpp",
            "6a52601c9a2ce7132742debeefa89038d3bcd006",
        ),
        (
            "mooncake-transfer-engine/tent/src/transport/tcp/tcp_transport.cpp",
            "973173ca723e292916166ae5f9e70b240d65b772",
        ),
        (
            "mooncake-transfer-engine/tent/src/runtime/control_plane.cpp",
            "0378593de88cfc90b7407b8ca3bd4f1d14516fee",
        ),
    ] {
        let output = Command::new("git")
            .arg("hash-object")
            .arg(source.join(path))
            .output()
            .expect("hash pinned Mooncake backport input");
        assert!(
            output.status.success(),
            "Mooncake source hash failed: {path}"
        );
        let actual = String::from_utf8(output.stdout).expect("Git object hash is UTF-8");
        assert_eq!(
            actual.trim(),
            expected,
            "Mooncake backport input differs: {path}"
        );
    }
    assert!(
        !source
            .join("mooncake-transfer-engine/tent/include/tent/platform/cuda_utils.h")
            .exists(),
        "Mooncake backport header already exists in the pinned input"
    );
}
