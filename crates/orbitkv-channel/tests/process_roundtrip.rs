#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant, SystemTime};

use orbitkv_channel::{CallOptions, Command, CommandCode, StatusCode, TransportClient};

fn wait(child: &mut std::process::Child) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "owned CPU child wait exceeded ten seconds: {}",
            child.id()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn output(command: &mut ProcessCommand) -> std::process::Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let status = wait(&mut child);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    use std::io::Read;
    child
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut stdout)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_end(&mut stderr)
        .unwrap();
    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

#[test]
fn request_response_crosses_a_real_process_boundary() {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let service_name = format!("orbitkv/test/{}/{nonce}", std::process::id());
    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
        .arg(&service_name)
        .arg("11")
        .current_dir("/tmp")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut ready = String::new();
        let result = BufReader::new(stdout).read_line(&mut ready).map(|_| ready);
        let _ = sender.send(result);
    });
    let ready = receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(ready.trim(), "READY");

    let client = TransportClient::connect(&service_name).unwrap();
    let mut ping = Command::ping(7, 11);
    ping.arg0 = 41;
    let response = client.call(ping, CallOptions::default()).unwrap();
    assert_eq!(response.status, StatusCode::Ok);
    assert_eq!(response.request_id, 7);
    assert_eq!(response.session_epoch, 11);
    assert_eq!(response.value0, 42);

    let stale = client
        .call(Command::ping(8, 10), CallOptions::default())
        .unwrap();
    assert_eq!(stale.status, StatusCode::StaleSession);
    assert_eq!(stale.request_id, 8);
    assert_eq!(stale.session_epoch, 11);

    let shutdown = Command {
        code: CommandCode::Shutdown,
        request_id: 9,
        session_epoch: 11,
        ..Command::ping(9, 11)
    };
    let response = client
        .call(
            shutdown,
            CallOptions {
                timeout: Duration::from_secs(2),
                ..CallOptions::default()
            },
        )
        .unwrap();
    assert_eq!(response.status, StatusCode::Ok);
    assert!(wait(&mut child).success());
}

#[test]
fn private_domain_child() {
    if std::env::var_os("ORBITKV_CHANNEL_CONFIG").is_some() {
        request_response_crosses_a_real_process_boundary();
    }
}

#[test]
fn private_domain_is_consumed_by_both_processes_without_touching_default_shm() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("domain");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(root.join("nodes")).unwrap();
    std::fs::create_dir(root.join("services")).unwrap();
    let config = directory.path().join("channel.toml");
    let prefix = format!("orbitkv_private_{}_", std::process::id());
    std::fs::write(
        &config,
        format!(
            "[global]\nroot-path = {:?}\nprefix = {prefix:?}\n",
            root.to_str().unwrap()
        ),
    )
    .unwrap();
    let snapshot = || -> std::collections::BTreeMap<_, _> {
        std::fs::read_dir("/dev/shm")
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("iox2_")
            })
            .map(|path| {
                let bytes = std::fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect()
    };
    let before = snapshot();
    let output = output(
        ProcessCommand::new(std::env::current_exe().unwrap())
            .args(["--exact", "private_domain_child", "--nocapture"])
            .env("ORBITKV_CHANNEL_CONFIG", &config)
            .current_dir("/tmp"),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        snapshot(),
        before,
        "private channel changed default-domain resources"
    );
    let owned: Vec<_> = std::fs::read_dir("/dev/shm")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&prefix)
        })
        .collect();
    assert_eq!(
        owned.len(),
        1,
        "only this private domain's persistent counter should remain"
    );
    std::fs::remove_file(&owned[0]).unwrap();
}

#[test]
fn invalid_explicit_domain_never_falls_back_to_default_node_creation() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("invalid.toml");
    for source in [
        "[global]\nprefix='iox2_'\n",
        "malformed[",
        "[global]\nroot-path='relative'\nprefix='private_'\n",
        &format!(
            "[global]\nroot-path={:?}\nprefix='private_'\n[global.node]\ndirectory='/tmp'\n",
            directory.path().to_str().unwrap()
        ),
    ] {
        std::fs::write(&config, source).unwrap();
        let output = output(
            ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
                .arg("private-invalid")
                .env("ORBITKV_CHANNEL_CONFIG", &config),
        );
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("READY"));
    }
    use std::os::unix::fs::symlink;
    let alias = directory.path().join("symlink.toml");
    symlink(&config, &alias).unwrap();
    for path in [&alias, &directory.path().join("missing.toml")] {
        let output = output(
            ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
                .arg("private-invalid")
                .env("ORBITKV_CHANNEL_CONFIG", path),
        );
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("READY"));
    }
}

#[test]
fn private_domain_refuses_symlink_directories_and_file_roots_before_shared_state() {
    use std::os::unix::fs::symlink;
    for case in [
        "node",
        "service",
        "intermediate",
        "file-root",
        "missing-directories",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("root");
        let outside = directory.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        if case == "file-root" {
            std::fs::write(&root, b"not a directory").unwrap();
        } else {
            std::fs::create_dir(&root).unwrap();
            if case != "missing-directories" {
                for name in ["nodes", "services"] {
                    if (case == "node" && name == "nodes")
                        || (case == "service" && name == "services")
                    {
                        symlink(&outside, root.join(name)).unwrap();
                    } else {
                        std::fs::create_dir(root.join(name)).unwrap();
                    }
                }
            }
            if case == "intermediate" {
                symlink(&outside, root.join("escape")).unwrap();
                std::fs::create_dir(outside.join("nodes")).unwrap();
                std::fs::create_dir(outside.join("services")).unwrap();
            }
        }
        let prefix = format!(
            "orbitkv_refused_{}_{}_",
            std::process::id(),
            case.replace('-', "_")
        );
        let config = directory.path().join("config.toml");
        let dirs = if case == "intermediate" {
            "[global.node]\ndirectory='escape/nodes'\n[global.service]\ndirectory='escape/services'\n"
        } else {
            ""
        };
        std::fs::write(
            &config,
            format!(
                "[global]\nroot-path={:?}\nprefix={prefix:?}\n{dirs}",
                root.to_str().unwrap()
            ),
        )
        .unwrap();
        let output = output(
            ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
                .arg("private-refusal")
                .env("ORBITKV_CHANNEL_CONFIG", &config),
        );
        assert!(!output.status.success(), "{case}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("READY"));
        assert!(
            !std::fs::read_dir("/dev/shm").unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(&prefix)),
            "{case} created shared state"
        );
    }
}
