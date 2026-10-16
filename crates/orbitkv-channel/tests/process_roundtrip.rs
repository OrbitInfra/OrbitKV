#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime};

use orbitkv_channel::{CallOptions, Command, CommandCode, StatusCode, TransportClient};

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
    let mut ready = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut ready)
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
    assert!(child.wait().unwrap().success());
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
    let output = ProcessCommand::new(std::env::current_exe().unwrap())
        .args(["--exact", "private_domain_child", "--nocapture"])
        .env("ORBITKV_CHANNEL_CONFIG", &config)
        .current_dir("/tmp")
        .output()
        .unwrap();
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
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
            .arg("private-invalid")
            .env("ORBITKV_CHANNEL_CONFIG", &config)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("READY"));
    }
    use std::os::unix::fs::symlink;
    let alias = directory.path().join("symlink.toml");
    symlink(&config, &alias).unwrap();
    for path in [&alias, &directory.path().join("missing.toml")] {
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-channel-echo"))
            .arg("private-invalid")
            .env("ORBITKV_CHANNEL_CONFIG", path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("READY"));
    }
}
