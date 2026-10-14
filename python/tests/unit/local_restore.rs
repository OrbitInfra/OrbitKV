use super::*;

#[test]
fn unknown_completion_keeps_pending_result_and_bounded_diagnostics() {
    let restore = LocalRestore {
        operation_id: 7,
        result: Mutex::new(RestoreResult::Pending {
            enqueued: false,
            retained: None,
        }),
        completed: Condvar::new(),
    };
    restore.retain_unproven("unknown completion 錯".repeat(200));
    for _ in 0..2 {
        let response = restore.poll().unwrap().unwrap();
        assert_eq!(response.operation_id, 7);
        assert_eq!(response.state, RestoreState::Pending);
        assert!(response.message.len() <= 1024);
        assert!(response.message.starts_with("unknown completion"));
    }
    assert!(restore.wait(Duration::from_millis(1)).unwrap().is_none());
    assert!(!restore.wait_enqueued(Duration::ZERO).unwrap());
    restore.mark_enqueued();
    assert!(restore.wait_enqueued(Duration::ZERO).unwrap());
    assert_eq!(
        restore.poll().unwrap().unwrap().state,
        RestoreState::Pending
    );
    *restore.result.lock().unwrap() = RestoreResult::Ready(
        RestoreResponse {
            operation_id: 7,
            state: RestoreState::Succeeded,
            message: String::new(),
        },
        None,
    );
    assert_eq!(
        restore.poll().unwrap().unwrap().state,
        RestoreState::Succeeded
    );
    assert!(restore.poll().is_err());
}

#[test]
#[ignore = "CPU process-owner gate; set ORBITKV_SHARED_COMPLETION_TEST_BINARY to the built channel fixture"]
fn managed_process_exit_retains_the_actual_restore_job() {
    use std::process::Command;
    let directory =
        std::env::temp_dir().join(format!("orbitkv-py-restore-owner-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut child = Command::new(
        std::env::var_os("ORBITKV_SHARED_COMPLETION_TEST_BINARY").expect("built CPU fixture"),
    )
    .args(["--exact", "shared_completion_child", "--nocapture"])
    .env("ORBITKV_COMPLETION_TEST_DIRECTORY", &directory)
    .spawn()
    .unwrap();
    let pid = child.id();
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    let start_ticks = stat
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .nth(19)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !directory.join("ready").exists() {
        assert!(child.try_wait().unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let client = CacheClient::connect(
        directory.join("cache.sock"),
        orbitkv_channel::CallOptions::default(),
    )
    .unwrap();
    let handle = client
        .start_restore(&orbitkv_channel::RestoreRequest {
            instance_id: "pending-until-exit".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: Vec::new(),
            loads: Vec::new(),
        })
        .unwrap();
    let result = Arc::new(LocalRestore {
        operation_id: handle.operation_id,
        result: Mutex::new(RestoreResult::Pending {
            enqueued: false,
            retained: None,
        }),
        completed: Condvar::new(),
    });
    let mut job = Job {
        handle,
        result: Arc::clone(&result),
        timing: None,
        layer_events: HashMap::new(),
        _layer_event_owners: Vec::new(),
    };
    let executor = Mutex::new(None);
    assert!(advance(&client, &executor, &mut job).is_none());
    std::fs::write(directory.join("exit"), b"").unwrap();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    };
    assert!(status.success());
    assert!(!client.channel().peer_alive().unwrap());
    assert!(advance(&client, &executor, &mut job).is_none());
    let response = result.poll().unwrap().unwrap();
    assert_eq!(response.state, RestoreState::Pending);
    assert!(response.message.contains("resources retained"));
    assert!(result.wait(Duration::from_millis(1)).unwrap().is_none());
    assert_eq!(
        client
            .restore_completions()
            .state(handle.operation_id)
            .unwrap(),
        GrantState::Managed
    );
    assert!(!std::path::Path::new(&format!("/proc/{pid}/stat")).exists());
    println!(
        "actual_advance=1 pending_job_owned=1 terminal_result=0 pid={pid} start_ticks={start_ticks} exit=0 reaped=1 forced_cleanup=0"
    );
    client.close();
    std::fs::remove_dir_all(directory).unwrap();
}
