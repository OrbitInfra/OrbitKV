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

pub(crate) struct ProcessOwner {
    process: ProcessGuard,
    pub(crate) client: Arc<CacheClient>,
    handle: RestoreHandle,
}

struct ProcessGuard {
    directory: std::path::PathBuf,
    child: std::process::Child,
    start_ticks: Option<String>,
    deadline: Instant,
}

impl ProcessOwner {
    pub(crate) fn start(instance: &str) -> Self {
        use std::sync::atomic::AtomicU64;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let directory = std::env::temp_dir().join(format!(
            "orbitkv-py-restore-owner-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let child = std::process::Command::new(
            std::env::var_os("ORBITKV_SHARED_COMPLETION_TEST_BINARY").expect("built CPU fixture"),
        )
        .args(["--exact", "shared_completion_child", "--nocapture"])
        .env("ORBITKV_COMPLETION_TEST_DIRECTORY", &directory)
        .spawn()
        .unwrap();
        let mut process = ProcessGuard {
            directory,
            child,
            start_ticks: None,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        process.start_ticks = process_start_ticks(process.child.id());
        assert!(process.start_ticks.is_some());
        while !process.directory.join("ready").exists() {
            assert!(process.child.try_wait().unwrap().is_none());
            assert!(Instant::now() < process.deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let client = Arc::new(
            CacheClient::connect(
                process.directory.join("cache.sock"),
                orbitkv_channel::CallOptions::default(),
            )
            .unwrap(),
        );
        let handle = client
            .start_restore(&orbitkv_channel::RestoreRequest {
                instance_id: instance.into(),
                tp_rank: 0,
                device_id: 0,
                layer_groups: Vec::new(),
                loads: Vec::new(),
            })
            .unwrap();
        Self {
            process,
            client,
            handle,
        }
    }

    fn job(&self) -> Job {
        Job {
            handle: self.handle,
            result: Arc::new(LocalRestore {
                operation_id: self.handle.operation_id,
                result: Mutex::new(RestoreResult::Pending {
                    enqueued: false,
                    retained: None,
                }),
                completed: Condvar::new(),
            }),
            timing: None,
            layer_events: HashMap::new(),
            _layer_event_owners: Vec::new(),
        }
    }

    pub(crate) fn stop(&mut self) {
        self.process.stop();
    }
}

impl Drop for ProcessOwner {
    fn drop(&mut self) {
        self.client.close();
    }
}

impl ProcessGuard {
    fn stop(&mut self) {
        std::fs::write(self.directory.join("exit"), b"").unwrap();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                assert!(process_start_ticks(self.child.id()).is_none());
                println!(
                    "pid={} start_ticks={} exit=0 reaped=1 forced_cleanup=0",
                    self.child.id(),
                    self.start_ticks.as_deref().unwrap_or("unknown")
                );
                return;
            }
            assert!(Instant::now() < self.deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            if self.start_ticks.is_none()
                || process_start_ticks(self.child.id()) != self.start_ticks
            {
                eprintln!("unknown process ownership; refusing signal and retaining directory");
                return;
            }
            eprintln!(
                "pid={} start_ticks={} forced_cleanup=1",
                self.child.id(),
                self.start_ticks.as_deref().unwrap_or("unknown")
            );
            let _ = self.child.kill();
            while Instant::now() < self.deadline {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = std::fs::remove_dir_all(&self.directory);
        } else {
            eprintln!("CPU owner did not reap within original budget; retained directory");
        }
    }
}

fn process_start_ticks(pid: u32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    Some(
        stat.rsplit_once(')')?
            .1
            .split_whitespace()
            .nth(19)?
            .to_string(),
    )
}

#[test]
#[ignore = "CPU process-owner gate; set ORBITKV_SHARED_COMPLETION_TEST_BINARY to the built channel fixture"]
fn managed_process_exit_retains_the_actual_restore_job() {
    let mut owner = ProcessOwner::start("pending-until-exit");
    let mut job = owner.job();
    let executor = Mutex::new(None);
    assert!(advance(&owner.client, &executor, &mut job).is_none());
    owner.stop();
    assert!(!owner.client.channel().peer_alive().unwrap());
    assert!(advance(&owner.client, &executor, &mut job).is_none());
    let response = job.result.poll().unwrap().unwrap();
    assert_eq!(response.state, RestoreState::Pending);
    assert!(response.message.contains("resources retained"));
    assert!(job.result.wait(Duration::from_millis(1)).unwrap().is_none());
    assert_eq!(
        owner
            .client
            .restore_completions()
            .state(owner.handle.operation_id)
            .unwrap(),
        GrantState::Managed
    );
    println!("actual_advance=1 pending_job_owned=1 terminal_result=0");
}

#[test]
#[cfg(feature = "test-hooks")]
#[ignore = "CPU claim-race gate; requires the built channel fixture and test-hooks"]
fn losing_claim_keeps_the_other_executors_active_source_owned() {
    use std::sync::mpsc;
    for instance in ["local", "local-parts"] {
        let mut owner = ProcessOwner::start(instance);
        let job = owner.job();
        let client = &owner.client;
        let handle = owner.handle;
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (outcome, job) = std::thread::scope(|scope| {
            let losing = scope.spawn(move || {
                orbitkv_channel::test_faults::pause_next_claim(entered_tx, resume_rx);
                let mut job = job;
                let outcome = advance(client, &Mutex::new(None), &mut job);
                (outcome, job)
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let winner = client.claim_local_restore(handle).unwrap().unwrap();
            assert_eq!(winner, b"cross-process bounded plan");
            resume_tx.send(()).unwrap();
            losing.join().unwrap()
        });
        let state = owner
            .client
            .restore_completions()
            .state(handle.operation_id)
            .unwrap();
        if matches!(state, GrantState::Active | GrantState::ActiveMore) {
            owner
                .client
                .finish_local_restore(
                    handle,
                    Err("CPU winner never submitted a device operation".into()),
                    None,
                )
                .unwrap();
        }
        owner.stop();
        assert!(
            outcome.is_none(),
            "losing claimant published a false terminal result"
        );
        assert_eq!(
            state,
            if instance == "local" {
                GrantState::Active
            } else {
                GrantState::ActiveMore
            }
        );
        assert_eq!(
            job.result.poll().unwrap().unwrap().state,
            RestoreState::Pending
        );
        println!(
            "actual_advance=1 cas_loser_retained=1 winner_active_before_own_drain=1 terminal_result=0"
        );
    }
}

pub(crate) fn idle_registration_worker(generation: [u8; 16], device: i32) -> LocalRestoreWorker {
    let (jobs, _receiver) = mpsc::sync_channel(1);
    LocalRestoreWorker {
        manager_device_id: device,
        registration_generation: generation,
        executor: Arc::new(Mutex::new(None)),
        jobs,
        admission: Arc::new(LocalCompletions::new().unwrap()),
        pending: Arc::new(AtomicUsize::new(0)),
        idle: Arc::new((Mutex::new(()), Condvar::new())),
        arena_count: 0,
        stopping: Arc::new(AtomicBool::new(false)),
        released: Arc::new((Mutex::new(true), Condvar::new())),
    }
}
