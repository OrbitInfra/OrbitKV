//! Explicitly enabled deterministic barriers for process/GPU qualification.
//! Absent from default builds; each test owns a private directory and process.
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub use crate::storage::ssd::process_test::{
    enabled as ssd_process_enabled, spawn as spawn_ssd_process,
};

pub fn active(name: &str) -> bool {
    let Some(dir) = std::env::var_os("ORBITKV_TEST_FAULTS") else {
        return false;
    };
    let dir = PathBuf::from(dir);
    if !dir.join(format!("{name}.pause")).exists() {
        return false;
    }
    let _ = std::fs::write(dir.join(format!("{name}.reached")), b"reached");
    true
}

pub async fn pause(name: &str) {
    let started = Instant::now();
    while active(name) {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "test fault barrier {name} was never released"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

pub fn pause_blocking(name: &str) {
    let started = Instant::now();
    while active(name) {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "test fault barrier {name} was never released"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}
