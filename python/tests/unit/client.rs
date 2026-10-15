use super::*;
use crate::local_restore::tests::{ProcessOwner, idle_registration_worker};

fn registration(client: &PyCacheManagerClient, generation: [u8; 16]) -> PyGpuRegistration {
    PyGpuRegistration {
        instance_id: "same-instance".into(),
        tp_rank: 1,
        pp_rank: 2,
        device_id: 3,
        generation,
        owner: Arc::downgrade(&client.inner),
        _tensors: Arc::new(Vec::new()),
    }
}

fn native_client(inner: Arc<CacheClient>, generation: [u8; 16]) -> PyCacheManagerClient {
    PyCacheManagerClient {
        inner,
        completions: Arc::new(LocalCompletions::new().unwrap()),
        local: Mutex::new(HashMap::from([(
            ("same-instance".into(), 1, 3),
            Arc::new(idle_registration_worker(generation, 7)),
        )])),
    }
}

#[test]
#[ignore = "CPU native authority gate; requires the built authenticated channel fixture"]
fn publish_authority_rejects_pre_snapshot_wait_and_late_callbacks() {
    use std::sync::mpsc;
    Python::initialize();
    let mut process = ProcessOwner::start("pending-until-exit");
    let client = native_client(Arc::clone(&process.client), [1; 16]);
    let old = registration(&client, [1; 16]);
    let active = client.registered_worker(&old).unwrap();
    assert_eq!(active.manager_device_id, 7);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let mut local = client.local.lock().unwrap();
    let rejected = std::thread::scope(|scope| {
        scope.spawn(|| {
            entered_tx.send(()).unwrap();
            let result = client.registered_worker(&old);
            result_tx.send(result.is_err()).unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            result_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        local.insert(
            ("same-instance".into(), 1, 3),
            Arc::new(idle_registration_worker([2; 16], 8)),
        );
        drop(local);
        result_rx.recv_timeout(Duration::from_secs(2)).unwrap()
    });
    let fresh = registration(&client, [2; 16]);
    let fresh_route = client.registered_worker(&fresh).unwrap().manager_device_id;
    let late_save_rejected =
        rejected && Python::attach(|py| client.save(py, &old, Vec::new()).is_err());
    let late_diagnostic_rejected =
        rejected && Python::attach(|py| client.save_diagnostic(py, &old, Vec::new()).is_err());
    client.local.lock().unwrap().clear();
    let after_unregister = client.registered_worker(&fresh).is_err();
    process.stop();
    assert!(
        rejected,
        "waiting call rebound to the replacement registration"
    );
    assert!(
        late_save_rejected,
        "late save used the replacement registration"
    );
    assert!(
        late_diagnostic_rejected,
        "diagnostic save bypassed authority"
    );
    assert_eq!(active.manager_device_id, 7);
    assert_eq!(fresh_route, 8);
    assert!(after_unregister);
    println!(
        "actual_native_authority=1 pre_snapshot_rejected=1 late_callback_rejected=1 diagnostic_rejected=1 fresh_route=8 retired_rejected=1 gpu_operations=0"
    );
}

#[test]
#[ignore = "CPU native authority gate; requires the built authenticated channel fixture"]
fn registration_cannot_move_to_a_different_native_client() {
    Python::initialize();
    let mut first = ProcessOwner::start("pending-until-exit");
    let mut second = ProcessOwner::start("pending-until-exit");
    let owner = native_client(Arc::clone(&first.client), [1; 16]);
    let foreign = native_client(Arc::clone(&second.client), [1; 16]);
    let authority = registration(&owner, [1; 16]);
    let foreign_save = Python::attach(|py| foreign.save(py, &authority, Vec::new()));
    let foreign_diagnostic =
        Python::attach(|py| foreign.save_diagnostic(py, &authority, Vec::new()));
    let accepted = owner.registered_worker(&authority).is_ok();
    first.stop();
    second.stop();
    assert!(foreign_save.is_err());
    assert!(foreign_diagnostic.is_err());
    assert!(accepted);
    println!(
        "actual_native_authority=1 foreign_client_rejected=1 own_client_accepted=1 gpu_operations=0"
    );
}

#[test]
fn registration_keeps_exporters_until_the_last_authority_is_dropped() {
    Python::initialize();
    Python::attach(|py| {
        let builtins = py.import("builtins").unwrap();
        let exporter_type = builtins
            .getattr("type")
            .unwrap()
            .call1((
                "Exporter",
                pyo3::types::PyTuple::empty(py),
                pyo3::types::PyDict::new(py),
            ))
            .unwrap();
        let exporter = exporter_type.call0().unwrap();
        let weak = py
            .import("weakref")
            .unwrap()
            .getattr("ref")
            .unwrap()
            .call1((&exporter,))
            .unwrap();
        let retained = Arc::new(vec![exporter.clone().unbind()]);
        let worker_hold = Arc::clone(&retained);
        let authority = PyGpuRegistration {
            instance_id: "retired".into(),
            tp_rank: 0,
            pp_rank: 0,
            device_id: 0,
            generation: [1; 16],
            owner: Weak::new(),
            _tensors: retained,
        };
        drop(exporter);
        drop(worker_hold);
        assert!(!weak.call0().unwrap().is_none());
        drop(authority);
        assert!(weak.call0().unwrap().is_none());
    });
}
