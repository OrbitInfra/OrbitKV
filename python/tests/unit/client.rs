use super::*;
use crate::local_restore::tests::{ProcessOwner, idle_registration_worker};

#[test]
#[ignore = "real CPU bootstrap/channel native Save owner control; no CUDA allocations or DMA"]
fn pending_save_retains_after_handle_drop_and_delayed_reply() {
    use orbitkv_channel::{
        BootstrapServer, CommandCode, RESPONSE_FLAG_REQUEST_CONSUMED, Response, TransportServer,
    };
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;
    Python::initialize();
    let directory = std::path::PathBuf::from(format!("/var/tmp/okv-save-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let socket = directory.join("save.sock");
    let name = format!("orbitkv/test/pending-save/{}", std::process::id());
    let bootstrap = BootstrapServer::bind(&socket, &name, 71, 64 * 1024, 4096).unwrap();
    bootstrap.set_nonblocking(true).unwrap();
    let transport = TransportServer::bind(&name).unwrap();
    let release = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&release);
    let stopping = Arc::clone(&stop);
    let peer = std::thread::spawn(move || {
        let mut sessions = HashMap::new();
        let mut pending = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !stopping.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline);
            while let Some(session) = bootstrap.try_accept().unwrap() {
                sessions.insert(session.client_token(), session);
            }
            transport
                .try_serve_deferred_for_epoch(71, |command, reply| {
                    assert_eq!(command.code, CommandCode::Publish);
                    let session = sessions.get_mut(&command.arg0).unwrap();
                    let slot = bootstrap
                        .descriptor_slot(command.descriptor.offset)
                        .unwrap();
                    session
                        .validate_request(command.descriptor, command.arg0, slot)
                        .unwrap();
                    PublishRequest::decode(&bootstrap.arena().read(command.descriptor).unwrap())
                        .unwrap();
                    pending.push((command, reply));
                    Ok(())
                })
                .unwrap();
            if gate.load(Ordering::Acquire) {
                for (command, reply) in pending.drain(..) {
                    let session = sessions.get_mut(&command.arg0).unwrap();
                    let mut response = Response::ok(command);
                    response.descriptor = bootstrap
                        .arena()
                        .write_response(command.descriptor, &[])
                        .unwrap();
                    response.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                    session.complete_request().unwrap();
                    reply
                        .send_and_notify(response, session.reply_notification_fd())
                        .unwrap();
                }
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    });
    let inner = Arc::new(CacheClient::connect(&socket, CallOptions::default()).unwrap());
    let client = native_client(Arc::clone(&inner), [37; 16]);
    let registration = registration(&client, [37; 16]);
    let tensors = Arc::downgrade(&registration._tensors);
    let handle = Python::attach(|py| client.start_save(py, &registration, Vec::new())).unwrap();
    {
        let mut usage = SAVE_USAGE.lock().unwrap();
        let saved = *usage;
        usage.0 = MAX_PENDING_SAVES;
        drop(usage);
        assert!(Python::attach(|py| client.start_save(py, &registration, Vec::new())).is_err());
        *SAVE_USAGE.lock().unwrap() = saved;
        assert_eq!(client.saves.lock().unwrap().len(), 1);
    }
    drop(registration);
    let started = Instant::now();
    let response =
        Python::attach(|py| client.wait_save(py, &handle.borrow(py), Some(0.001))).unwrap();
    assert!(!response.done);
    assert!(started.elapsed() < Duration::from_secs(1));
    let foreign = native_client(
        Arc::new(CacheClient::connect(&socket, CallOptions::default()).unwrap()),
        [37; 16],
    );
    assert!(Python::attach(|py| foreign.poll_save(py, &handle.borrow(py))).is_err());
    let observed = Python::attach(|py| Arc::clone(&handle.borrow(py).pending));
    drop(handle);
    assert!(
        tensors.upgrade().is_some(),
        "native Save lost exporter ownership with the handle"
    );
    assert!(!observed.status(Duration::ZERO).done);
    release.store(true, Ordering::Release);
    assert!(observed.status(Duration::from_secs(5)).success);
    let deadline = Instant::now() + Duration::from_secs(2);
    while tensors.upgrade().is_some() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    stop.store(true, Ordering::Release);
    peer.join().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    println!(
        "actual_channel_Save=1 bounded_pending=1 foreign_rejected=1 handle_drop_retained=1 late_receipt_release=1 no_DMA=true"
    );
}

#[test]
fn pending_save_wait_is_a_receipt_wait_not_a_timeout_completion() {
    let pending = Arc::new(PendingSave {
        instance: "cpu".into(),
        bytes: 0,
        result: Mutex::new(None),
        completed: Condvar::new(),
    });
    assert!(!pending.status(Duration::from_millis(1)).done);
    let receipt = Arc::clone(&pending);
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(5));
        *receipt.result.lock().unwrap() = Some(Err("failed after drained copy".into()));
        receipt.completed.notify_all();
    });
    let result = pending.wait();
    assert!(result.done);
    assert!(!result.success);
    assert!(result.message.contains("drained copy"));
    worker.join().unwrap();
}

#[test]
fn save_request_admission_accounts_for_spare_capacity() {
    let mut hashes = Vec::with_capacity(37);
    hashes.push(Vec::with_capacity(4096));
    let request = PublishRequest {
        instance_id: "capacity".into(),
        registration_generation: [37; 16],
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: vec![PublishLayer {
            layer_name: "layer".into(),
            block_ids: Vec::with_capacity(1024),
            block_hashes: hashes,
        }],
    };
    let bytes = save_request_bytes(&request).unwrap();
    assert!(bytes >= 37 * std::mem::size_of::<Vec<u8>>() + 4096 + 1024 * 4);
}

fn registration(client: &PyCacheManagerClient, generation: [u8; 16]) -> PyGpuRegistration {
    PyGpuRegistration {
        instance_id: "same-instance".into(),
        tp_rank: 1,
        pp_rank: 2,
        device_id: 3,
        generation,
        export_owner_generation: [4; 16],
        ipc_counters: Vec::new(),
        ipc_credit_evidence: String::new(),
        owner: Arc::downgrade(&client.inner),
        _tensors: Arc::new(Vec::new()),
    }
}

fn native_client(inner: Arc<CacheClient>, generation: [u8; 16]) -> PyCacheManagerClient {
    PyCacheManagerClient {
        inner,
        saves: Arc::new(Mutex::new(HashMap::new())),
        next_save: AtomicU64::new(1),
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
            export_owner_generation: [4; 16],
            ipc_counters: Vec::new(),
            ipc_credit_evidence: String::new(),
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

#[test]
#[ignore = "CPU registration entry control; requires authenticated channel fixture and explicit layout stub"]
fn duplicate_registration_never_calls_the_cuda_exporter() {
    use pyo3::ffi::c_str;
    use pyo3::types::PyModule;
    Python::initialize();
    let mut process = ProcessOwner::start("pending-until-exit");
    let client = native_client(Arc::clone(&process.client), [1; 16]);
    Python::attach(|py| {
        let fixture = PyModule::from_code(
            py,
            c_str!(
                r#"
from types import SimpleNamespace
exports = 0
class Storage:
    def data_ptr(self): return 4096
    def nbytes(self): return 64
    def _share_cuda_(self):
        global exports
        exports += 1
        raise AssertionError('duplicate registration created an export credit')
class Tensor:
    device = SimpleNamespace(type='cuda', index=3)
    shape = (8,)
    dtype = 'torch.float16'
    def data_ptr(self): return 4096
    def untyped_storage(self): return Storage()
    def element_size(self): return 2
    def storage_offset(self): return 0
    def stride(self): return (1,)
def device_uuid(device): return 'GPU-c4930667-4d65-aae5-8205-fe785c3654be'
"#
            ),
            c_str!("duplicate_entry.py"),
            c_str!("duplicate_entry"),
        )
        .unwrap();
        let modules = py.import("sys").unwrap().getattr("modules").unwrap();
        let previous = modules
            .call_method1("get", ("orbitkv.client.gpu",))
            .unwrap()
            .unbind();
        modules.set_item("orbitkv.client.gpu", &fixture).unwrap();
        let tensor = fixture.getattr("Tensor").unwrap().call0().unwrap().unbind();
        let result = client.register_context_batch(
            py,
            "same-instance".into(),
            "namespace".into(),
            1,
            2,
            1,
            1,
            3,
            vec!["layer".into()],
            vec![8],
            vec![2],
            vec![0],
            vec![1],
            "direct",
            false,
            None,
            None,
            None,
            vec![tensor],
        );
        if previous.bind(py).is_none() {
            modules.del_item("orbitkv.client.gpu").unwrap();
        } else {
            modules.set_item("orbitkv.client.gpu", previous).unwrap();
        }
        assert!(result.is_err());
        assert_eq!(
            fixture
                .getattr("exports")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            0
        );
        assert!(!crate::registration::instance_pending(
            &client.inner,
            "same-instance"
        ));
        println!(
            "actual_register_entry=1 local_duplicate_rejected=1 export_calls=0 CUDA_operations=0"
        );
    });
    process.stop();
}

#[test]
fn storage_rebinding_cannot_change_the_retained_registration_allocation() {
    use pyo3::ffi::c_str;
    use pyo3::types::PyModule;
    Python::initialize();
    Python::attach(|py| {
        let fixture = PyModule::from_code(
            py,
            c_str!(
                r#"
from types import SimpleNamespace
import weakref
class Storage:
    def __init__(self, address): self.address = address
    def data_ptr(self): return self.address
    def nbytes(self): return 64
class Tensor:
    device = SimpleNamespace(index=0)
    dtype = 'torch.float16'
    shape = (8,)
    def __init__(self): self.storage = Storage(4096)
    def data_ptr(self): return self.storage.data_ptr()
    def storage_offset(self): return 0
    def stride(self): return (1,)
    def element_size(self): return 2
tensor = Tensor()
original_storage = weakref.ref(tensor.storage)
"#
            ),
            c_str!("storage_rebind.py"),
            c_str!("storage_rebind"),
        )
        .unwrap();
        let tensor = fixture.getattr("tensor").unwrap();
        let storage = tensor.getattr("storage").unwrap();
        let tensors = vec![tensor.clone().unbind()];
        let storages = vec![storage.clone().unbind()];
        let layout = CudaIpcTensorRegistration {
            storage_size_bytes: 64,
            shape: vec![8],
            strides: vec![1],
            dtype: "float16".into(),
            element_size: 2,
            ..Default::default()
        };
        validate_export_snapshot(
            py,
            &tensors,
            &storages,
            &[4096],
            std::slice::from_ref(&layout),
            0,
        )
        .unwrap();
        let replacement = fixture.getattr("Storage").unwrap().call1((8192,)).unwrap();
        tensor.setattr("storage", replacement).unwrap();
        drop(storage);
        assert!(
            !fixture
                .getattr("original_storage")
                .unwrap()
                .call0()
                .unwrap()
                .is_none()
        );
        assert!(validate_export_snapshot(py, &tensors, &storages, &[4096], &[layout], 0).is_err());
        assert_eq!(
            storages[0]
                .bind(py)
                .call_method0("data_ptr")
                .unwrap()
                .extract::<u64>()
                .unwrap(),
            4096
        );
        drop(storages);
        assert!(
            fixture
                .getattr("original_storage")
                .unwrap()
                .call0()
                .unwrap()
                .is_none()
        );
    });
}
