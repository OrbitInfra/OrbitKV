use super::*;
use crate::local_restore::tests::ProcessOwner;
use pyo3::ffi::c_str;
use pyo3::types::PyModule;

#[test]
#[ignore = "CPU IPC-owner control; requires authenticated channel fixture and local Torch ABI stub"]
fn actual_attempt_owner_refunds_never_sent_and_retains_unknown_credits() {
    Python::initialize();
    let mut process = ProcessOwner::start("pending-until-exit");
    Python::attach(|py| {
        let fixture = PyModule::from_code(
            py,
            c_str!(
                r#"
from types import SimpleNamespace
calls = []
class Storage:
    fail_release = False
    fail_export = False
    slot = 0
    def _share_cuda_(self):
        if self.fail_export:
            raise RuntimeError('controlled export failure')
        self.slot += 1
        return (0, b'h'*64, 64, 0, b'/torch-counter', self.slot, b'', False)
    @staticmethod
    def _release_ipc_counter_cuda(name, offset):
        calls.append((name, offset))
        if Storage.fail_release:
            raise RuntimeError('controlled release return lost')
torch = SimpleNamespace(UntypedStorage=Storage)
"#
            ),
            c_str!("test_ipc_owner.py"),
            c_str!("test_ipc_owner"),
        )
        .unwrap();
        let modules = py.import("sys").unwrap().getattr("modules").unwrap();
        let previous = modules.call_method1("get", ("torch",)).unwrap().unbind();
        modules
            .set_item("torch", fixture.getattr("torch").unwrap())
            .unwrap();
        let exporter = fixture.getattr("Storage").unwrap().call0().unwrap();
        let tensors = Arc::new(vec![exporter.clone().unbind()]);
        let key = ("refund".into(), 0, 0);
        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            key.clone(),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        assert!(
            AttemptOwner::reserve(Arc::clone(&process.client), key, Arc::clone(&tensors), 64)
                .is_err()
        );
        owner.export(py, &exporter).unwrap();
        owner.rollback(py).unwrap();
        assert!(!instance_pending(&process.client, "refund"));
        assert_eq!(fixture.getattr("calls").unwrap().len().unwrap(), 1);
        drop(owner);

        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("never-imported".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        owner.export(py, &exporter).unwrap();
        owner.submitted().unwrap();
        owner
            .resolve(py, &[CudaCreditDisposition::NeverImported as i32])
            .unwrap();
        assert!(!instance_pending(&process.client, "never-imported"));
        assert_eq!(fixture.getattr("calls").unwrap().len().unwrap(), 2);

        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("unknown".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        owner.export(py, &exporter).unwrap();
        owner.submitted().unwrap();
        let generation = owner.0;
        owner
            .resolve(py, &[CudaCreditDisposition::Unknown as i32])
            .unwrap();
        assert!(owner.accepted().is_err());
        drop(owner);
        assert!(instance_pending(&process.client, "unknown"));
        assert_eq!(fixture.getattr("calls").unwrap().len().unwrap(), 2);
        // This test ABI submits no importer/device work. Only its exact test
        // owner may remove the retained stub; production has no such release.
        ATTEMPTS.lock().unwrap().remove(&generation);

        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("release-error".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        owner.export(py, &exporter).unwrap();
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("fail_release", true)
            .unwrap();
        assert!(owner.rollback(py).is_err());
        let calls = fixture.getattr("calls").unwrap().len().unwrap();
        owner.rollback(py).unwrap();
        assert_eq!(
            fixture.getattr("calls").unwrap().len().unwrap(),
            calls,
            "rollback retried an unknown decrement"
        );
        assert!(instance_pending(&process.client, "release-error"));
        ATTEMPTS.lock().unwrap().remove(&owner.0);
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("fail_release", false)
            .unwrap();

        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("partial".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        owner.export(py, &exporter).unwrap();
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("fail_export", true)
            .unwrap();
        assert!(owner.export(py, &exporter).is_err());
        owner.rollback(py).unwrap();
        assert!(instance_pending(&process.client, "partial"));
        ATTEMPTS.lock().unwrap().remove(&owner.0);

        let owner = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("accepted".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("fail_export", false)
            .unwrap();
        owner.export(py, &exporter).unwrap();
        owner.submitted().unwrap();
        owner
            .resolve(py, &[CudaCreditDisposition::ImporterOwned as i32])
            .unwrap();
        assert!(instance_pending(&process.client, "accepted"));
        owner.accepted().unwrap();
        assert!(!instance_pending(&process.client, "accepted"));
        assert!(
            AttemptOwner::reserve(
                Arc::clone(&process.client),
                ("too-large".into(), 0, 0),
                Arc::clone(&tensors),
                MAX_RETAINED_BYTES + 1
            )
            .is_err()
        );
        let mut reserved = Vec::new();
        for index in 0..MAX_ATTEMPTS {
            reserved.push(
                AttemptOwner::reserve(
                    Arc::clone(&process.client),
                    (format!("bounded-{index}"), 0, 0),
                    Arc::clone(&tensors),
                    64,
                )
                .unwrap(),
            );
        }
        assert!(
            AttemptOwner::reserve(
                Arc::clone(&process.client),
                ("admission-full".into(), 0, 0),
                Arc::clone(&tensors),
                64
            )
            .is_err()
        );
        for owner in reserved {
            owner.rollback(py).unwrap();
        }
        if previous.bind(py).is_none() {
            modules.del_item("torch").unwrap();
        } else {
            modules.set_item("torch", previous).unwrap();
        }
        println!(
            "actual_attempt_owner=1 duplicate_preexport=1 never_sent_refunded=1 unknown_retained=1 lost_release_not_retried=1 partial_export_retained=1 consumed_owner_handoff=1 budget_reject=1 CUDA_operations=0"
        );
    });
    process.stop();
}
