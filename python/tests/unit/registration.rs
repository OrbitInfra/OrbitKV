use super::*;
use crate::local_restore::tests::ProcessOwner;
use pyo3::ffi::c_str;
use pyo3::types::PyModule;

#[test]
fn retained_storage_bytes_counts_exact_backing_aliases_once() {
    for layers in [32_u64, 36] {
        let backing_bytes = 26144 * 8 * 16 * 256 * 2 * layers;
        let shared = vec![(0x1000, backing_bytes); layers as usize];
        assert_eq!(retained_storage_bytes(shared), Ok(backing_bytes));
        assert!(backing_bytes <= MAX_RETAINED_BYTES);
        assert!(backing_bytes * layers > MAX_RETAINED_BYTES);
    }

    for (name, storages, expected) in [
        ("empty", vec![], 0),
        ("exact_alias", vec![(0x1000, 64), (0x1000, 64)], 64),
        ("distinct_bases", vec![(0x1000, 64), (0x2000, 64)], 128),
        ("different_extents", vec![(0x1000, 64), (0x1000, 32)], 96),
        ("partial_overlap", vec![(0x1000, 64), (0x1020, 64)], 128),
    ] {
        assert_eq!(retained_storage_bytes(storages), Ok(expected), "{name}");
    }
    assert!(retained_storage_bytes([(1, u64::MAX), (2, 1)]).is_err());
    assert_eq!(retained_storage_bytes([(1, u64::MAX); 2]), Ok(u64::MAX));
}

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
import os
import struct
import uuid
calls = []
counter_path = '/dev/shm/orbitkv-credit-control-' + uuid.uuid4().hex
with open(counter_path, 'xb') as stream:
    stream.write(bytes(64 + 10000 * 8))
class Storage:
    fail_release = False
    fail_export = False
    silent_release = False
    replace_counter = False
    remove_counter = False
    slot = 0
    def _share_cuda_(self):
        if self.fail_export:
            raise RuntimeError('controlled export failure')
        self.slot += 1
        with open(counter_path, 'r+b') as stream:
            stream.seek(64 + self.slot * 8)
            stream.write(struct.pack('=q', 1))
        return (0, b'h'*64, 64, 0, os.path.basename(counter_path).encode(), self.slot, b'', False)
    @staticmethod
    def _release_ipc_counter_cuda(name, offset):
        calls.append((name, offset))
        if Storage.fail_release:
            raise RuntimeError('controlled release return lost')
        if Storage.remove_counter:
            os.unlink(counter_path)
            return
        if Storage.replace_counter:
            os.unlink(counter_path)
            with open(counter_path, 'xb') as stream:
                stream.write(bytes(64 + 10000 * 8))
            return
        if not Storage.silent_release:
            with open(counter_path, 'r+b') as stream:
                stream.seek(64 + offset * 8)
                stream.write(struct.pack('=q', 0))
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
        let evidence: serde_json::Value =
            serde_json::from_str(&retained_credit_evidence(&process.client, "unknown").unwrap())
                .unwrap();
        let credit = &evidence["attempts"][0]["credit_evidence"]["credits"][0];
        assert_eq!(credit["disposition"], "unknown");
        assert_eq!(credit["observation"]["same_object"], true);
        assert_eq!(credit["observation"]["observed_value"], 1);
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
            ("silent-release".into(), 0, 0),
            Arc::clone(&tensors),
            64,
        )
        .unwrap();
        owner.export(py, &exporter).unwrap();
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("silent_release", true)
            .unwrap();
        assert!(owner.rollback(py).is_err());
        assert!(instance_pending(&process.client, "silent-release"));
        let calls = fixture.getattr("calls").unwrap().len().unwrap();
        owner.rollback(py).unwrap();
        assert_eq!(fixture.getattr("calls").unwrap().len().unwrap(), calls);
        ATTEMPTS.lock().unwrap().remove(&owner.0);
        fixture
            .getattr("Storage")
            .unwrap()
            .setattr("silent_release", false)
            .unwrap();

        for mode in ["replace_counter", "remove_counter"] {
            let owner = AttemptOwner::reserve(
                Arc::clone(&process.client),
                (mode.into(), 0, 0),
                Arc::clone(&tensors),
                64,
            )
            .unwrap();
            owner.export(py, &exporter).unwrap();
            fixture
                .getattr("Storage")
                .unwrap()
                .setattr(mode, true)
                .unwrap();
            assert!(owner.rollback(py).is_err());
            assert!(instance_pending(&process.client, mode));
            let calls = fixture.getattr("calls").unwrap().len().unwrap();
            owner.rollback(py).unwrap();
            assert_eq!(fixture.getattr("calls").unwrap().len().unwrap(), calls);
            ATTEMPTS.lock().unwrap().remove(&owner.0);
            fixture
                .getattr("Storage")
                .unwrap()
                .setattr(mode, false)
                .unwrap();
            let path: String = fixture.getattr("counter_path").unwrap().extract().unwrap();
            if !std::path::Path::new(&path).exists() {
                std::fs::write(path, vec![0; 64 + 10_000 * 8]).unwrap();
            }
        }

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
        let credits = owner.accepted().unwrap();
        let evidence = credit_evidence(&credits);
        assert_eq!(evidence["credits"][0]["disposition"], "importer_owned");
        assert_eq!(evidence["credits"][0]["observation"]["same_object"], true);
        let path: String = fixture.getattr("counter_path").unwrap().extract().unwrap();
        let name = std::path::Path::new(&path);
        let backup = name.with_extension("held");
        std::fs::rename(name, &backup).unwrap();
        std::fs::write(name, vec![0; 64 + 10_000 * 8]).unwrap();
        let replaced = credit_evidence(&credits);
        assert_eq!(replaced["credits"][0]["observation"]["same_object"], false);
        assert_eq!(
            replaced["credits"][0]["observation"]["inode"],
            evidence["credits"][0]["observation"]["inode"]
        );
        std::fs::remove_file(name).unwrap();
        let absent = credit_evidence(&credits);
        assert!(!absent["credits"][0]["observation"]["path_error"].is_null());
        assert_eq!(
            absent["credits"][0]["observation"]["inode"],
            evidence["credits"][0]["observation"]["inode"]
        );
        std::fs::rename(backup, name).unwrap();
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
        let backing_bytes = 26144_u64 * 8 * 16 * 256 * 2 * 36;
        let shared_bytes = retained_storage_bytes([(0x1000, backing_bytes); 36]).unwrap();
        let shared = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("shared-backing".into(), 0, 0),
            Arc::clone(&tensors),
            shared_bytes,
        )
        .unwrap();
        let distinct_bytes = retained_storage_bytes(
            (0..36).map(|index| (0x1000 + index * backing_bytes, backing_bytes)),
        )
        .unwrap();
        assert!(distinct_bytes > MAX_RETAINED_BYTES);
        assert!(
            AttemptOwner::reserve(
                Arc::clone(&process.client),
                ("distinct-over-budget".into(), 0, 0),
                Arc::clone(&tensors),
                distinct_bytes,
            )
            .is_err()
        );
        shared.rollback(py).unwrap();
        let charged = AttemptOwner::reserve(
            Arc::clone(&process.client),
            ("already-charged".into(), 0, 0),
            Arc::clone(&tensors),
            MAX_RETAINED_BYTES / 2 + 1,
        )
        .unwrap();
        assert!(
            AttemptOwner::reserve(
                Arc::clone(&process.client),
                ("separate-attempt".into(), 0, 0),
                Arc::clone(&tensors),
                MAX_RETAINED_BYTES / 2 + 1,
            )
            .is_err()
        );
        charged.rollback(py).unwrap();
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
        let counter: String = fixture.getattr("counter_path").unwrap().extract().unwrap();
        std::fs::remove_file(counter).unwrap();
        if previous.bind(py).is_none() {
            modules.del_item("torch").unwrap();
        } else {
            modules.set_item("torch", previous).unwrap();
        }
        println!(
            "actual_attempt_owner=1 duplicate_preexport=1 never_sent_refunded=1 unknown_retained=1 lost_release_not_retried=1 silent_release_retained=1 replaced_counter_retained=1 missing_counter_retained=1 partial_export_retained=1 consumed_owner_handoff=1 budget_reject=1 CUDA_operations=0"
        );
    });
    process.stop();
}
