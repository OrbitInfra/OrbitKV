//! Native ownership of CUDA exports until registration has an import receipt.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use orbitkv_channel::CacheClient;
use orbitkv_proto::proto::engine::CudaCreditDisposition;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use uuid::Uuid;

pub(crate) type RetainedAttempt = (Vec<u8>, String, u64, Vec<String>, bool);

const MAX_ATTEMPTS: usize = 64;
const MAX_CREDITS: usize = 4096;
const MAX_RETAINED_BYTES: u64 = 1 << 40;

static ATTEMPTS: LazyLock<Mutex<HashMap<[u8; 16], RegistrationAttempt>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) struct RegistrationAttempt {
    client: Arc<CacheClient>,
    key: (String, u32, i32),
    _tensors: Arc<Vec<Py<PyAny>>>,
    bytes: u64,
    shared: Vec<Py<PyAny>>,
    credits: Vec<ExportCredit>,
    exporting: bool,
    message: String,
}

struct ExportCredit {
    name: Vec<u8>,
    offset: u64,
    state: CreditState,
    counter: Option<File>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CreditState {
    NeverImported,
    Unknown,
    ImporterOwned,
    Repaid,
}

pub(crate) struct AttemptOwner(pub(crate) [u8; 16]);

impl AttemptOwner {
    pub(crate) fn reserve(
        client: Arc<CacheClient>,
        key: (String, u32, i32),
        tensors: Arc<Vec<Py<PyAny>>>,
        bytes: u64,
    ) -> Result<Self, String> {
        let mut attempts = ATTEMPTS
            .lock()
            .map_err(|_| "registration attempt lock poisoned")?;
        if attempts
            .values()
            .any(|attempt| Arc::ptr_eq(&attempt.client, &client) && attempt.key == key)
        {
            return Err("GPU registration already pending or retained on this client".into());
        }
        let total = attempts
            .values()
            .try_fold(bytes, |total, attempt| total.checked_add(attempt.bytes));
        if attempts.len() >= MAX_ATTEMPTS
            || tensors.is_empty()
            || tensors.len() > MAX_CREDITS * 2
            || total.is_none_or(|total| total > MAX_RETAINED_BYTES)
        {
            return Err("CUDA registration retention admission limit reached".into());
        }
        let generation = Uuid::new_v4().into_bytes();
        attempts.insert(
            generation,
            RegistrationAttempt {
                client,
                key,
                _tensors: tensors,
                bytes,
                shared: Vec::new(),
                credits: Vec::new(),
                exporting: false,
                message: String::new(),
            },
        );
        Ok(Self(generation))
    }

    pub(crate) fn export(&self, py: Python<'_>, storage: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.update(|attempt| attempt.exporting = true)?;
        let shared = storage.call_method0("_share_cuda_")?.unbind();
        self.update(|attempt| attempt.shared.push(shared.clone_ref(py)))?;
        let tuple = shared.bind(py).cast::<PyTuple>()?;
        // Own the credit before parsing any other field in the returned tuple.
        let name: Vec<u8> = tuple.get_item(4)?.extract()?;
        let offset: u64 = tuple.get_item(5)?.extract()?;
        if name.is_empty() || name.len() > 255 || name.contains(&0) || offset >= 10_000 {
            return Err(crate::OrbitKVError::new_err(
                "invalid CUDA export counter; owner retained",
            ));
        }
        let mut duplicate = false;
        let counter = counter_file(&name, offset)
            .map_err(|error| crate::OrbitKVError::new_err(error.to_string()))?;
        if counter_value(&counter, offset)? != 1 {
            return Err(crate::OrbitKVError::new_err(
                "CUDA export credit is not exclusively outstanding; owner retained",
            ));
        }
        self.update(|attempt| {
            duplicate = attempt
                .credits
                .iter()
                .any(|prior| prior.name == name && prior.offset == offset);
            if duplicate {
                for credit in &mut attempt.credits {
                    credit.state = CreditState::Unknown;
                }
            } else {
                attempt.credits.push(ExportCredit {
                    name,
                    offset,
                    state: CreditState::NeverImported,
                    counter: Some(counter),
                });
                attempt.exporting = false;
            }
        })?;
        if duplicate {
            return Err(crate::OrbitKVError::new_err(
                "duplicate CUDA export credit; owner retained",
            ));
        }
        Ok(shared)
    }

    pub(crate) fn submitted(&self) -> PyResult<()> {
        self.update(|attempt| {
            for credit in &mut attempt.credits {
                credit.state = CreditState::Unknown;
            }
        })
    }

    pub(crate) fn resolve(&self, py: Python<'_>, credits: &[i32]) -> PyResult<()> {
        self.update(|attempt| {
            if credits.len() == attempt.credits.len() {
                for (credit, value) in attempt.credits.iter_mut().zip(credits) {
                    credit.state = match CudaCreditDisposition::try_from(*value) {
                        Ok(CudaCreditDisposition::NeverImported) => CreditState::NeverImported,
                        Ok(CudaCreditDisposition::ImporterOwned) => CreditState::ImporterOwned,
                        _ => CreditState::Unknown,
                    };
                }
            }
        })?;
        // Only NeverImported receipts authorize a decrement. ImporterOwned
        // credits on a failed attempt remain retained until separately reconciled.
        self.rollback(py)
    }

    pub(crate) fn rollback(&self, py: Python<'_>) -> PyResult<()> {
        loop {
            let mut next = None;
            self.update(|attempt| {
                for (index, credit) in attempt.credits.iter_mut().enumerate() {
                    if credit.state == CreditState::NeverImported {
                        credit.state = CreditState::Unknown;
                        next = Some((index, credit.name.clone(), credit.offset));
                        break;
                    }
                }
            })?;
            let Some((index, name, offset)) = next else {
                break;
            };
            let counter = {
                let attempts = ATTEMPTS.lock().map_err(|_| {
                    crate::OrbitKVInternal::new_err("registration attempt lock poisoned")
                })?;
                attempts
                    .get(&self.0)
                    .and_then(|attempt| attempt.credits[index].counter.as_ref())
                    .ok_or_else(|| crate::OrbitKVError::new_err("CUDA credit file owner absent"))?
                    .try_clone()?
            };
            let current = counter_file(&name, offset)
                .map_err(|error| crate::OrbitKVError::new_err(error.to_string()))?;
            let prior = counter.metadata()?;
            let observed = current.metadata()?;
            if (prior.dev(), prior.ino()) != (observed.dev(), observed.ino()) {
                return Err(crate::OrbitKVError::new_err(
                    "CUDA export counter identity changed; owner retained",
                ));
            }
            if counter_value(&counter, offset)? != 1 {
                return Err(crate::OrbitKVError::new_err(
                    "CUDA export credit is not exclusively outstanding; owner retained",
                ));
            }
            // Do not hold the registry mutex while calling the framework ABI.
            // A lost return leaves exactly this credit Unknown; no second repay.
            py.import("torch")?
                .getattr("UntypedStorage")?
                .call_method1(
                    "_release_ipc_counter_cuda",
                    (PyBytes::new(py, &name), offset),
                )?;
            let current = counter_file(&name, offset)
                .map_err(|error| crate::OrbitKVError::new_err(error.to_string()))?;
            let prior = counter.metadata()?;
            let observed = current.metadata()?;
            if (prior.dev(), prior.ino()) != (observed.dev(), observed.ino())
                || counter_value(&counter, offset)? != 0
                || counter_value(&current, offset)? != 0
            {
                return Err(crate::OrbitKVError::new_err(
                    "CUDA export counter repayment unproven; owner retained",
                ));
            }
            self.update(|attempt| {
                attempt.credits[index].state = CreditState::Repaid;
                attempt.credits[index].counter = None;
            })?;
        }
        let mut attempts = ATTEMPTS
            .lock()
            .map_err(|_| crate::OrbitKVInternal::new_err("registration attempt lock poisoned"))?;
        if attempts.get(&self.0).is_some_and(|attempt| {
            !attempt.exporting
                && attempt
                    .credits
                    .iter()
                    .all(|credit| credit.state == CreditState::Repaid)
        }) {
            attempts.remove(&self.0);
        }
        Ok(())
    }

    pub(crate) fn accepted(&self) -> PyResult<()> {
        let mut attempts = ATTEMPTS
            .lock()
            .map_err(|_| crate::OrbitKVInternal::new_err("registration attempt lock poisoned"))?;
        if attempts.get(&self.0).is_none_or(|attempt| {
            attempt.exporting
                || attempt
                    .credits
                    .iter()
                    .any(|credit| credit.state != CreditState::ImporterOwned)
        }) {
            return Err(crate::OrbitKVError::new_err(
                "CUDA registration lacks complete importer receipt; owner retained",
            ));
        }
        attempts.remove(&self.0);
        Ok(())
    }

    fn update(&self, apply: impl FnOnce(&mut RegistrationAttempt)) -> PyResult<()> {
        let mut attempts = ATTEMPTS
            .lock()
            .map_err(|_| crate::OrbitKVInternal::new_err("registration attempt lock poisoned"))?;
        let attempt = attempts
            .get_mut(&self.0)
            .ok_or_else(|| crate::OrbitKVInternal::new_err("registration attempt owner absent"))?;
        apply(attempt);
        Ok(())
    }
}

fn counter_file(name: &[u8], offset: u64) -> std::io::Result<File> {
    let name = name.strip_prefix(b"/").unwrap_or(name);
    if name.is_empty()
        || name == b"."
        || name == b".."
        || name.contains(&b'/')
        || name.contains(&0)
        || offset >= 10_000
    {
        return Err(std::io::Error::other(
            "invalid CUDA export counter identity",
        ));
    }
    let path = Path::new("/dev/shm").join(std::ffi::OsStr::from_bytes(name));
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() < 64 + (offset + 1) * 8 {
        return Err(std::io::Error::other("CUDA export counter slot is absent"));
    }
    Ok(file)
}

fn counter_value(file: &File, offset: u64) -> PyResult<i64> {
    let mut bytes = [0; 8];
    file.read_exact_at(&mut bytes, 64 + offset * 8)?;
    Ok(i64::from_ne_bytes(bytes))
}

impl Drop for AttemptOwner {
    fn drop(&mut self) {
        if let Ok(mut attempts) = ATTEMPTS.lock()
            && let Some(attempt) = attempts.get_mut(&self.0)
        {
            attempt.message =
                "registration receipt/credit termination unproven; resources retained".into();
        }
    }
}

pub(crate) fn retention() -> (usize, u64, usize) {
    let attempts = ATTEMPTS.lock().unwrap_or_else(|poison| poison.into_inner());
    let bytes = attempts.values().map(|attempt| attempt.bytes).sum();
    let credits = attempts.values().map(|attempt| attempt.credits.len()).sum();
    (attempts.len(), bytes, credits)
}

#[cfg(test)]
#[path = "../tests/unit/registration.rs"]
mod tests;

pub(crate) fn instance_pending(client: &Arc<CacheClient>, instance: &str) -> bool {
    let attempts = ATTEMPTS.lock().unwrap_or_else(|poison| poison.into_inner());
    attempts
        .values()
        .any(|attempt| Arc::ptr_eq(&attempt.client, client) && attempt.key.0 == instance)
}

pub(crate) fn retained_attempts() -> Vec<RetainedAttempt> {
    let attempts = ATTEMPTS.lock().unwrap_or_else(|poison| poison.into_inner());
    attempts
        .iter()
        .map(|(generation, attempt)| {
            let states = attempt
                .credits
                .iter()
                .map(|credit| {
                    match credit.state {
                        CreditState::NeverImported => "never_imported",
                        CreditState::Unknown => "unknown",
                        CreditState::ImporterOwned => "importer_owned",
                        CreditState::Repaid => "repaid",
                    }
                    .to_string()
                })
                .collect();
            (
                generation.to_vec(),
                attempt.key.0.clone(),
                attempt.bytes,
                states,
                attempt.exporting,
            )
        })
        .collect()
}
