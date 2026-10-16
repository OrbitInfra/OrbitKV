//! Native ownership of CUDA exports until registration has an import receipt.

use std::collections::HashMap;
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
    credits: Vec<(Vec<u8>, u64, CreditState)>,
    exporting: bool,
    message: String,
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
            || tensors.len() > MAX_CREDITS
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
        self.update(|attempt| {
            duplicate = attempt
                .credits
                .iter()
                .any(|(prior, slot, _)| *prior == name && *slot == offset);
            if duplicate {
                for (_, _, state) in &mut attempt.credits {
                    *state = CreditState::Unknown;
                }
            } else {
                attempt
                    .credits
                    .push((name, offset, CreditState::NeverImported));
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
            for (_, _, state) in &mut attempt.credits {
                *state = CreditState::Unknown;
            }
        })
    }

    pub(crate) fn resolve(&self, py: Python<'_>, credits: &[i32]) -> PyResult<()> {
        self.update(|attempt| {
            if credits.len() == attempt.credits.len() {
                for ((_, _, state), value) in attempt.credits.iter_mut().zip(credits) {
                    *state = match CudaCreditDisposition::try_from(*value) {
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
                for (index, (name, offset, state)) in attempt.credits.iter_mut().enumerate() {
                    if *state == CreditState::NeverImported {
                        *state = CreditState::Unknown;
                        next = Some((index, name.clone(), *offset));
                        break;
                    }
                }
            })?;
            let Some((index, name, offset)) = next else {
                break;
            };
            // Do not hold the registry mutex while calling the framework ABI.
            // A lost return leaves exactly this credit Unknown; no second repay.
            py.import("torch")?
                .getattr("UntypedStorage")?
                .call_method1(
                    "_release_ipc_counter_cuda",
                    (PyBytes::new(py, &name), offset),
                )?;
            self.update(|attempt| attempt.credits[index].2 = CreditState::Repaid)?;
        }
        let mut attempts = ATTEMPTS
            .lock()
            .map_err(|_| crate::OrbitKVInternal::new_err("registration attempt lock poisoned"))?;
        if attempts.get(&self.0).is_some_and(|attempt| {
            !attempt.exporting
                && attempt
                    .credits
                    .iter()
                    .all(|(_, _, state)| *state == CreditState::Repaid)
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
                    .any(|(_, _, state)| *state != CreditState::ImporterOwned)
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
                .map(|(_, _, state)| {
                    match state {
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
