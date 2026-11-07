use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::block::RestoreSource;
use crate::{EngineError, QueryOwner, QueryReservation};

const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(600);
const DEFAULT_LEASE_SWEEP_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryLeaseId([u8; 16]);

impl QueryLeaseId {
    pub fn fresh() -> Self {
        Self(*Uuid::new_v4().as_bytes())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("query lease id must be non-empty".to_string());
        }
        let token: [u8; 16] = bytes
            .try_into()
            .map_err(|_| format!("query lease id must be 16 bytes, got {}", bytes.len()))?;
        Ok(Self(token))
    }

    pub fn to_bytes(&self) -> [u8; 16] {
        self.0
    }
}

impl fmt::Debug for QueryLeaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryLeaseId({})", Uuid::from_bytes(self.0))
    }
}

struct QueryLease {
    instance_id: String,
    blocks: Vec<RestoreSource>,
    remaining_consumers: usize,
    expires_at: Instant,
    ownership: Option<(QueryOwner, QueryReservation)>,
}

pub(crate) struct QueryLeaseManager {
    inner: Arc<QueryLeaseInner>,
    sweeper: Option<JoinHandle<()>>,
}

struct QueryLeaseInner {
    leases: Mutex<HashMap<QueryLeaseId, QueryLease>>,
}

impl Default for QueryLeaseManager {
    fn default() -> Self {
        Self::new(DEFAULT_LEASE_SWEEP_INTERVAL)
    }
}

impl QueryLeaseManager {
    fn new(sweep_interval: Duration) -> Self {
        let inner = Arc::new(QueryLeaseInner {
            leases: Mutex::new(HashMap::new()),
        });
        let sweeper = tokio::runtime::Handle::try_current().ok().map(|handle| {
            let inner = Arc::clone(&inner);
            handle.spawn(async move {
                let mut interval = tokio::time::interval(sweep_interval);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    if let Err(error) = inner.sweep_expired() {
                        log::error!("Query lease sweep retained owners: {error}");
                    }
                }
            })
        });

        Self { inner, sweeper }
    }

    pub(crate) fn create(
        &self,
        instance_id: &str,
        blocks: Vec<RestoreSource>,
        consumers: usize,
        ownership: Option<(QueryOwner, QueryReservation)>,
    ) -> Result<QueryLeaseId, EngineError> {
        self.sweep_expired()?;
        debug_assert!(!blocks.is_empty(), "query leases require ready blocks");

        let token = QueryLeaseId::fresh();
        let lease = QueryLease {
            instance_id: instance_id.to_string(),
            blocks,
            remaining_consumers: consumers.max(1),
            expires_at: Instant::now() + DEFAULT_LEASE_TTL,
            ownership,
        };
        self.inner.insert(token, lease)?;
        Ok(token)
    }

    pub(crate) fn claim(&self, token: &QueryLeaseId) -> Result<(), EngineError> {
        let leases = self
            .inner
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?;
        if let Some(lease) = leases.get(token)
            && let Some((_, reservation)) = &lease.ownership
        {
            reservation.claim();
        }
        Ok(())
    }

    pub(crate) fn validate_registration<'a>(
        &self,
        tokens: impl Iterator<Item = &'a QueryLeaseId>,
        generation: [u8; 16],
    ) -> Result<(), EngineError> {
        let leases = self
            .inner
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?;
        for token in tokens {
            let lease = leases
                .get(token)
                .ok_or_else(|| EngineError::Storage("query lease is unknown or expired".into()))?;
            if let Some((_, reservation)) = &lease.ownership
                && reservation.registration_generation() != Some(generation)
            {
                return Err(EngineError::InvalidArgument(
                    "query lease registration changed".into(),
                ));
            }
        }
        Ok(())
    }

    /// Validate the whole batch before consuming any lease share. The validator
    /// runs under the lease lock and must not perform I/O or re-enter this manager.
    pub(crate) fn consume_batch<T>(
        &self,
        instance_id: &str,
        tokens: &[QueryLeaseId],
        validate: impl FnOnce(&[&[RestoreSource]]) -> Result<T, EngineError>,
    ) -> Result<(T, Vec<RestoreSource>, Vec<QueryReservation>), EngineError> {
        let mut leases = self
            .inner
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?;
        let now = Instant::now();
        let mut seen = HashSet::with_capacity(tokens.len());
        for token in tokens {
            if !seen.insert(*token) {
                return Err(EngineError::InvalidArgument(
                    "restore batch contains duplicate query lease".to_string(),
                ));
            }
            let lease = leases
                .get(token)
                .filter(|lease| lease.expires_at > now)
                .ok_or_else(|| {
                    EngineError::Storage("query lease is unknown or expired".to_string())
                })?;
            if lease.instance_id != instance_id {
                return Err(EngineError::Storage(format!(
                    "query lease belongs to instance {}, got {}",
                    lease.instance_id, instance_id
                )));
            }
        }
        let sources: Vec<_> = tokens
            .iter()
            .map(|token| leases[token].blocks.as_slice())
            .collect();
        let validated = validate(&sources)?;

        let mut blocks = Vec::with_capacity(sources.iter().map(|blocks| blocks.len()).sum());
        let mut reservations = Vec::with_capacity(tokens.len());
        for token in tokens {
            let lease = leases.get_mut(token).ok_or_else(|| {
                EngineError::Storage("query lease disappeared during consume".into())
            })?;
            if lease.remaining_consumers > 1 {
                lease.remaining_consumers -= 1;
                blocks.extend(lease.blocks.iter().cloned());
                if let Some((_, reservation)) = &lease.ownership {
                    reservations.push(reservation.clone());
                }
            } else {
                let lease = leases.remove(token).ok_or_else(|| {
                    EngineError::Storage("query lease disappeared during consume".into())
                })?;
                blocks.extend(lease.blocks);
                if let Some((_, reservation)) = lease.ownership {
                    reservations.push(reservation);
                }
            }
        }
        Ok((validated, blocks, reservations))
    }

    pub(crate) fn release(&self, token: &QueryLeaseId) -> Result<bool, EngineError> {
        self.sweep_expired()?;
        self.inner.remove(token)
    }

    pub(crate) fn release_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let mut leases = self
            .inner
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?;
        leases.retain(|_, lease| lease.instance_id != instance_id);
        Ok(())
    }

    pub(crate) fn release_owner(
        &self,
        matches: impl Fn(QueryOwner) -> bool,
    ) -> Result<(), EngineError> {
        self.inner
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?
            .retain(|_, lease| {
                !lease
                    .ownership
                    .as_ref()
                    .is_some_and(|(owner, _)| matches(*owner))
            });
        Ok(())
    }

    pub(crate) fn sweep_expired(&self) -> Result<(), EngineError> {
        self.inner.sweep_expired()
    }
}

impl Drop for QueryLeaseManager {
    fn drop(&mut self) {
        if let Some(sweeper) = self.sweeper.take() {
            sweeper.abort();
        }
    }
}

impl QueryLeaseInner {
    fn insert(&self, token: QueryLeaseId, lease: QueryLease) -> Result<(), EngineError> {
        self.leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?
            .insert(token, lease);
        Ok(())
    }

    fn remove(&self, token: &QueryLeaseId) -> Result<bool, EngineError> {
        Ok(self
            .leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?
            .remove(token)
            .is_some())
    }

    fn sweep_expired(&self) -> Result<(), EngineError> {
        let now = Instant::now();
        self.leases
            .lock()
            .map_err(|_| EngineError::Poisoned("query leases"))?
            .retain(|_, lease| lease.expires_at > now);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/query/lease.rs"]
mod tests;
