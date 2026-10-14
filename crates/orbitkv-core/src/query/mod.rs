//! Byte reservations held from query admission through the last GPU consumer.

pub(crate) mod lease;
pub(crate) mod read;
mod tier_attribution;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use opentelemetry::{KeyValue, global};
use parking_lot::Mutex;

use crate::metrics::core_metrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryOwner {
    pub session: u64,
    pub operation: u64,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryMode {
    Demand,
    WaitForFullPrefix,
    Warmup,
    Prepare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Warming,
    Preloading,
    Prepared,
    Preparing,
    Ready,
    Restoring,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Preloading => "preloading",
            Self::Prepared => "prepared",
            Self::Preparing => "preparing",
            Self::Ready => "ready",
            Self::Restoring => "restoring",
        }
    }

    fn speculative(self) -> bool {
        matches!(self, Self::Warming | Self::Preloading | Self::Prepared)
    }
}

#[derive(Default)]
struct Usage {
    total: u64,
    instances: HashMap<String, u64>,
    warming: u64,
    warming_instances: HashMap<String, u64>,
}

pub(crate) struct QueryBudget {
    global: u64,
    per_instance: u64,
    usage: Mutex<Usage>,
    total_peak: Arc<AtomicU64>,
    instance_peak: Arc<AtomicU64>,
}

pub enum QueryAdmission {
    Admitted(QueryReservation),
    Busy,
    TooLarge,
}

/// Accounting is conservative: two owners of the same pages each reserve bytes.
/// The pinned allocator separately measures physical memory shared by owners.
#[derive(Clone)]
pub struct QueryReservation(Arc<Reservation>, Option<[u8; 16]>);

struct Reservation {
    budget: Arc<QueryBudget>,
    pub(crate) instance: String,
    pub(crate) namespace: String,
    state: Mutex<(u64, Phase)>,
}

impl QueryBudget {
    pub(crate) fn new(global: u64, per_instance: u64) -> Result<Arc<Self>, String> {
        if global == 0 || per_instance == 0 || per_instance > global || global > i64::MAX as u64 {
            return Err("query byte limits must satisfy 0 < instance <= global <= i64::MAX".into());
        }
        for phase in [
            Phase::Warming,
            Phase::Preloading,
            Phase::Prepared,
            Phase::Preparing,
            Phase::Ready,
            Phase::Restoring,
        ] {
            account(0, phase);
        }
        let total_peak = Arc::new(AtomicU64::new(0));
        let instance_peak = Arc::new(AtomicU64::new(0));
        let meter = global::meter("orbitkv-core");
        let weak = Arc::downgrade(&total_peak);
        meter
            .u64_observable_gauge("orbitkv_query_reserved_peak_bytes")
            .with_description("Maximum concurrent conservative query bytes since budget creation")
            .with_unit("bytes")
            .with_callback(move |observer| {
                if let Some(peak) = weak.upgrade() {
                    observer.observe(peak.load(Ordering::Relaxed), &[]);
                }
            })
            .build();
        let weak = Arc::downgrade(&instance_peak);
        meter
            .u64_observable_gauge("orbitkv_query_instance_reserved_peak_bytes")
            .with_description("Maximum concurrent conservative query bytes of any one instance since budget creation")
            .with_unit("bytes")
            .with_callback(move |observer| {
                if let Some(peak) = weak.upgrade() {
                    observer.observe(peak.load(Ordering::Relaxed), &[]);
                }
            })
            .build();
        Ok(Arc::new(Self {
            global,
            per_instance,
            usage: Mutex::new(Usage::default()),
            total_peak,
            instance_peak,
        }))
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        instance: &str,
        namespace: &str,
        bytes: u64,
        mode: QueryMode,
    ) -> QueryAdmission {
        let warming = matches!(mode, QueryMode::Warmup | QueryMode::Prepare);
        let limit = if warming {
            self.per_instance / 4
        } else {
            self.per_instance
        };
        if bytes > limit {
            core_metrics().query_budget_bypasses.add(1, &[]);
            return QueryAdmission::TooLarge;
        }
        let mut usage = self.usage.lock();
        if mode == QueryMode::Warmup && usage.total > usage.warming {
            core_metrics().warmup_foreground_skips.add(1, &[]);
            return QueryAdmission::Busy;
        }
        let instance_used = usage.instances.get(instance).copied().unwrap_or(0);
        let warm_used = usage.warming_instances.get(instance).copied().unwrap_or(0);
        if bytes > self.global - usage.total
            || bytes > self.per_instance - instance_used
            || (warming
                && (bytes > self.global / 4 - usage.warming
                    || bytes > self.per_instance / 4 - warm_used))
        {
            core_metrics().query_budget_waits.add(1, &[]);
            return QueryAdmission::Busy;
        }
        usage.total += bytes;
        self.total_peak.fetch_max(usage.total, Ordering::Relaxed);
        self.instance_peak
            .fetch_max(instance_used + bytes, Ordering::Relaxed);
        core_metrics().query_reserved_bytes.add(bytes as i64, &[]);
        *usage.instances.entry(instance.into()).or_default() += bytes;
        let phase = if warming {
            usage.warming += bytes;
            core_metrics()
                .query_speculative_reserved_bytes
                .add(bytes as i64, &[]);
            *usage.warming_instances.entry(instance.into()).or_default() += bytes;
            if mode == QueryMode::Prepare {
                Phase::Preloading
            } else {
                Phase::Warming
            }
        } else {
            Phase::Preparing
        };
        account(bytes as i64, phase);
        QueryAdmission::Admitted(QueryReservation(
            Arc::new(Reservation {
                budget: Arc::clone(self),
                instance: instance.into(),
                namespace: namespace.into(),
                state: Mutex::new((bytes, phase)),
            }),
            None,
        ))
    }
}

fn account(bytes: i64, phase: Phase) {
    core_metrics()
        .query_reserved_bytes_by_phase
        .add(bytes, &[KeyValue::new("phase", phase.label())]);
}

impl QueryReservation {
    pub(crate) fn bind_registration(mut self, generation: [u8; 16]) -> Self {
        self.1 = Some(generation);
        self
    }

    pub(crate) fn registration_generation(&self) -> Option<[u8; 16]> {
        self.1
    }

    pub(crate) fn instance(&self) -> &str {
        &self.0.instance
    }

    pub(crate) fn namespace(&self) -> &str {
        &self.0.namespace
    }

    pub(crate) fn ready(&self, bytes: u64) -> Result<(), String> {
        let mut state = self.0.state.lock();
        if !matches!(state.1, Phase::Preparing | Phase::Preloading) || bytes > state.0 {
            return Err("query result exceeds its registered layout byte reservation".into());
        }
        let released = state.0 - bytes;
        self.0.release_bytes(released, state.1);
        account(-(state.0 as i64), state.1);
        let phase = if state.1 == Phase::Preloading {
            Phase::Prepared
        } else {
            Phase::Ready
        };
        account(bytes as i64, phase);
        *state = (bytes, phase);
        Ok(())
    }

    /// Bound a read submission using this group's registered bytes per page.
    pub fn batch_blocks(&self, requested: usize, byte_limit: u64) -> usize {
        let bytes = self.0.state.lock().0;
        if requested == 0 || byte_limit == 0 || bytes == 0 {
            return requested.max(1);
        }
        (byte_limit / (bytes / requested as u64).max(1)).max(1) as usize
    }

    pub(crate) fn claim(&self) {
        let mut state = self.0.state.lock();
        if state.1 == Phase::Prepared {
            let mut usage = self.0.budget.usage.lock();
            release_speculative(&mut usage, &self.0.instance, state.0);
            account(-(state.0 as i64), state.1);
            state.1 = Phase::Ready;
            account(state.0 as i64, state.1);
        }
    }

    pub(crate) fn restoring(&self) {
        let mut state = self.0.state.lock();
        if state.1 == Phase::Ready {
            account(-(state.0 as i64), state.1);
            state.1 = Phase::Restoring;
            account(state.0 as i64, state.1);
        }
    }
}

impl Reservation {
    fn release_bytes(&self, bytes: u64, phase: Phase) {
        let mut usage = self.budget.usage.lock();
        usage.total -= bytes;
        core_metrics()
            .query_reserved_bytes
            .add(-(bytes as i64), &[]);
        if phase.speculative() {
            release_speculative(&mut usage, &self.instance, bytes);
        }
        if let Some(instance) = usage.instances.get_mut(&self.instance) {
            *instance -= bytes;
            if *instance == 0 {
                usage.instances.remove(&self.instance);
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let (bytes, phase) = *self.state.get_mut();
        self.release_bytes(bytes, phase);
        account(-(bytes as i64), phase);
    }
}

fn release_speculative(usage: &mut Usage, instance: &str, bytes: u64) {
    usage.warming -= bytes;
    core_metrics()
        .query_speculative_reserved_bytes
        .add(-(bytes as i64), &[]);
    if let Some(used) = usage.warming_instances.get_mut(instance) {
        *used -= bytes;
        if *used == 0 {
            usage.warming_instances.remove(instance);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/query/mod.rs"]
mod tests;
