//! Fixed publication lifetimes, shared by resident, file and in-flight owners.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Absolute cache retention bounds carried by SSD records and peer copies.
/// Reading or moving a replica never renews these bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheExpiry {
    pub(crate) published_unix_millis: u64,
    pub(crate) expires_unix_millis: u64,
}

impl CacheExpiry {
    /// Original publication time, in milliseconds since the Unix epoch.
    pub fn published_unix_millis(&self) -> u64 {
        self.published_unix_millis
    }

    /// Exclusive lifetime bound, in milliseconds since the Unix epoch.
    pub fn expires_unix_millis(&self) -> u64 {
        self.expires_unix_millis
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.published_unix_millis == 0 || self.expires_unix_millis <= self.published_unix_millis
        {
            return Err("invalid cache publication lifetime".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct Expiration {
    pub(crate) record: CacheExpiry,
    deadline: Instant,
    expired: AtomicBool,
}

pub(crate) fn unix_millis() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .ok_or_else(|| "cache TTL requires a valid Unix clock".into())
}

impl Expiration {
    pub(crate) fn new(ttl: Option<Duration>) -> Result<Option<Arc<Self>>, String> {
        let Some(ttl) = ttl else { return Ok(None) };
        let millis = u64::try_from(ttl.as_millis())
            .ok()
            .filter(|&millis| millis != 0)
            .ok_or("cache TTL must be positive and fit in milliseconds")?;
        let published_unix_millis = unix_millis()?;
        let expires_unix_millis = published_unix_millis
            .checked_add(millis)
            .ok_or("cache TTL deadline overflow")?;
        Self::from_record(Some(CacheExpiry {
            published_unix_millis,
            expires_unix_millis,
        }))
    }

    pub(crate) fn from_record(record: Option<CacheExpiry>) -> Result<Option<Arc<Self>>, String> {
        record
            .map(|record| Self::at(record, unix_millis()?, Instant::now()).map(Arc::new))
            .transpose()
    }

    fn at(record: CacheExpiry, now: u64, monotonic: Instant) -> Result<Self, String> {
        record.validate()?;
        let expired = now < record.published_unix_millis || now >= record.expires_unix_millis;
        let remaining = if expired {
            0
        } else {
            record.expires_unix_millis - now
        };
        let deadline = monotonic
            .checked_add(Duration::from_millis(remaining))
            .ok_or("cache TTL monotonic deadline overflow")?;
        Ok(Self {
            record,
            deadline,
            expired: AtomicBool::new(expired),
        })
    }

    pub(crate) fn is_expired(&self) -> bool {
        self.expired_at(unix_millis(), Instant::now())
    }

    fn expired_at(&self, now: Result<u64, String>, monotonic: Instant) -> bool {
        if self.expired.load(Ordering::Acquire) {
            return true;
        }
        if monotonic >= self.deadline
            || now.map_or(true, |now| {
                now < self.record.published_unix_millis || now >= self.record.expires_unix_millis
            })
        {
            self.expired.store(true, Ordering::Release);
            return true;
        }
        false
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/expiry.rs"]
mod tests;
