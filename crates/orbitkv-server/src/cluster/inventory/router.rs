//! Read-only router projections. The source inventory owns the resume watermark.
use std::time::Duration;

use orbitkv_core::InventoryReadError;
use orbitkv_state::{
    CacheOwner, DiscoveryCoverage, InventoryRecord, InventoryScope, ReplicaLocation,
    ReplicaMetadata, StateKey, validate_discovery_query,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::InventoryRuntime;

#[derive(Debug)]
pub(crate) enum RouterError {
    Invalid(&'static str),
    Unavailable,
    Resync,
    Inventory,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Cursor {
    stream: String,
    sequence: u64,
}

#[derive(Serialize)]
struct Placement {
    hash: Vec<u8>,
    peers: Vec<ReplicaLocation>,
    peer_coverage: DiscoveryCoverage,
}

#[derive(Serialize)]
struct Residence {
    hash: Vec<u8>,
    sequence: u64,
    present: bool,
    metadata: ReplicaMetadata,
}

impl TryFrom<InventoryRecord> for Residence {
    type Error = RouterError;

    fn try_from(record: InventoryRecord) -> Result<Self, Self::Error> {
        Ok(Self {
            hash: record.key.hash,
            sequence: record.sequence,
            present: record.present,
            metadata: record.metadata.ok_or(RouterError::Inventory)?,
        })
    }
}

#[derive(Serialize)]
pub(crate) struct Lookup {
    source: CacheOwner,
    cursor: Cursor,
    local: Vec<Residence>,
    placements: Vec<Placement>,
}

#[derive(Serialize)]
pub(crate) struct Events {
    cursor: Cursor,
    changes: Vec<Residence>,
}

impl InventoryRuntime {
    fn router_stream(&self, scope: &InventoryScope) -> Result<String, RouterError> {
        let generation = self
            .shared
            .membership
            .admitted_generation()
            .ok_or(RouterError::Unavailable)?;
        let mut digest = Sha256::new();
        digest.update(b"orbitkv/router-source/v1\0");
        digest.update(self.shared.format.cluster_uuid.as_bytes());
        digest.update(self.shared.own.owner.incarnation.as_bytes());
        digest.update(self.shared.own.epoch.to_be_bytes());
        digest.update(generation.to_be_bytes());
        digest.update(scope.digest());
        Ok(format!("{:x}", digest.finalize()))
    }

    pub(crate) fn router_lookup(
        &self,
        scope: &InventoryScope,
        hashes: Vec<Vec<u8>>,
    ) -> Result<Lookup, RouterError> {
        let Some([namespace]) = scope.namespaces() else {
            return Err(RouterError::Invalid(
                "one exact storage namespace is required",
            ));
        };
        validate_discovery_query(namespace, &hashes).map_err(RouterError::Invalid)?;
        let stream = self.router_stream(scope)?;
        let keys: Vec<_> = hashes
            .into_iter()
            .map(|hash| StateKey::new(namespace.to_owned(), hash))
            .collect();
        let (sequence, records) = self
            .shared
            .inventory
            .lookup_at_sequence(&keys)
            .map_err(|_| RouterError::Inventory)?;
        let placements = self
            .shared
            .index
            .lookup(&keys)
            .into_iter()
            .map(|row| Placement {
                hash: row.key.hash,
                peers: row.replicas,
                peer_coverage: row.coverage,
            })
            .collect();
        if self.router_stream(scope)? != stream {
            return Err(RouterError::Unavailable);
        }
        Ok(Lookup {
            source: self.shared.own.owner.clone(),
            cursor: Cursor { stream, sequence },
            local: records
                .into_iter()
                .map(Residence::try_from)
                .collect::<Result<_, _>>()?,
            placements,
        })
    }

    pub(crate) async fn router_events(
        &self,
        scope: &InventoryScope,
        cursor: Cursor,
        wait_ms: u64,
    ) -> Result<Events, RouterError> {
        if wait_ms > 1000 {
            return Err(RouterError::Invalid("wait_ms must be between 0 and 1000"));
        }
        if cursor.stream != self.router_stream(scope)? {
            return Err(RouterError::Resync);
        }
        // Subscribe before sampling the head so an insertion cannot miss the wakeup.
        let mut changed = self.shared.inventory.changed();
        let mut membership = self.shared.member_changes.subscribe();
        if wait_ms != 0 && cursor.sequence == self.shared.inventory.sequence() {
            tokio::select! {
                _ = changed.changed() => {},
                _ = membership.changed() => {},
                _ = tokio::time::sleep(Duration::from_millis(wait_ms)) => {},
            }
        }
        let head = self.shared.inventory.sequence();
        let delta = self
            .shared
            .inventory
            .coalesced_changes_scoped(cursor.sequence, head, scope)
            .map_err(|error| match error {
                InventoryReadError::HistoryGap => RouterError::Resync,
                _ => RouterError::Inventory,
            })?;
        if cursor.stream != self.router_stream(scope)? {
            return Err(RouterError::Resync);
        }
        Ok(Events {
            cursor: Cursor {
                sequence: delta.through,
                ..cursor
            },
            changes: delta
                .records
                .into_iter()
                .map(Residence::try_from)
                .collect::<Result<_, _>>()?,
        })
    }
}
