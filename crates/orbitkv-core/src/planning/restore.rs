use std::collections::HashSet;

use crate::SsdReadPath;
use crate::block::RestoreSource;
use crate::cost::resource_id;

/// Device and aggregate shape of validated restore destinations.
/// This metadata owns neither engine pages nor a device admission permit.
#[derive(Debug)]
pub(crate) struct RestoreTargetShape {
    device_id: i32,
    bytes: u64,
    fragments: usize,
}

/// Physical restore intent after the engine has allocated destination pages.
/// It owns no source or device resource; execution owners acquire those next.
#[derive(Debug)]
pub(crate) struct RestorePlan {
    device_id: i32,
    ssd_path: Option<SsdReadPath>,
    allow_uring_fallback: bool,
    ssd_source_bytes: u64,
    ssd_source_fragments: usize,
    source_bytes: u64,
    source_fragments: usize,
    source_set_hash: u64,
    has_memory: bool,
    target_shape: Option<RestoreTargetShape>,
}

impl RestorePlan {
    pub(crate) fn new<'a>(
        device_id: i32,
        sources: impl IntoIterator<Item = (usize, &'a RestoreSource)>,
    ) -> Result<Self, String> {
        if device_id < 0 {
            return Err("restore target device must be non-negative".into());
        }
        let mut seen = HashSet::new();
        let mut source_domains = Vec::new();
        let mut plan = Self {
            device_id,
            ssd_path: None,
            allow_uring_fallback: false,
            ssd_source_bytes: 0,
            ssd_source_fragments: 0,
            source_bytes: 0,
            source_fragments: 0,
            source_set_hash: 0,
            has_memory: false,
            target_shape: None,
        };
        for (source_id, source) in sources {
            if let RestoreSource::Ssd {
                path,
                allow_uring_fallback,
                ..
            } = source
            {
                if plan.ssd_path.is_some_and(|selected| selected != *path) {
                    return Err("one restore plan cannot mix SSD read routes".into());
                }
                if plan.ssd_path.is_some() && plan.allow_uring_fallback != *allow_uring_fallback {
                    return Err("one restore plan cannot mix SSD fallback policies".into());
                }
                plan.ssd_path = Some(*path);
                plan.allow_uring_fallback = *allow_uring_fallback;
            }
            if !seen.insert(source_id) {
                continue;
            }
            match source {
                RestoreSource::Memory(block) => {
                    plan.has_memory = true;
                    plan.source_bytes = plan
                        .source_bytes
                        .checked_add(block.memory_footprint())
                        .ok_or("restore source bytes overflow")?;
                    plan.source_fragments = plan
                        .source_fragments
                        .checked_add(block.slots().iter().map(|slot| slot.num_segments()).sum())
                        .ok_or("restore source fragment count overflow")?;
                    source_domains.push((0u8, 0u64));
                }
                #[cfg(any(feature = "mooncake", feature = "orbite"))]
                RestoreSource::Peer(block) => {
                    source_domains.push((2u8, block.source_id()));
                    plan.source_bytes = plan
                        .source_bytes
                        .checked_add(block.bytes())
                        .ok_or("restore source bytes overflow")?;
                    plan.source_fragments = plan
                        .source_fragments
                        .checked_add(
                            block
                                .slots
                                .iter()
                                .map(|slot| slot.meta.num_segments())
                                .sum(),
                        )
                        .ok_or("restore fragment count overflow")?;
                }
                RestoreSource::Ssd { lease, .. } => {
                    source_domains.push((1u8, resource_id(&lease.cost_resource())));
                    for slot in &lease.entry.slots {
                        let bytes = slot.total_size();
                        let fragments = slot.num_segments();
                        plan.ssd_source_bytes = plan
                            .ssd_source_bytes
                            .checked_add(bytes)
                            .ok_or("restore source bytes overflow")?;
                        plan.ssd_source_fragments = plan
                            .ssd_source_fragments
                            .checked_add(fragments)
                            .ok_or("restore source fragment count overflow")?;
                        plan.source_bytes = plan
                            .source_bytes
                            .checked_add(bytes)
                            .ok_or("restore source bytes overflow")?;
                        plan.source_fragments = plan
                            .source_fragments
                            .checked_add(fragments)
                            .ok_or("restore source fragment count overflow")?;
                    }
                }
            }
        }
        source_domains.sort_unstable();
        source_domains.dedup();
        plan.source_set_hash = resource_id(&source_domains);
        Ok(plan)
    }

    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn ssd_path(&self) -> Option<SsdReadPath> {
        self.ssd_path
    }

    pub(crate) fn fallback_from_cufile(&mut self) -> Result<bool, String> {
        if self.ssd_path != Some(SsdReadPath::Cufile) {
            return Ok(false);
        }
        if !self.allow_uring_fallback {
            return Err("explicit cuFile restore has no device staging owner".into());
        }
        self.ssd_path = Some(SsdReadPath::Uring);
        self.allow_uring_fallback = false;
        Ok(true)
    }

    pub(crate) fn ssd_source_bytes(&self) -> u64 {
        self.ssd_source_bytes
    }

    pub(crate) fn ssd_source_fragments(&self) -> usize {
        self.ssd_source_fragments
    }

    pub(crate) fn has_memory(&self) -> bool {
        self.has_memory
    }

    pub(crate) fn bind_target_shape(&mut self, bytes: u64, fragments: usize) -> Result<(), String> {
        if bytes == 0 || fragments == 0 {
            return Err("restore target shape requires non-empty target ranges".into());
        }
        if self.target_shape.is_some() {
            return Err("restore target shape was already bound".into());
        }
        self.target_shape = Some(RestoreTargetShape {
            device_id: self.device_id,
            bytes,
            fragments,
        });
        Ok(())
    }

    pub(crate) fn target_shape(&self) -> Option<&RestoreTargetShape> {
        self.target_shape.as_ref()
    }

    pub(crate) fn source_bytes(&self) -> u64 {
        self.source_bytes
    }

    pub(crate) fn source_fragments(&self) -> usize {
        self.source_fragments
    }

    pub(crate) fn source_set_hash(&self) -> u64 {
        self.source_set_hash
    }
}

impl RestoreTargetShape {
    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn fragments(&self) -> usize {
        self.fragments
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/restore.rs"]
mod tests;
