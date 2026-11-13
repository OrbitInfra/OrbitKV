use crate::SsdReadPath;
use crate::block::RestoreSource;

/// Physical restore intent after the engine has allocated destination pages.
/// It owns no source or device resource; execution owners acquire those next.
#[derive(Debug)]
pub(crate) struct RestorePlan {
    device_id: i32,
    ssd_path: Option<SsdReadPath>,
    allow_uring_fallback: bool,
    target_bound: bool,
}

impl RestorePlan {
    pub(crate) fn new<'a>(
        device_id: i32,
        sources: impl IntoIterator<Item = &'a RestoreSource>,
    ) -> Result<Self, String> {
        if device_id < 0 {
            return Err("restore target device must be non-negative".into());
        }
        let mut plan = Self {
            device_id,
            ssd_path: None,
            allow_uring_fallback: false,
            target_bound: false,
        };
        for source in sources {
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
        }
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

    pub(crate) fn bind_target_shape(&mut self, bytes: u64, fragments: usize) -> Result<(), String> {
        if bytes == 0 || fragments == 0 {
            return Err("restore target shape requires non-empty target ranges".into());
        }
        if self.target_bound {
            return Err("restore target shape was already bound".into());
        }
        self.target_bound = true;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/restore.rs"]
mod tests;
