//! Submission limits are independent from the lifetime of already submitted I/O.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Debug)]
pub(crate) struct ReadControl {
    cancelled: AtomicBool,
    deadline: Instant,
    pub(crate) batch_bytes: u64,
    pub(crate) max_batches: usize,
}

impl ReadControl {
    pub(crate) fn new(deadline: Instant, batch_bytes: u64, max_batches: usize) -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            deadline,
            batch_bytes,
            max_batches,
        }
    }

    pub(crate) fn deadline(&self) -> Instant {
        self.deadline
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn can_submit(&self, submitted: usize) -> bool {
        !self.cancelled.load(Ordering::Acquire)
            && Instant::now() < self.deadline
            && submitted < self.max_batches
    }

    pub(crate) fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
}
