use std::ptr::NonNull;

pub(crate) const INVALID_BATCH: u64 = 0;
pub(crate) const STATUS_WAITING: i32 = 0;
pub(crate) const STATUS_PENDING: i32 = 1;
pub(crate) const STATUS_INVALID: i32 = 2;
pub(crate) const STATUS_CANCELED: i32 = 3;
pub(crate) const STATUS_COMPLETED: i32 = 4;
pub(crate) const STATUS_TIMEOUT: i32 = 5;
pub(crate) const STATUS_FAILED: i32 = 6;

pub const P2P_METADATA: &str = "P2PHANDSHAKE";
pub const AUTO_MEMORY_LOCATION: &str = "*";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

impl TransferOp {
    pub(crate) const fn code(self) -> i32 {
        match self {
            Self::Read => 0,
            Self::Write => 1,
        }
    }
}

/// TENT's C ABI priority values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TransferPriority {
    High = 0,
    Low = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct TransferSlice {
    pub priority: TransferPriority,
    pub local: NonNull<u8>,
    pub remote_address: u64,
    pub length: usize,
}

unsafe impl Send for TransferSlice {}
unsafe impl Sync for TransferSlice {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub name: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NicLoadStat {
    pub device_name: String,
    pub inflight_bytes: u64,
    pub ewma_bandwidth_bps: f64,
}
