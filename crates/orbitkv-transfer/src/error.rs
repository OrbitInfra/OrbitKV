use thiserror::Error;

#[derive(Debug, Error)]
pub enum MooncakeError {
    #[error("Mooncake native runtime unavailable: {0}")]
    NativeRuntime(String),
    #[error("invalid string: {0}")]
    InvalidString(#[from] std::ffi::NulError),
    #[error("Mooncake TENT creation failed")]
    Create,
    #[error("Mooncake operation {operation} failed with status {status}")]
    Operation {
        operation: &'static str,
        status: i32,
    },
    #[error("Mooncake returned an invalid batch id")]
    InvalidBatch,
    #[error("Mooncake native batch admission exhausted or descriptor bytes overflow")]
    BatchAdmissionExhausted,
    #[error("Mooncake transfer task {task} failed with state {state}")]
    TransferFailed { task: usize, state: i32 },
    #[error("Mooncake transfer batch timed out")]
    Timeout,
    #[error("Mooncake notification buffer has invalid size {0}")]
    InvalidNotificationCount(i32),
    #[error("Mooncake returned a null notification buffer for {0} messages")]
    InvalidNotificationBuffer(i32),
    #[error("invalid Mooncake notification wait")]
    InvalidNotificationWait,
    #[error("Mooncake notification wait for {0} timed out")]
    NotificationTimeout(String),
}

pub type Result<T> = std::result::Result<T, MooncakeError>;
