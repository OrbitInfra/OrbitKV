use super::*;

/// Real CPU engine for owner tests; CUDA work still requires its explicit shim.
pub(crate) fn engine() -> Arc<TransferEngine> {
    Arc::new(
        TransferEngine::new(
            64 << 20,
            "127.0.0.1:0",
            orbite::Transport::Tcp,
            None,
            #[cfg(feature = "orbite-rdma")]
            None,
        )
        .unwrap(),
    )
}

pub(crate) fn demand() -> TransferOptions {
    TransferOptions::read(
        crate::QueryMode::Demand,
        Instant::now() + std::time::Duration::from_secs(30),
    )
}

#[test]
fn kv_intents_preserve_deadlines_and_assign_priority() {
    use crate::QueryMode;
    let deadline = demand().deadline;
    for (mode, intent, priority) in [
        (
            QueryMode::Demand,
            TransferIntent::ForegroundGet,
            TransferPriority::High,
        ),
        (
            QueryMode::WaitForFullPrefix,
            TransferIntent::ForegroundGet,
            TransferPriority::High,
        ),
        (
            QueryMode::Prepare,
            TransferIntent::BackgroundPrefetch,
            TransferPriority::Low,
        ),
        (
            QueryMode::Warmup,
            TransferIntent::BackgroundPrefetch,
            TransferPriority::Low,
        ),
    ] {
        let options = TransferOptions::read(mode, deadline);
        assert_eq!(
            (options.intent, options.intent.priority(), options.deadline),
            (intent, priority, deadline)
        );
    }
    let save = TransferOptions::checkpoint(deadline);
    assert_eq!(
        (save.intent, save.intent.priority()),
        (TransferIntent::Checkpoint, TransferPriority::Low)
    );
    assert_eq!(save.deadline, deadline);
}
