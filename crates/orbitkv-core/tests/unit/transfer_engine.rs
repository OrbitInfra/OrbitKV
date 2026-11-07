use super::*;

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

#[test]
fn local_only_orbite_cannot_choose_the_tent_copy_implementation() {
    use crate::PeerTransferBackend::{Mooncake, Orbite, OrbiteRdma, OrbiteTcp};
    let tent: Arc<dyn TransferBackend> = Arc::new(crate::peer::restore::tests::Control::default());
    for selected in [Orbite, OrbiteTcp, OrbiteRdma] {
        for engine in [None, Some(&tent)] {
            let error = local::validate_engine(selected, engine).unwrap_err();
            assert_eq!(error.kind, ErrorKind::Unavailable);
        }
    }
    local::validate_engine(Mooncake, None).unwrap();
    local::validate_engine(Mooncake, Some(&tent)).unwrap();
}
