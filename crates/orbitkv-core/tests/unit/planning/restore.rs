use std::sync::Arc;

use super::*;
use crate::block::SealedBlock;

#[test]
fn memory_sources_bind_engine_target() {
    let block = RestoreSource::Memory(Arc::new(SealedBlock::from_slots(Vec::new())));
    let plan = RestorePlan::new(3, [&block, &block]).unwrap();
    assert_eq!(plan.device_id(), 3);
    assert_eq!(plan.ssd_path(), None);
    assert!(!plan.target_bound);
    assert!(RestorePlan::new(-1, [&block]).is_err());
}

#[test]
fn restore_target_shape_is_device_bound_nonempty_and_bound_once() {
    let mut plan = RestorePlan::new(3, std::iter::empty::<&crate::block::RestoreSource>()).unwrap();
    assert!(plan.bind_target_shape(0, 1).is_err());
    assert!(plan.bind_target_shape(4096, 0).is_err());
    plan.bind_target_shape(4096, 2).unwrap();
    assert!(plan.bind_target_shape(4096, 2).is_err());
}

#[test]
fn automatic_cufile_can_fall_back_but_explicit_route_cannot() {
    let mut automatic = RestorePlan {
        device_id: 0,
        ssd_path: Some(SsdReadPath::Cufile),
        allow_uring_fallback: true,
        target_bound: false,
    };
    assert_eq!(automatic.fallback_from_cufile(), Ok(true));
    assert_eq!(automatic.ssd_path(), Some(SsdReadPath::Uring));
    assert_eq!(automatic.fallback_from_cufile(), Ok(false));

    let mut explicit = RestorePlan {
        device_id: 0,
        ssd_path: Some(SsdReadPath::Cufile),
        allow_uring_fallback: false,
        target_bound: false,
    };
    assert!(explicit.fallback_from_cufile().is_err());
    assert_eq!(explicit.ssd_path(), Some(SsdReadPath::Cufile));
}
