use super::*;

#[test]
fn tent_c_abi_layouts_match_the_pinned_header() {
    assert_eq!(std::mem::size_of::<TransferRequest>(), 88);
    assert_eq!(std::mem::align_of::<TransferRequest>(), 8);
    assert_eq!(
        std::mem::offset_of!(TransferRequest, expected_source_epoch),
        48
    );
    assert_eq!(std::mem::size_of::<TransferStatus>(), 16);
    assert_eq!(std::mem::size_of::<MemoryOptions>(), 344);
    assert_eq!(std::mem::size_of::<NotificationRecord>(), 4360);
    assert_eq!(std::mem::size_of::<NotificationInfo>(), 16);
    assert_eq!(std::mem::size_of::<NicLoadStat>(), 80);
    assert_eq!(INVALID_BATCH, 0);
    assert_eq!(INVALID_SEGMENT, u64::MAX);
}

#[test]
fn forced_tcp_temporarily_suppresses_general_tent_config() {
    let original_force = env::var_os("MC_FORCE_TCP");
    let original_config = env::var_os("MC_TENT_CONF");
    unsafe {
        env::set_var("MC_FORCE_TCP", "1");
        env::set_var("MC_TENT_CONF", "original");
    }
    let guard = SuspendedTentConfig::for_forced_tcp().unwrap();
    assert!(env::var_os("MC_TENT_CONF").is_none());
    drop(guard);
    assert_eq!(
        env::var_os("MC_TENT_CONF").as_deref(),
        Some(std::ffi::OsStr::new("original"))
    );
    unsafe {
        match original_force {
            Some(value) => env::set_var("MC_FORCE_TCP", value),
            None => env::remove_var("MC_FORCE_TCP"),
        }
        match original_config {
            Some(value) => env::set_var("MC_TENT_CONF", value),
            None => env::remove_var("MC_TENT_CONF"),
        }
    }
}
