use super::*;

#[test]
fn runtime_requires_all_four_libraries_and_consumed_symbols() {
    let source = Path::new(WORKSPACE_RUNTIME_LIB_DIR);
    // SAFETY: source is the pinned built TENT bundle; the returned Api
    // owns every library for the duration of symbol inspection.
    let loaded = unsafe { load_api_from(source) };
    assert!(loaded.as_ref().unwrap().files.is_ok());
    let incomplete =
        env::temp_dir().join(format!("orbitkv-tent-three-libs-{}", std::process::id()));
    std::fs::create_dir(&incomplete).unwrap();
    for name in ["libtent_shared.so", "libmooncake_common.so", "libasio.so"] {
        std::fs::copy(source.join(name), incomplete.join(name)).unwrap();
    }
    // SAFETY: the directory contains copies of the trusted pinned bundle;
    // the absent fourth library causes loader rejection before symbol use.
    assert!(unsafe { load_api_from(&incomplete) }.is_err());
    std::fs::remove_dir_all(incomplete).unwrap();
}

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
    assert_eq!(std::mem::size_of::<files::FileOptions>(), 20);
    assert_eq!(std::mem::size_of::<files::FileElement>(), 24);
    assert_eq!(std::mem::align_of::<files::FileElement>(), 8);
    assert_eq!(std::mem::offset_of!(files::FileElement, address), 8);
    assert_eq!(std::mem::offset_of!(files::FileElement, length), 16);
    assert_eq!((files::RANGE, files::PADDING), (0, 1));
    assert_eq!(std::mem::size_of::<files::FileOwner>(), 24);
    assert_eq!(std::mem::size_of::<files::FileResult>(), 16);
    assert_eq!(
        std::mem::offset_of!(files::FileResult, transferred_bytes),
        8
    );
    assert_eq!(INVALID_BATCH, 0);
    assert_eq!(INVALID_SEGMENT, u64::MAX);
}

#[test]
fn forced_tcp_rejects_config_conflict_without_mutating_environment() {
    const CHILD: &str = "ORBITKV_TEST_CONFIG_CONFLICT";
    if env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::forced_tcp_rejects_config_conflict_without_mutating_environment",
            ])
            .env(CHILD, "1")
            .env("MC_FORCE_TCP", "1")
            .env("MC_TENT_CONF", "original")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    // SAFETY: every string is static and terminated; rejection precedes native
    // construction, and no environment value is changed by this call.
    let error = unsafe {
        create(
            c"P2PHANDSHAKE".as_ptr(),
            c"127.0.0.1:0".as_ptr(),
            c"127.0.0.1".as_ptr(),
            0,
            c"null",
        )
    }
    .unwrap_err();
    assert!(error.contains("unset MC_TENT_CONF"));
    assert_eq!(env::var("MC_TENT_CONF").unwrap(), "original");
}

#[test]
fn failed_terminal_query_preserves_unknown_effect() {
    ensure_loaded().unwrap();
    // SAFETY: a null engine is the C API's explicit invalid-handle control.
    let (status, code, known) = unsafe { terminal_result(std::ptr::null_mut(), 0, usize::MAX) };
    assert_eq!((status, code, known), (-1, 135, false));
}
