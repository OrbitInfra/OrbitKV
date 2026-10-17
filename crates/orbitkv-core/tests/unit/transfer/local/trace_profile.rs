use super::*;

#[test]
fn cpu_timespec_rejects_invalid_or_overflowing_values() {
    for (seconds, nanos) in [(-1, 0), (0, -1), (0, 1_000_000_000), (i64::MAX, 0)] {
        assert!(
            timespec_ns(libc::timespec {
                tv_sec: seconds,
                tv_nsec: nanos
            })
            .is_err()
        );
    }
    assert_eq!(
        timespec_ns(libc::timespec {
            tv_sec: 2,
            tv_nsec: 3
        })
        .unwrap(),
        2_000_000_003
    );
}

#[test]
fn clock_windows_use_matched_brackets_and_preserve_negative_residuals() {
    let begin = ThreadStamp {
        wall_before_ns: 10,
        cpu_ns: 100,
        wall_after_ns: 20,
    };
    let end = ThreadStamp {
        wall_before_ns: 30,
        cpu_ns: 125,
        wall_after_ns: 40,
    };
    assert_eq!(
        begin.interval(end).unwrap(),
        json!({
            "cpu_ns": 25, "wall_lower_ns": 10, "wall_upper_ns": 30,
            "residual_lower_ns": -15, "residual_upper_ns": 5,
        })
    );
    for end in [
        ThreadStamp { cpu_ns: 99, ..end },
        ThreadStamp {
            wall_after_ns: 29,
            ..end
        },
        ThreadStamp {
            wall_before_ns: 19,
            ..end
        },
    ] {
        assert!(begin.interval(end).is_err());
    }
}

#[test]
fn thread_usage_is_checked_and_real_clock_reads_are_monotonic() {
    let epoch = Instant::now();
    let before = ThreadUsage::read().unwrap();
    let start = ThreadStamp::read(epoch).unwrap();
    let end = ThreadStamp::read(epoch).unwrap();
    let after = ThreadUsage::read().unwrap();
    assert!(thread_clock(true).unwrap() > 0);
    assert!(start.interval(end).is_ok());
    assert!(before.delta(after).is_ok());
    let synthetic = ThreadUsage {
        minor_faults: 3,
        major_faults: 2,
        voluntary_switches: 1,
        involuntary_switches: 4,
    };
    let smaller = ThreadUsage {
        minor_faults: 2,
        ..synthetic
    };
    assert!(synthetic.delta(smaller).is_err());
}
