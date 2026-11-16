use super::*;

impl Expiration {
    pub(crate) fn expire_for_test(&self) {
        self.expired.store(true, Ordering::Release);
    }
}

#[test]
fn fixed_deadline_survives_rollback_and_forward_jump() {
    let start = Instant::now();
    let record = CacheExpiry {
        published_unix_millis: 100,
        expires_unix_millis: 200,
    };
    let expiry = Expiration::at(record, 100, start).unwrap();
    assert!(!expiry.expired_at(Ok(199), start + Duration::from_millis(99)));
    assert!(expiry.expired_at(Ok(150), start + Duration::from_millis(100)));
    assert!(expiry.expired_at(Ok(100), start));

    let expiry = Expiration::at(record, 100, start).unwrap();
    assert!(expiry.expired_at(Ok(200), start));
    assert!(expiry.expired_at(Ok(150), start));
    let expiry = Expiration::at(record, 100, start).unwrap();
    assert!(expiry.expired_at(Ok(99), start));
    assert!(expiry.expired_at(Ok(150), start));
}

#[test]
fn reconstructed_lifetimes_use_only_the_remaining_time() {
    let start = Instant::now();
    let record = CacheExpiry {
        published_unix_millis: 100,
        expires_unix_millis: 200,
    };
    let expiry = Expiration::at(record, 190, start).unwrap();
    assert_eq!(expiry.record, record);
    assert!(!expiry.expired_at(Ok(190), start + Duration::from_millis(9)));
    assert!(expiry.expired_at(Ok(190), start + Duration::from_millis(10)));
    for now in [0, 99, 200, 201] {
        assert!(
            Expiration::at(record, now, start)
                .unwrap()
                .expired_at(Ok(now), start)
        );
    }
    assert!(
        Expiration::at(record, 150, start)
            .unwrap()
            .expired_at(Err("clock".into()), start)
    );
}

#[test]
fn invalid_lifetimes_fail_without_enabling_a_default_ttl() {
    assert!(Expiration::new(None).unwrap().is_none());
    assert!(Expiration::new(Some(Duration::ZERO)).is_err());
    assert!(Expiration::new(Some(Duration::from_nanos(1))).is_err());
    assert!(Expiration::new(Some(Duration::MAX)).is_err());
    for (published, expires) in [(0, 1), (100, 100), (100, 99)] {
        let record = CacheExpiry {
            published_unix_millis: published,
            expires_unix_millis: expires,
        };
        assert!(Expiration::at(record, 100, Instant::now()).is_err());
    }
}

#[test]
fn partial_lifetimes_keep_the_owner_or_reject_conflicting_clock_order() {
    let wall = unix_millis().unwrap();
    let mono = Instant::now();
    let make = |expires, observed| {
        Arc::new(
            Expiration::at(
                CacheExpiry {
                    published_unix_millis: wall - 60_000,
                    expires_unix_millis: wall + expires,
                },
                observed,
                mono,
            )
            .unwrap(),
        )
    };
    for equal_wall in [false, true] {
        for reverse in [false, true] {
            let early = make(if equal_wall { 60_000 } else { 30_000 }, wall);
            let late = make(60_000, if equal_wall { wall - 10_000 } else { wall });
            let merged = if reverse {
                late.merge(early.clone())
            } else {
                early.clone().merge(late)
            };
            assert!(Arc::ptr_eq(&merged, &early));
            assert!(!merged.is_expired());
        }
    }
    let old = make(60_000, wall);
    let next = make(30_000, wall - 40_000);
    assert!(!old.is_expired());
    assert!(!next.is_expired());
    let merged = old.clone().merge(next);
    assert!(Arc::ptr_eq(&merged, &old));
    assert!(merged.is_expired());
    assert!(old.is_expired());
}
