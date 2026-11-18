use super::*;
use axum::body::to_bytes;
use axum::http::HeaderValue;
use serde_json::json;

#[test]
fn credentials_require_one_admitted_namespace_per_fingerprint() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let address = "127.0.0.1:9000".parse().unwrap();
    let namespace = format!("orbitkv:v2:{}", "1".repeat(64));
    let token = "a".repeat(64);
    let entry =
        json!({"namespace":namespace,"token_sha256":format!("{:x}", Sha256::digest(&token))});
    std::fs::write(file.path(), json!([entry]).to_string()).unwrap();
    let credentials =
        Credentials::load(file.path(), address, &InventoryScope::AllNamespaces).unwrap();
    let mut headers = HeaderMap::new();
    assert!(credentials.authenticate(&headers).is_none());
    let bearer = HeaderValue::from_str(&format!("Bearer {token}")).unwrap();
    headers.insert(header::AUTHORIZATION, bearer.clone());
    assert_eq!(
        credentials
            .authenticate(&headers)
            .unwrap()
            .namespaces()
            .unwrap(),
        [namespace]
    );
    headers.append(header::AUTHORIZATION, bearer);
    assert!(credentials.authenticate(&headers).is_none());
    assert!(
        Credentials::load(
            file.path(),
            "0.0.0.0:9000".parse().unwrap(),
            &InventoryScope::AllNamespaces
        )
        .is_err()
    );
    assert!(Credentials::load(file.path(), address, &InventoryScope::exact([]).unwrap()).is_err());
    std::fs::write(file.path(), json!([entry, entry]).to_string()).unwrap();
    assert!(Credentials::load(file.path(), address, &InventoryScope::AllNamespaces).is_err());
    std::fs::write(file.path(), vec![b' '; CONFIG_BYTES + 1]).unwrap();
    assert!(Credentials::load(file.path(), address, &InventoryScope::AllNamespaces).is_err());
}

#[tokio::test]
async fn encoded_body_holds_capacity_until_its_final_reader_drops_it() {
    let slots = Arc::new(Semaphore::new(1));
    let slot = slots.clone().try_acquire_owned().unwrap();
    let response = reply(&vec![7u8; 100], slot).unwrap();
    assert_eq!(slots.available_permits(), 0);
    let bytes = to_bytes(response.into_body(), RESPONSE_BYTES)
        .await
        .unwrap();
    let retained = bytes.clone();
    drop(bytes);
    assert_eq!(slots.available_permits(), 0);
    drop(retained);
    assert_eq!(slots.available_permits(), 1);
    let slot = slots.clone().try_acquire_owned().unwrap();
    assert!(reply(&"x".repeat(RESPONSE_BYTES + 1), slot).is_err());
    assert_eq!(slots.available_permits(), 1);
}
