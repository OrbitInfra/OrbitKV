use super::*;
use crate::http_server::router::{Credentials, routes};
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn namespace(c: char) -> String {
    format!("orbitkv:v2:{}", c.to_string().repeat(64))
}

async fn http(runtime: InventoryRuntime) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    let entries: Vec<_> = ['a', 'b']
        .into_iter()
        .map(|c| {
            json!({
                "namespace": namespace(c),
                "token_sha256": format!("{:x}", Sha256::digest(c.to_string().repeat(64))),
            })
        })
        .collect();
    std::fs::write(file.path(), serde_json::to_vec(&entries).unwrap()).unwrap();
    let app = routes(
        Credentials::load(file.path(), address, &InventoryScope::AllNamespaces).unwrap(),
        runtime,
    );
    (
        address,
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
    )
}

async fn post(address: SocketAddr, path: &str, token: char, body: Value, expected: u16) -> Value {
    let json = body.to_string();
    let header = format!(
        "POST /v1/router/{path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAuthorization: Bearer {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        token.to_string().repeat(64),
        json.len()
    );
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream.write_all(header.as_bytes()).await.unwrap();
    stream.write_all(json.as_bytes()).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(5), stream.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(
        headers.starts_with(&format!("HTTP/1.1 {expected} ")),
        "{response}"
    );
    if expected == 200 {
        serde_json::from_str(body).unwrap()
    } else {
        Value::String(body.into())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; starts a real etcd process and HTTP/gRPC listeners"]
async fn http_router_recovers_scoped_events_after_gaps_and_owner_restart() {
    let etcd = Etcd::start(1).await;
    let port = free_port();
    let source_view = view(port);
    let (source, _, inventory) = join(&etcd, "router", "source", source_view.clone(), 60).await;
    let runtime = source.inventory();
    let grpc = serve(runtime.clone(), port).await;
    let other = scoped_node(
        &etcd,
        "router",
        "other",
        free_port(),
        InventoryScope::AllNamespaces,
    )
    .await;
    wait_for(|| {
        source_view.permits(other.cluster.view.owner())
            && other.cluster.view.permits(source_view.owner())
    })
    .await;
    let (addr, http_source) = http(runtime.clone()).await;
    let (peer, http_peer) = http(other.runtime.clone()).await;
    let query = json!({"hashes": [[9], [10]]});
    let key = StateKey::new(namespace('a'), vec![9]);
    let hidden = StateKey::new(namespace('b'), vec![99]);
    let medium = ReplicaMedium::Dram;
    inventory
        .test_change(&key, medium, Some(metadata(medium)))
        .unwrap();
    other
        .runtime
        .await_fence(
            &runtime.capture_fence().unwrap(),
            &other.index.scope().digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

    post(addr, "lookup", 'c', query.clone(), 401).await;
    post(
        addr,
        "lookup",
        'a',
        json!({"hashes":[[9]], "namespace":namespace('b')}),
        422,
    )
    .await;
    post(
        addr,
        "lookup",
        'a',
        json!({"hashes":vec![vec![0]; 129]}),
        400,
    )
    .await;
    let baseline = post(addr, "lookup", 'a', query.clone(), 200).await;
    assert_eq!(
        baseline["source"]["incarnation"],
        source_view.owner().incarnation.to_string()
    );
    assert_eq!(baseline["local"][0]["hash"], json!([9]));
    let remote = post(peer, "lookup", 'a', query.clone(), 200).await;
    assert_eq!(
        remote["placements"][0]["peer_coverage"],
        "complete_at_watermarks"
    );
    assert_eq!(
        remote["placements"][0]["peers"][0]["owner"],
        baseline["source"]
    );
    assert_eq!(remote["placements"][1]["peers"], json!([]));
    assert_eq!(remote["local"], json!([]));
    let initial = json!({"cursor":baseline["cursor"]});
    post(peer, "events", 'a', initial.clone(), 409).await;
    post(addr, "events", 'b', initial.clone(), 409).await;
    inventory
        .test_change(&hidden, medium, Some(metadata(medium)))
        .unwrap();
    let filtered = post(addr, "events", 'a', initial.clone(), 200).await;
    assert_eq!(filtered["changes"], json!([]));
    assert!(filtered["cursor"]["sequence"].as_u64() > baseline["cursor"]["sequence"].as_u64());

    inventory.test_change(&key, medium, None).unwrap();
    let ssd = ReplicaMedium::Ssd;
    inventory
        .test_change(&key, ssd, Some(metadata(ssd)))
        .unwrap();
    let replay = post(addr, "events", 'a', initial.clone(), 200).await;
    assert_eq!(
        post(addr, "events", 'a', initial.clone(), 200).await,
        replay
    );
    let changes = replay["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 2);
    assert!(changes.iter().all(|r| r["hash"] == json!([9])));
    assert!(
        changes
            .iter()
            .any(|r| r["metadata"]["medium"] == "Dram" && r["present"] == false)
    );
    assert!(
        changes
            .iter()
            .any(|r| r["metadata"]["medium"] == "Ssd" && r["present"] == true)
    );
    let idle = post(
        addr,
        "events",
        'a',
        json!({"cursor":replay["cursor"], "wait_ms":1}),
        200,
    )
    .await;
    assert_eq!(idle["cursor"], replay["cursor"]);

    for id in 0u32..300 {
        inventory
            .test_change(
                &StateKey::new(namespace('b'), id.to_le_bytes().to_vec()),
                medium,
                Some(metadata(medium)),
            )
            .unwrap();
    }
    assert_eq!(
        post(addr, "events", 'a', initial, 409).await,
        "resync_required"
    );
    let fresh = post(addr, "lookup", 'a', query.clone(), 200).await;
    assert_eq!(fresh["local"].as_array().unwrap().len(), 1);
    assert_eq!(fresh["local"][0]["metadata"]["medium"], "Ssd");

    grpc.abort();
    grpc.await.unwrap_err();
    source.shutdown().await;
    post(addr, "lookup", 'a', query.clone(), 503).await;
    http_source.abort();
    let replacement = scoped_node(
        &etcd,
        "router",
        "source",
        port,
        InventoryScope::AllNamespaces,
    )
    .await;
    let (addr, http_rejoined) = http(replacement.runtime.clone()).await;
    post(addr, "events", 'a', json!({"cursor":fresh["cursor"]}), 409).await;
    let empty = post(addr, "lookup", 'a', query, 200).await;
    assert_ne!(
        empty["source"]["incarnation"],
        fresh["source"]["incarnation"]
    );
    assert_eq!(empty["local"], json!([]));
    http_peer.abort();
    http_rejoined.abort();
    replacement.server.abort();
    replacement.cluster.shutdown().await;
    other.server.abort();
    other.cluster.shutdown().await;
}
