use super::*;
use crate::endpoint::ProcessEndpoint;
use crate::metric::hll::MultiWindowHllTracker;
use crate::proto::engine::{RegisterContextRequest, SessionRequest};
use crate::registry::CudaTensorRegistry;
use orbitkv_channel::lifecycle::LifecycleCommand;
use orbitkv_channel::{CallOptions, ChannelClient, ChannelError, QueryBundleRequest};
use orbitkv_core::EngineConfig;
use prost::Message;
use std::time::Duration;
use tokio::sync::Notify;

#[test]
fn registration_requires_a_complete_stable_gpu_identity() {
    for (device_uuid, valid) in [
        ("GPU-c4930667-4d65-aae5-8205-fe785c3654be", true),
        ("", false),
        ("7", false),
        ("GPU-c4930667", false),
        ("MIG-c4930667-4d65-aae5-8205-fe785c3654be", false),
    ] {
        let request = crate::wire::registration(RegisterContextRequest {
            instance_id: "identity".into(),
            namespace: "identity".into(),
            client_version: env!("CARGO_PKG_VERSION").into(),
            device_uuid: device_uuid.into(),
            tp_size: 1,
            world_size: 1,
            ..Default::default()
        });
        assert_eq!(
            LifecycleService::validate_register_context_request(&request).is_ok(),
            valid,
            "UUID {device_uuid:?}"
        );
    }
}

pub(crate) fn test_engine() -> Arc<OrbitKVEngine> {
    Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, EngineConfig::default()).unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_channel_lifecycle_and_cache_control_need_no_grpc() {
    let engine = test_engine();
    let lifecycle = LifecycleService::new(
        Arc::clone(&engine),
        RegistryHandle::spawn(CudaTensorRegistry::empty()),
    );
    let shutdown = Arc::new(Notify::new());
    let id = uuid::Uuid::new_v4();
    let socket = std::env::temp_dir().join(format!("orbitkv-lifecycle-{id}.sock"));
    let mut endpoint = ProcessEndpoint::start(
        format!("orbitkv/test/lifecycle/{id}"),
        42,
        socket.clone(),
        1 << 20,
        1 << 16,
        engine,
        tokio::runtime::Handle::current(),
        Arc::new(std::sync::Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        ))),
        Arc::clone(&shutdown),
        lifecycle.clone(),
        0,
        None,
        usize::MAX,
        None,
    )
    .unwrap();
    let (first, second) = tokio::task::spawn_blocking(move || {
        let first = ChannelClient::connect(&socket, CallOptions::default()).unwrap();
        first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Register, &[0xff]),
            Err(ChannelError::Lifecycle { code: 1, .. })
        ));
        let bad_version = RegisterContextRequest {
            instance_id: "inst".into(),
            namespace: "ns".into(),
            client_version: "old".into(),
            ..Default::default()
        };
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Register, &bad_version.encode_to_vec()),
            Err(ChannelError::Lifecycle { code: 2, .. })
        ));
        first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
        assert!(
            first
                .query_bundle(
                    1,
                    &orbitkv_channel::QueryCommand::Submit(QueryBundleRequest {
                        ticket: orbitkv_channel::QueryTicket {
                            operation_id: 1,
                            revision: 1
                        },
                        instance_id: "missing".into(),
                        request_id: "query".into(),
                        block_hashes: vec![],
                        wait_for_full_prefix: true,
                        warmup: false,
                        discover: false,
                        materialize: false,
                        prepare: false,
                        demand: None,
                        group_id: 0,
                    })
                )
                .is_err()
        );
        let session = SessionRequest {
            instance_id: "inst".into(),
            namespace: "ns".into(),
            tp_size: 1,
            world_size: 1,
        }
        .encode_to_vec();
        first
            .lifecycle(LifecycleCommand::Session, &session)
            .unwrap();
        let incompatible = SessionRequest {
            instance_id: "inst".into(),
            namespace: "different-model".into(),
            tp_size: 1,
            world_size: 1,
        }
        .encode_to_vec();
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Session, &incompatible),
            Err(ChannelError::Lifecycle { code: 2, .. })
        ));
        let second = ChannelClient::connect(&socket, CallOptions::default()).unwrap();
        second
            .lifecycle(LifecycleCommand::Session, &session)
            .unwrap();
        (first, second)
    })
    .await
    .unwrap();
    // Old connection teardown must leave the replacement session intact.
    first.close();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(lifecycle.sessions.topology("inst").is_some());
    second.close();
    tokio::time::timeout(Duration::from_secs(2), async {
        while lifecycle.sessions.topology("inst").is_some() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("disconnect must clean up the current session");
    assert!(matches!(
        second.lifecycle(LifecycleCommand::Health, &[]),
        Err(ChannelError::SessionRequiresReconnect)
    ));
    endpoint.stop();
    shutdown.notify_waiters();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_shutdown_latches_while_an_accepted_request_finishes() {
    let engine = test_engine();
    let lifecycle = LifecycleService::new(
        Arc::clone(&engine),
        RegistryHandle::spawn(CudaTensorRegistry::empty()),
    );
    let shutdown = Arc::new(Notify::new());
    let id = uuid::Uuid::new_v4();
    let socket = std::env::temp_dir().join(format!("orbitkv-lifecycle-shutdown-{id}.sock"));
    let mut endpoint = ProcessEndpoint::start(
        format!("orbitkv/test/lifecycle-shutdown/{id}"),
        42,
        socket.clone(),
        1 << 20,
        1 << 16,
        engine,
        tokio::runtime::Handle::current(),
        Arc::new(std::sync::Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        ))),
        Arc::clone(&shutdown),
        lifecycle.clone(),
        0,
        None,
        usize::MAX,
        None,
    )
    .unwrap();
    let client = Arc::new(
        tokio::task::spawn_blocking({
            let socket = socket.clone();
            move || ChannelClient::connect(socket, CallOptions::default()).unwrap()
        })
        .await
        .unwrap(),
    );
    let session = SessionRequest {
        instance_id: "shutdown-inst".into(),
        namespace: "shutdown-ns".into(),
        tp_size: 1,
        world_size: 1,
    }
    .encode_to_vec();
    tokio::task::spawn_blocking({
        let client = Arc::clone(&client);
        move || client.lifecycle(LifecycleCommand::Session, &session)
    })
    .await
    .unwrap()
    .unwrap();
    assert!(lifecycle.sessions.topology("shutdown-inst").is_some());

    let (request_accepted, resume_request) =
        crate::endpoint::test_pause::install("lifecycle_after_request");
    let health = tokio::task::spawn_blocking({
        let client = Arc::clone(&client);
        move || client.lifecycle(LifecycleCommand::Health, &[])
    });
    tokio::time::timeout(Duration::from_secs(2), request_accepted)
        .await
        .expect("lifecycle request must reach its full-frame acceptance boundary")
        .expect("lifecycle request hook must remain connected");

    let late;
    {
        let drain = endpoint.stop_lifecycle_and_drain_connections();
        tokio::pin!(drain);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), drain.as_mut())
                .await
                .is_err(),
            "shutdown must preserve an accepted lifecycle operation"
        );
        resume_request.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), health)
            .await
            .expect("accepted lifecycle response must finish")
            .expect("lifecycle client task must not panic")
            .expect("accepted lifecycle request must receive its response");
        tokio::time::timeout(Duration::from_secs(2), drain.as_mut())
            .await
            .expect("latched shutdown must close an idle client after its response");
        assert_eq!(endpoint.active_lifecycle_connections(), 0);
        assert!(lifecycle.sessions.topology("shutdown-inst").is_none());

        let idle_result = tokio::task::spawn_blocking({
            let client = Arc::clone(&client);
            move || client.lifecycle(LifecycleCommand::Health, &[])
        })
        .await
        .unwrap();
        assert!(idle_result.is_err());

        late = Arc::new(
            tokio::task::spawn_blocking({
                let socket = socket.clone();
                move || ChannelClient::connect(socket, CallOptions::default()).unwrap()
            })
            .await
            .unwrap(),
        );
        let late_result = tokio::task::spawn_blocking({
            let late = Arc::clone(&late);
            move || late.lifecycle(LifecycleCommand::Health, &[])
        })
        .await
        .unwrap();
        assert!(late_result.is_err());
        assert_eq!(endpoint.active_lifecycle_connections(), 0);
    }

    client.close();
    late.close();
    endpoint.stop();
    shutdown.notify_waiters();
}
