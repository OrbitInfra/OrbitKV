use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use orbitkv_channel::{CallOptions, ChannelClient, ChannelError, PublishRequest, StatusCode};
use orbitkv_core::{EngineConfig, OrbitKVEngine};
use tokio::sync::Notify;

use super::super::{ProcessEndpoint, test_pause};
use crate::cache::lifecycle::LifecycleService;
use crate::metric::hll::MultiWindowHllTracker;
use crate::registry::{CudaTensorRegistry, RegistryHandle};

fn start_endpoint() -> (ProcessEndpoint, PathBuf, Arc<Notify>) {
    let engine =
        Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, EngineConfig::default()).unwrap());
    let lifecycle = LifecycleService::new(
        Arc::clone(&engine),
        RegistryHandle::spawn(CudaTensorRegistry::empty()),
    );
    let shutdown = Arc::new(Notify::new());
    let id = uuid::Uuid::new_v4();
    let socket = std::env::temp_dir().join(format!("orbitkv-admission-{id}.sock"));
    let endpoint = ProcessEndpoint::start(
        format!("orbitkv/test/admission/{id}"),
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
        lifecycle,
        0,
        None,
        usize::MAX,
        None,
    )
    .unwrap();
    (endpoint, socket, shutdown)
}

fn empty_publish(instance_id: &str) -> PublishRequest {
    PublishRequest {
        instance_id: instance_id.into(),
        registration_generation: [1; 16],
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: Vec::new(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_fences_prechecked_publish_before_active_registration() {
    let (mut endpoint, socket, shutdown) = start_endpoint();
    let first = Arc::new(
        tokio::task::spawn_blocking({
            let socket = socket.clone();
            move || ChannelClient::connect(socket, CallOptions::default()).unwrap()
        })
        .await
        .unwrap(),
    );
    let second = Arc::new(
        tokio::task::spawn_blocking({
            let socket = socket.clone();
            move || ChannelClient::connect(socket, CallOptions::default()).unwrap()
        })
        .await
        .unwrap(),
    );

    let (first_admitted, finish_first) = test_pause::install("publish_after_admission");
    let first_call = tokio::task::spawn_blocking({
        let client = Arc::clone(&first);
        move || client.publish(1, &empty_publish("missing-first"))
    });
    tokio::time::timeout(Duration::from_secs(2), first_admitted)
        .await
        .expect("first Publish must reach the accepted side of the fence")
        .expect("first Publish admission hook must remain connected");

    let (second_prechecked, resume_second) = test_pause::install("publish_before_admission");
    let second_call = tokio::task::spawn_blocking({
        let client = Arc::clone(&second);
        move || client.publish(2, &empty_publish("missing-second"))
    });
    tokio::time::timeout(Duration::from_secs(2), second_prechecked)
        .await
        .expect("second Publish must pause after the control-loop check")
        .expect("second Publish admission hook must remain connected");

    {
        let drain = endpoint.stop_admission_and_drain_publishes();
        tokio::pin!(drain);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), drain.as_mut())
                .await
                .is_err(),
            "shutdown must wait for work accepted before the close boundary"
        );

        finish_first.send(()).unwrap();
        let first_result = tokio::time::timeout(Duration::from_secs(2), first_call)
            .await
            .expect("accepted Publish must finish")
            .expect("accepted Publish task must not panic");
        assert!(matches!(
            first_result,
            Err(ChannelError::Status(StatusCode::Invalid))
        ));
        tokio::time::timeout(Duration::from_secs(2), drain.as_mut())
            .await
            .expect("shutdown must drain the accepted Publish");
        assert_eq!(endpoint.active_publishes.active_count(), 0);

        resume_second.send(()).unwrap();
        let second_result = tokio::time::timeout(Duration::from_secs(2), second_call)
            .await
            .expect("prechecked Publish must receive a shutdown rejection")
            .expect("prechecked Publish task must not panic");
        assert!(matches!(
            second_result,
            Err(ChannelError::Status(StatusCode::StaleSession))
        ));
        assert_eq!(endpoint.active_publishes.active_count(), 0);
        assert!(!endpoint.active_publishes.is_accepting());
    }

    first.close();
    second.close();
    endpoint.stop();
    shutdown.notify_waiters();
}
