use super::*;
use crate::peer::completion::{TransferCompletions, tests::Fixture};
use crate::peer::segment::{PeerIdentity, PeerSegments};
use orbitkv_catalog::MembershipView;
use orbitkv_state::CacheOwner;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;

const WAIT: Duration = Duration::from_secs(15);

fn requester(capacity: usize) -> Arc<TransferEngine> {
    let mut backend = TransferEngine::new(
        4096,
        "127.0.0.1:0",
        Transport::Tcp,
        None,
        #[cfg(feature = "orbite-rdma")]
        None,
    )
    .unwrap();
    // A small native identity table makes leaked descriptor handles observable
    // in a short churn control. The real TCP and retirement owners are unchanged.
    let mut config = EngineConfig::default();
    config.max_peer_segments = capacity;
    config.max_failover_attempts = 0;
    config.enable_auto_failover_on_poll = false;
    backend.engine = Arc::new(NativeEngine::new(config).unwrap());
    backend.endpoint = format!("orbite+tcp://{}", backend.engine.endpoint());
    backend.epoch = backend.engine.source_epoch().to_string();
    Arc::new(backend)
}

fn owner(endpoint: &str) -> CacheOwner {
    CacheOwner {
        endpoint: endpoint.into(),
        incarnation: uuid::Uuid::new_v4(),
    }
}

fn view(peers: &[CacheOwner]) -> Arc<MembershipView> {
    let view = Arc::new(MembershipView::new(owner("local")));
    assert!(view.renew(Instant::now(), WAIT * 4));
    install(&view, 1, peers);
    view
}

fn install(view: &MembershipView, epoch: u64, peers: &[CacheOwner]) {
    view.replace_members(
        std::iter::once(("local".into(), 1, view.owner().clone())).chain(
            peers
                .iter()
                .map(|peer| (peer.incarnation.to_string(), epoch, peer.clone())),
        ),
    );
}

fn identity(
    cache: &PeerSegments,
    view: &MembershipView,
    owner: &CacheOwner,
    endpoint: &str,
    epoch: &str,
) -> Arc<PeerIdentity> {
    cache
        .get(view.lease(owner).unwrap(), endpoint, epoch)
        .unwrap()
}

async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(WAIT, async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("peer retirement made no progress");
}

fn read(peer: &Arc<PeerIdentity>, local: &RegisteredMemory, remote: &RegisteredMemory) {
    let options = crate::transfer_engine::TransferOptions::read(
        crate::QueryMode::Demand,
        Instant::now() + WAIT,
    );
    let op = peer.open(options.deadline).unwrap();
    let slices = [TransferSlice {
        local: std::ptr::NonNull::new(local.base_address() as *mut u8).unwrap(),
        remote_address: remote.base_address(),
        length: local.len(),
    }];
    // SAFETY: real host registrations and source bytes remain owned through drain.
    assert_eq!(
        // SAFETY: the source export and destination registration remain owned
        // through this transfer and its retirement wait; host reads follow drain.
        unsafe {
            op.transfer(
                crate::transfer_engine::TransferOp::Read,
                &slices,
                options,
                &Cancellation::default(),
            )
        }
        .unwrap(),
        local.len()
    );
}

/// A bounded transparent relay; it does not parse or synthesize engine frames.
/// Holding a response drives the production TCP drain owner, not a fake backend.
struct Relay {
    endpoint: String,
    target: Arc<parking_lot::RwLock<SocketAddr>>,
    held: Arc<AtomicBool>,
    arrived: Arc<Semaphore>,
    release: Arc<Semaphore>,
    task: tokio::task::JoinHandle<()>,
}

impl Relay {
    async fn new(target: SocketAddr) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("orbite+tcp://{}", listener.local_addr().unwrap());
        let target = Arc::new(parking_lot::RwLock::new(target));
        let held = Arc::new(AtomicBool::new(false));
        let arrived = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let state = (
            target.clone(),
            held.clone(),
            arrived.clone(),
            release.clone(),
        );
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    _ = connections.join_next(), if !connections.is_empty() => {},
                    accepted = listener.accept(), if connections.len() < 16 => {
                        let (client, _) = accepted.unwrap();
                        let target = *state.0.read();
                        let hold = state.1.load(Ordering::Acquire);
                        let (arrived, release) = (state.2.clone(), state.3.clone());
                        connections.spawn(async move {
                            let upstream = tokio::net::TcpStream::connect(target).await.unwrap();
                            let (mut cr, mut cw) = client.into_split();
                            let (mut ur, mut uw) = upstream.into_split();
                            let _ = tokio::try_join!(
                                async {
                                    tokio::io::copy(&mut cr, &mut uw).await?;
                                    uw.shutdown().await
                                },
                                async {
                                    let mut first = [0; 4096];
                                    let n = ur.read(&mut first).await?;
                                    if hold {
                                        arrived.add_permits(1);
                                        release.acquire().await.unwrap().forget();
                                    }
                                    cw.write_all(&first[..n]).await?;
                                    tokio::io::copy(&mut ur, &mut cw).await?;
                                    cw.shutdown().await
                                },
                            );
                        });
                    }
                }
            }
        });
        Self {
            endpoint,
            target,
            held,
            arrived,
            release,
            task,
        }
    }
}
impl Drop for Relay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn membership_removal_drains_each_requester_and_keeps_restore_owners() {
    let authority = Fixture::new().await;
    let completions = TransferCompletions::default();
    let (grant, _) = completions
        .authorize(
            &authority.segment(),
            uuid::Uuid::new_v4(),
            crate::transfer_engine::tests::demand(),
        )
        .await
        .unwrap();
    let source = requester(8);
    let unrelated = requester(8);
    let remote = source
        .engine
        .register_memory(vec![0x59; 4096].into(), Permission::RemoteRead)
        .unwrap();
    let other_remote = unrelated
        .engine
        .register_memory(vec![0x87; 4096].into(), Permission::RemoteRead)
        .unwrap();
    let relay = Relay::new(source.engine.endpoint()).await;
    let other_owner = owner("unrelated");
    let view = view(&[authority.owner.clone(), other_owner.clone()]);
    let backends = [requester(8), requester(8)];
    let caches = backends
        .each_ref()
        .map(|b| PeerSegments::new(b.clone(), view.clone()));
    // Both requester engines open both identities of this member. A handle
    // obtained on one requester is never borrowed by the other requester.
    let peers = caches
        .each_ref()
        .map(|c| identity(c, &view, &authority.owner, &relay.endpoint, &source.epoch));
    let direct = caches
        .each_ref()
        .map(|c| identity(c, &view, &authority.owner, &source.endpoint, &source.epoch));
    for peer in peers.iter().chain(&direct) {
        drop(peer.open(Instant::now() + WAIT).unwrap());
    }
    let other = identity(
        &caches[0],
        &view,
        &other_owner,
        &unrelated.endpoint,
        &unrelated.epoch,
    );
    let local = Arc::new(
        backends[0]
            .engine
            .register_memory(vec![0xcc; 4096].into(), Permission::LocalOnly)
            .unwrap(),
    );
    let buffer_owner = Arc::downgrade(&local);
    let remote_address = remote.base_address();
    let op = peers[0].open(Instant::now() + WAIT).unwrap();
    relay.held.store(true, Ordering::Release);
    let bytes = local.clone();
    let result = tokio::spawn(async move {
        grant
            .run_with_buffers(local, move || {
                let slice = TransferSlice {
                    local: std::ptr::NonNull::new(bytes.base_address() as *mut u8).unwrap(),
                    remote_address,
                    length: bytes.len(),
                };
                // SAFETY: actual source registration and grant, destination registration
                // and query buffers all stay owned until this blocking operation drains.
                let result = unsafe {
                    op.transfer(
                        crate::transfer_engine::TransferOp::Read,
                        &[slice],
                        crate::transfer_engine::TransferOptions::read(
                            crate::QueryMode::Demand,
                            Instant::now() + WAIT,
                        ),
                        &Cancellation::default(),
                    )
                };
                if result.is_ok() {
                    let mut got = vec![0; bytes.len()];
                    bytes.read(0, &mut got).unwrap();
                    assert_eq!(got, vec![0x59; got.len()]);
                }
                result.map_err(PeerTransferError::from)
            })
            .await
            .unwrap()
    });
    tokio::time::timeout(WAIT, relay.arrived.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert!(backends[0].engine.stats().unwrap().active_peer_operations > 0);
    install(&view, 1, std::slice::from_ref(&other_owner));
    for peer in &peers {
        assert!(peer.open(Instant::now() + WAIT).is_err());
    }
    until(|| {
        backends[0].engine.stats().unwrap().unquiesced_peer_segments == 1
            && backends[1].engine.stats().unwrap().peer_segments == 0
    })
    .await;
    assert!(buffer_owner.upgrade().is_some());
    assert_eq!(authority.pins_and_releases(), (1, 0));
    assert!(
        !result.is_finished(),
        "retirement timeout cannot complete the restore"
    );
    let unaffected = backends[0]
        .engine
        .register_memory(vec![0; 4096].into(), Permission::LocalOnly)
        .unwrap();
    read(&other, &unaffected, &other_remote);
    let mut got = vec![0; 4096];
    unaffected.read(0, &mut got).unwrap();
    assert_eq!(got, vec![0x87; 4096]);
    relay.held.store(false, Ordering::Release);
    relay.release.add_permits(16);
    let (buffers, transferred) = result.await.unwrap();
    assert_eq!(transferred.unwrap(), 4096);
    let local = buffers.unwrap();
    backends[0].engine.unregister_memory(&local).unwrap();
    drop(local);
    until(|| {
        authority.pins_and_releases() == (0, 1)
            && backends[0].engine.stats().unwrap().peer_segments == 1
    })
    .await;
    assert!(buffer_owner.upgrade().is_none());
    view.fence();
    until(|| {
        backends
            .iter()
            .all(|b| b.engine.stats().unwrap().peer_segments == 0)
    })
    .await;
    backends[0].engine.unregister_memory(&unaffected).unwrap();
    source.engine.unregister_memory(&remote).unwrap();
    unrelated.engine.unregister_memory(&other_remote).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn epoch_replacement_reopens_and_churn_releases_native_descriptor_handles() {
    let old = requester(2);
    let new = requester(2);
    let dst = requester(2);
    let owner = owner("peer");
    let view = view(std::slice::from_ref(&owner));
    let cache = PeerSegments::new(dst.clone(), view.clone());
    let relay = Relay::new(old.engine.endpoint()).await;
    let stale = identity(&cache, &view, &owner, &relay.endpoint, &old.epoch);
    drop(stale.open(Instant::now() + WAIT).unwrap());
    *relay.target.write() = new.engine.endpoint();
    install(&view, 2, std::slice::from_ref(&owner));
    assert!(stale.open(Instant::now() + WAIT).is_err());
    let fresh = identity(&cache, &view, &owner, &relay.endpoint, &new.epoch);
    drop(fresh.open(Instant::now() + WAIT).unwrap());
    until(|| dst.engine.stats().unwrap().peer_segments == 1).await;
    assert!(
        matches!(dst.open_peer(&relay.endpoint, &old.epoch, Instant::now() + WAIT), Err(error) if error.kind == ErrorKind::EpochMismatch)
    );
    let backing = Arc::new(orbite::HostMemory::new(vec![0x41; 64].into()).unwrap());
    let remote = new
        .engine
        .register_host_memory(backing.clone(), 0, 64, Permission::RemoteRead)
        .unwrap();
    let local = dst
        .engine
        .register_memory(vec![0; 64].into(), Permission::LocalOnly)
        .unwrap();
    read(&fresh, &local, &remote);
    let mut got = [0; 64];
    local.read(0, &mut got).unwrap();
    assert_eq!(got, [0x41; 64]);
    let address = remote.base_address();
    new.engine.unregister_memory(&remote).unwrap();
    backing.write(0, &[0x63; 64]).unwrap();
    let remote = new
        .engine
        .register_host_memory(backing, 0, 64, Permission::RemoteRead)
        .unwrap();
    assert_eq!(remote.base_address(), address);
    read(&fresh, &local, &remote);
    local.read(0, &mut got).unwrap();
    assert_eq!(
        got, [0x63; 64],
        "cached identity refreshes re-registered source keys"
    );
    // Keep old query identities alive. Native descriptors must be dropped by
    // retirement itself, otherwise this engine's two-slot table exhausts.
    let mut grants = vec![stale, fresh];
    for epoch in 3..19 {
        install(&view, epoch, std::slice::from_ref(&owner));
        until(|| dst.engine.stats().unwrap().peer_segments == 0).await;
        let next = identity(&cache, &view, &owner, &relay.endpoint, &new.epoch);
        read(&next, &local, &remote);
        grants.push(next);
    }
    assert!(dst.engine.stats().unwrap().reclaimed_peer_segments >= 17);
    dst.engine.unregister_memory(&local).unwrap();
    new.engine.unregister_memory(&remote).unwrap();
    dst.shutdown(Instant::now() + WAIT).unwrap();
    view.fence();
    until(|| dst.engine.stats().unwrap().peer_segments == 0).await;
    drop(grants);
}
