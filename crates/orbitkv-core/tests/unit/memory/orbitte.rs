use super::*;

use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use orbit_te::{EngineConfig, Permission, TransferEngine, TransferOp, TransferRequest, Transport};

const TIMEOUT: Duration = Duration::from_secs(5);

#[test]
#[ignore = "OrbitTE allocator hardware gate: requires a real CUDA runtime and GPU"]
fn pinned_aliases_transfer_bytes_and_release_the_actual_slot() {
    let allocator = PinnedAllocator::new_global(1024, 1, false, None);
    let source_allocation = allocator
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let destination_allocation = allocator
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let source_weak = Arc::downgrade(&source_allocation);
    let destination_weak = Arc::downgrade(&destination_allocation);
    let generation = source_allocation.id;
    let aliases = thread::scope(|scope| {
        let first = scope.spawn(|| unsafe { source_allocation.orbitte_memory().unwrap() });
        let second = scope.spawn(|| unsafe { source_allocation.orbitte_memory().unwrap() });
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert!(Arc::ptr_eq(&aliases[0], &aliases[1]));
    // No CPU/device access outside these canonical OrbitTE views is active.
    let destination_memory = unsafe { destination_allocation.orbitte_memory().unwrap() };
    let original = (0..512)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    aliases[0].write(0, &original).unwrap();
    destination_memory.write(0, &[0xee; 512]).unwrap();
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let requester = Arc::new(TransferEngine::new(EngineConfig::default()).unwrap());
    let registrations = source
        .register_host_memory_batch(
            &[
                (Arc::clone(&aliases[0]), 0, 512),
                (Arc::clone(&aliases[1]), 128, 128),
                (Arc::clone(&aliases[1]), 0, 512),
            ],
            Permission::RemoteReadWrite,
        )
        .unwrap();
    assert_eq!(registrations[0].id(), registrations[2].id());
    let destination = requester
        .register_host_memory(destination_memory.clone(), 0, 512, Permission::LocalOnly)
        .unwrap();
    let peer = requester
        .open_segment(source.endpoint(), source.source_epoch(), TIMEOUT)
        .unwrap();
    let requests = [
        TransferRequest {
            transport: Transport::Tcp,
            local_address: destination.base_address(),
            remote_address: registrations[0].base_address(),
            length: 256,
        },
        TransferRequest {
            transport: Transport::Tcp,
            local_address: destination.base_address() + 256,
            remote_address: registrations[0].base_address() + 256,
            length: 256,
        },
    ];
    let mut batch = requester
        .submit_async(TransferOp::Read, &peer, &requests, None, TIMEOUT)
        .unwrap();
    let status = batch.wait(TIMEOUT).unwrap();
    assert_eq!((status.completed_tasks, status.transferred_bytes), (2, 512));
    assert!(status.locally_drained);
    batch.free().unwrap();
    let mut bytes = [0; 512];
    destination.read(0, &mut bytes).unwrap();
    assert_eq!(bytes.as_slice(), original);
    destination.write(64, &[0x5a; 64]).unwrap();
    let write = requester
        .submit_and_wait(
            TransferOp::Write,
            &peer,
            &[TransferRequest {
                local_address: destination.base_address() + 64,
                remote_address: registrations[0].base_address() + 64,
                length: 64,
                transport: Transport::Tcp,
            }],
            TIMEOUT,
        )
        .unwrap();
    assert!(write.is_complete() && write.locally_drained);
    registrations[0].read(0, &mut bytes).unwrap();
    assert_eq!(&bytes[..64], &original[..64]);
    assert_eq!(&bytes[64..128], &[0x5a; 64]);
    assert_eq!(&bytes[128..], &original[128..]);

    for registration in &registrations {
        source.unregister_memory(registration).unwrap();
    }
    requester.unregister_memory(&destination).unwrap();
    drop((
        aliases,
        destination_memory,
        source_allocation,
        destination_allocation,
    ));
    assert!(source_weak.upgrade().is_some());
    assert!(destination_weak.upgrade().is_some());
    assert!(
        allocator
            .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
            .is_none()
    );
    drop((registrations, destination));
    requester.close_requester(TIMEOUT).unwrap();
    source.close_source(TIMEOUT).unwrap();
    assert!(source_weak.upgrade().is_none());
    assert!(destination_weak.upgrade().is_none());
    assert_eq!(allocator.usage(), (0, 1024));
    let reused = allocator
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    assert!(reused.id > generation);
}

#[test]
#[ignore = "OrbitTE allocator hardware gate: requires a real CUDA runtime and GPU"]
fn detached_transfer_retains_pinned_slot_after_cancel_unregister_and_wait_timeout() {
    let allocator = PinnedAllocator::new_global(512, 1, false, None);
    let allocation = allocator
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let weak_allocation = Arc::downgrade(&allocation);
    let weak_pool = Arc::downgrade(&allocation.pool);
    // The test owns this allocation exclusively outside the OrbitTE access lock.
    let memory = unsafe { allocation.orbitte_memory().unwrap() };
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let remote = source
        .register_memory(vec![0x7b; 512].into_boxed_slice(), Permission::RemoteRead)
        .unwrap();
    let requester = Arc::new(TransferEngine::new(EngineConfig::default()).unwrap());
    let local = requester
        .register_host_memory(memory.clone(), 0, 512, Permission::LocalOnly)
        .unwrap();
    let mut gate = GatedPeer::new(source.endpoint());
    let peer = requester
        .open_segment(gate.endpoint, source.source_epoch(), TIMEOUT)
        .unwrap();
    let mut batch = requester
        .submit_async(
            TransferOp::Read,
            &peer,
            &[TransferRequest {
                local_address: local.base_address(),
                remote_address: remote.base_address(),
                length: 512,
                transport: Transport::Tcp,
            }],
            None,
            TIMEOUT,
        )
        .unwrap();
    gate.started.recv_timeout(TIMEOUT).unwrap();
    assert!(matches!(
        batch.wait(Duration::from_millis(10)),
        Err(orbit_te::Error::Timeout)
    ));
    assert!(batch.cancel(0).unwrap());
    assert!(!batch.status().unwrap().locally_drained);
    assert!(batch.free().is_err());
    assert!(matches!(
        requester.close_requester(Duration::ZERO),
        Err(orbit_te::Error::Timeout)
    ));
    requester.unregister_memory(&local).unwrap();
    drop((batch, local, memory, allocation));
    assert!(weak_allocation.upgrade().is_some());
    assert_eq!(allocator.usage(), (512, 512));
    assert!(
        allocator
            .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
            .is_none()
    );
    drop(allocator);
    assert!(weak_pool.upgrade().is_some());
    gate.release();
    requester.close_requester(TIMEOUT).unwrap();
    assert!(weak_allocation.upgrade().is_none());
    assert!(weak_pool.upgrade().is_none());
    let stats = requester.stats().unwrap();
    assert_eq!(
        (
            stats.retained_bytes,
            stats.allocated_batches,
            stats.requester_workers
        ),
        (0, 0, 0)
    );
    source.unregister_memory(&remote).unwrap();
    source.close_source(TIMEOUT).unwrap();
}

/// Forwards the actual engine's metadata, then delays its data connection.
/// No OrbitTE wire format or transfer behavior is duplicated here.
struct GatedPeer {
    endpoint: SocketAddr,
    started: Receiver<()>,
    release: Option<SyncSender<()>>,
    worker: Option<JoinHandle<io::Result<()>>>,
}

impl GatedPeer {
    fn new(source: SocketAddr) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = listener.local_addr().unwrap();
        let (started_tx, started) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let metadata = accept(&listener)?;
            relay(metadata, source)?;
            let data = accept(&listener)?;
            started_tx.send(()).unwrap();
            released.recv_timeout(TIMEOUT).unwrap();
            relay(data, source)
        });
        Self {
            endpoint,
            started,
            release: Some(release),
            worker: Some(worker),
        }
    }

    fn release(&mut self) {
        self.release.take().unwrap().send(()).unwrap();
        self.worker.take().unwrap().join().unwrap().unwrap();
    }
}

impl Drop for GatedPeer {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn accept(listener: &TcpListener) -> io::Result<TcpStream> {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(error),
        }
    }
}

fn relay(mut client: TcpStream, endpoint: SocketAddr) -> io::Result<()> {
    let mut server = TcpStream::connect_timeout(&endpoint, TIMEOUT)?;
    for socket in [&client, &server] {
        socket.set_read_timeout(Some(TIMEOUT))?;
        socket.set_write_timeout(Some(TIMEOUT))?;
    }
    let mut client_reader = client.try_clone()?;
    let mut server_writer = server.try_clone()?;
    thread::scope(|scope| {
        let outbound = scope.spawn(move || {
            io::copy(&mut client_reader, &mut server_writer)?;
            server_writer.shutdown(Shutdown::Write)
        });
        io::copy(&mut server, &mut client)?;
        client.shutdown(Shutdown::Write)?;
        outbound.join().unwrap()
    })
}
