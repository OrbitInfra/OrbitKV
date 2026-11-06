use super::*;

#[test]
fn rejects_incompatible_and_unbounded_frames_before_allocating() {
    let header = LifecycleHeader {
        code: 2,
        epoch: 42,
        payload_len: 512,
    };
    let mut bytes = header.encode().unwrap();
    assert_eq!(LifecycleHeader::decode(bytes).unwrap().epoch, 42);
    bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(LifecycleHeader::decode(bytes).is_err());
    bytes = header.encode().unwrap();
    for version in [10_u16, 99] {
        bytes[4..6].copy_from_slice(&version.to_le_bytes());
        assert!(LifecycleHeader::decode(bytes).is_err());
    }
}

#[test]
fn registration_receives_owned_fds_and_fragmented_stream_frames() {
    use std::io::Write;
    use std::os::fd::AsFd;
    use std::thread;

    let (client, mut server) = UnixStream::pair().unwrap();
    let original = tempfile::tempfile().unwrap();
    original.set_len(8192).unwrap();
    thread::scope(|scope| {
        scope.spawn(|| {
            send_lifecycle_fds(&server, &[original.as_fd()]).unwrap();
            let header = LifecycleHeader {
                code: 0,
                epoch: 91,
                payload_len: 7,
            }
            .encode()
            .unwrap();
            for byte in header {
                server.write_all(&[byte]).unwrap();
            }
            server.write_all(b"arena:1").unwrap();
        });
        let (header, reply) = receive_lifecycle_reply(&client, MAX_LIFECYCLE_PAYLOAD).unwrap();
        assert_eq!(header.epoch, 91);
        assert_eq!(reply.payload, b"arena:1");
        assert_eq!(reply.fds.len(), 1);
        let file = std::fs::File::from(reply.fds.into_iter().next().unwrap());
        assert_eq!(file.metadata().unwrap().len(), 8192);
        assert!(
            rustix::io::fcntl_getfd(&file)
                .unwrap()
                .contains(rustix::io::FdFlags::CLOEXEC)
        );
    });
}

#[test]
fn lifecycle_descriptor_marker_is_required_and_descriptor_count_is_bounded() {
    use std::io::Write;
    use std::os::fd::AsFd;
    let (client, mut server) = UnixStream::pair().unwrap();
    server.write_all(&[0]).unwrap();
    assert!(receive_lifecycle_reply(&client, MAX_LIFECYCLE_PAYLOAD).is_err());
    let file = tempfile::tempfile().unwrap();
    let fds = vec![file.as_fd(); MAX_LIFECYCLE_FDS + 1];
    assert!(send_lifecycle_fds(&server, &fds).is_err());
}
