use super::*;

#[test]
fn removed_observation_opcode_and_old_abi_are_rejected() {
    assert!(CommandCode::try_from(8).is_err());
    let mut message = Command::ping(1, 2).encode();
    message[0] = (u64::from(MAGIC) << 32) | (12u64 << 16) | CommandCode::Ping as u64;
    assert!(Command::decode(message).is_err());
}

#[test]
fn command_round_trip_preserves_identity_guards() {
    let command = Command {
        code: CommandCode::Restore,
        request_id: 42,
        session_epoch: 7,
        descriptor: DescriptorRef {
            offset: 4096,
            len: 128,
            generation: 9,
        },
        arg0: 10,
        arg1: 11,
    };
    assert_eq!(Command::decode(command.encode()).unwrap(), command);
}

#[test]
fn decoder_rejects_wrong_magic_and_version() {
    let mut message = Command::ping(1, 2).encode();
    message[0] = 0;
    assert!(matches!(
        Command::decode(message),
        Err(ProtocolError::InvalidMagic(0))
    ));

    let mut message = Command::ping(1, 2).encode();
    message[0] = (u64::from(MAGIC) << 32) | (u64::from(ABI_VERSION + 1) << 16) | 1;
    assert!(matches!(
        Command::decode(message),
        Err(ProtocolError::UnsupportedVersion(_))
    ));
}
