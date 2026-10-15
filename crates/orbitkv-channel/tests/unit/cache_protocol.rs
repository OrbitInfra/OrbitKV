use super::*;

fn completion_observation() -> CompletionObservationRequest {
    CompletionObservationRequest {
        instance_id: "decode-instance".into(),
        destination_device_id: 3,
        source_endpoint: "tent://prefill-7".into(),
        transfer_generation: 11,
        intent: CompletionIntent::EngineRestore,
        route: CompletionRoute::PrefillToDecodeHandoff,
        representation: ReplicaRepresentation::Raw,
        logical_bytes: 32 * 1024,
        wire_bytes: 32 * 1024,
        fragment_count: 8,
        elapsed_ns: 400_000,
        decode_page_bytes: 32 * 1024,
        handoff_queue_depth: 3,
        handoff_queue_parallelism: 16,
        tent_inflight_bytes: 64 * 1024,
        tent_bandwidth_bytes_per_second: 20_000_000_000,
        admission: CompletionAdmission::Admitted,
        outcome: CompletionOutcome::Completed,
    }
}

#[test]
fn completion_observation_round_trip_is_bounded_and_rejects_malformed_frames() {
    let request = completion_observation();
    let bytes = request.encode().unwrap();
    assert_eq!(
        CompletionObservationRequest::decode(&bytes).unwrap(),
        request
    );
    assert_eq!(bytes.len(), COMPLETION_OBSERVATION_HEADER_BYTES + 15 + 16);
    for end in 0..bytes.len() {
        assert!(CompletionObservationRequest::decode(&bytes[..end]).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        CompletionObservationRequest::decode(&trailing),
        Err(CacheProtocolError::TrailingBytes(1))
    );

    let mut too_long = request.clone();
    too_long.instance_id = "x".repeat(MAX_COMPLETION_INSTANCE_ID_BYTES + 1);
    assert!(matches!(
        too_long.encode(),
        Err(CacheProtocolError::CompletionFieldTooLong {
            field: "completion_instance_id",
            ..
        })
    ));
    too_long = request;
    too_long.source_endpoint = "x".repeat(MAX_COMPLETION_SOURCE_ENDPOINT_BYTES + 1);
    assert!(matches!(
        too_long.encode(),
        Err(CacheProtocolError::CompletionFieldTooLong {
            field: "completion_source_endpoint",
            ..
        })
    ));
}

#[test]
fn completion_observation_rejects_invalid_evidence_combinations() {
    let mut request = completion_observation();
    request.transfer_generation = 0;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::ZeroCompletionField(
            "transfer_generation"
        ))
    );

    request = completion_observation();
    request.intent = CompletionIntent::HostReady;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::InvalidCompletionTarget)
    );

    request = completion_observation();
    request.admission = CompletionAdmission::Rejected;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::InvalidCompletionState)
    );

    request.outcome = CompletionOutcome::Failed;
    request.wire_bytes = 0;
    request.decode_page_bytes = 0;
    assert!(request.encode().is_ok());

    request = completion_observation();
    request.wire_bytes = 0;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::InvalidCompletionState)
    );

    request = completion_observation();
    request.decode_page_bytes = 0;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::InvalidCompletionResources)
    );

    request = completion_observation();
    request.handoff_queue_depth = 4097;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::CompletionQueueTooDeep(4097))
    );

    request = completion_observation();
    request.handoff_queue_parallelism = 0;
    assert_eq!(
        request.encode(),
        Err(CacheProtocolError::InvalidCompletionResources)
    );
}

#[test]
fn cancel_query_preserves_scope_and_rejects_malformed_frames() {
    let request = CancelQueryRequest {
        ticket: QueryTicket {
            operation_id: 7,
            revision: 2,
        },
    };
    let mut bytes = request.encode().unwrap();
    assert_eq!(CancelQueryRequest::decode(&bytes).unwrap(), request);
    for end in 0..bytes.len() {
        assert!(CancelQueryRequest::decode(&bytes[..end]).is_err());
    }
    bytes.push(0);
    assert!(CancelQueryRequest::decode(&bytes).is_err());
}

#[test]
fn request_round_trip_preserves_variable_hashes() {
    let request = QueryBundleRequest {
        ticket: QueryTicket {
            operation_id: 7,
            revision: 2,
        },
        instance_id: "model-a".to_string(),
        request_id: "request-7".to_string(),
        block_hashes: vec![vec![1; 32], vec![2; 17]],
        group_id: 3,
        wait_for_full_prefix: true,
        warmup: false,
        discover: false,
        materialize: false,
        prepare: false,
        demand: None,
    };
    assert_eq!(
        QueryBundleRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );
    let mut previous_schema = request.encode().unwrap();
    previous_schema[4..6].copy_from_slice(&(CACHE_PROTOCOL_VERSION - 1).to_le_bytes());
    assert_eq!(
        QueryBundleRequest::decode(&previous_schema),
        Err(CacheProtocolError::UnsupportedVersion(
            CACHE_PROTOCOL_VERSION - 1
        ))
    );
    for command in [
        QueryCommand::Submit(request.clone()),
        QueryCommand::Poll(request.ticket),
        QueryCommand::Claim {
            ticket: request.ticket,
            count_lookup: true,
        },
        QueryCommand::Claim {
            ticket: request.ticket,
            count_lookup: false,
        },
    ] {
        let bytes = command.encode().unwrap();
        assert_eq!(QueryCommand::decode(&bytes).unwrap(), command);
        for end in 0..bytes.len() {
            assert!(QueryCommand::decode(&bytes[..end]).is_err());
        }
    }
    for ticket in [
        QueryTicket {
            operation_id: 0,
            revision: 1,
        },
        QueryTicket {
            operation_id: 1,
            revision: 0,
        },
    ] {
        assert_eq!(
            QueryCommand::Poll(ticket).encode(),
            Err(CacheProtocolError::InvalidQueryTicket)
        );
    }
}

#[test]
fn selected_recovery_round_trip_preserves_all_group_ranges_and_rejects_bad_frames() {
    let span = TokenRange {
        start: 64,
        end: 128,
    };
    let request = QueryBundleRequest {
        ticket: QueryTicket {
            operation_id: 9,
            revision: 2,
        },
        instance_id: "registered-shard".into(),
        request_id: "recovery".into(),
        block_hashes: vec![vec![1; 32], vec![2; 32]],
        group_id: 1,
        wait_for_full_prefix: false,
        warmup: false,
        discover: false,
        materialize: true,
        prepare: true,
        demand: Some(RecoveryDemand {
            page_tokens: 16,
            span,
            groups: vec![
                (0, span),
                (
                    1,
                    TokenRange {
                        start: 96,
                        end: 128,
                    },
                ),
                (
                    2,
                    TokenRange {
                        start: 112,
                        end: 128,
                    },
                ),
            ],
        }),
    };
    let bytes = request.encode().unwrap();
    assert_eq!(QueryBundleRequest::decode(&bytes).unwrap(), request);
    for end in 0..bytes.len() {
        assert!(QueryBundleRequest::decode(&bytes[..end]).is_err());
    }
    let mut oversized = bytes.clone();
    let group_count_offset = bytes.len() - 3 * 20 - 4;
    oversized[group_count_offset..group_count_offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        QueryBundleRequest::decode(&oversized),
        Err(CacheProtocolError::Truncated)
    );
    let mut duplicate = bytes.clone();
    let last_group_offset = bytes.len() - 20;
    duplicate[last_group_offset..last_group_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        QueryBundleRequest::decode(&duplicate),
        Err(CacheProtocolError::Recovery(_))
    ));
    let mut invalid_range = bytes.clone();
    invalid_range[last_group_offset + 4..last_group_offset + 12]
        .copy_from_slice(&129u64.to_le_bytes());
    assert!(matches!(
        QueryBundleRequest::decode(&invalid_range),
        Err(CacheProtocolError::Recovery(_))
    ));
    let mut missing = request.clone();
    missing.demand = None;
    assert_eq!(
        missing.encode(),
        Err(CacheProtocolError::InvalidRecoveryDemand)
    );
    let mut unselected = request.clone();
    unselected.materialize = false;
    assert_eq!(
        unselected.encode(),
        Err(CacheProtocolError::InvalidRecoveryDemand)
    );
    let mut wrong_count = request;
    wrong_count.block_hashes.pop();
    assert!(matches!(
        wrong_count.encode(),
        Err(CacheProtocolError::Recovery(_))
    ));
}

#[test]
fn response_round_trip_preserves_lease_and_positions() {
    let response = QueryBundleResponse {
        outcome: QueryOutcomeCode::Ready,
        num_hit_blocks: 2,
        lease: vec![3; 16],
        hit_positions: vec![1, 4],
    };
    assert_eq!(
        QueryBundleResponse::decode(&response.encode().unwrap()).unwrap(),
        response
    );
}

#[test]
fn decoder_rejects_trailing_bytes_and_invalid_loading_payload() {
    let mut encoded = QueryBundleResponse::loading().encode().unwrap();
    encoded.push(0);
    assert_eq!(
        QueryBundleResponse::decode(&encoded),
        Err(CacheProtocolError::TrailingBytes(1))
    );

    let invalid = QueryBundleResponse {
        outcome: QueryOutcomeCode::Loading,
        num_hit_blocks: 1,
        lease: Vec::new(),
        hit_positions: Vec::new(),
    };
    assert_eq!(
        QueryBundleResponse::decode(&invalid.encode().unwrap()),
        Err(CacheProtocolError::InvalidLoadingPayload)
    );
}

#[test]
fn release_request_round_trip_and_empty_rejection() {
    let request = ReleaseRequest { lease: vec![7; 16] };
    assert_eq!(
        ReleaseRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );
    assert_eq!(
        ReleaseRequest::decode(&ReleaseRequest { lease: Vec::new() }.encode().unwrap()),
        Err(CacheProtocolError::EmptyLease)
    );
}

#[test]
fn publish_request_round_trip_and_shape_validation() {
    let request = PublishRequest {
        instance_id: "model-a".to_string(),
        registration_generation: [1; 16],
        tp_rank: 2,
        pp_rank: 1,
        device_id: 3,
        layers: vec![PublishLayer {
            layer_name: "layer.0".to_string(),
            block_ids: vec![4, 7],
            block_hashes: vec![vec![1; 32], vec![2; 32]],
        }],
    };
    assert_eq!(
        PublishRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );

    let invalid = PublishRequest {
        layers: vec![PublishLayer {
            layer_name: "bad".to_string(),
            block_ids: vec![1],
            block_hashes: Vec::new(),
        }],
        ..request
    };
    assert!(matches!(
        invalid.encode(),
        Err(CacheProtocolError::PublishShapeMismatch { .. })
    ));
}

#[test]
fn restore_request_round_trip() {
    let request = RestoreRequest {
        instance_id: "model-a".to_string(),
        tp_rank: 1,
        device_id: 2,
        layer_groups: vec![vec!["layer.0".to_string()], Vec::new()],
        loads: vec![RestoreLease {
            lease: vec![9; 16],
            block_ids_by_group: vec![vec![Some(3), None], vec![None, Some(7)]],
        }],
    };
    assert_eq!(
        RestoreRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );
}

#[test]
fn candidate_hints_cannot_carry_leases_or_ambiguous_positions() {
    for (lease, positions, count) in [
        (vec![1], vec![1, 2], 2),
        (vec![], vec![2, 1], 2),
        (vec![], vec![1, 1], 2),
        (vec![], vec![1, 2], 1),
    ] {
        let response = QueryBundleResponse {
            outcome: QueryOutcomeCode::Candidates,
            num_hit_blocks: count,
            lease,
            hit_positions: positions,
        };
        assert_eq!(
            QueryBundleResponse::decode(&response.encode().unwrap()),
            Err(CacheProtocolError::InvalidCandidates)
        );
    }
}

#[test]
fn encoded_size_rejects_overflow_without_changing_the_budget() {
    for (initial, count, width) in [
        (0, usize::MAX, 2),
        (usize::MAX, 1, 1),
        (isize::MAX as usize, 1, 1),
    ] {
        let mut size = initial;
        assert!(matches!(
            add_encoded_size(&mut size, count, width),
            Err(CacheProtocolError::FieldTooLarge {
                field: "payload",
                ..
            })
        ));
        assert_eq!(size, initial);
    }
}

#[test]
fn shard_response_checks_all_leases_and_control_before_decode_allocation() {
    let ready = ShardQueryResponse {
        outcome: QueryOutcomeCode::Ready,
        num_hit_blocks: 2,
        leases: vec![vec![1; 16], vec![2; 16]],
        control_id: vec![3; 16],
    };
    let encoded = ready.encode().unwrap();
    assert_eq!(ShardQueryResponse::decode(&encoded).unwrap(), ready);
    for case in 0..4 {
        let mut invalid = ready.clone();
        match case {
            0 => invalid.control_id.clear(),
            1 => invalid.leases[1].clear(),
            2 => invalid.leases = vec![vec![1; 16]; 33],
            _ => invalid.leases[0] = vec![0; 16],
        }
        assert!(invalid.encode().is_err());
    }
    let mut invalid = encoded.clone();
    invalid[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(ShardQueryResponse::decode(&invalid).is_err());
    let mut extra = encoded;
    extra.push(0);
    assert!(ShardQueryResponse::decode(&extra).is_err());
}

#[test]
fn publish_generation_is_required_and_preserved_in_every_wire_chunk() {
    let mut request = PublishRequest {
        instance_id: "a:b".into(),
        registration_generation: [7; 16],
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: vec![PublishLayer {
            layer_name: "layer".into(),
            block_ids: vec![0, 1],
            block_hashes: vec![vec![1; 8], vec![2; 8]],
        }],
    };
    let encoded = request.encode().unwrap();
    assert_eq!(PublishRequest::decode(&encoded).unwrap(), request);
    let chunk = request.encode_range(Some(1..2)).unwrap();
    assert_eq!(
        PublishRequest::decode(&chunk)
            .unwrap()
            .registration_generation,
        [7; 16]
    );
    let mut missing = encoded.clone();
    missing[28..44].fill(0);
    assert!(matches!(
        PublishRequest::decode(&missing),
        Err(CacheProtocolError::InvalidRegistrationGeneration)
    ));
    let mut legacy = encoded;
    legacy[4..6].copy_from_slice(&9u16.to_le_bytes());
    assert!(matches!(
        PublishRequest::decode(&legacy),
        Err(CacheProtocolError::UnsupportedVersion(9))
    ));
    request.registration_generation = [0; 16];
    assert!(matches!(
        request.encode(),
        Err(CacheProtocolError::InvalidRegistrationGeneration)
    ));
}
