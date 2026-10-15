use std::ops::Range;

use orbitkv_state::{RecoveryDemand, ReplicaRepresentation, TokenRange};
use thiserror::Error;

const QUERY_REQUEST_MAGIC: u32 = 0x4f52_5151; // ORQQ
const QUERY_RESPONSE_MAGIC: u32 = 0x4f52_5152; // ORQR
const QUERY_POLL_MAGIC: u32 = 0x4f52_5150; // ORQP
const CANCEL_QUERY_MAGIC: u32 = 0x4f52_5143; // ORQC
const RELEASE_REQUEST_MAGIC: u32 = 0x4f52_4c51; // ORLQ
const PUBLISH_REQUEST_MAGIC: u32 = 0x4f52_5051; // ORPQ
const RESTORE_REQUEST_MAGIC: u32 = 0x4f52_5251; // ORRQ
const COMPLETION_OBSERVATION_MAGIC: u32 = 0x4f52_434f; // ORCO
const CACHE_PROTOCOL_VERSION: u16 = 10;
const REQUEST_HEADER_BYTES: usize = 40;
const RESPONSE_HEADER_BYTES: usize = 24;
const RELEASE_HEADER_BYTES: usize = 12;
const PUBLISH_HEADER_BYTES: usize = 44;
const RESTORE_HEADER_BYTES: usize = 28;
const COMPLETION_OBSERVATION_HEADER_BYTES: usize = 104;
const MAX_COMPLETION_INSTANCE_ID_BYTES: usize = 256;
const MAX_COMPLETION_SOURCE_ENDPOINT_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CompletionIntent {
    HostReady = 1,
    EngineRestore = 2,
    SourceRelease = 3,
}

impl TryFrom<u16> for CompletionIntent {
    type Error = CacheProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HostReady),
            2 => Ok(Self::EngineRestore),
            3 => Ok(Self::SourceRelease),
            _ => Err(CacheProtocolError::UnknownCompletionIntent(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CompletionRoute {
    PrefillToDecodeHandoff = 1,
}

impl TryFrom<u16> for CompletionRoute {
    type Error = CacheProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::PrefillToDecodeHandoff),
            _ => Err(CacheProtocolError::UnknownCompletionRoute(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CompletionAdmission {
    Admitted = 1,
    Rejected = 2,
}

impl TryFrom<u16> for CompletionAdmission {
    type Error = CacheProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Admitted),
            2 => Ok(Self::Rejected),
            _ => Err(CacheProtocolError::UnknownCompletionAdmission(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CompletionOutcome {
    Completed = 1,
    Failed = 2,
    Cancelled = 3,
    TimedOut = 4,
}

impl TryFrom<u16> for CompletionOutcome {
    type Error = CacheProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Completed),
            2 => Ok(Self::Failed),
            3 => Ok(Self::Cancelled),
            4 => Ok(Self::TimedOut),
            _ => Err(CacheProtocolError::UnknownCompletionOutcome(value)),
        }
    }
}

/// Bounded evidence for one physical operation observed by an engine process.
///
/// Request identifiers and cache keys are deliberately absent: they are not
/// execution-resource identities and must never become cost dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionObservationRequest {
    pub instance_id: String,
    pub destination_device_id: i32,
    pub source_endpoint: String,
    pub transfer_generation: u64,
    pub intent: CompletionIntent,
    pub route: CompletionRoute,
    pub representation: ReplicaRepresentation,
    pub logical_bytes: u64,
    pub wire_bytes: u64,
    pub fragment_count: u32,
    pub elapsed_ns: u64,
    pub decode_page_bytes: u64,
    pub handoff_queue_depth: u32,
    pub handoff_queue_parallelism: u32,
    pub tent_inflight_bytes: u64,
    pub tent_bandwidth_bytes_per_second: u64,
    pub admission: CompletionAdmission,
    pub outcome: CompletionOutcome,
}

impl CompletionObservationRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        self.validate()?;
        let instance_id = self.instance_id.as_bytes();
        let source_endpoint = self.source_endpoint.as_bytes();
        let mut bytes = Vec::with_capacity(
            COMPLETION_OBSERVATION_HEADER_BYTES + instance_id.len() + source_endpoint.len(),
        );
        push_u32(&mut bytes, COMPLETION_OBSERVATION_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, self.intent as u16);
        push_u16(&mut bytes, self.route as u16);
        push_u16(&mut bytes, representation_code(self.representation));
        push_u16(&mut bytes, self.admission as u16);
        push_u16(&mut bytes, self.outcome as u16);
        push_u16(&mut bytes, 0);
        push_i32(&mut bytes, self.destination_device_id);
        push_u32(
            &mut bytes,
            checked_u32(instance_id.len(), "completion_instance_id")?,
        );
        push_u32(
            &mut bytes,
            checked_u32(source_endpoint.len(), "completion_source_endpoint")?,
        );
        push_u64(&mut bytes, self.transfer_generation);
        push_u64(&mut bytes, self.logical_bytes);
        push_u64(&mut bytes, self.wire_bytes);
        push_u32(&mut bytes, self.fragment_count);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, self.elapsed_ns);
        push_u64(&mut bytes, self.decode_page_bytes);
        push_u32(&mut bytes, self.handoff_queue_depth);
        push_u32(&mut bytes, self.handoff_queue_parallelism);
        push_u64(&mut bytes, self.tent_inflight_bytes);
        push_u64(&mut bytes, self.tent_bandwidth_bytes_per_second);
        bytes.extend_from_slice(instance_id);
        bytes.extend_from_slice(source_endpoint);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(COMPLETION_OBSERVATION_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let intent = CompletionIntent::try_from(decoder.u16()?)?;
        let route = CompletionRoute::try_from(decoder.u16()?)?;
        let representation = decode_representation(decoder.u16()?)?;
        let admission = CompletionAdmission::try_from(decoder.u16()?)?;
        let outcome = CompletionOutcome::try_from(decoder.u16()?)?;
        let reserved = decoder.u16()?;
        if reserved != 0 {
            return Err(CacheProtocolError::InvalidCompletionReserved(
                reserved as u32,
            ));
        }
        let destination_device_id = decoder.i32()?;
        let instance_len = decoder.usize_u32()?;
        let source_len = decoder.usize_u32()?;
        let transfer_generation = decoder.u64()?;
        let logical_bytes = decoder.u64()?;
        let wire_bytes = decoder.u64()?;
        let fragment_count = decoder.u32()?;
        let reserved = decoder.u32()?;
        if reserved != 0 {
            return Err(CacheProtocolError::InvalidCompletionReserved(reserved));
        }
        let elapsed_ns = decoder.u64()?;
        let decode_page_bytes = decoder.u64()?;
        let handoff_queue_depth = decoder.u32()?;
        let handoff_queue_parallelism = decoder.u32()?;
        let tent_inflight_bytes = decoder.u64()?;
        let tent_bandwidth_bytes_per_second = decoder.u64()?;
        let instance_id = decoder.string(instance_len, "completion_instance_id")?;
        let source_endpoint = decoder.string(source_len, "completion_source_endpoint")?;
        decoder.finish()?;
        let request = Self {
            instance_id,
            destination_device_id,
            source_endpoint,
            transfer_generation,
            intent,
            route,
            representation,
            logical_bytes,
            wire_bytes,
            fragment_count,
            elapsed_ns,
            decode_page_bytes,
            handoff_queue_depth,
            handoff_queue_parallelism,
            tent_inflight_bytes,
            tent_bandwidth_bytes_per_second,
            admission,
            outcome,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), CacheProtocolError> {
        validate_completion_string(
            &self.instance_id,
            "completion_instance_id",
            MAX_COMPLETION_INSTANCE_ID_BYTES,
        )?;
        validate_completion_string(
            &self.source_endpoint,
            "completion_source_endpoint",
            MAX_COMPLETION_SOURCE_ENDPOINT_BYTES,
        )?;
        if self.destination_device_id < 0 {
            return Err(CacheProtocolError::InvalidCompletionDevice(
                self.destination_device_id,
            ));
        }
        if self.transfer_generation == 0 {
            return Err(CacheProtocolError::ZeroCompletionField(
                "transfer_generation",
            ));
        }
        if self.logical_bytes == 0 {
            return Err(CacheProtocolError::ZeroCompletionField("logical_bytes"));
        }
        if self.fragment_count == 0 {
            return Err(CacheProtocolError::ZeroCompletionField("fragment_count"));
        }
        if self.elapsed_ns == 0 {
            return Err(CacheProtocolError::ZeroCompletionField("elapsed_ns"));
        }
        if self.handoff_queue_depth > 4096 {
            return Err(CacheProtocolError::CompletionQueueTooDeep(
                self.handoff_queue_depth,
            ));
        }
        if self.handoff_queue_parallelism > 4096 {
            return Err(CacheProtocolError::CompletionQueueTooWide(
                self.handoff_queue_parallelism,
            ));
        }
        if self.representation == ReplicaRepresentation::Unknown {
            return Err(CacheProtocolError::UnknownCompletionRepresentation(0));
        }
        if self.route == CompletionRoute::PrefillToDecodeHandoff
            && self.intent != CompletionIntent::EngineRestore
        {
            return Err(CacheProtocolError::InvalidCompletionTarget);
        }
        match (self.admission, self.outcome, self.wire_bytes) {
            (CompletionAdmission::Admitted, CompletionOutcome::Completed, 0) => {
                Err(CacheProtocolError::InvalidCompletionState)
            }
            (CompletionAdmission::Rejected, CompletionOutcome::Completed, _)
            | (CompletionAdmission::Rejected, _, 1..) => {
                Err(CacheProtocolError::InvalidCompletionState)
            }
            _ => Ok(()),
        }?;
        match self.admission {
            CompletionAdmission::Admitted
                if self.decode_page_bytes != self.logical_bytes
                    || self.handoff_queue_depth == 0
                    || self.handoff_queue_parallelism == 0 =>
            {
                Err(CacheProtocolError::InvalidCompletionResources)
            }
            CompletionAdmission::Rejected if self.decode_page_bytes != 0 => {
                Err(CacheProtocolError::InvalidCompletionResources)
            }
            _ => Ok(()),
        }
    }
}

fn validate_completion_string(
    value: &str,
    field: &'static str,
    max_bytes: usize,
) -> Result<(), CacheProtocolError> {
    if value.is_empty() {
        return Err(CacheProtocolError::EmptyCompletionField(field));
    }
    if value.len() > max_bytes {
        return Err(CacheProtocolError::CompletionFieldTooLong {
            field,
            len: value.len(),
            max: max_bytes,
        });
    }
    Ok(())
}

fn representation_code(representation: ReplicaRepresentation) -> u16 {
    match representation {
        ReplicaRepresentation::Unknown => 0,
        ReplicaRepresentation::Raw => 1,
        ReplicaRepresentation::Ans => 2,
        ReplicaRepresentation::Fp8 => 3,
        ReplicaRepresentation::TurboQuant => 4,
        ReplicaRepresentation::Mixed => 5,
    }
}

fn decode_representation(value: u16) -> Result<ReplicaRepresentation, CacheProtocolError> {
    match value {
        1 => Ok(ReplicaRepresentation::Raw),
        2 => Ok(ReplicaRepresentation::Ans),
        3 => Ok(ReplicaRepresentation::Fp8),
        4 => Ok(ReplicaRepresentation::TurboQuant),
        5 => Ok(ReplicaRepresentation::Mixed),
        _ => Err(CacheProtocolError::UnknownCompletionRepresentation(value)),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CancelQueryRequest {
    pub ticket: QueryTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryTicket {
    pub operation_id: u64,
    pub revision: u64,
}

impl QueryTicket {
    fn encode_into(self, bytes: &mut Vec<u8>) -> Result<(), CacheProtocolError> {
        if self.operation_id == 0 || self.revision == 0 {
            return Err(CacheProtocolError::InvalidQueryTicket);
        }
        push_u64(bytes, self.operation_id);
        push_u64(bytes, self.revision);
        Ok(())
    }

    fn decode_from(decoder: &mut Decoder<'_>) -> Result<Self, CacheProtocolError> {
        let ticket = Self {
            operation_id: decoder.u64()?,
            revision: decoder.u64()?,
        };
        if ticket.operation_id == 0 || ticket.revision == 0 {
            return Err(CacheProtocolError::InvalidQueryTicket);
        }
        Ok(ticket)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryCommand {
    Submit(QueryBundleRequest),
    Poll(QueryTicket),
    Claim {
        ticket: QueryTicket,
        count_lookup: bool,
    },
}

impl QueryCommand {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        match self {
            Self::Submit(request) => request.encode(),
            Self::Poll(ticket) | Self::Claim { ticket, .. } => {
                let mut bytes = Vec::with_capacity(24);
                push_u32(&mut bytes, QUERY_POLL_MAGIC);
                push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
                let flags = match self {
                    Self::Claim { count_lookup, .. } => 1 | (u16::from(*count_lookup) << 1),
                    _ => 0,
                };
                push_u16(&mut bytes, flags);
                ticket.encode_into(&mut bytes)?;
                Ok(bytes)
            }
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        let magic = decoder.u32()?;
        if magic == QUERY_REQUEST_MAGIC {
            return QueryBundleRequest::decode(bytes).map(Self::Submit);
        }
        if magic != QUERY_POLL_MAGIC {
            return Err(CacheProtocolError::InvalidMagic(magic));
        }
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags > 3 || flags == 2 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        decoder.finish()?;
        Ok(if flags & 1 != 0 {
            Self::Claim {
                ticket,
                count_lookup: flags & 2 != 0,
            }
        } else {
            Self::Poll(ticket)
        })
    }
}

impl CancelQueryRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        let mut bytes = Vec::with_capacity(24);
        push_u32(&mut bytes, CANCEL_QUERY_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, 0);
        self.ticket.encode_into(&mut bytes)?;
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(CANCEL_QUERY_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        decoder.finish()?;
        Ok(Self { ticket })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreLease {
    pub lease: Vec<u8>,
    pub block_ids_by_group: Vec<Vec<Option<u32>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreRequest {
    pub instance_id: String,
    pub tp_rank: u32,
    pub device_id: i32,
    pub layer_groups: Vec<Vec<String>>,
    pub loads: Vec<RestoreLease>,
}

impl RestoreRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        let instance = self.instance_id.as_bytes();
        let mut size = RESTORE_HEADER_BYTES;
        add_encoded_size(&mut size, instance.len(), 1)?;
        for group in &self.layer_groups {
            add_encoded_size(&mut size, 1, 4)?;
            for layer in group {
                add_encoded_size(&mut size, 1, 4)?;
                add_encoded_size(&mut size, layer.len(), 1)?;
            }
        }
        for load in &self.loads {
            add_encoded_size(&mut size, 1, 8)?;
            add_encoded_size(&mut size, load.lease.len(), 1)?;
            for targets in &load.block_ids_by_group {
                add_encoded_size(&mut size, 1, 4)?;
                add_encoded_size(&mut size, targets.len(), 4)?;
            }
        }
        let mut bytes = Vec::with_capacity(size);
        push_u32(&mut bytes, RESTORE_REQUEST_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, self.tp_rank);
        push_i32(&mut bytes, self.device_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        push_u32(
            &mut bytes,
            checked_u32(self.layer_groups.len(), "layer_groups")?,
        );
        push_u32(&mut bytes, checked_u32(self.loads.len(), "loads")?);
        bytes.extend_from_slice(instance);
        for group in &self.layer_groups {
            push_u32(&mut bytes, checked_u32(group.len(), "layer_group")?);
            for layer in group {
                push_bytes(&mut bytes, layer.as_bytes(), "layer_name")?;
            }
        }
        for load in &self.loads {
            push_bytes(&mut bytes, &load.lease, "lease")?;
            push_u32(
                &mut bytes,
                checked_u32(load.block_ids_by_group.len(), "block_ids_by_group")?,
            );
            for targets in &load.block_ids_by_group {
                push_u32(&mut bytes, checked_u32(targets.len(), "block_targets")?);
                for target in targets {
                    push_u32(&mut bytes, target.map_or(u32::MAX, |block_id| block_id));
                }
            }
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RESTORE_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let tp_rank = decoder.u32()?;
        let device_id = decoder.i32()?;
        let instance_len = decoder.usize_u32()?;
        let group_count = decoder.usize_u32()?;
        let load_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        if group_count > decoder.remaining() / 4 {
            return Err(CacheProtocolError::Truncated);
        }
        let mut layer_groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let layer_count = decoder.usize_u32()?;
            if layer_count > decoder.remaining() / 4 {
                return Err(CacheProtocolError::Truncated);
            }
            let mut group = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let len = decoder.usize_u32()?;
                group.push(decoder.string(len, "layer_name")?);
            }
            layer_groups.push(group);
        }
        if load_count > decoder.remaining() / 8 {
            return Err(CacheProtocolError::Truncated);
        }
        let mut loads = Vec::with_capacity(load_count);
        for _ in 0..load_count {
            let lease_len = decoder.usize_u32()?;
            let lease = decoder.bytes(lease_len)?.to_vec();
            if lease.is_empty() {
                return Err(CacheProtocolError::EmptyLease);
            }
            let target_group_count = decoder.usize_u32()?;
            if target_group_count > decoder.remaining() / 4 {
                return Err(CacheProtocolError::Truncated);
            }
            let mut block_ids_by_group = Vec::with_capacity(target_group_count);
            for _ in 0..target_group_count {
                let target_count = decoder.usize_u32()?;
                if target_count > decoder.remaining() / 4 {
                    return Err(CacheProtocolError::Truncated);
                }
                let mut targets = Vec::with_capacity(target_count);
                for _ in 0..target_count {
                    let target = decoder.u32()?;
                    targets.push((target != u32::MAX).then_some(target));
                }
                block_ids_by_group.push(targets);
            }
            loads.push(RestoreLease {
                lease,
                block_ids_by_group,
            });
        }
        decoder.finish()?;
        Ok(Self {
            instance_id,
            tp_rank,
            device_id,
            layer_groups,
            loads,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RestoreState {
    Pending = 1,
    Succeeded = 2,
    Failed = 3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreResponse {
    pub operation_id: u64,
    pub state: RestoreState,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishLayer {
    pub layer_name: String,
    pub block_ids: Vec<u32>,
    pub block_hashes: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishRequest {
    pub instance_id: String,
    pub registration_generation: [u8; 16],
    pub tp_rank: u32,
    pub pp_rank: u32,
    pub device_id: i32,
    pub layers: Vec<PublishLayer>,
}

impl PublishRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        self.encode_range(None)
    }

    pub(crate) fn encoded_len(
        &self,
        range: Option<Range<usize>>,
    ) -> Result<usize, CacheProtocolError> {
        if self.registration_generation == [0; 16] {
            return Err(CacheProtocolError::InvalidRegistrationGeneration);
        }
        validate_publish_layers(&self.layers)?;
        checked_u32(self.instance_id.len(), "instance_id")?;
        checked_u32(self.layers.len(), "layers")?;
        let mut size = PUBLISH_HEADER_BYTES;
        add_encoded_size(&mut size, self.instance_id.len(), 1)?;
        for layer in &self.layers {
            let selected = match &range {
                Some(range) => {
                    let end = range.end.min(layer.block_ids.len());
                    if range.start >= end {
                        continue;
                    }
                    range.start..end
                }
                None => 0..layer.block_ids.len(),
            };
            checked_u32(layer.layer_name.len(), "layer_name")?;
            checked_u32(selected.len(), "block_ids")?;
            add_encoded_size(&mut size, 1, 8)?;
            add_encoded_size(&mut size, layer.layer_name.len(), 1)?;
            add_encoded_size(&mut size, selected.len(), 8)?;
            for hash in &layer.block_hashes[selected] {
                checked_u32(hash.len(), "block_hash")?;
                add_encoded_size(&mut size, hash.len(), 1)?;
            }
        }
        Ok(size)
    }

    pub(crate) fn encode_range(
        &self,
        range: Option<Range<usize>>,
    ) -> Result<Vec<u8>, CacheProtocolError> {
        let size = self.encoded_len(range.clone())?;
        let instance = self.instance_id.as_bytes();
        let mut bytes = Vec::with_capacity(size);
        push_u32(&mut bytes, PUBLISH_REQUEST_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, self.tp_rank);
        push_u32(&mut bytes, self.pp_rank);
        push_i32(&mut bytes, self.device_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        let layer_count = self
            .layers
            .iter()
            .filter(|layer| {
                range
                    .as_ref()
                    .is_none_or(|range| range.start < range.end.min(layer.block_ids.len()))
            })
            .count();
        push_u32(&mut bytes, checked_u32(layer_count, "layers")?);
        bytes.extend_from_slice(&self.registration_generation);
        bytes.extend_from_slice(instance);
        for layer in &self.layers {
            let selected = match &range {
                Some(range) => {
                    let end = range.end.min(layer.block_ids.len());
                    if range.start >= end {
                        continue;
                    }
                    range.start..end
                }
                None => 0..layer.block_ids.len(),
            };
            let name = layer.layer_name.as_bytes();
            push_u32(&mut bytes, checked_u32(name.len(), "layer_name")?);
            push_u32(&mut bytes, checked_u32(selected.len(), "block_ids")?);
            bytes.extend_from_slice(name);
            for (block_id, hash) in layer.block_ids[selected.clone()]
                .iter()
                .zip(&layer.block_hashes[selected])
            {
                push_u32(&mut bytes, *block_id);
                push_u32(&mut bytes, checked_u32(hash.len(), "block_hash")?);
                bytes.extend_from_slice(hash);
            }
        }
        Ok(bytes)
    }

    pub(crate) fn block_payload_len(&self, index: usize) -> Result<usize, CacheProtocolError> {
        let mut size = 0;
        for layer in &self.layers {
            if let Some(hash) = layer.block_hashes.get(index) {
                add_encoded_size(&mut size, 1, 8)?;
                add_encoded_size(&mut size, hash.len(), 1)?;
            }
        }
        Ok(size)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(PUBLISH_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let tp_rank = decoder.u32()?;
        let pp_rank = decoder.u32()?;
        let device_id = decoder.i32()?;
        let instance_len = decoder.usize_u32()?;
        let layer_count = decoder.usize_u32()?;
        let registration_generation: [u8; 16] = decoder
            .bytes(16)?
            .try_into()
            .map_err(|_| CacheProtocolError::Truncated)?;
        if registration_generation == [0; 16] {
            return Err(CacheProtocolError::InvalidRegistrationGeneration);
        }
        let instance_id = decoder.string(instance_len, "instance_id")?;
        if layer_count > decoder.remaining() / 8 {
            return Err(CacheProtocolError::Truncated);
        }
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let name_len = decoder.usize_u32()?;
            let block_count = decoder.usize_u32()?;
            let layer_name = decoder.string(name_len, "layer_name")?;
            if block_count > decoder.remaining() / 8 {
                return Err(CacheProtocolError::Truncated);
            }
            let mut block_ids = Vec::with_capacity(block_count);
            let mut block_hashes = Vec::with_capacity(block_count);
            for _ in 0..block_count {
                block_ids.push(decoder.u32()?);
                let hash_len = decoder.usize_u32()?;
                block_hashes.push(decoder.bytes(hash_len)?.to_vec());
            }
            layers.push(PublishLayer {
                layer_name,
                block_ids,
                block_hashes,
            });
        }
        decoder.finish()?;
        validate_publish_layers(&layers)?;
        Ok(Self {
            instance_id,
            registration_generation,
            tp_rank,
            pp_rank,
            device_id,
            layers,
        })
    }
}

fn validate_publish_layers(layers: &[PublishLayer]) -> Result<(), CacheProtocolError> {
    for layer in layers {
        if layer.block_ids.len() != layer.block_hashes.len() {
            return Err(CacheProtocolError::PublishShapeMismatch {
                layer: layer.layer_name.clone(),
                block_ids: layer.block_ids.len(),
                block_hashes: layer.block_hashes.len(),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseRequest {
    pub lease: Vec<u8>,
}

impl ReleaseRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        let mut bytes = Vec::with_capacity(RELEASE_HEADER_BYTES + self.lease.len());
        push_u32(&mut bytes, RELEASE_REQUEST_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, checked_u32(self.lease.len(), "lease")?);
        bytes.extend_from_slice(&self.lease);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RELEASE_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let lease_len = decoder.usize_u32()?;
        let lease = decoder.bytes(lease_len)?.to_vec();
        decoder.finish()?;
        if lease.is_empty() {
            return Err(CacheProtocolError::EmptyLease);
        }
        Ok(Self { lease })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryBundleRequest {
    pub ticket: QueryTicket,
    pub instance_id: String,
    pub request_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub group_id: u32,
    pub wait_for_full_prefix: bool,
    /// Best-effort preparation without a restore lease. Never waits for publication.
    pub warmup: bool,
    /// Metadata hints only: no payload read, byte reservation, or restore lease.
    pub discover: bool,
    /// Read a selected recovery range already counted by candidate discovery.
    pub materialize: bool,
    /// Keep a consumer-owned result at the Manager until its first demand poll.
    pub prepare: bool,
    /// Every group's compiled range at the selected boundary, scoped by this
    /// request's registered instance. Present exactly for materialized recovery.
    pub demand: Option<RecoveryDemand>,
}

impl QueryBundleRequest {
    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        if self.request_id.is_empty() {
            return Err(CacheProtocolError::EmptyRequestId);
        }
        if self.materialize != self.demand.is_some() {
            return Err(CacheProtocolError::InvalidRecoveryDemand);
        }
        if let Some(demand) = &self.demand {
            demand.validate(self.group_id, self.block_hashes.len())?;
        }
        let instance = self.instance_id.as_bytes();
        let request = self.request_id.as_bytes();
        let mut size = REQUEST_HEADER_BYTES;
        add_encoded_size(&mut size, instance.len(), 1)?;
        add_encoded_size(&mut size, request.len(), 1)?;
        add_encoded_size(&mut size, self.block_hashes.len(), 4)?;
        for hash in &self.block_hashes {
            add_encoded_size(&mut size, hash.len(), 1)?;
        }
        if let Some(demand) = &self.demand {
            add_encoded_size(&mut size, 1, 28)?;
            add_encoded_size(&mut size, demand.groups.len(), 20)?;
        }
        let mut bytes = Vec::with_capacity(size);
        push_u32(&mut bytes, QUERY_REQUEST_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(
            &mut bytes,
            u16::from(self.wait_for_full_prefix)
                | (u16::from(self.warmup) << 1)
                | (u16::from(self.discover) << 2)
                | (u16::from(self.materialize) << 3)
                | (u16::from(self.prepare) << 4),
        );
        self.ticket.encode_into(&mut bytes)?;
        push_u32(&mut bytes, self.group_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        push_u32(&mut bytes, checked_u32(request.len(), "request_id")?);
        push_u32(
            &mut bytes,
            checked_u32(self.block_hashes.len(), "block_hashes")?,
        );
        bytes.extend_from_slice(instance);
        bytes.extend_from_slice(request);
        for hash in &self.block_hashes {
            push_u32(&mut bytes, checked_u32(hash.len(), "block_hash")?);
            bytes.extend_from_slice(hash);
        }
        if let Some(demand) = &self.demand {
            push_u64(&mut bytes, demand.page_tokens);
            push_u64(&mut bytes, demand.span.start);
            push_u64(&mut bytes, demand.span.end);
            push_u32(
                &mut bytes,
                checked_u32(demand.groups.len(), "recovery_groups")?,
            );
            for &(group, range) in &demand.groups {
                push_u32(&mut bytes, group);
                push_u64(&mut bytes, range.start);
                push_u64(&mut bytes, range.end);
            }
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(QUERY_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags & !31 != 0 {
            return Err(CacheProtocolError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        let group_id = decoder.u32()?;
        let instance_len = decoder.usize_u32()?;
        let request_len = decoder.usize_u32()?;
        let hash_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        let request_id = decoder.string(request_len, "request_id")?;
        if request_id.is_empty() {
            return Err(CacheProtocolError::EmptyRequestId);
        }
        if hash_count > decoder.remaining() / 4 {
            return Err(CacheProtocolError::Truncated);
        }
        let mut block_hashes = Vec::with_capacity(hash_count);
        for _ in 0..hash_count {
            let len = decoder.usize_u32()?;
            block_hashes.push(decoder.bytes(len)?.to_vec());
        }
        let demand = if flags & 8 != 0 {
            let page_tokens = decoder.u64()?;
            let span = TokenRange {
                start: decoder.u64()?,
                end: decoder.u64()?,
            };
            let count = decoder.usize_u32()?;
            if count > decoder.remaining() / 20 {
                return Err(CacheProtocolError::Truncated);
            }
            let mut groups = Vec::with_capacity(count);
            for _ in 0..count {
                groups.push((
                    decoder.u32()?,
                    TokenRange {
                        start: decoder.u64()?,
                        end: decoder.u64()?,
                    },
                ));
            }
            let demand = RecoveryDemand {
                page_tokens,
                span,
                groups,
            };
            demand.validate(group_id, hash_count)?;
            Some(demand)
        } else {
            None
        };
        decoder.finish()?;
        Ok(Self {
            ticket,
            instance_id,
            request_id,
            block_hashes,
            group_id,
            wait_for_full_prefix: flags & 1 != 0,
            warmup: flags & 2 != 0,
            discover: flags & 4 != 0,
            materialize: flags & 8 != 0,
            prepare: flags & 16 != 0,
            demand,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum QueryOutcomeCode {
    Ready = 1,
    Loading = 2,
    Busy = 3,
    Candidates = 4,
}

impl TryFrom<u16> for QueryOutcomeCode {
    type Error = CacheProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ready),
            2 => Ok(Self::Loading),
            3 => Ok(Self::Busy),
            4 => Ok(Self::Candidates),
            _ => Err(CacheProtocolError::UnknownOutcome(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryBundleResponse {
    pub outcome: QueryOutcomeCode,
    pub num_hit_blocks: u64,
    pub lease: Vec<u8>,
    pub hit_positions: Vec<u32>,
}

impl QueryBundleResponse {
    pub fn loading() -> Self {
        Self {
            outcome: QueryOutcomeCode::Loading,
            num_hit_blocks: 0,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        let mut bytes = Vec::with_capacity(
            RESPONSE_HEADER_BYTES + self.lease.len() + self.hit_positions.len() * 4,
        );
        push_u32(&mut bytes, QUERY_RESPONSE_MAGIC);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, self.outcome as u16);
        push_u64(&mut bytes, self.num_hit_blocks);
        push_u32(&mut bytes, checked_u32(self.lease.len(), "lease")?);
        push_u32(
            &mut bytes,
            checked_u32(self.hit_positions.len(), "hit_positions")?,
        );
        bytes.extend_from_slice(&self.lease);
        for position in &self.hit_positions {
            push_u32(&mut bytes, *position);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(QUERY_RESPONSE_MAGIC)?;
        decoder.expect_version()?;
        let outcome = QueryOutcomeCode::try_from(decoder.u16()?)?;
        let num_hit_blocks = decoder.u64()?;
        let lease_len = decoder.usize_u32()?;
        let positions_len = decoder.usize_u32()?;
        let lease = decoder.bytes(lease_len)?.to_vec();
        if positions_len > decoder.remaining() / 4 {
            return Err(CacheProtocolError::Truncated);
        }
        let mut hit_positions = Vec::with_capacity(positions_len);
        for _ in 0..positions_len {
            hit_positions.push(decoder.u32()?);
        }
        decoder.finish()?;
        if matches!(outcome, QueryOutcomeCode::Loading | QueryOutcomeCode::Busy)
            && (num_hit_blocks != 0 || !lease.is_empty() || !hit_positions.is_empty())
        {
            return Err(CacheProtocolError::InvalidLoadingPayload);
        }
        if outcome == QueryOutcomeCode::Candidates
            && (!lease.is_empty()
                || num_hit_blocks != hit_positions.len() as u64
                || hit_positions.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(CacheProtocolError::InvalidCandidates);
        }
        Ok(Self {
            outcome,
            num_hit_blocks,
            lease,
            hit_positions,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CacheProtocolError {
    #[error("GPU registration generation must be nonzero")]
    InvalidRegistrationGeneration,
    #[error("invalid dense shard query response")]
    InvalidShardPayload,
    #[error("only selected recovery reads must carry complete compiled demand")]
    InvalidRecoveryDemand,
    #[error(transparent)]
    Recovery(#[from] orbitkv_state::RecoveryError),
    #[error("candidate hints must be ordered, unique and unleased")]
    InvalidCandidates,
    #[error("query operation and revision must be nonzero")]
    InvalidQueryTicket,
    #[error("cache protocol payload is truncated")]
    Truncated,
    #[error("cache protocol payload has {0} trailing bytes")]
    TrailingBytes(usize),
    #[error("invalid cache protocol magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported cache protocol version: {0}")]
    UnsupportedVersion(u16),
    #[error("invalid cache protocol flags: {0:#x}")]
    InvalidFlags(u16),
    #[error("unknown query outcome: {0}")]
    UnknownOutcome(u16),
    #[error("query request_id must not be empty")]
    EmptyRequestId,
    #[error("release lease must not be empty")]
    EmptyLease,
    #[error("cache protocol field {field} is too large: {len}")]
    FieldTooLarge { field: &'static str, len: usize },
    #[error("cache protocol field {field} is not valid UTF-8")]
    InvalidUtf8 { field: &'static str },
    #[error("loading query outcome must not carry ready data")]
    InvalidLoadingPayload,
    #[error("publish layer {layer} has {block_ids} block ids but {block_hashes} block hashes")]
    PublishShapeMismatch {
        layer: String,
        block_ids: usize,
        block_hashes: usize,
    },
    #[error("unknown completion intent: {0}")]
    UnknownCompletionIntent(u16),
    #[error("unknown completion route: {0}")]
    UnknownCompletionRoute(u16),
    #[error("unknown completion representation: {0}")]
    UnknownCompletionRepresentation(u16),
    #[error("unknown completion admission: {0}")]
    UnknownCompletionAdmission(u16),
    #[error("unknown completion outcome: {0}")]
    UnknownCompletionOutcome(u16),
    #[error("completion field {0} must not be empty")]
    EmptyCompletionField(&'static str),
    #[error("completion field {field} is too long: {len} bytes (maximum {max})")]
    CompletionFieldTooLong {
        field: &'static str,
        len: usize,
        max: usize,
    },
    #[error("completion field {0} must be non-zero")]
    ZeroCompletionField(&'static str),
    #[error("completion destination device must be non-negative, got {0}")]
    InvalidCompletionDevice(i32),
    #[error("completion route and target intent are inconsistent")]
    InvalidCompletionTarget,
    #[error("completion admission, outcome and wire bytes are inconsistent")]
    InvalidCompletionState,
    #[error("completion reserved field must be zero, got {0}")]
    InvalidCompletionReserved(u32),
    #[error("completion handoff queue depth exceeds the bounded limit: {0}")]
    CompletionQueueTooDeep(u32),
    #[error("completion handoff queue parallelism exceeds the bounded limit: {0}")]
    CompletionQueueTooWide(u32),
    #[error("completion resource evidence is inconsistent with admission")]
    InvalidCompletionResources,
}

/// Common dense prefix with one local lease per Manager and one control hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardQueryResponse {
    pub outcome: QueryOutcomeCode,
    pub num_hit_blocks: u64,
    pub leases: Vec<Vec<u8>>,
    pub control_id: Vec<u8>,
}

impl ShardQueryResponse {
    pub const MAX_SHARDS: usize = 32;

    pub fn loading() -> Self {
        Self {
            outcome: QueryOutcomeCode::Loading,
            num_hit_blocks: 0,
            leases: Vec::new(),
            control_id: Vec::new(),
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, CacheProtocolError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(24 + 16 * (self.leases.len() + 1));
        push_u32(&mut bytes, 0x4f52_5352);
        push_u16(&mut bytes, CACHE_PROTOCOL_VERSION);
        push_u16(&mut bytes, self.outcome as u16);
        push_u64(&mut bytes, self.num_hit_blocks);
        push_u32(&mut bytes, self.leases.len() as u32);
        push_u32(&mut bytes, self.control_id.len() as u32);
        bytes.extend_from_slice(&self.control_id);
        for lease in &self.leases {
            push_u16(&mut bytes, lease.len() as u16);
            bytes.extend_from_slice(lease);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CacheProtocolError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(0x4f52_5352)?;
        decoder.expect_version()?;
        let outcome = QueryOutcomeCode::try_from(decoder.u16()?)?;
        let num_hit_blocks = decoder.u64()?;
        let count = decoder.usize_u32()?;
        let control_len = decoder.usize_u32()?;
        if count > Self::MAX_SHARDS || !matches!(control_len, 0 | 16) {
            return Err(CacheProtocolError::InvalidShardPayload);
        }
        let control_id = decoder.bytes(control_len)?.to_vec();
        let mut leases = Vec::with_capacity(count);
        for _ in 0..count {
            let len = decoder.u16()? as usize;
            if !matches!(len, 0 | 16) {
                return Err(CacheProtocolError::InvalidShardPayload);
            }
            leases.push(decoder.bytes(len)?.to_vec());
        }
        decoder.finish()?;
        let response = Self {
            outcome,
            num_hit_blocks,
            leases,
            control_id,
        };
        response.validate()?;
        Ok(response)
    }

    fn validate(&self) -> Result<(), CacheProtocolError> {
        let nonzero_id = |bytes: &[u8]| bytes.len() == 16 && bytes.iter().any(|byte| *byte != 0);
        let valid = self.leases.len() <= Self::MAX_SHARDS
            && match self.outcome {
                QueryOutcomeCode::Ready if self.num_hit_blocks > 0 => {
                    !self.leases.is_empty()
                        && nonzero_id(&self.control_id)
                        && self.leases.iter().all(|lease| nonzero_id(lease))
                }
                QueryOutcomeCode::Ready => {
                    !self.leases.is_empty()
                        && self.control_id.is_empty()
                        && self.leases.iter().all(Vec::is_empty)
                }
                QueryOutcomeCode::Loading | QueryOutcomeCode::Busy => {
                    self.num_hit_blocks == 0 && self.leases.is_empty() && self.control_id.is_empty()
                }
                QueryOutcomeCode::Candidates => false,
            };
        if valid {
            Ok(())
        } else {
            Err(CacheProtocolError::InvalidShardPayload)
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self, expected: u32) -> Result<(), CacheProtocolError> {
        let actual = self.u32()?;
        if actual != expected {
            return Err(CacheProtocolError::InvalidMagic(actual));
        }
        Ok(())
    }

    fn expect_version(&mut self) -> Result<(), CacheProtocolError> {
        let version = self.u16()?;
        if version != CACHE_PROTOCOL_VERSION {
            return Err(CacheProtocolError::UnsupportedVersion(version));
        }
        Ok(())
    }

    fn u16(&mut self) -> Result<u16, CacheProtocolError> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?.try_into().expect("fixed slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, CacheProtocolError> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn i32(&mut self) -> Result<i32, CacheProtocolError> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn u64(&mut self) -> Result<u64, CacheProtocolError> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("fixed slice"),
        ))
    }

    fn usize_u32(&mut self) -> Result<usize, CacheProtocolError> {
        Ok(self.u32()? as usize)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], CacheProtocolError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(CacheProtocolError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(CacheProtocolError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn string(&mut self, len: usize, field: &'static str) -> Result<String, CacheProtocolError> {
        String::from_utf8(self.bytes(len)?.to_vec())
            .map_err(|_| CacheProtocolError::InvalidUtf8 { field })
    }

    fn finish(self) -> Result<(), CacheProtocolError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(CacheProtocolError::TrailingBytes(
                self.bytes.len() - self.offset,
            ))
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, CacheProtocolError> {
    u32::try_from(value).map_err(|_| CacheProtocolError::FieldTooLarge { field, len: value })
}

fn add_encoded_size(
    size: &mut usize,
    count: usize,
    width: usize,
) -> Result<(), CacheProtocolError> {
    *size = count
        .checked_mul(width)
        .and_then(|bytes| size.checked_add(bytes))
        .filter(|bytes| isize::try_from(*bytes).is_ok())
        .ok_or(CacheProtocolError::FieldTooLarge {
            field: "payload",
            len: usize::MAX,
        })?;
    Ok(())
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(
    bytes: &mut Vec<u8>,
    value: &[u8],
    field: &'static str,
) -> Result<(), CacheProtocolError> {
    push_u32(bytes, checked_u32(value.len(), field)?);
    bytes.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/cache_protocol.rs"]
mod tests;
