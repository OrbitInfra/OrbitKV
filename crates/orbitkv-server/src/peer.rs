//! Embeddable P2P transfer gRPC service.
//!
//! A node opens reusable transfer windows, authorizes single-use tickets with
//! `QueryBlocksForTransfer`, and closes them with `ReleaseTransferLock`.
//! The Cache Manager serves only these peer RPCs plus
//! `Health`; inference processes use the node-local UDS/iceoryx2 endpoint.
//!
//! Source pins are budgeted and retained after timeout. An overdue session
//! requires terminal completion or transport revocation before memory reuse.

use std::sync::Arc;

use log::debug;
use tonic::{Request, Response, Status, async_trait};

use orbitkv_proto::proto::engine::engine_server::Engine;
use orbitkv_proto::proto::engine::{
    HealthRequest, HealthResponse, OpenTransferWindowRequest, OpenTransferWindowResponse,
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, ReleaseTransferLockRequest,
    ReleaseTransferLockResponse, ResponseStatus, TransferBlockInfo, TransferSlotInfo,
    TransferTicket as WireTicket,
};

use orbitkv_core::{LayerBlock, OrbitKVEngine, PeerError, TransferTicket};

/// Peer-only gRPC service for fetching blocks from this Cache Manager or
/// an embedded `OrbitKVEngine`.
pub struct P2pTransferService {
    engine: Arc<OrbitKVEngine>,
}

impl P2pTransferService {
    pub fn new(engine: Arc<OrbitKVEngine>) -> Self {
        Self { engine }
    }

    fn parse_ticket(ticket: Option<WireTicket>) -> Result<TransferTicket, Status> {
        let ticket = ticket.ok_or_else(|| Status::invalid_argument("missing transfer ticket"))?;
        let window = ticket
            .window_id
            .parse::<uuid::Uuid>()
            .map_err(|_| Status::invalid_argument("invalid transfer window"))?;
        TransferTicket::new(window, ticket.slot as usize, ticket.generation)
            .map_err(Self::authorization_error)
    }

    fn authorization_error(error: PeerError) -> Status {
        match error {
            PeerError::Unavailable => {
                Status::failed_precondition("Mooncake transfer engine is not configured")
            }
            PeerError::StagingFailed => Status::unavailable("source SSD staging failed"),
            PeerError::StaleReplica => {
                Status::failed_precondition("stale owner or residency candidate")
            }
            PeerError::InvalidRequest(reason) => Status::invalid_argument(reason),
            PeerError::UnknownWindow => Status::not_found("unknown transfer window"),
            PeerError::StaleTicket => {
                Status::failed_precondition("closed or occupied transfer ticket")
            }
            PeerError::BudgetExhausted => {
                Status::resource_exhausted("source transfer reservation budget exhausted")
            }
        }
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn build_transfer_slot_info(
        raw_block: &orbitkv_core::RawBlock,
        numa_node: orbitkv_core::NumaNode,
    ) -> TransferSlotInfo {
        let layer_block = LayerBlock::new(raw_block);
        TransferSlotInfo {
            k_ptr: layer_block.k_ptr() as u64,
            k_size: layer_block.k_size() as u64,
            v_ptr: layer_block.v_ptr().map(|p| p as u64).unwrap_or(0),
            v_size: layer_block.v_size().unwrap_or(0) as u64,
            numa_node: numa_node.0,
            encoding: raw_block
                .encoding()
                .map(|meta| serde_json::to_vec(meta).expect("serializable codec metadata"))
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl Engine for P2pTransferService {
    async fn open_transfer_window(
        &self,
        request: Request<OpenTransferWindowRequest>,
    ) -> Result<Response<OpenTransferWindowResponse>, Status> {
        let req = request.into_inner();
        let owner = req
            .owner_incarnation
            .parse()
            .map_err(|_| Status::invalid_argument("invalid owner incarnation"))?;
        let requester = req
            .requester_incarnation
            .parse::<uuid::Uuid>()
            .map_err(|_| Status::invalid_argument("invalid requester incarnation"))?;
        let window = self
            .engine
            .peer_exports()
            .open(owner, requester)
            .map_err(Self::authorization_error)?;
        Ok(Response::new(OpenTransferWindowResponse {
            window_id: window.to_string(),
        }))
    }

    async fn query_blocks_for_transfer(
        &self,
        request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        let req = request.into_inner();

        if !self.engine.has_remote_transport() {
            return Err(Status::failed_precondition(
                "Mooncake transfer engine is not configured",
            ));
        }

        if req.residency_sequences.len() != req.block_hashes.len()
            || req.residency_sequences.contains(&0)
        {
            return Err(Status::invalid_argument("missing residency sequences"));
        }
        let owner = req
            .owner_incarnation
            .parse()
            .map_err(|_| Status::invalid_argument("invalid owner incarnation"))?;
        let records: Vec<_> = req
            .block_hashes
            .iter()
            .zip(&req.residency_sequences)
            .map(|(hash, &sequence)| orbitkv_state::InventoryRecord {
                key: orbitkv_state::StateKey::new(req.namespace.clone(), hash.clone()),
                sequence,
                present: true,
                metadata: None,
            })
            .collect();
        let ticket = Self::parse_ticket(req.ticket)?;
        let found_blocks = self
            .engine
            .peer_exports()
            .authorize(owner, ticket, &records)
            .await
            .map_err(Self::authorization_error)?;

        let blocks: Vec<TransferBlockInfo> = found_blocks
            .iter()
            .map(|(key, block)| {
                let slots: Vec<TransferSlotInfo> = block
                    .slots()
                    .iter()
                    .zip(block.slot_numas())
                    .map(|(raw, &numa)| Self::build_transfer_slot_info(raw, numa))
                    .collect();
                TransferBlockInfo {
                    block_hash: key.hash.clone(),
                    slots,
                }
            })
            .collect();

        debug!(
            "P2P query_blocks_for_transfer: requested={} found={} ticket={:?}",
            req.block_hashes.len(),
            blocks.len(),
            ticket,
        );

        Ok(Response::new(QueryBlocksForTransferResponse {
            status: Some(Self::ok_status()),
            blocks,
            lock_timeout_secs: self.engine.peer_exports().lock_timeout().as_secs() as u32,
            native_source_epoch: self
                .engine
                .transfer_source_epoch()
                .ok_or_else(|| {
                    Status::failed_precondition("registered native source epoch is unavailable")
                })?
                .to_string(),
            transfer_endpoint: self
                .engine
                .transfer_endpoint()
                .ok_or_else(|| {
                    Status::failed_precondition("Mooncake transfer engine is not configured")
                })?
                .to_string(),
        }))
    }

    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let ticket = Self::parse_ticket(request.into_inner().ticket)?;
        let released = self
            .engine
            .peer_exports()
            .release(ticket)
            .map_err(Self::authorization_error)?;
        Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(Self::ok_status()),
            released_blocks: released as u64,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: Some(Self::ok_status()),
        }))
    }
}
