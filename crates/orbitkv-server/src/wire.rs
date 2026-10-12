//! Decode shared protobuf messages at protocol boundaries.

use orbitkv_core::TransferMode;

use crate::cache::lifecycle::{Registration, SessionSpec};
use crate::proto::engine::{
    RegisterContextRequest, SessionRequest, TransferMode as WireTransferMode,
};

pub(crate) fn registration(request: RegisterContextRequest) -> Registration {
    let transfer_mode = match request.transfer_mode() {
        WireTransferMode::Direct => TransferMode::Direct,
        WireTransferMode::Kernel => TransferMode::Kernel,
    };
    Registration {
        instance_id: request.instance_id,
        namespace: request.namespace,
        tp_rank: request.tp_rank,
        pp_rank: request.pp_rank,
        tp_size: request.tp_size,
        world_size: request.world_size,
        device_id: request.device_id,
        device_uuid: request.device_uuid,
        layer_names: request.layer_names,
        wrapper_bytes: request.wrapper_bytes,
        num_blocks: request.num_blocks,
        bytes_per_block: request.bytes_per_block,
        kv_stride_bytes: request.kv_stride_bytes,
        segments: request.segments,
        client_version: request.client_version,
        transfer_mode,
        page_first: request.page_first,
        layer_group_ids: request.layer_group_ids,
        layer_formats: request.layer_formats,
        layer_attention: request
            .layer_attention
            .into_iter()
            .map(|a| (a.head_dim, a.role, a.layer_index, a.layer_count))
            .collect(),
    }
}

pub(crate) fn session(request: SessionRequest) -> SessionSpec {
    SessionSpec {
        instance_id: request.instance_id,
        namespace: request.namespace,
        tp_size: request.tp_size,
        world_size: request.world_size,
    }
}
