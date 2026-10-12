"""Type stubs for the orbitkv Rust extension module (PyO3 bindings).

This module provides local Cache Manager bindings for LLM inference.
"""

from collections.abc import Sequence

__version__: str

# Custom exceptions for error classification

class OrbitKVError(Exception):
    """Base exception for all OrbitKV errors."""

    ...

class OrbitKVInternal(OrbitKVError):
    """Internal server error."""

    ...

class MooncakeTransferEngine:
    def __init__(self, *, bind_host: str, nics: list[str] = ...) -> None: ...
    @property
    def endpoint(self) -> str: ...
    def register_memory(self, regions: list[dict[str, int | str]]) -> None: ...
    def unregister_memory(self, addresses: list[int]) -> None: ...
    def write(
        self,
        remote_endpoint: str,
        slices: list[tuple[int, int, int]],
        timeout_s: float = 30.0,
        notify_name: str | None = None,
        notify_message: str | None = None,
    ) -> int: ...
    def read(
        self,
        remote_endpoint: str,
        slices: list[tuple[int, int, int]],
        timeout_s: float = 30.0,
    ) -> int: ...
    def send_notification(self, remote_endpoint: str, name: str, message: str) -> None: ...
    def open_notification_scope(self, name: str) -> int: ...
    def wait_for_status(
        self,
        name: str,
        generation: int,
        expected_done_count: int = 1,
        timeout_s: float = 30.0,
    ) -> str | None: ...
    def close_notification_scope(self, name: str, generation: int) -> None: ...
    def nic_load_stats(self) -> list[tuple[str, int, float]]: ...
    def invalidate_segment(self, remote_endpoint: str) -> None: ...

class QueryLoading:
    admitted: bool
    def __init__(self, admitted: bool = True) -> None: ...

class ShardedQueryReady:
    @property
    def num_hit_blocks(self) -> int: ...
    @property
    def leases(self) -> tuple[bytes, ...]: ...
    @property
    def control_id(self) -> bytes: ...

class QueryReady:
    """Query result whose lease token is returned as owned, immutable bytes."""

    num_hit_blocks: int
    lease: bytes
    hit_positions: list[int]
    def __init__(
        self,
        num_hit_blocks: int,
        lease: bytes,
        hit_positions: list[int] = ...,
    ) -> None: ...

class QueryCandidates:
    """Provisional metadata only; cannot be passed to GPU restore."""

    num_hit_blocks: int
    hit_positions: list[int]
    def __init__(self, hit_positions: list[int]) -> None: ...

class RecoveryContract:
    """Compile attention, MLA, window, recurrent or convolution groups into page demand."""

    def __init__(
        self, namespace: str, page_tokens: int, groups: list[tuple[int, str, int]]
    ) -> None: ...
    def required_ranges(self, namespace: str, start: int, end: int) -> list[tuple[int, int, int]]:
        """Group/start/end demand from a valid HBM origin; does not promise hits."""
        ...
    def select_boundary(
        self,
        namespace: str,
        start: int,
        end: int,
        shards: list[list[tuple[int, list[int]]]],
        limit: int,
    ) -> int | None: ...
    def common_boundaries(
        self, namespace: str, start: int, end: int, shards: list[list[tuple[int, list[int]]]]
    ) -> list[int]: ...
    def restorable_boundaries(
        self, namespace: str, start: int, end: int, groups: list[tuple[int, list[int]]]
    ) -> list[int]: ...

class ChannelProbeClient:
    """Low-level iceoryx2 probe for Cache Manager channel diagnostics."""

    def __init__(
        self,
        service_name: str,
        session_epoch: int,
        timeout_ms: int = 5000,
        spin_iterations: int = 64,
    ) -> None: ...
    @property
    def service_name(self) -> str: ...
    @property
    def session_epoch(self) -> int: ...
    def ping(self, value: int = 0, request_id: int = 1) -> int: ...
    def shutdown(self, request_id: int = 1) -> None: ...

class BlockHashes:
    """Immutable Rust-owned query hashes; slices share storage and are cheap to poll."""

    def __init__(self, hashes: Sequence[bytes]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, view: slice) -> BlockHashes: ...

class ChannelCallObservation:
    """One opt-in same-host process-channel call observation."""

    @property
    def request_id(self) -> int: ...
    @property
    def session_epoch(self) -> int: ...
    @property
    def session_token(self) -> int: ...
    @property
    def submitted_mono_ns(self) -> int: ...
    @property
    def returned_mono_ns(self) -> int: ...

class CacheManagerClient:
    """Native owner of query, publish and restore lifetimes."""

    def __init__(
        self,
        bootstrap_socket: str,
        *,
        timeout_ms: int = 5000,
        spin_iterations: int = 64,
    ) -> None: ...
    def close(self) -> None: ...
    def health(self) -> tuple[bool, str]: ...
    def configure_shard_queries(
        self,
        instance_id: str,
        namespace: str,
        tp_size: int,
        world_size: int,
        shards: list[tuple[str, str, bytes]],
    ) -> None: ...
    def query_shards(
        self, instance_id: str, block_hashes: BlockHashes, req_id: str
    ) -> QueryLoading | ShardedQueryReady: ...
    def release_shard_query(self, control_id: bytes) -> None: ...
    def export_query_target(
        self, instance_id: str, namespace: str, tp_size: int, world_size: int
    ) -> bytes: ...
    def unregister_context(self, instance_id: str) -> tuple[bool, str]: ...
    def start_session_watcher(
        self, instance_id: str, namespace: str, tp_size: int, world_size: int
    ) -> None: ...
    def register_context_batch(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        pp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
        transfer_backend: str,
        page_first: bool,
        layer_group_ids: list[int] | None = None,
        layer_formats: list[str] | None = None,
        layer_attention: list[tuple[int, str, int, int]] | None = None,
        *,
        tensors: list[object],
    ) -> tuple[bool, str]:
        """Register client-local device_id and retain the UUID-resolved Manager route."""
        ...
    @property
    def transport(self) -> str: ...
    @property
    def bootstrap_socket(self) -> str: ...
    @property
    def service_name(self) -> str: ...
    @property
    def session_epoch(self) -> int: ...
    @property
    def notification_fd(self) -> int: ...
    def query_candidates(
        self, instance_id: str, block_hashes: BlockHashes, req_id: str, group_id: int = 0
    ) -> QueryCandidates | QueryLoading: ...
    def read_recovery(
        self,
        instance_id: str,
        block_hashes: BlockHashes,
        req_id: str,
        contract: RecoveryContract,
        namespace: str,
        start: int,
        end: int,
        group_id: int,
    ) -> QueryReady | QueryLoading:
        """Read required_ranges and return only complete, revalidated group leases."""
        ...
    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: BlockHashes,
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
    ) -> QueryLoading | QueryReady: ...
    def query_prefetch_diagnostic(
        self,
        instance_id: str,
        block_hashes: BlockHashes,
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
    ) -> tuple[QueryLoading | QueryReady, ChannelCallObservation]: ...
    def prepare_prefix(self, instance_id: str, block_hashes: BlockHashes, req_id: str) -> bool:
        """Prepare a partial attention prefix for a later counted query_prefetch claim."""

    def prepare_recovery(
        self,
        instance_id: str,
        block_hashes: BlockHashes,
        req_id: str,
        contract: RecoveryContract,
        namespace: str,
        start: int,
        end: int,
        group_id: int,
    ) -> bool:
        """Keep a bounded, expiring compiled range at the Manager for this consumer."""

    def warm_prefix(self, instance_id: str, block_hashes: BlockHashes, req_id: str) -> bool: ...
    def release(self, lease: bytes) -> None: ...
    def cancel_query(self, instance_id: str, req_id: str, group_id: int = 0) -> None: ...
    def save(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]: ...
    def save_diagnostic(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str, ChannelCallObservation]: ...
    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
        *,
        ready_stream: int,
        layer_events: list[tuple[str, object]] | None = None,
    ) -> RestoreHandle:
        """Reserve an operation before submission; claimed errors arrive in RestoreStatus.

        The caller supplies its previous-user CUDA stream in ready_stream.
        Native code retains registered tensors and captures that stream before
        submission. Hold destination pages until the terminal result, including
        after timeout. A lost submission ACK retains any claimed operation.
        """
        ...
    def wait_restore_enqueued(self, handle: RestoreHandle, *, timeout: float) -> None:
        """Wait until layer events describe this operation's enqueued copies; keep its final fence."""
        ...

    def poll_restore(self, handle: RestoreHandle) -> RestoreStatus: ...
    def wait_restore(self, handle: RestoreHandle, *, timeout: float) -> RestoreStatus:
        """Consume the terminal result without the GIL; timeout preserves ownership."""
        ...
    def restore_completions_ready(self, *, timeout: float = 0.0) -> bool: ...

class RestoreHandle:
    """Client-bound GPU restore identity, created only by start_restore."""

    @property
    def operation_id(self) -> int: ...
    @property
    def session_epoch(self) -> int: ...
    @property
    def key(self) -> str: ...

class RestoreStatus:
    done: bool
    success: bool
    message: str
    def __init__(self, done: bool, success: bool, message: str = "") -> None: ...

def rekey_hashes(hashes: tuple[bytes, ...], generation: bytes) -> tuple[bytes, ...]: ...
