use crate::memory::numa::NumaNode;
use crate::planning::peer::{FetchPlan, FetchSegment};

use crate::block::{SealedBlock, StateKey};
#[cfg(test)]
use crate::storage::MaterializedBlocks;
use std::sync::Arc;

const MAX_AUTHORIZATION_RETRIES: usize = 2;

#[derive(Clone, Copy, Debug)]
pub(super) enum AuthorizationError {
    /// Rejected during authorization, before any payload transfer was submitted.
    /// Stale evidence and transient source admission both permit another owner.
    Rejected,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AuthorizationMode {
    Demand,
    Lookahead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FetchStatus {
    Complete,
    AuthorizationExhausted,
    PayloadFailed,
}

pub(crate) struct FetchResult<B = Arc<SealedBlock>> {
    pub(crate) blocks: Vec<(StateKey, B)>,
    pub(crate) status: FetchStatus,
    pub(crate) attempts: usize,
    pub(crate) completed_segments: usize,
}

impl<B> FetchResult<B> {
    pub(crate) fn can_replan(&self) -> bool {
        self.status == FetchStatus::AuthorizationExhausted && self.blocks.is_empty()
    }
}

#[tonic::async_trait]
pub(super) trait SegmentFetcher {
    /// The grant owns its source pins, including cleanup when never consumed.
    type Grant: Send;
    type Block: Send;

    async fn authorize_segment(
        &self,
        segment: &FetchSegment,
        mode: AuthorizationMode,
    ) -> Result<Self::Grant, AuthorizationError>;

    /// Destination admission happens here, after the preceding READ drains.
    async fn fetch_segment(
        &self,
        segment: &FetchSegment,
        grant: Self::Grant,
        req_id: &str,
        destination_nodes: &[NumaNode],
    ) -> Result<Vec<(StateKey, Self::Block)>, ()>;
}

pub(super) async fn execute_fetch_plan<F: SegmentFetcher>(
    fetcher: &F,
    mut plan: FetchPlan<'_>,
    req_id: &str,
    destination_nodes: &[NumaNode],
) -> FetchResult<F::Block> {
    let mut fetched = Vec::with_capacity(plan.block_count());
    let mut attempts = 0;
    let mut completed = 0;
    let mut rejected = 0;
    let mut status = FetchStatus::AuthorizationExhausted;
    let mut prepared = None;
    loop {
        let (segment, authorization) = if let Some((segment, grant)) = prepared.take() {
            (segment, Ok(grant))
        } else {
            let Some(segment) = plan.next_segment(fetched.len()) else {
                break;
            };
            attempts += 1;
            let authorization = fetcher
                .authorize_segment(&segment, AuthorizationMode::Demand)
                .await;
            (segment, authorization)
        };
        let grant = match authorization {
            Err(AuthorizationError::Rejected) => {
                // Only discard this batch's evidence. A later key may still be valid.
                plan.reject(fetched.len(), &segment);
                rejected += 1;
                if rejected > MAX_AUTHORIZATION_RETRIES {
                    break;
                }
                continue;
            }
            Err(AuthorizationError::Failed) => {
                status = FetchStatus::PayloadFailed;
                break;
            }
            Ok(grant) => grant,
        };

        // At most one READ and one following authorization are in flight. The
        // latter allocates no destination buffers and stays in this future, so
        // a failed prefix or cancellation drops its known-ticket cleanup owner.
        let next = plan.next_segment(fetched.len() + segment.records.len());
        let read = fetcher.fetch_segment(&segment, grant, req_id, destination_nodes);
        let authorize_next = async {
            let Some(next) = &next else {
                return std::future::pending().await;
            };
            attempts += 1;
            fetcher
                .authorize_segment(next, AuthorizationMode::Lookahead)
                .await
        };
        let (returned, next_grant) = {
            tokio::pin!(read, authorize_next);
            let (returned, authorized) = tokio::select! {
                // Start the READ before asking for another source grant. A
                // synchronous allocation failure never speculates at all.
                biased;
                returned = &mut read => (returned, None),
                authorized = &mut authorize_next => (read.await, Some(authorized)),
            };
            let complete = returned.as_ref().is_ok_and(|blocks| {
                blocks.len() == segment.records.len()
                    && blocks
                        .iter()
                        .zip(&segment.records)
                        .all(|((key, _), record)| *key == record.key)
            });
            let next_grant = if complete && next.is_some() {
                // A speculative admission failure can be caused by the current
                // READ's source budget. Retry on demand after it drains rather
                // than discarding an otherwise usable owner or its evidence.
                match authorized {
                    Some(result) => result.ok(),
                    None => authorize_next.await.ok(),
                }
            } else {
                None
            };
            (returned, next_grant)
        };
        match returned {
            Err(()) => {
                status = FetchStatus::PayloadFailed;
                break;
            }
            Ok(returned) => {
                let contiguous = returned
                    .iter()
                    .zip(&segment.records)
                    .take_while(|((key, _), record)| *key == record.key)
                    .count();
                let returned_count = returned.len();
                fetched.extend(returned.into_iter().take(contiguous));
                if contiguous != segment.records.len() || returned_count != segment.records.len() {
                    status = FetchStatus::PayloadFailed;
                    break;
                }
                completed += 1;
            }
        }
        prepared = next.zip(next_grant);
    }
    if fetched.len() == plan.block_count() {
        status = FetchStatus::Complete;
    }
    FetchResult {
        blocks: fetched,
        status,
        attempts,
        completed_segments: completed,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/peer/execute.rs"]
mod tests;
