use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, EventType, GetOptions, KeyValue, WatchOptions};
use orbitkv_catalog::MembershipView;

use super::format::ClusterFormat;
use super::inventory::InventoryRuntime;
use super::{MAX_MEMBERS, MEMBER_BYTES, Member, RPC_TIMEOUT, cluster_id, parse_label, rpc};

pub(super) enum FollowError {
    Disconnected(String),
    Rebuild(String),
}

impl From<etcd_client::Error> for FollowError {
    fn from(error: etcd_client::Error) -> Self {
        match &error {
            etcd_client::Error::IoError(_) | etcd_client::Error::TransportError(_) => {
                Self::Disconnected(error.to_string())
            }
            etcd_client::Error::GRpcStatus(status) if disconnected_status(status) => {
                Self::Disconnected(error.to_string())
            }
            _ => Self::Rebuild(error.to_string()),
        }
    }
}

fn disconnected_status(status: &tonic::Status) -> bool {
    if matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::Cancelled | tonic::Code::DeadlineExceeded
    ) {
        return true;
    }
    status.code() == tonic::Code::Unknown
        && (status.message().contains("h2 protocol error")
            || status.message().contains("transport error"))
}

async fn follow_rpc<T>(
    request: impl Future<Output = Result<T, etcd_client::Error>>,
) -> Result<T, FollowError> {
    tokio::time::timeout(RPC_TIMEOUT, request)
        .await
        .map_err(|_| FollowError::Disconnected("coordinator request timed out".into()))?
        .map_err(FollowError::from)
}

pub(super) async fn run(
    mut client: Client,
    prefix: String,
    expected_cluster: u64,
    format: ClusterFormat,
    registration: Member,
    view: Arc<MembershipView>,
    inventory: InventoryRuntime,
) {
    let mut snapshot = None;
    while view.registration_valid() {
        if snapshot.is_none() {
            view.invalidate_snapshot();
            inventory.membership_unavailable();
            match bootstrap(
                &mut client,
                &prefix,
                expected_cluster,
                &format,
                &registration,
                &view,
                &inventory,
            )
            .await
            {
                Ok(state) => snapshot = Some(state),
                Err(error) => {
                    log::warn!("Membership snapshot needs repair: {error}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            }
        }
        let Some((members, revision)) = snapshot.as_mut() else {
            continue;
        };
        match follow(
            &mut client,
            &prefix,
            expected_cluster,
            &format,
            &registration,
            &view,
            &inventory,
            members,
            revision,
        )
        .await
        {
            Ok(()) => break,
            Err(FollowError::Disconnected(error)) => {
                log::warn!("Membership Watch will resume at revision {revision}: {error}");
            }
            Err(FollowError::Rebuild(error)) => {
                log::warn!("Membership Watch requires a fresh snapshot: {error}");
                view.invalidate_snapshot();
                inventory.membership_unavailable();
                snapshot = None;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    view.invalidate_snapshot();
    inventory.membership_unavailable();
}

async fn bootstrap(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
    expected_format: &ClusterFormat,
    registration: &Member,
    view: &MembershipView,
    inventory: &InventoryRuntime,
) -> Result<(BTreeMap<String, Member>, i64), String> {
    let mut members = BTreeMap::new();
    let mut start = prefix.as_bytes().to_vec();
    let mut end = start.clone();
    *end.last_mut().ok_or("empty metadata prefix")? += 1;
    let mut revision = 0;
    let mut format_seen = false;
    loop {
        let response = rpc(client.get(
            start.clone(),
            Some(
                GetOptions::new()
                    .with_range(end.clone())
                    .with_revision(revision)
                    .with_limit(128),
            ),
        ))
        .await?;
        if cluster_id(response.header())? != expected_cluster {
            view.fence();
            return Err("coordinator changed during membership snapshot".into());
        }
        if revision == 0 {
            revision = response
                .header()
                .ok_or("missing membership snapshot revision")?
                .revision();
        }
        for kv in response.kvs() {
            let _ = decode(
                prefix,
                kv,
                false,
                expected_format,
                &mut members,
                &mut format_seen,
            )?;
        }
        if !response.more() {
            break;
        }
        start = response
            .kvs()
            .last()
            .ok_or("empty membership continuation page")?
            .key()
            .to_vec();
        start.push(0);
        tokio::task::yield_now().await;
    }
    if !format_seen || members.get(&registration.node_id) != Some(registration) {
        view.fence();
        return Err("cluster format or own registration changed".into());
    }
    install_members(view, inventory, revision, &members);
    Ok((members, revision))
}

#[allow(
    clippy::too_many_arguments,
    reason = "membership identity and inventory consumers share one Watch boundary"
)]
async fn follow(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
    expected_format: &ClusterFormat,
    registration: &Member,
    view: &MembershipView,
    inventory: &InventoryRuntime,
    members: &mut BTreeMap<String, Member>,
    applied: &mut i64,
) -> Result<(), FollowError> {
    let mut stream = follow_rpc(
        client.watch(
            prefix,
            Some(
                WatchOptions::new()
                    .with_prefix()
                    .with_start_revision(*applied + 1)
                    .with_prev_key(),
            ),
        ),
    )
    .await?;
    loop {
        if !view.registration_valid() {
            return Ok(());
        }
        let response = match tokio::time::timeout(Duration::from_secs(2), stream.message()).await {
            Ok(response) => response.map_err(FollowError::from)?,
            Err(_) => {
                follow_rpc(stream.request_progress()).await?;
                continue;
            }
        }
        .ok_or_else(|| FollowError::Disconnected("membership Watch closed".into()))?;
        if cluster_id(response.header()).map_err(FollowError::Rebuild)? != expected_cluster {
            view.fence();
            return Ok(());
        }
        if response.canceled() || response.compact_revision() > 0 {
            return Err(FollowError::Rebuild(format!(
                "Watch canceled or compacted: {}",
                response.cancel_reason()
            )));
        }
        if response.created() {
            continue;
        }
        let mut through = *applied;
        let mut format_valid = true;
        for event in response.events() {
            let kv = event
                .kv()
                .ok_or_else(|| FollowError::Rebuild("missing membership event key".into()))?;
            if kv.mod_revision() <= *applied {
                continue;
            }
            through = through.max(kv.mod_revision());
            let deleted = event.event_type() == EventType::Delete;
            let value = if deleted {
                event.prev_kv().ok_or_else(|| {
                    FollowError::Rebuild("membership delete lacks previous value".into())
                })?
            } else {
                kv
            };
            if let Some(previous) = decode(
                prefix,
                value,
                deleted,
                expected_format,
                members,
                &mut format_valid,
            )
            .map_err(FollowError::Rebuild)?
            {
                view.revoke(&previous);
            }
            if !format_valid || members.get(&registration.node_id) != Some(registration) {
                view.fence();
                return Ok(());
            }
        }
        if response.events().is_empty() {
            through = response
                .header()
                .ok_or_else(|| FollowError::Rebuild("missing Watch revision".into()))?
                .revision()
                .max(through);
        }
        if through > *applied {
            install_members(view, inventory, through, members);
            *applied = through;
        }
    }
}

fn install_members(
    view: &MembershipView,
    inventory: &InventoryRuntime,
    revision: i64,
    members: &BTreeMap<String, Member>,
) {
    view.replace_members(
        members
            .values()
            .map(|member| (member.node_id.clone(), member.epoch, member.owner.clone())),
    );
    inventory.replace_members(revision, members.clone());
}

fn decode(
    prefix: &str,
    kv: &KeyValue,
    deleted: bool,
    expected_format: &ClusterFormat,
    members: &mut BTreeMap<String, Member>,
    format_valid: &mut bool,
) -> Result<Option<orbitkv_state::CacheOwner>, String> {
    let suffix = kv
        .key_str()
        .map_err(|error| error.to_string())?
        .strip_prefix(prefix)
        .ok_or("metadata key outside cluster")?;
    if kv.value().len() > MEMBER_BYTES {
        return Err("membership record exceeds byte limit".into());
    }
    let mut removed = None;
    if suffix == "format" {
        *format_valid = !deleted
            && kv.lease() == 0
            && ClusterFormat::decode(kv.value())
                .is_ok_and(|observed| observed.same_identity(expected_format));
        if !*format_valid {
            return Err("cluster inventory format changed".into());
        }
    } else if let Some(node) = suffix.strip_prefix("members/") {
        let member = decode_member(node, kv)?;
        removed = members
            .get(node)
            .filter(|previous| deleted || *previous != &member)
            .map(|previous| previous.owner.clone());
        if deleted {
            members.remove(node);
        } else {
            members.insert(node.into(), member);
        }
        if members.len() > MAX_MEMBERS {
            return Err("metadata member limit exceeded".into());
        }
    } else if let Some(node) = suffix.strip_prefix("epochs/") {
        parse_label(node)?;
        if deleted
            || kv.lease() != 0
            || kv
                .value_str()
                .map_err(|error| error.to_string())?
                .parse::<u64>()
                .is_err()
        {
            return Err("invalid persistent node epoch".into());
        }
    } else {
        return Err("block metadata is forbidden by the inventory-stream format".into());
    }
    Ok(removed)
}

fn decode_member(node: &str, kv: &KeyValue) -> Result<Member, String> {
    if kv.value().len() > MEMBER_BYTES || kv.lease() == 0 {
        return Err("member record oversized or unleased".into());
    }
    let mut member: Member =
        serde_json::from_slice(kv.value()).map_err(|error| error.to_string())?;
    member.lease = kv.lease();
    parse_label(node)?;
    let address: std::net::SocketAddr = member
        .owner
        .endpoint
        .parse()
        .map_err(|_| "invalid member endpoint")?;
    if node != member.node_id
        || member.epoch == 0
        || member.protocol != orbitkv_state::INVENTORY_STREAM_PROTOCOL
        || member.owner.incarnation.is_nil()
        || address.ip().is_unspecified()
        || address.port() == 0
    {
        return Err("invalid member identity".into());
    }
    Ok(member)
}

#[cfg(test)]
#[path = "../../tests/unit/cluster/watch.rs"]
mod tests;
