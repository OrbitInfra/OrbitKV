mod format;
mod inventory;
mod registration;
mod watch;

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use etcd_client::{Client, ConnectOptions, ResponseHeader};
use log::{info, warn};
use orbitkv_catalog::{GlobalIndex, MembershipView};
use orbitkv_core::ResidencyInventory;
use orbitkv_state::CacheOwner;
use serde::{Deserialize, Serialize};
use tokio::sync::watch as signal;
use tokio::task::JoinHandle;

pub(crate) use inventory::{InventoryRuntime, InventoryRuntimeStatus, router};

pub(super) const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_MEMBERS: usize = 4096;
const MEMBER_BYTES: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub(super) enum BootstrapError {
    #[error("{0}")]
    Transport(String),
    #[error("{0}")]
    Rejected(String),
}

impl From<String> for BootstrapError {
    fn from(error: String) -> Self {
        Self::Transport(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    node_id: String,
    epoch: u64,
    owner: CacheOwner,
    protocol: String,
    #[serde(skip)]
    lease: i64,
}

/// Owns leased registration, metadata publication and local-index synchronization.
pub(crate) struct Cluster {
    view: Arc<MembershipView>,
    client: Client,
    lease: i64,
    inventory: InventoryRuntime,
    stop: signal::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
}

impl Cluster {
    pub(crate) async fn join(
        endpoints: &[String],
        cluster: &str,
        node: &str,
        ttl: i64,
        view: Arc<MembershipView>,
        inventory: Arc<ResidencyInventory>,
        index: Arc<GlobalIndex>,
    ) -> Result<Self, String> {
        parse_label(cluster)?;
        parse_label(node)?;
        if endpoints.is_empty() || !(12..=3600).contains(&ttl) {
            return Err(
                "membership requires endpoints and a lease TTL in 12..=3600 seconds".into(),
            );
        }
        let address: std::net::SocketAddr = view
            .owner()
            .endpoint
            .parse()
            .map_err(|_| "invalid member endpoint")?;
        if address.ip().is_unspecified() || address.port() == 0 || view.owner().incarnation.is_nil()
        {
            return Err(
                "membership requires a concrete peer address and runtime incarnation".into(),
            );
        }
        let mut client = rpc(Client::connect(
            endpoints,
            Some(ConnectOptions::new().with_connect_timeout(RPC_TIMEOUT)),
        ))
        .await?;
        // A balanced channel can initially select an unavailable endpoint.
        // Retry before registration; any unacknowledged empty lease expires.
        let deadline = Instant::now() + RPC_TIMEOUT * 4;
        let (sent_at, grant) = loop {
            let sent_at = Instant::now();
            match rpc(client.lease_grant(ttl, None)).await {
                Ok(grant) => break (sent_at, grant),
                Err(error) if Instant::now() < deadline => {
                    warn!("Lease grant will retry another coordinator connection: {error}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(error) => return Err(format!("lease grant: {error}")),
            }
        };
        let lease = grant.id();
        let (stop, _) = signal::channel(false);
        let placeholder_format = format::ClusterFormat {
            protocol: orbitkv_state::INVENTORY_STREAM_PROTOCOL.into(),
            cluster_uuid: uuid::Uuid::nil(),
            revision: 0,
            encoded: Vec::new(),
        };
        let placeholder_member = Member {
            node_id: node.into(),
            epoch: 0,
            owner: membership_owner(&view),
            protocol: orbitkv_state::INVENTORY_STREAM_PROTOCOL.into(),
            lease,
        };
        let runtime_view = Arc::clone(&view);
        let mut membership = Self {
            view,
            client,
            lease,
            inventory: InventoryRuntime::new(
                placeholder_format,
                placeholder_member,
                inventory.clone(),
                index.clone(),
                runtime_view,
            ),
            stop,
            tasks: Vec::new(),
        };
        let result = async {
            let cluster_id = cluster_id(grant.header())?;
            if grant.ttl() <= 0
                || !membership
                    .view
                    .renew(sent_at, Duration::from_secs(grant.ttl() as u64))
            {
                return Err("initial membership lease acknowledgement expired".into());
            }
            let prefix = format!("/orbitkv/v2/{cluster}/");
            let deadline = Instant::now() + RPC_TIMEOUT * 4;
            let (format, member) = loop {
                let attempt = async {
                    let format = format::install(&mut membership.client, &prefix, cluster_id).await?;
                    let member = registration::register(
                        &mut membership.client,
                        &prefix,
                        node,
                        membership.view.owner(),
                        lease,
                        cluster_id,
                        &format,
                    ).await?;
                    Ok::<_, BootstrapError>((format, member))
                }.await;
                match attempt {
                    Ok(result) if membership.view.registration_valid() => break result,
                    Ok(_) => return Err("membership expired during metadata bootstrap".into()),
                    Err(BootstrapError::Transport(error))
                        if membership.view.registration_valid() && Instant::now() < deadline =>
                    {
                        warn!("Metadata bootstrap will retry without changing its lease or incarnation: {error}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(error) => return Err(format!("metadata bootstrap: {error}")),
                }
            };
            info!(
                "Membership registered: node={} epoch={} incarnation={} endpoint={}",
                member.node_id, member.epoch, member.owner.incarnation, member.owner.endpoint
            );
            membership.inventory = InventoryRuntime::new(
                format.clone(),
                member.clone(),
                inventory.clone(),
                index.clone(),
                membership.view.clone(),
            );

            let mut lease_client = membership.client.clone();
            let view = Arc::clone(&membership.view);
            let mut stop = membership.stop.subscribe();
            membership.tasks.push(tokio::spawn(async move {
                tokio::select! {
                    _ = stop.changed() => {}
                    _ = maintain_lease(&mut lease_client, lease, ttl, cluster_id, &view) => {}
                }
                view.fence();
            }));
            let client = membership.client.clone();
            let view = Arc::clone(&membership.view);
            let mut stop = membership.stop.subscribe();
            let watch_prefix = prefix.clone();
            let watch_member = member.clone();
            let watch_inventory = membership.inventory.clone();
            membership.tasks.push(tokio::spawn(async move {
                tokio::select! {
                    _ = stop.changed() => {}
                    _ = watch::run(client, watch_prefix, cluster_id, format, watch_member, view.clone(), watch_inventory.clone()) => {}
                }
                view.invalidate_snapshot();
                watch_inventory.membership_unavailable();
            }));
            let stop = membership.stop.subscribe();
            let followers = membership.inventory.clone();
            membership.tasks.push(tokio::spawn(async move {
                inventory::follow_members(followers, stop).await;
            }));
            Ok(())
        }
        .await;
        if let Err(error) = result {
            membership.shutdown().await;
            return Err(error);
        }
        Ok(membership)
    }

    pub(crate) fn inventory(&self) -> InventoryRuntime {
        self.inventory.clone()
    }

    pub(crate) async fn shutdown(mut self) {
        self.view.fence();
        let _ = self.stop.send(true);
        for task in self.tasks.drain(..) {
            let _ = task.await;
        }
        if let Err(error) = rpc(self.client.lease_revoke(self.lease)).await {
            warn!("Membership revoke failed; registration will expire: {error}");
        }
    }
}

fn membership_owner(view: &MembershipView) -> CacheOwner {
    view.owner().clone()
}

impl Drop for Cluster {
    fn drop(&mut self) {
        self.view.fence();
        let _ = self.stop.send(true);
        for task in &self.tasks {
            task.abort();
        }
    }
}

async fn maintain_lease(
    client: &mut Client,
    lease: i64,
    ttl: i64,
    expected_cluster: u64,
    view: &MembershipView,
) {
    loop {
        if !view.registration_valid() {
            warn!("Membership expired or was fenced; restart this Manager for a new incarnation");
            return;
        }
        let result = async {
            let (mut keeper, mut stream) = rpc(client.lease_keep_alive(lease)).await?;
            loop {
                let sent_at = Instant::now();
                rpc(keeper.keep_alive()).await?;
                let response = rpc(stream.message()).await?.ok_or("lease stream closed")?;
                if cluster_id(response.header())? != expected_cluster || response.id() != lease {
                    view.fence();
                    return Err("coordinator cluster or lease identity changed".to_string());
                }
                if response.ttl() <= 0
                    || !view.renew(sent_at, Duration::from_secs(response.ttl() as u64))
                {
                    view.fence();
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_secs(ttl as u64 / 6)).await;
            }
        }
        .await;
        match result {
            Ok(()) => {
                warn!(
                    "Membership lease expired; restart this Manager to register a new incarnation"
                );
                return;
            }
            Err(error) => warn!("Membership keepalive failed: {error}"),
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

pub(crate) fn parse_label(value: &str) -> Result<String, String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(
            "cluster and node IDs must be 1..=128 ASCII letters, digits, '_', '-' or '.'".into(),
        );
    }
    Ok(value.into())
}

fn cluster_id(header: Option<&ResponseHeader>) -> Result<u64, String> {
    header
        .map(ResponseHeader::cluster_id)
        .filter(|id| *id != 0)
        .ok_or_else(|| "missing coordinator cluster ID".into())
}

async fn rpc<T>(request: impl Future<Output = Result<T, etcd_client::Error>>) -> Result<T, String> {
    tokio::time::timeout(RPC_TIMEOUT, request)
        .await
        .map_err(|_| "coordinator request timed out".to_string())?
        .map_err(|error| error.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/cluster/mod.rs"]
mod tests;
