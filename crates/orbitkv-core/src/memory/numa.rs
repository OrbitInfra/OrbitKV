// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA (Non-Uniform Memory Access) utilities
//!
//! This module provides:
//! - NUMA node abstraction (`NumaNode`)
//! - System NUMA topology detection (CPU-to-node mapping)
//! - GPU to NUMA node affinity detection
//! - Thread-to-NUMA-node pinning for first-touch allocation policy

use std::collections::HashMap;
use std::fs;
use std::mem;
use std::process::Command;

use cudarc::driver::result;
use uuid::Uuid;

/// Represents a NUMA node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNode(pub u32);

impl NumaNode {
    /// Represents an unknown or invalid NUMA node
    pub const UNKNOWN: NumaNode = NumaNode(u32::MAX);

    /// Check if this is the unknown node
    pub fn is_unknown(&self) -> bool {
        self.0 == u32::MAX
    }

    /// Check if this is a valid NUMA node
    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl std::fmt::Display for NumaNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "UNKNOWN")
        } else {
            write!(f, "NUMA{}", self.0)
        }
    }
}

// ============================================================================
// CPU Topology from sysfs
// ============================================================================

/// Read CPU-to-NUMA mapping from sysfs
///
/// Returns a map of NUMA node ID -> list of CPU IDs.
fn read_cpu_topology_from_sysfs() -> Result<HashMap<u32, Vec<usize>>, String> {
    let mut node_to_cpus: HashMap<u32, Vec<usize>> = HashMap::new();

    let node_dir = std::path::Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return Err("NUMA not supported: /sys/devices/system/node not found".to_string());
    }

    let entries =
        fs::read_dir(node_dir).map_err(|e| format!("Failed to read node directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if !name.starts_with("node") {
            continue;
        }

        let node_id: u32 = name[4..]
            .parse()
            .map_err(|_| format!("Invalid node directory name: {}", name))?;

        let cpulist_path = path.join("cpulist");
        if !cpulist_path.exists() {
            continue;
        }

        let cpulist = fs::read_to_string(&cpulist_path)
            .map_err(|e| format!("Failed to read {}: {}", cpulist_path.display(), e))?;

        let cpus = parse_cpulist(cpulist.trim())?;
        node_to_cpus.insert(node_id, cpus);
    }

    if node_to_cpus.is_empty() {
        return Err("No NUMA nodes found".to_string());
    }

    Ok(node_to_cpus)
}

/// Parse Linux cpulist format (e.g. "0-3,8-11" -> [0,1,2,3,8,9,10,11])
fn parse_cpulist(cpulist: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();

    if cpulist.is_empty() {
        return Ok(cpus);
    }

    for part in cpulist.split(',') {
        if part.contains('-') {
            let range: Vec<&str> = part.split('-').collect();
            if range.len() != 2 {
                return Err(format!("Invalid CPU range format: {}", part));
            }

            let start: usize = range[0]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[0]))?;
            let end: usize = range[1]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[1]))?;

            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            let cpu: usize = part
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", part))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

// ============================================================================
// Thread pinning
// ============================================================================

/// Pin the current thread to CPUs on a specific NUMA node.
///
/// Sets the CPU affinity of the calling thread to only run on CPUs
/// belonging to the specified NUMA node. Critical for ensuring
/// first-touch memory allocations land on the correct node.
pub(crate) fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    if node.is_unknown() {
        return Err("Cannot pin to unknown NUMA node".to_string());
    }

    let node_to_cpus = read_cpu_topology_from_sysfs()
        .map_err(|e| format!("Failed to get NUMA topology: {}", e))?;

    let cpus = node_to_cpus
        .get(&node.0)
        .ok_or_else(|| format!("No CPUs found for NUMA node {}", node.0))?;

    if cpus.is_empty() {
        return Err(format!("CPU list is empty for NUMA node {}", node.0));
    }

    // SAFETY: cpu_set_t is a plain C struct safe to zero-initialize. CPU_SET
    // writes to valid indices. sched_setaffinity(0, ...) targets the calling
    // thread; the cpu_set is correctly sized.
    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();

        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut cpu_set);
        }

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(format!("sched_setaffinity failed: {}", err));
        }
    }

    Ok(())
}

/// Run a closure on a thread pinned to a specific NUMA node.
///
/// Spawns a temporary thread, pins it to the specified NUMA node,
/// runs the closure, and returns the result. Useful for first-touch
/// memory allocation policy.
pub(crate) fn run_on_numa<T, F>(node: NumaNode, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if node.is_unknown() {
        return Err("Cannot run on unknown NUMA node".to_string());
    }

    let (tx, rx) = std::sync::mpsc::channel();

    let handle = std::thread::Builder::new()
        .name(format!("numa{}-init", node.0))
        .spawn(move || {
            if let Err(e) = pin_thread_to_numa_node(node) {
                let _ = tx.send(Err(e));
                return;
            }

            let result = f();
            let _ = tx.send(Ok(result));
        })
        .map_err(|e| format!("Failed to spawn NUMA thread: {}", e))?;

    let result = rx
        .recv()
        .map_err(|_| "NUMA thread panicked or closed channel".to_string())?;

    handle
        .join()
        .map_err(|_| "NUMA thread panicked".to_string())?;

    result
}

// ============================================================================
// GPU NUMA affinity
// ============================================================================

/// Get the NUMA node for a GPU device via nvidia-smi.
///
/// Returns `NumaNode::UNKNOWN` if nvidia-smi is unavailable or fails.
fn get_device_numa_node(device_uuid: &str) -> NumaNode {
    let output = match Command::new("nvidia-smi")
        .args(["topo", "--get-numa-id-of-nearby-cpu", "-i", device_uuid])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            return NumaNode::UNKNOWN;
        }
    };

    if let Ok(stdout) = std::str::from_utf8(&output.stdout) {
        return parse_closest_cpu_numa_node(stdout);
    }

    NumaNode::UNKNOWN
}

fn parse_closest_cpu_numa_node(output: &str) -> NumaNode {
    output
        .lines()
        .next()
        .and_then(|line| line.split(':').nth(1))
        .and_then(|numa_ids| numa_ids.split(',').next())
        .and_then(|numa_id| numa_id.trim().parse::<u32>().ok())
        .map(NumaNode)
        .unwrap_or(NumaNode::UNKNOWN)
}

/// Get NUMA affinity for CUDA-visible ordinals through their stable UUIDs.
///
/// Returns (device_id, numa_node) pairs. Empty if nvidia-smi is unavailable.
fn get_gpu_numa_affinity() -> Vec<(i32, NumaNode)> {
    let visible = (|| {
        result::init()?;
        (0..result::device::get_count()?)
            .map(|ordinal| {
                let device = result::device::get(ordinal)?;
                let uuid = result::device::get_uuid(device)?;
                Ok((
                    ordinal,
                    format!("GPU-{}", Uuid::from_bytes(uuid.bytes.map(|b| b as u8))),
                ))
            })
            .collect::<Result<Vec<_>, result::DriverError>>()
    })();
    match visible {
        Ok(devices) => gpu_numa_affinity(&devices, get_device_numa_node),
        Err(error) => {
            log::warn!("Cannot enumerate CUDA-visible GPU NUMA affinity: {error}");
            Vec::new()
        }
    }
}

fn gpu_numa_affinity(
    visible_devices: &[(i32, String)],
    mut lookup: impl FnMut(&str) -> NumaNode,
) -> Vec<(i32, NumaNode)> {
    visible_devices
        .iter()
        .map(|(ordinal, uuid)| {
            let node = lookup(uuid);
            log::info!("CUDA device {ordinal} ({uuid}) -> {node}");
            (*ordinal, node)
        })
        .collect()
}

// ============================================================================
// NumaTopology
// ============================================================================

/// GPU-to-NUMA topology for the system.
///
/// Built once during engine initialization. Provides efficient lookup
/// of NUMA affinity for GPU devices.
#[derive(Debug, Clone)]
pub(crate) struct NumaTopology {
    gpu_numa_map: HashMap<i32, NumaNode>,
    numa_nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Detect and build the GPU-NUMA topology.
    ///
    /// Queries nvidia-smi for GPU NUMA affinity and reads system NUMA topology.
    pub(crate) fn detect() -> Self {
        let gpu_affinity = get_gpu_numa_affinity();
        let gpu_numa_map: HashMap<i32, NumaNode> = gpu_affinity.into_iter().collect();

        let numa_nodes = match read_cpu_topology_from_sysfs() {
            Ok(node_to_cpus) => {
                let mut node_ids: Vec<u32> = node_to_cpus.keys().copied().collect();
                node_ids.sort_unstable();
                node_ids.into_iter().map(NumaNode).collect()
            }
            Err(_) => {
                vec![NumaNode(0)]
            }
        };

        Self {
            gpu_numa_map,
            numa_nodes,
        }
    }

    /// Get the preferred NUMA node for a GPU device.
    ///
    /// Returns `NumaNode::UNKNOWN` if the device is not found.
    pub(crate) fn numa_for_gpu(&self, device_id: i32) -> NumaNode {
        self.gpu_numa_map
            .get(&device_id)
            .copied()
            .unwrap_or(NumaNode::UNKNOWN)
    }

    /// Get all NUMA nodes in the system.
    pub(crate) fn numa_nodes(&self) -> &[NumaNode] {
        &self.numa_nodes
    }

    /// Get all valid NUMA nodes that have at least one GPU attached.
    pub(crate) fn gpu_numa_nodes(&self) -> Vec<NumaNode> {
        let mut nodes: Vec<NumaNode> = self
            .gpu_numa_map
            .values()
            .copied()
            .filter(NumaNode::is_valid)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// Get the number of NUMA nodes.
    pub(crate) fn num_nodes(&self) -> usize {
        self.numa_nodes.len()
    }

    /// Check if this is a multi-NUMA system.
    pub(crate) fn is_multi_numa(&self) -> bool {
        self.numa_nodes.len() > 1
    }

    /// Log the detected topology.
    pub(crate) fn log_summary(&self) {
        log::info!("=== GPU-NUMA Topology ===");
        log::info!("NUMA nodes: {}", self.num_nodes());

        if self.gpu_numa_map.is_empty() {
            log::warn!("No GPU NUMA affinity detected (nvidia-smi unavailable?)");
        } else {
            let mut devices: Vec<_> = self.gpu_numa_map.iter().collect();
            devices.sort_by_key(|(dev, _)| *dev);
            for (dev, node) in devices {
                log::info!("  GPU {} -> {}", dev, node);
            }
        }
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/memory/numa.rs"]
mod tests;
