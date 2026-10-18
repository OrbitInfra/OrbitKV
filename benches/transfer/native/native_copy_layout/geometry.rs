use std::collections::HashSet;

use serde::Serialize;

use super::CopyDesc;

pub(super) const LAYERS: usize = 64;
pub(super) const FRAGMENT_BYTES: usize = 65_536;
pub(super) const GUARD: usize = 32;
pub(super) const SLOT_BYTES: usize = FRAGMENT_BYTES + 2 * GUARD;

#[derive(Clone)]
pub(super) struct Layout {
    pub(super) name: &'static str,
    pub(super) host_order: Vec<usize>,
    pub(super) split_host: bool,
    pub(super) expected_dma_ranges: usize,
    pub(super) observed_batches: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct CopyRange {
    pub(super) layer: usize,
    pub(super) device_offset: usize,
    pub(super) host_registration: usize,
    pub(super) host_offset: usize,
    pub(super) bytes: usize,
}

pub(super) fn layouts() -> Vec<Layout> {
    let mut layouts = Vec::new();
    for (name, groups, observed_batches) in [
        ("d64-r64", vec![1], 27),
        ("d192-r64", vec![3], 1),
        ("d256-r192", vec![2, 1, 1], 1),
        ("d448-r192", vec![3, 2, 2], 1),
        ("d448-r448", vec![1; 7], 1),
        ("d512-r192", vec![3, 3, 2], 1),
        ("d512-r512", vec![1; 8], 1),
        ("d640-r320", vec![2; 5], 1),
    ] {
        let mut end = groups.iter().sum::<usize>();
        let mut host_order = Vec::with_capacity(end);
        for &length in &groups {
            let start = end - length;
            host_order.extend(start..end);
            end = start;
        }
        layouts.push(Layout {
            name,
            host_order,
            split_host: false,
            expected_dma_ranges: groups.len() * LAYERS,
            observed_batches,
        });
    }
    layouts.push(Layout {
        name: "host-registration-split",
        host_order: (0..4).collect(),
        split_host: true,
        expected_dma_ranges: 4 * LAYERS,
        observed_batches: 0,
    });
    layouts
}

impl Layout {
    pub(super) fn pages(&self) -> usize {
        self.host_order.len()
    }

    pub(super) fn host_lengths(&self) -> Vec<usize> {
        if self.split_host {
            vec![2 * GUARD + LAYERS * SLOT_BYTES; self.pages()]
        } else {
            vec![2 * GUARD + LAYERS * self.pages() * SLOT_BYTES]
        }
    }

    pub(super) fn device_lengths(&self) -> Vec<usize> {
        vec![2 * GUARD + self.pages() * SLOT_BYTES; LAYERS]
    }

    pub(super) fn ranges(&self) -> Result<Vec<CopyRange>, String> {
        let pages = self.pages();
        if pages == 0 || pages > 10 {
            return Err("layout requires one to ten pages per layer".into());
        }
        let distinct: HashSet<_> = self.host_order.iter().copied().collect();
        if distinct.len() != pages || distinct.iter().any(|&page| page >= pages) {
            return Err("host order must be a complete unique page permutation".into());
        }
        let mut ranges = Vec::with_capacity(LAYERS * pages);
        for layer in 0..LAYERS {
            for (page, &host_page) in self.host_order.iter().enumerate() {
                ranges.push(CopyRange {
                    layer,
                    device_offset: GUARD + page * SLOT_BYTES,
                    host_registration: if self.split_host { host_page } else { 0 },
                    host_offset: GUARD
                        + (if self.split_host {
                            layer
                        } else {
                            host_page * LAYERS + layer
                        }) * SLOT_BYTES,
                    bytes: FRAGMENT_BYTES,
                });
            }
        }
        let device_lengths = self.device_lengths();
        let host_lengths = self.host_lengths();
        let mut device_targets = HashSet::new();
        let mut host_targets = HashSet::new();
        for range in &ranges {
            if range.device_offset + range.bytes > device_lengths[range.layer]
                || range.host_offset + range.bytes > host_lengths[range.host_registration]
                || !device_targets.insert((range.layer, range.device_offset))
                || !host_targets.insert((range.host_registration, range.host_offset))
            {
                return Err("layout has an out-of-bounds or repeated fragment".into());
            }
        }
        Ok(ranges)
    }

    pub(super) fn bind(
        &self,
        devices: &[u64],
        hosts: &[(*mut u8, u64)],
    ) -> Result<Vec<CopyDesc>, String> {
        if devices.len() != LAYERS || hosts.len() != self.host_lengths().len() {
            return Err("physical allocation count differs from the layout".into());
        }
        let device_ids: HashSet<_> = devices.iter().copied().collect();
        let host_ids: HashSet<_> = hosts.iter().map(|(host, _)| *host as usize).collect();
        if device_ids.len() != LAYERS
            || device_ids.contains(&0)
            || host_ids.len() != hosts.len()
            || host_ids.contains(&0)
            || hosts.iter().any(|(_, mapped)| *mapped == 0)
        {
            return Err("physical allocations must have distinct nonzero addresses".into());
        }
        let mut copies = Vec::with_capacity(LAYERS * self.pages());
        for range in self.ranges()? {
            let device = devices[range.layer]
                .checked_add(range.device_offset as u64)
                .ok_or("device address overflow")?;
            let (host, mapped) = hosts[range.host_registration];
            let host_address = (host as usize)
                .checked_add(range.host_offset)
                .ok_or("host address overflow")?;
            let host_device = mapped
                .checked_add(range.host_offset as u64)
                .ok_or("mapped host address overflow")?;
            if device.checked_add(range.bytes as u64).is_none()
                || host_address.checked_add(range.bytes).is_none()
                || host_device.checked_add(range.bytes as u64).is_none()
            {
                return Err("fragment end address overflow".into());
            }
            copies.push(CopyDesc {
                device,
                host: host_address as *mut u8,
                host_device,
                size: range.bytes,
                device_allocation: devices[range.layer] as usize,
                host_registration: host as usize,
            });
        }
        copies.sort_unstable_by_key(|copy| copy.device);
        Ok(copies)
    }

    pub(super) fn virtual_copies(&self) -> Result<Vec<CopyDesc>, String> {
        let devices: Vec<_> = (0..LAYERS)
            .map(|layer| 0x1_0000_0000 + layer as u64 * 0x1000_0000)
            .collect();
        let hosts: Vec<_> = (0..self.host_lengths().len())
            .map(|registration| {
                let address = 0x10_0000_0000 + registration as u64 * 0x1000_0000;
                (address as *mut u8, address + 0x10_0000_0000)
            })
            .collect();
        self.bind(&devices, &hosts)
    }
}
