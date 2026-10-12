//! Compile explicit rows within retained allocation and registration boundaries.

use super::CopyDesc;

/// One contiguous row or a run of explicit equal-width, constant-pitch rows.
/// Registration identities constrain CUDA access; the caller retains every
/// logical allocation covered by the original descriptors through stream drain.
#[derive(Debug)]
pub(crate) struct DmaCopy {
    pub(super) device: u64,
    pub(super) host: *mut u8,
    pub(super) width: usize,
    pub(super) rows: usize,
    pub(super) device_pitch: usize,
    pub(super) host_pitch: usize,
    pub(super) device_allocation: usize,
    pub(super) host_registration: usize,
}

fn contiguous_ranges(mut copies: &[CopyDesc]) -> impl Iterator<Item = DmaCopy> + '_ {
    std::iter::from_fn(move || {
        let (&start, remaining) = copies.split_first()?;
        copies = remaining;
        let mut width = start.size;
        while let Some((&next, remaining)) = copies.split_first() {
            if start.device_allocation != next.device_allocation
                || start.host_registration != next.host_registration
                || start.device.checked_add(width as u64) != Some(next.device)
                || (start.host as usize).checked_add(width) != Some(next.host as usize)
                || next.device.checked_add(next.size as u64).is_none()
                || (next.host as usize).checked_add(next.size).is_none()
            {
                break;
            }
            let Some(combined) = width.checked_add(next.size) else {
                break;
            };
            width = combined;
            copies = remaining;
        }
        Some(DmaCopy {
            device: start.device,
            host: start.host,
            width,
            rows: 1,
            device_pitch: width,
            host_pitch: width,
            device_allocation: start.device_allocation,
            host_registration: start.host_registration,
        })
    })
}

/// Every row comes from validated descriptors. Neither a missing row nor a
/// gap between rows can become part of the copy. Descending or overlapping
/// rows remain separate submissions, preserving their original order.
pub(crate) fn dma_copies(
    copies: &[CopyDesc],
    max_pitch: usize,
) -> impl Iterator<Item = DmaCopy> + '_ {
    let mut ranges = contiguous_ranges(copies).peekable();
    std::iter::from_fn(move || {
        let mut copy = ranges.next()?;
        let mut previous_device = copy.device;
        let mut previous_host = copy.host as usize;
        while let Some(next) = ranges.peek() {
            if next.width != copy.width
                || copy.width == 0
                || next.device_allocation != copy.device_allocation
                || next.host_registration != copy.host_registration
                || next.device.checked_add(next.width as u64).is_none()
                || (next.host as usize).checked_add(next.width).is_none()
            {
                break;
            }
            let Some(device_pitch) = next
                .device
                .checked_sub(previous_device)
                .and_then(|pitch| usize::try_from(pitch).ok())
            else {
                break;
            };
            let Some(host_pitch) = (next.host as usize).checked_sub(previous_host) else {
                break;
            };
            if device_pitch < copy.width
                || host_pitch < copy.width
                || device_pitch > max_pitch
                || host_pitch > max_pitch
                || (copy.rows > 1
                    && (device_pitch != copy.device_pitch || host_pitch != copy.host_pitch))
            {
                break;
            }
            copy.device_pitch = device_pitch;
            copy.host_pitch = host_pitch;
            copy.rows += 1;
            previous_device = next.device;
            previous_host = next.host as usize;
            ranges.next();
        }
        Some(copy)
    })
}
