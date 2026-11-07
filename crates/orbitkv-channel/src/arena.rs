use std::fs::File;
use std::os::fd::OwnedFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};
use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, ftruncate, memfd_create};
use thiserror::Error;

use crate::DescriptorRef;

const ARENA_MAGIC: u32 = 0x4f52_4241; // ORBA
const ARENA_VERSION: u16 = 1;
const ARENA_HEADER_BYTES: usize = 4096;
const SLOT_HEADER_BYTES: usize = 16;

pub const DEFAULT_ARENA_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_SLOT_CAPACITY: usize = 64 * 1024;

#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("descriptor arena size {arena_bytes} cannot hold a {slot_capacity}-byte slot")]
    TooSmall {
        arena_bytes: usize,
        slot_capacity: usize,
    },
    #[error("descriptor arena field {field} exceeds its wire representation")]
    FieldOverflow { field: &'static str },
    #[error("descriptor arena system operation failed: {0}")]
    System(#[from] std::io::Error),
    #[error("invalid descriptor arena magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported descriptor arena version: {0}")]
    UnsupportedVersion(u16),
    #[error("descriptor arena session mismatch: expected {expected}, got {actual}")]
    SessionMismatch { expected: u64, actual: u64 },
    #[error("descriptor slot {slot} is out of range (slot_count={slot_count})")]
    SlotOutOfRange { slot: usize, slot_count: usize },
    #[error("descriptor offset {offset} does not identify a slot payload")]
    InvalidOffset { offset: u64 },
    #[error("descriptor payload length {len} exceeds slot capacity {capacity}")]
    PayloadTooLarge { len: usize, capacity: usize },
    #[error("descriptor generation must be non-zero")]
    ZeroGeneration,
    #[error("stale descriptor generation: expected {expected}, got {actual}")]
    StaleGeneration { expected: u64, actual: u64 },
    #[error("descriptor length mismatch: reference={expected}, slot={actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("descriptor arena lock poisoned")]
    Poisoned,
    #[error("descriptor arena changed during a read: expected generation {expected}, got {actual}")]
    ConcurrentWrite { expected: u64, actual: u64 },
}

pub struct DescriptorArena {
    file: File,
    map: Mutex<MmapMut>,
    arena_bytes: usize,
    session_epoch: u64,
    slot_capacity: usize,
    slot_stride: usize,
    slot_count: usize,
}

impl DescriptorArena {
    pub fn create(
        session_epoch: u64,
        arena_bytes: usize,
        slot_capacity: usize,
    ) -> Result<Self, ArenaError> {
        if session_epoch == 0 {
            return Err(ArenaError::ZeroGeneration);
        }
        let layout = ArenaLayout::new(arena_bytes, slot_capacity)?;
        let fd = memfd_create(
            "orbitkv-descriptor-arena",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(std::io::Error::from)?;
        ftruncate(&fd, arena_bytes as u64).map_err(std::io::Error::from)?;
        fcntl_add_seals(&fd, SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL)
            .map_err(std::io::Error::from)?;
        let file = File::from(fd);
        let mut map = map_file(&file, arena_bytes)?;
        map.fill(0);
        write_u32(&mut map, 0, ARENA_MAGIC);
        write_u16(&mut map, 4, ARENA_VERSION);
        write_u64(&mut map, 8, session_epoch);
        write_u64(&mut map, 16, arena_bytes as u64);
        write_u32(
            &mut map,
            24,
            u32::try_from(slot_capacity).map_err(|_| ArenaError::FieldOverflow {
                field: "slot_capacity",
            })?,
        );
        write_u32(
            &mut map,
            28,
            u32::try_from(layout.slot_count).map_err(|_| ArenaError::FieldOverflow {
                field: "slot_count",
            })?,
        );
        Ok(Self {
            file,
            map: Mutex::new(map),
            arena_bytes,
            session_epoch,
            slot_capacity,
            slot_stride: layout.slot_stride,
            slot_count: layout.slot_count,
        })
    }

    pub fn open(
        fd: OwnedFd,
        expected_session_epoch: u64,
        arena_bytes: usize,
    ) -> Result<Self, ArenaError> {
        let file = File::from(fd);
        let map = map_file(&file, arena_bytes)?;
        let magic = read_u32(&map, 0)?;
        if magic != ARENA_MAGIC {
            return Err(ArenaError::InvalidMagic(magic));
        }
        let version = read_u16(&map, 4)?;
        if version != ARENA_VERSION {
            return Err(ArenaError::UnsupportedVersion(version));
        }
        let session_epoch = read_u64(&map, 8)?;
        if session_epoch != expected_session_epoch {
            return Err(ArenaError::SessionMismatch {
                expected: expected_session_epoch,
                actual: session_epoch,
            });
        }
        let encoded_bytes =
            usize::try_from(read_u64(&map, 16)?).map_err(|_| ArenaError::FieldOverflow {
                field: "arena_bytes",
            })?;
        if encoded_bytes != arena_bytes {
            return Err(ArenaError::LengthMismatch {
                expected: arena_bytes,
                actual: encoded_bytes,
            });
        }
        let slot_capacity = read_u32(&map, 24)? as usize;
        let layout = ArenaLayout::new(arena_bytes, slot_capacity)?;
        let encoded_slots = read_u32(&map, 28)? as usize;
        if encoded_slots != layout.slot_count {
            return Err(ArenaError::LengthMismatch {
                expected: layout.slot_count,
                actual: encoded_slots,
            });
        }
        Ok(Self {
            file,
            map: Mutex::new(map),
            arena_bytes,
            session_epoch,
            slot_capacity,
            slot_stride: layout.slot_stride,
            slot_count: layout.slot_count,
        })
    }

    pub fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    pub fn len(&self) -> usize {
        self.arena_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn slot_offset(&self, slot: usize) -> Result<u64, ArenaError> {
        if slot >= self.slot_count {
            return Err(ArenaError::SlotOutOfRange {
                slot,
                slot_count: self.slot_count,
            });
        }
        let offset = ARENA_HEADER_BYTES + slot * self.slot_stride + SLOT_HEADER_BYTES;
        u64::try_from(offset).map_err(|_| ArenaError::FieldOverflow {
            field: "slot_offset",
        })
    }

    pub fn write_slot(
        &self,
        slot: usize,
        generation: u64,
        payload: &[u8],
    ) -> Result<DescriptorRef, ArenaError> {
        let offset = self.slot_offset(slot)?;
        self.write_at(offset, generation, payload)
    }

    pub fn write_response(
        &self,
        request: DescriptorRef,
        payload: &[u8],
    ) -> Result<DescriptorRef, ArenaError> {
        let generation = request
            .generation
            .checked_add(1)
            .ok_or(ArenaError::FieldOverflow {
                field: "generation",
            })?;
        self.write_at(request.offset, generation, payload)
    }

    pub(crate) fn reset_slot(&self, slot: usize, generation: u64) -> Result<(), ArenaError> {
        let offset = self.slot_offset(slot)?;
        let (slot_header, _) = self.validate_offset(offset)?;
        let mut map = self.map.lock().map_err(|_| ArenaError::Poisoned)?;
        write_u32(&mut map, slot_header + 8, 0);
        store_generation(&mut map, slot_header, generation);
        Ok(())
    }

    pub(crate) fn slot_generation(&self, slot: usize) -> Result<u64, ArenaError> {
        let offset = self.slot_offset(slot)?;
        let (slot_header, _) = self.validate_offset(offset)?;
        let map = self.map.lock().map_err(|_| ArenaError::Poisoned)?;
        Ok(load_generation(&map, slot_header))
    }

    pub fn read(&self, descriptor: DescriptorRef) -> Result<Vec<u8>, ArenaError> {
        let (slot_header, payload_offset) = self.validate_offset(descriptor.offset)?;
        let map = self.map.lock().map_err(|_| ArenaError::Poisoned)?;
        let actual_generation = load_generation(&map, slot_header);
        if actual_generation != descriptor.generation {
            return Err(ArenaError::StaleGeneration {
                expected: descriptor.generation,
                actual: actual_generation,
            });
        }
        let actual_len = read_u32(&map, slot_header + 8)? as usize;
        if actual_len != descriptor.len as usize {
            return Err(ArenaError::LengthMismatch {
                expected: descriptor.len as usize,
                actual: actual_len,
            });
        }
        if actual_len > self.slot_capacity {
            return Err(ArenaError::PayloadTooLarge {
                len: actual_len,
                capacity: self.slot_capacity,
            });
        }
        let payload = map[payload_offset..payload_offset + actual_len].to_vec();
        let final_generation = load_generation(&map, slot_header);
        if final_generation != actual_generation {
            return Err(ArenaError::ConcurrentWrite {
                expected: actual_generation,
                actual: final_generation,
            });
        }
        Ok(payload)
    }

    fn write_at(
        &self,
        offset: u64,
        generation: u64,
        payload: &[u8],
    ) -> Result<DescriptorRef, ArenaError> {
        if generation == 0 {
            return Err(ArenaError::ZeroGeneration);
        }
        if payload.len() > self.slot_capacity {
            return Err(ArenaError::PayloadTooLarge {
                len: payload.len(),
                capacity: self.slot_capacity,
            });
        }
        let (slot_header, payload_offset) = self.validate_offset(offset)?;
        let len = u32::try_from(payload.len()).map_err(|_| ArenaError::FieldOverflow {
            field: "payload_len",
        })?;
        let mut map = self.map.lock().map_err(|_| ArenaError::Poisoned)?;
        store_generation(&mut map, slot_header, 0);
        map[payload_offset..payload_offset + payload.len()].copy_from_slice(payload);
        write_u32(&mut map, slot_header + 8, len);
        store_generation(&mut map, slot_header, generation);
        Ok(DescriptorRef {
            offset,
            len,
            generation,
        })
    }

    fn validate_offset(&self, offset: u64) -> Result<(usize, usize), ArenaError> {
        let slot = self.descriptor_slot(offset)?;
        let payload_offset = ARENA_HEADER_BYTES + slot * self.slot_stride + SLOT_HEADER_BYTES;
        Ok((payload_offset - SLOT_HEADER_BYTES, payload_offset))
    }

    pub(crate) fn descriptor_slot(&self, offset: u64) -> Result<usize, ArenaError> {
        let payload_offset = usize::try_from(offset).map_err(|_| ArenaError::FieldOverflow {
            field: "descriptor_offset",
        })?;
        let relative = payload_offset
            .checked_sub(ARENA_HEADER_BYTES + SLOT_HEADER_BYTES)
            .ok_or(ArenaError::InvalidOffset { offset })?;
        if !relative.is_multiple_of(self.slot_stride) {
            return Err(ArenaError::InvalidOffset { offset });
        }
        let slot = relative / self.slot_stride;
        if slot >= self.slot_count {
            return Err(ArenaError::InvalidOffset { offset });
        }
        Ok(slot)
    }

    pub fn file(&self) -> &File {
        &self.file
    }
}

struct ArenaLayout {
    slot_stride: usize,
    slot_count: usize,
}

impl ArenaLayout {
    fn new(arena_bytes: usize, slot_capacity: usize) -> Result<Self, ArenaError> {
        let unaligned_stride =
            SLOT_HEADER_BYTES
                .checked_add(slot_capacity)
                .ok_or(ArenaError::FieldOverflow {
                    field: "slot_stride",
                })?;
        let slot_stride = unaligned_stride
            .checked_add(std::mem::align_of::<AtomicU64>() - 1)
            .ok_or(ArenaError::FieldOverflow {
                field: "slot_stride",
            })?
            & !(std::mem::align_of::<AtomicU64>() - 1);
        let usable = arena_bytes.saturating_sub(ARENA_HEADER_BYTES);
        let slot_count = usable / slot_stride;
        if slot_capacity == 0 || slot_count == 0 {
            return Err(ArenaError::TooSmall {
                arena_bytes,
                slot_capacity,
            });
        }
        Ok(Self {
            slot_stride,
            slot_count,
        })
    }
}

fn map_file(file: &File, len: usize) -> Result<MmapMut, ArenaError> {
    // SAFETY: the memfd is kept alive by DescriptorArena for the full mapping
    // lifetime, its length is fixed before mapping, and no code truncates it.
    unsafe { MmapOptions::new().len(len).map_mut(file) }.map_err(ArenaError::System)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArenaError> {
    let value = bytes
        .get(offset..)
        .and_then(|bytes| bytes.first_chunk::<2>())
        .ok_or(ArenaError::InvalidOffset {
            offset: offset as u64,
        })?;
    Ok(u16::from_le_bytes(*value))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ArenaError> {
    let value = bytes
        .get(offset..)
        .and_then(|bytes| bytes.first_chunk::<4>())
        .ok_or(ArenaError::InvalidOffset {
            offset: offset as u64,
        })?;
    Ok(u32::from_le_bytes(*value))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ArenaError> {
    let value = bytes
        .get(offset..)
        .and_then(|bytes| bytes.first_chunk::<8>())
        .ok_or(ArenaError::InvalidOffset {
            offset: offset as u64,
        })?;
    Ok(u64::from_le_bytes(*value))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn load_generation(bytes: &[u8], offset: usize) -> u64 {
    let address = bytes.as_ptr().wrapping_add(offset);
    debug_assert_eq!(address.align_offset(std::mem::align_of::<AtomicU64>()), 0);
    // SAFETY: arena slot headers are 8-byte aligned, the mapping covers the
    // entire AtomicU64, and every cross-process generation access is atomic.
    unsafe { &*address.cast::<AtomicU64>() }.load(Ordering::Acquire)
}

fn store_generation(bytes: &mut [u8], offset: usize, value: u64) {
    let address = bytes.as_mut_ptr().wrapping_add(offset);
    debug_assert_eq!(address.align_offset(std::mem::align_of::<AtomicU64>()), 0);
    // SAFETY: see load_generation. Release publishes payload and length writes
    // before the new generation becomes visible to the peer process.
    unsafe { &*address.cast::<AtomicU64>() }.store(value, Ordering::Release);
}

#[cfg(test)]
#[path = "../tests/unit/arena.rs"]
mod tests;
