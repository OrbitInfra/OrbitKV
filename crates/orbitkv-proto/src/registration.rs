use uuid::Uuid;

use crate::proto::engine::CudaIpcTensorRegistration;

impl CudaIpcTensorRegistration {
    /// Validate the wire geometry before any importer or CUDA operation.
    pub fn validate(&self, expected_device_uuid: &str) -> Result<u64, String> {
        if self.format_version != 1 || self.allocation_kind != "torch-cuda-ipc-storage" {
            return Err("unsupported CUDA registration format or allocation kind".into());
        }
        let id = self
            .device_uuid
            .strip_prefix("GPU-")
            .and_then(|id| Uuid::parse_str(id).ok());
        if id.is_none_or(|id| format!("GPU-{id}") != self.device_uuid)
            || self.device_uuid != expected_device_uuid
        {
            return Err("CUDA registration GPU UUID is invalid or differs from its owner".into());
        }
        if self.allocation_handle.len() != 64
            || self.export_owner_generation.len() != 16
            || self.export_owner_generation.iter().all(|byte| *byte == 0)
            || self.ref_counter_handle.is_empty()
            || self.ref_counter_handle.len() > 255
            || self.ref_counter_handle.contains(&0)
            || self.ref_counter_offset > usize::MAX as u64
            || (!self.event_handle.is_empty() && self.event_handle.len() != 64)
            || (self.event_sync_required && self.event_handle.len() != 64)
        {
            return Err("invalid CUDA IPC handle, exporter generation, counter or event".into());
        }
        self.validate_layout()
    }

    pub fn validate_layout(&self) -> Result<u64, String> {
        let width = match self.dtype.as_str() {
            "uint8" | "int8" | "bool" | "float8_e4m3fn" | "float8_e5m2" => 1,
            "float16" | "bfloat16" | "int16" => 2,
            "float32" | "int32" => 4,
            "float64" | "int64" => 8,
            _ => return Err("unsupported CUDA registration dtype".into()),
        };
        if self.element_size != width
            || self.storage_size_bytes == 0
            || !self.view_offset_bytes.is_multiple_of(width)
            || self.shape.is_empty()
            || self.shape.len() > 8
            || self.shape.len() != self.strides.len()
            || self.shape.contains(&0)
            || self.strides.contains(&0)
        {
            return Err("invalid CUDA registration dtype, storage or tensor layout".into());
        }
        let mut elements = 1_u64;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides) {
            elements = dimension
                .checked_sub(1)
                .and_then(|last| last.checked_mul(stride))
                .and_then(|extent| elements.checked_add(extent))
                .ok_or("CUDA tensor layout overflow")?;
        }
        let view_size = elements
            .checked_mul(width)
            .ok_or("CUDA tensor byte extent overflow")?;
        if self
            .view_offset_bytes
            .checked_add(view_size)
            .is_none_or(|end| end > self.storage_size_bytes)
            || self
                .storage_allocation_offset_bytes
                .checked_add(self.storage_size_bytes)
                .is_none_or(|end| end > usize::MAX as u64)
        {
            return Err("CUDA tensor view exceeds its exported storage extent".into());
        }
        Ok(view_size)
    }
}

#[cfg(test)]
#[path = "../tests/unit/registration.rs"]
mod tests;
