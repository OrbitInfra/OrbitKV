use super::*;
use prost::Message;

fn registration() -> CudaIpcTensorRegistration {
    CudaIpcTensorRegistration {
        format_version: 1,
        allocation_kind: "torch-cuda-ipc-storage".into(),
        allocation_handle: vec![1; 64],
        device_uuid: "GPU-c4930667-4d65-aae5-8205-fe785c3654be".into(),
        storage_size_bytes: 128,
        storage_allocation_offset_bytes: 512,
        view_offset_bytes: 16,
        shape: vec![2, 3],
        strides: vec![4, 1],
        dtype: "float16".into(),
        element_size: 2,
        ref_counter_handle: b"/torch-counter".to_vec(),
        ref_counter_offset: 3,
        event_handle: vec![2; 64],
        event_sync_required: true,
        export_owner_generation: vec![3; 16],
    }
}

#[test]
fn explicit_registration_preserves_storage_suballocation_and_strided_view() {
    let value = registration();
    assert_eq!(value.validate(&value.device_uuid), Ok(14));
    let decoded = CudaIpcTensorRegistration::decode(value.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, value);
    assert_eq!(decoded.storage_allocation_offset_bytes, 512);
    assert_eq!(decoded.view_offset_bytes, 16);
}

#[test]
fn malformed_registration_is_rejected_before_import() {
    for case in 0..15 {
        let mut value = registration();
        let expected = value.device_uuid.clone();
        match case {
            0 => value.format_version = 0,
            1 => value.allocation_kind = "unknown".into(),
            2 => value.allocation_handle.pop().map(|_| ()).unwrap(),
            3 => value.device_uuid = "GPU-foreign".into(),
            4 => value.export_owner_generation = vec![0; 16],
            5 => value.ref_counter_handle = Vec::new(),
            6 => value.event_handle = Vec::new(),
            7 => value.element_size = 8,
            8 => value.view_offset_bytes = 127,
            9 => value.strides = vec![u64::MAX, 1],
            10 => value.shape = vec![u64::MAX, 3],
            11 => value.storage_allocation_offset_bytes = u64::MAX,
            12 => value.ref_counter_offset = 10_000,
            13 => {
                value.shape = vec![1];
                value.strides = vec![u64::MAX];
            }
            14 => value.storage_allocation_offset_bytes = 1,
            _ => unreachable!(),
        }
        assert!(value.validate(&expected).is_err(), "case {case}");
    }
}
