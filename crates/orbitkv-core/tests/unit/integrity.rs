use super::*;

/// Each counter control runs alone because the production meter is process-global.
pub(crate) fn isolate(name: &str) -> Option<prometheus::Registry> {
    if std::env::var("ORBITKV_CRC_CONTROL").as_deref() != Ok(name) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--ignored", "--nocapture"])
            .env("ORBITKV_CRC_CONTROL", name)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let registry = prometheus::Registry::new();
    let reader = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .unwrap();
    opentelemetry::global::set_meter_provider(
        opentelemetry_sdk::metrics::SdkMeterProvider::builder()
            .with_reader(reader)
            .build(),
    );
    Some(registry)
}

pub(crate) fn bytes(registry: &prometheus::Registry, tier: &str) -> u64 {
    registry
        .gather()
        .iter()
        .filter(|family| family.name() == "orbitkv_integrity_bytes_total")
        .flat_map(|family| family.get_metric())
        .filter(|metric| {
            metric
                .get_label()
                .iter()
                .any(|label| label.name() == "tier" && label.value() == tier)
        })
        .map(|metric| metric.get_counter().value() as u64)
        .sum()
}

#[test]
fn layout_ranges_and_padding_combine_without_reading_payload_again() {
    let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let mut references = vec![];
    for (start, end) in [(0, 137), (137, 512), (512, 1024), (1024, 4096)] {
        references.push(Checksum {
            end: end as u64,
            crc: crc32fast::hash(&data[start..end]),
        });
    }
    for (start, end) in [(0, 137), (137, 512), (512, 4096), (0, 4096)] {
        assert_eq!(
            range(&references, start as u64, (end - start) as u64).unwrap(),
            crc32fast::hash(&data[start..end])
        );
    }
    assert!(range(&references, 1, 511).is_err());
    for length in [0, 1, 511, 4096, 1 << 20] {
        assert_eq!(zeros(length), crc32fast::hash(&vec![0; length as usize]));
    }
    append_zeros(&mut references, 8192, &[4608]);
    let mut padded = data;
    padded.resize(8192, 0);
    assert_eq!(
        range(&references, 0, 8192).unwrap(),
        crc32fast::hash(&padded)
    );
    validate(&references, 8192).unwrap();
    assert!(from_wire(b"[]", 8192).is_err());
    assert!(from_wire(&vec![0; (1 << 20) + 1], 8192).is_err());
}
