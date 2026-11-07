use super::*;
use orbitkv_state::ReplicaRepresentation;
use std::sync::Arc;

fn key(n: u32) -> StateKey {
    StateKey::new("model".into(), n.to_le_bytes().to_vec())
}

fn scoped_key(namespace: &str, n: u32) -> StateKey {
    StateKey::new(namespace.into(), n.to_le_bytes().to_vec())
}
fn metadata(medium: ReplicaMedium) -> ReplicaMetadata {
    ReplicaMetadata {
        medium,
        representation: ReplicaRepresentation::Raw,
        stored_bytes: Some(1024),
    }
}

#[test]
fn media_have_independent_generations_and_deletions_keep_identity() {
    let inventory = ResidencyInventory::new(4096);
    let k = key(1);
    inventory
        .change(&k, ReplicaMedium::Dram, Some(metadata(ReplicaMedium::Dram)))
        .unwrap();
    inventory
        .change(&k, ReplicaMedium::Ssd, Some(metadata(ReplicaMedium::Ssd)))
        .unwrap();
    let records = inventory.page(None).unwrap();
    assert_eq!(records.len(), 2);
    let dram = records
        .iter()
        .find(|r| r.metadata.unwrap().medium == ReplicaMedium::Dram)
        .unwrap();
    let ssd = records
        .iter()
        .find(|r| r.metadata.unwrap().medium == ReplicaMedium::Ssd)
        .unwrap();
    assert!(inventory.contains_record(dram, ReplicaMedium::Dram));
    assert!(!inventory.contains_record(ssd, ReplicaMedium::Dram));
    inventory.change(&k, ReplicaMedium::Dram, None).unwrap();
    assert!(inventory.contains_record(ssd, ReplicaMedium::Ssd));
    let removed = inventory.changes(2, 3).unwrap();
    assert!(!removed[0].present);
    assert_eq!(removed[0].metadata.unwrap().medium, ReplicaMedium::Dram);
    inventory
        .change(&k, ReplicaMedium::Dram, Some(metadata(ReplicaMedium::Dram)))
        .unwrap();
    assert!(!inventory.contains_record(dram, ReplicaMedium::Dram));
}

#[test]
fn bounded_journal_reports_snapshot_gaps_and_batch_overflow() {
    let inventory = ResidencyInventory::new(300);
    for n in 0..30 {
        inventory
            .change(
                &key(n),
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
    }
    assert_eq!(
        inventory.changes(0, 30),
        Err(InventoryReadError::HistoryGap)
    );
    let status = inventory.status();
    assert_eq!(status.sequence, 30);
    assert_eq!(status.resident_records, 30);
    assert!(status.journal_records < 30);
    assert!(status.journal_bytes <= status.journal_capacity_bytes);
    assert!(status.journal_bytes_peak <= status.journal_capacity_bytes);
    assert_eq!(status.history_gaps, 1);
    assert!(status.last_change_mono_ns > 0);
    inventory
        .change(
            &key(29),
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    inventory.capture_fence();
    assert_eq!(
        inventory.status().last_change_mono_ns,
        status.last_change_mono_ns
    );
    assert_eq!(inventory.status().sequence, 30);
    assert_eq!(inventory.changes(29, 30).unwrap().len(), 1);
    assert_eq!(inventory.page(None).unwrap().len(), 30);
    inventory
        .change(
            &StateKey::new("m".into(), vec![1; INVENTORY_BATCH_BYTES]),
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    assert_eq!(
        inventory.page(None),
        Err(InventoryReadError::RecordTooLarge)
    );
}

#[test]
fn coalescing_preserves_contiguous_coverage_and_latest_key_medium_generation() {
    let inventory = ResidencyInventory::new(4096);
    let first = key(1);
    let second = key(2);
    inventory
        .change(
            &first,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    inventory
        .change(
            &first,
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    inventory.change(&first, ReplicaMedium::Dram, None).unwrap();
    inventory
        .change(
            &second,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    inventory
        .change(
            &first,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();

    let delta = inventory.coalesced_changes(0, 5).unwrap();
    assert_eq!(delta.through, 5);
    assert_eq!(delta.input_records, 5);
    assert_eq!(delta.records.len(), 3);
    assert_eq!(
        delta
            .records
            .iter()
            .map(|record| (
                record.key.clone(),
                record.metadata.unwrap().medium,
                record.sequence,
                record.present,
            ))
            .collect::<Vec<_>>(),
        vec![
            (first.clone(), ReplicaMedium::Dram, 5, true),
            (first, ReplicaMedium::Ssd, 2, true),
            (second, ReplicaMedium::Dram, 4, true),
        ]
    );

    inventory
        .change(&key(2), ReplicaMedium::Dram, None)
        .unwrap();
    let deletion = inventory.coalesced_changes(5, 6).unwrap();
    assert_eq!(deletion.through, 6);
    assert_eq!(deletion.input_records, 1);
    assert_eq!(deletion.records.len(), 1);
    assert!(!deletion.records[0].present);
    assert_eq!(deletion.records[0].sequence, 6);
}

#[test]
fn captured_fence_is_monotonic_and_wakes_publishers() {
    let inventory = ResidencyInventory::new(4096);
    inventory
        .change(
            &key(1),
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    assert_eq!(inventory.capture_fence(), 1);
    assert_eq!(inventory.status().flush_through_sequence, 1);
    inventory
        .change(
            &key(2),
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    inventory.request_flush(1);
    assert_eq!(inventory.status().flush_through_sequence, 1);
    assert_eq!(inventory.capture_fence(), 2);
    assert_eq!(inventory.status().flush_through_sequence, 2);
}

#[tokio::test(start_paused = true)]
async fn coalescing_wait_is_bounded_and_flush_bypasses_it() {
    for window in [0, 2, 5] {
        let inventory = Arc::new(
            ResidencyInventory::with_publish_coalescing(1 << 20, Duration::from_millis(window))
                .unwrap(),
        );
        inventory
            .change(
                &key(1),
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
        assert_eq!(
            inventory.wait_to_publish(0).await,
            Duration::from_millis(window)
        );
        inventory.request_flush(1);
        assert_eq!(inventory.wait_to_publish(0).await, Duration::ZERO);
        // An already fulfilled flush notification cannot skip the next window.
        inventory
            .change(
                &key(2),
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
        assert_eq!(
            inventory.wait_to_publish(1).await,
            Duration::from_millis(window)
        );
    }
    let inventory = Arc::new(
        ResidencyInventory::with_publish_coalescing(1 << 20, Duration::from_millis(2)).unwrap(),
    );
    let writer = tokio::spawn({
        let inventory = inventory.clone();
        async move {
            for id in 0..10 {
                tokio::time::sleep(Duration::from_millis(1)).await;
                inventory
                    .change(
                        &key(id),
                        ReplicaMedium::Dram,
                        Some(metadata(ReplicaMedium::Dram)),
                    )
                    .unwrap();
            }
        }
    });
    assert_eq!(inventory.wait_to_publish(0).await, Duration::from_millis(5));
    writer.await.unwrap();
    let waiting = tokio::spawn({
        let inventory = inventory.clone();
        async move { inventory.wait_to_publish(0).await }
    });
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(1)).await;
    inventory.request_flush(10);
    assert_eq!(waiting.await.unwrap(), Duration::from_millis(1));
}

#[test]
fn coalescing_partitions_input_before_deduplication() {
    let inventory = ResidencyInventory::new(4 << 20);
    for id in 0..600 {
        inventory
            .change(
                &key(id),
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
        inventory
            .change(&key(id), ReplicaMedium::Dram, None)
            .unwrap();
    }
    let first = inventory.coalesced_changes(0, 1200).unwrap();
    assert_eq!(
        (first.input_records, first.through, first.records.len()),
        (1024, 1024, 512)
    );
    assert!(first.records.iter().all(|r| !r.present));
    let second = inventory.coalesced_changes(first.through, 1200).unwrap();
    assert_eq!(
        (second.input_records, second.through, second.records.len()),
        (176, 1200, 88)
    );
    let large = ResidencyInventory::new(4 << 20);
    for id in 0..100 {
        large
            .change(
                &StateKey::new("large".into(), vec![id; 32 * 1024]),
                ReplicaMedium::Ssd,
                Some(metadata(ReplicaMedium::Ssd)),
            )
            .unwrap();
    }
    let delta = large.coalesced_changes(0, 100).unwrap();
    assert!(delta.input_records < 100);
    assert!(delta.input_bytes <= INVENTORY_BATCH_BYTES);
    assert!(
        delta.input_bytes + large.changes(delta.through, 100).unwrap()[0].estimated_size()
            > INVENTORY_BATCH_BYTES
    );
}

#[test]
fn scoped_scan_and_delta_preserve_source_coverage_across_empty_intervals() {
    let inventory = ResidencyInventory::new(4 << 20);
    let scope = InventoryScope::exact(["selected".into()]).unwrap();
    for id in 0..700 {
        for namespace in ["outside", "selected"] {
            let key = scoped_key(namespace, id);
            inventory
                .change(
                    &key,
                    ReplicaMedium::Dram,
                    Some(metadata(ReplicaMedium::Dram)),
                )
                .unwrap();
        }
    }
    let mut cursor = None;
    let mut selected = Vec::new();
    loop {
        let page = inventory.scoped_page(cursor.as_ref(), &scope).unwrap();
        selected.extend(page.records);
        cursor = page.next;
        if page.complete {
            break;
        }
    }
    assert_eq!(selected.len(), 700);
    assert!(
        selected
            .iter()
            .all(|record| record.key.namespace == "selected")
    );

    let before = inventory.sequence();
    inventory
        .change(
            &scoped_key("outside", 999),
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    let empty = inventory
        .coalesced_changes_scoped(before, inventory.sequence(), &scope)
        .unwrap();
    assert_eq!(empty.through, inventory.sequence());
    assert_eq!(empty.input_records, 1);
    assert!(empty.records.is_empty());

    let selected_key = scoped_key("selected", 0);
    let before = inventory.sequence();
    inventory
        .change(&selected_key, ReplicaMedium::Dram, None)
        .unwrap();
    inventory
        .change(
            &selected_key,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    let replacement = inventory
        .coalesced_changes_scoped(before, inventory.sequence(), &scope)
        .unwrap();
    assert_eq!(replacement.records.len(), 1);
    assert!(replacement.records[0].present);
    assert_eq!(replacement.records[0].sequence, inventory.sequence());
    let status = inventory.status();
    assert!(status.scope_filter_input_records > status.scope_filter_output_records);
    assert!(status.scope_filter_micros > 0);
}

#[test]
fn exhausted_sequence_rejects_mutation_and_advertisement() {
    let inventory = ResidencyInventory::new(4096);
    inventory.state.lock().sequence = u64::MAX - 1;
    inventory
        .change(
            &key(1),
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    assert_eq!(
        inventory.change(&key(1), ReplicaMedium::Dram, None),
        Err(InventoryReadError::SequenceExhausted)
    );
    assert_eq!(
        inventory.changes(u64::MAX, u64::MAX),
        Err(InventoryReadError::SequenceExhausted)
    );
    assert_eq!(
        inventory.page(None),
        Err(InventoryReadError::SequenceExhausted)
    );
    assert_eq!(inventory.sequence(), u64::MAX);
}
