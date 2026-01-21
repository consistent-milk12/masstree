//! True-inline storage tests.

use masstree::{MassTree15Inline, RangeBound};
use seize::LocalGuard;

#[test]
#[expect(clippy::indexing_slicing, clippy::unwrap_used)]
fn masstree15_inline_no_leak_with_long_keys() {
    fn make_key(i: u64) -> [u8; 64] {
        let mut key = [0u8; 64];
        key[0..8].copy_from_slice(&i.to_be_bytes());

        // Fill rest with pattern to ensure suffix is used
        (8..64).for_each(|j| {
            key[j] = ((i + (j as u64)) & 0xFF) as u8;
        });

        key
    }

    // Create tree, insert enough keys to trigger external suffix bags
    // Inline capacity is 256 bytes, ~56 bytes per suffix, so ~5 keys fill it
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard: LocalGuard<'_> = tree.guard();

    for i in 0..20 {
        let key = make_key(i);
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    drop(guard);
    drop(tree); // Should free external suffix bags via `Drop`

    // If `Drop` is missing memory tools (miri, asan etc) will detect the leak.
}

// ============================================================================
// Scan Tests for Inline Storage
// ============================================================================

/// Test basic `for_each` scan on inline tree.
#[test]
#[expect(clippy::unwrap_used)]
fn inline_scan_for_each() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    // Insert some values
    for i in 0u64..20 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i * 10, &guard).unwrap();
    }

    // Scan using for_each
    let mut results: Vec<(Vec<u8>, u64)> = Vec::new();
    tree.iter(&guard).for_each(|key, value| {
        results.push((key.to_vec(), value));
        true
    });

    assert_eq!(results.len(), 20);

    // Values should be in order and correct
    for (i, (key, value)) in results.into_iter().enumerate() {
        let expected_key = (i as u64).to_be_bytes();
        assert_eq!(key, expected_key);
        assert_eq!(value, (i as u64) * 10);
    }
}

/// Test intra-leaf batch scan on inline tree.
#[test]
#[expect(clippy::unwrap_used)]
fn inline_scan_intra_leaf_batch() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    // Insert enough values to span multiple leaves
    for i in 0u64..100 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i * 100, &guard).unwrap();
    }

    // Scan using for_each_intra_leaf_batch
    let mut results: Vec<(Vec<u8>, u64)> = Vec::new();
    let count = tree.iter(&guard).for_each_intra_leaf_batch(|key, value| {
        results.push((key.to_vec(), value));
        true
    });

    assert_eq!(count, 100);
    assert_eq!(results.len(), 100);

    // Values should be in order and correct
    for (i, (key, value)) in results.into_iter().enumerate() {
        let expected_key = (i as u64).to_be_bytes();
        assert_eq!(key, expected_key, "Key mismatch at index {i}");
        assert_eq!(value, (i as u64) * 100, "Value mismatch at index {i}");
    }
}

/// Test intra-leaf batch scan with early stop.
#[test]
#[expect(clippy::unwrap_used)]
fn inline_scan_batch_early_stop() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    for i in 0u64..50 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    // Stop after 10 entries
    let mut seen = 0usize;
    let count = tree.iter(&guard).for_each_intra_leaf_batch(|_key, _value| {
        seen += 1;
        seen < 10
    });

    assert_eq!(count, 10);
    assert_eq!(seen, 10);
}

/// Test intra-leaf batch scan with range bounds.
#[test]
#[expect(clippy::unwrap_used)]
fn inline_scan_batch_range() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    for i in 0u64..100 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    // Range scan from 25 to 75
    let start = 25u64.to_be_bytes();
    let end = 75u64.to_be_bytes();

    let mut results: Vec<u64> = Vec::new();
    tree.range(
        RangeBound::Included(&start),
        RangeBound::Excluded(&end),
        &guard,
    )
    .for_each_intra_leaf_batch(|_key, value| {
        results.push(value);
        true
    });

    // Should get 25..75 = 50 values
    assert_eq!(results.len(), 50);
    for (i, value) in results.into_iter().enumerate() {
        assert_eq!(value, (25 + i) as u64);
    }
}

/// Test intra-leaf batch scan with multi-layer keys (keys > 8 bytes).
#[test]
#[expect(clippy::unwrap_used, clippy::indexing_slicing)]
fn inline_scan_batch_multi_layer() {
    // Helper to create keys that will create layers (> 8 bytes with common prefix)
    let make_key = |i: u64| -> Vec<u8> {
        let mut key = vec![0u8; 16];
        // Common 8-byte prefix
        key[0..8].copy_from_slice(&[0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A]);
        // Varying suffix
        key[8..16].copy_from_slice(&i.to_be_bytes());
        key
    };

    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    for i in 0u64..30 {
        let key = make_key(i);
        tree.insert_with_guard(&key, i * 11, &guard).unwrap();
    }

    // Scan all entries
    let mut results: Vec<(Vec<u8>, u64)> = Vec::new();
    let count = tree.iter(&guard).for_each_intra_leaf_batch(|key, value| {
        results.push((key.to_vec(), value));
        true
    });

    assert_eq!(count, 30);
    assert_eq!(results.len(), 30);

    // Results should be in sorted order
    for i in 0..results.len() - 1 {
        assert!(results[i].0 < results[i + 1].0, "Keys not in sorted order");
    }
}
