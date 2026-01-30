use super::compute_prefix_upper_bound;

#[test]
fn test_compute_prefix_upper_bound() {
    // Normal case
    assert_eq!(compute_prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));

    // Last byte is 0xFF
    assert_eq!(compute_prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));

    // Multiple trailing 0xFF
    assert_eq!(
        compute_prefix_upper_bound(b"a\xff\xff"),
        Some(b"b".to_vec())
    );

    // All 0xFF
    assert_eq!(compute_prefix_upper_bound(b"\xff\xff\xff"), None);

    // Empty
    assert_eq!(compute_prefix_upper_bound(b""), None);

    // Single byte
    assert_eq!(compute_prefix_upper_bound(b"a"), Some(b"b".to_vec()));
    assert_eq!(compute_prefix_upper_bound(b"\xff"), None);
}

#[test]
fn test_scan_ref_returns_same_values_as_scan() {
    use super::RangeBound;
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert some test data
    for i in 0u64..100 {
        let key = format!("key{i:03}");
        let _ = tree.insert_with_guard(key.as_bytes(), i, &guard);
    }

    // Collect values using scan (cloning)
    let mut values_scan: Vec<u64> = Vec::new();
    tree.scan(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_key, value| {
            values_scan.push(*value);
            true
        },
        &guard,
    );

    // Collect values using scan_ref (zero-copy)
    let mut values_scan_ref: Vec<u64> = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_key, value| {
            values_scan_ref.push(*value);
            true
        },
        &guard,
    );

    // Should produce identical results
    assert_eq!(values_scan.len(), 100);
    assert_eq!(values_scan, values_scan_ref);
}

#[test]
fn test_scan_ref_with_range_bounds() {
    use super::RangeBound;
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert keys "a", "b", "c", "d", "e"
    for (i, c) in ['a', 'b', 'c', 'd', 'e'].iter().enumerate() {
        let _ = tree.insert_with_guard(&[*c as u8], i as u64, &guard);
    }

    // First check what regular scan returns
    let mut values_scan: Vec<u64> = Vec::new();
    tree.scan(
        RangeBound::Included(b"b"),
        RangeBound::Included(b"d"),
        |_key, value| {
            values_scan.push(*value);
            true
        },
        &guard,
    );

    // Scan from "b" to "d" inclusive using scan_ref
    let mut values_ref: Vec<u64> = Vec::new();
    tree.scan_ref(
        RangeBound::Included(b"b"),
        RangeBound::Included(b"d"),
        |_key, value| {
            values_ref.push(*value);
            true
        },
        &guard,
    );

    // scan_ref should match scan exactly
    assert_eq!(values_scan, values_ref, "scan_ref should match scan");
}

#[test]
fn test_scan_ref_early_stop() {
    use super::RangeBound;
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert 100 entries
    for i in 0u64..100 {
        let key = format!("key{i:03}");
        let _ = tree.insert_with_guard(key.as_bytes(), i, &guard);
    }

    // Scan with early stop after 10 entries
    let mut count = 0usize;
    let visited = tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_key, _value| {
            count += 1;
            count < 10
        },
        &guard,
    );

    assert_eq!(visited, 10);
    assert_eq!(count, 10);
}
