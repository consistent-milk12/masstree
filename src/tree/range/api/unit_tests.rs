use crate::key::MAX_KEY_LENGTH;

use super::compute_prefix_upper_bound_into;

/// Helper: call `compute_prefix_upper_bound_into` and return the result as Option<Vec<u8>>
/// for easy comparison in tests.
fn upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut buf: [u8; 256] = [0u8; MAX_KEY_LENGTH];
    compute_prefix_upper_bound_into(prefix, &mut buf).map(|len| buf[..len].to_vec())
}

#[test]
fn test_compute_prefix_upper_bound() {
    // Normal case
    assert_eq!(upper_bound(b"abc"), Some(b"abd".to_vec()));

    // Last byte is 0xFF
    assert_eq!(upper_bound(b"ab\xff"), Some(b"ac".to_vec()));

    // Multiple trailing 0xFF
    assert_eq!(upper_bound(b"a\xff\xff"), Some(b"b".to_vec()));

    // All 0xFF
    assert_eq!(upper_bound(b"\xff\xff\xff"), None);

    // Empty
    assert_eq!(upper_bound(b""), None);

    // Single byte
    assert_eq!(upper_bound(b"a"), Some(b"b".to_vec()));
    assert_eq!(upper_bound(b"\xff"), None);
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

#[test]
fn test_scan_intra_leaf_batch_exact_bound_when_ikey_equal() {
    use super::RangeBound;
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // First two keys share the same 8-byte ikey.
    let k0: &[u8] = b"prefix00a";
    let k1: &[u8] = b"prefix00b";
    let k2: &[u8] = b"prefix01a";

    let _ = tree.insert_with_guard(k0, 10, &guard);
    let _ = tree.insert_with_guard(k1, 20, &guard);
    let _ = tree.insert_with_guard(k2, 30, &guard);

    let mut seen: Vec<Vec<u8>> = Vec::new();
    let visited = tree.scan_intra_leaf_batch(
        RangeBound::Unbounded,
        RangeBound::Excluded(k1),
        |key, _value| {
            seen.push(key.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(visited, 1);
    assert_eq!(seen, vec![k0.to_vec()]);
}

#[test]
fn test_scan_prefix_exact_8byte_with_layer_includes_exact_first() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix: &[u8] = b"PREFIXXX";
    let mut k1 = prefix.to_vec();
    k1.extend_from_slice(b"aaaaaaaa");
    let mut k2 = prefix.to_vec();
    k2.extend_from_slice(b"bbbbbbbb");

    let _ = tree.insert_with_guard(&k1, 1, &guard);
    let _ = tree.insert_with_guard(&k2, 2, &guard);
    let _ = tree.insert_with_guard(prefix, 3, &guard);

    let mut keys = Vec::new();
    let visited = tree.scan_prefix(
        prefix,
        |key, _| {
            keys.push(key.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(visited, 3);
    assert_eq!(keys[0], prefix);
    assert_eq!(keys[1], k1);
    assert_eq!(keys[2], k2);
}

#[test]
fn test_scan_prefix_descends_multiple_chunks() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix = [b"AAAAAAAA".as_ref(), b"BBBBBBBB".as_ref()].concat();
    let mut k1 = prefix.clone();
    k1.extend_from_slice(b"11111111");
    let mut k2 = prefix.clone();
    k2.extend_from_slice(b"22222222");
    let mut other = b"AAAAAAAA".to_vec();
    other.extend_from_slice(b"CCCCCCCC");
    other.extend_from_slice(b"33333333");

    let _ = tree.insert_with_guard(&k1, 1, &guard);
    let _ = tree.insert_with_guard(&k2, 2, &guard);
    let _ = tree.insert_with_guard(&other, 3, &guard);

    let mut keys = Vec::new();
    let visited = tree.scan_prefix(
        &prefix,
        |key, _| {
            keys.push(key.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(visited, 2);
    assert_eq!(keys, vec![k1, k2]);
}

#[test]
fn test_scan_prefix_descended_remainder_range_is_exact() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix: &[u8] = b"PREFIXXXab";
    let mut k1 = prefix.to_vec();
    k1.extend_from_slice(b"111111");
    let mut k2 = prefix.to_vec();
    k2.extend_from_slice(b"222222");

    // Same first 8 bytes, different remainder prefix ("ac" > "ab").
    let mut out_of_prefix = b"PREFIXXXac".to_vec();
    out_of_prefix.extend_from_slice(b"000000");

    let _ = tree.insert_with_guard(&k1, 1, &guard);
    let _ = tree.insert_with_guard(&k2, 2, &guard);
    let _ = tree.insert_with_guard(&out_of_prefix, 3, &guard);

    let mut keys = Vec::new();
    let visited = tree.scan_prefix(
        prefix,
        |key, _| {
            keys.push(key.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(visited, 2);
    assert_eq!(keys, vec![k1, k2]);
}

#[test]
fn test_scan_prefix_values_empty_prefix_visits_all() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let _ = tree.insert_with_guard(b"a", 1, &guard);
    let _ = tree.insert_with_guard(b"b", 2, &guard);
    let _ = tree.insert_with_guard(b"c", 3, &guard);

    let mut sum = 0u64;
    let visited = tree.scan_prefix_values(
        b"",
        |value| {
            sum += *value;
            true
        },
        &guard,
    );

    assert_eq!(visited, 3);
    assert_eq!(sum, 6);
}

#[test]
fn test_scan_prefix_values_all_ff_prefix() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let _ = tree.insert_with_guard(b"\xff\xfe", 1, &guard);
    let _ = tree.insert_with_guard(b"\xff\xff", 2, &guard);
    let _ = tree.insert_with_guard(b"\xff\xff\x00", 3, &guard);
    let _ = tree.insert_with_guard(b"\xff\xff\xff", 4, &guard);

    let mut values = Vec::new();
    let visited = tree.scan_prefix_values(
        b"\xff\xff",
        |value| {
            values.push(*value);
            true
        },
        &guard,
    );

    assert_eq!(visited, 3);
    assert_eq!(values, vec![2, 3, 4]);
}

#[test]
fn test_scan_prefix_values_early_stop() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    for i in 0u64..20 {
        let key = format!("k{i:02}");
        let _ = tree.insert_with_guard(key.as_bytes(), i, &guard);
    }

    let mut seen = 0usize;
    let visited = tree.scan_prefix_values(
        b"k",
        |_value| {
            seen += 1;
            seen < 7
        },
        &guard,
    );

    assert_eq!(visited, 7);
    assert_eq!(seen, 7);
}

#[test]
fn test_scan_prefix_values_non_aligned_can_over_include_boundary_ikey() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let _ = tree.insert_with_guard(b"abc0", 10, &guard);
    let _ = tree.insert_with_guard(b"abc1", 11, &guard);
    // This key is >= "abd" and should be out of the exact [abc, abd) range,
    // but values-only scans use ikey-only end checks and may include it.
    let _ = tree.insert_with_guard(b"abd\x00", 12, &guard);
    let _ = tree.insert_with_guard(b"abe0", 13, &guard);

    let mut values = Vec::new();
    let visited = tree.scan_prefix_values(
        b"abc",
        |value| {
            values.push(*value);
            true
        },
        &guard,
    );

    assert_eq!(visited, values.len());
    assert!(values.contains(&10));
    assert!(values.contains(&11));
    assert!(values.contains(&12));
    assert!(!values.contains(&13));
}

#[test]
fn test_scan_prefix_values_exact_8byte_with_layer_includes_exact_first() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix: &[u8] = b"PREFIXXX";
    let mut k1 = prefix.to_vec();
    k1.extend_from_slice(b"aaaaaaaa");
    let mut k2 = prefix.to_vec();
    k2.extend_from_slice(b"bbbbbbbb");

    let _ = tree.insert_with_guard(&k1, 1, &guard);
    let _ = tree.insert_with_guard(&k2, 2, &guard);
    let _ = tree.insert_with_guard(prefix, 3, &guard);

    let mut values = Vec::new();
    let visited = tree.scan_prefix_values(
        prefix,
        |value| {
            values.push(*value);
            true
        },
        &guard,
    );

    assert_eq!(visited, 3);
    assert_eq!(values, vec![3, 1, 2]);
}

#[test]
fn test_scan_prefix_values_descends_multiple_chunks() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix = [b"AAAAAAAA".as_ref(), b"BBBBBBBB".as_ref()].concat();
    let mut k1 = prefix.clone();
    k1.extend_from_slice(b"11111111");
    let mut k2 = prefix.clone();
    k2.extend_from_slice(b"22222222");
    let mut other = b"AAAAAAAA".to_vec();
    other.extend_from_slice(b"CCCCCCCC");
    other.extend_from_slice(b"33333333");

    let _ = tree.insert_with_guard(&k1, 1, &guard);
    let _ = tree.insert_with_guard(&k2, 2, &guard);
    let _ = tree.insert_with_guard(&other, 3, &guard);

    let mut values = Vec::new();
    let visited = tree.scan_prefix_values(
        &prefix,
        |value| {
            values.push(*value);
            true
        },
        &guard,
    );

    assert_eq!(visited, 2);
    assert_eq!(values, vec![1, 2]);
}

#[test]
fn test_scan_prefix_values_descended_remainder_does_not_under_return() {
    use crate::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let prefix: &[u8] = b"PREFIXXXab";
    let mut k1 = prefix.to_vec();
    k1.extend_from_slice(b"111111");
    let mut k2 = prefix.to_vec();
    k2.extend_from_slice(b"222222");
    let mut other = b"PREFIXXXac".to_vec();
    other.extend_from_slice(b"000000");

    let _ = tree.insert_with_guard(&k1, 10, &guard);
    let _ = tree.insert_with_guard(&k2, 20, &guard);
    let _ = tree.insert_with_guard(&other, 30, &guard);

    let mut values = Vec::new();
    let visited = tree.scan_prefix_values(
        prefix,
        |value| {
            values.push(*value);
            true
        },
        &guard,
    );

    // Non-aligned value-only scans are approximate, so over-inclusion is allowed,
    // but descended scans must not under-return matches.
    assert_eq!(visited, values.len());
    assert!(values.contains(&10));
    assert!(values.contains(&20));
}
