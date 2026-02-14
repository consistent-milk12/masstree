#![expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

use super::{RangeBound, ScanEntry};
use crate::MassTree;

// ============================================================================
//  RangeBound Tests
// ============================================================================

#[test]
fn test_range_bound_contains() {
    // Unbounded contains everything
    assert!(RangeBound::Unbounded.contains(b"anything"));
    assert!(RangeBound::Unbounded.contains(b""));

    // Included: key <= bound
    let included = RangeBound::Included(b"middle");
    assert!(included.contains(b"aaa"));
    assert!(included.contains(b"middle"));
    assert!(!included.contains(b"zzz"));

    // Excluded: key < bound
    let excluded = RangeBound::Excluded(b"middle");
    assert!(excluded.contains(b"aaa"));
    assert!(!excluded.contains(b"middle"));
    assert!(!excluded.contains(b"zzz"));
}

#[test]
fn test_range_bound_to_start_params() {
    let (key, emit) = RangeBound::Unbounded.to_start_params();
    assert_eq!(key, b"");
    assert!(emit);

    let (key, emit) = RangeBound::Included(b"start").to_start_params();
    assert_eq!(key, b"start");
    assert!(emit);

    let (key, emit) = RangeBound::Excluded(b"start").to_start_params();
    assert_eq!(key, b"start");
    assert!(!emit);
}

#[test]
fn test_range_bound_from_std_bound() {
    use std::ops::Bound;

    let rb: RangeBound = Bound::Unbounded.into();
    assert!(matches!(rb, RangeBound::Unbounded));

    let rb: RangeBound = Bound::Included(b"key".as_slice()).into();
    assert!(matches!(rb, RangeBound::Included(k) if k == b"key"));

    let rb: RangeBound = Bound::Excluded(b"key".as_slice()).into();
    assert!(matches!(rb, RangeBound::Excluded(k) if k == b"key"));
}

#[test]
fn test_scan_entry() {
    let entry = ScanEntry::new(b"key".to_vec(), 42u64);

    assert_eq!(entry.key(), b"key");
    assert_eq!(*entry.value(), 42);

    let (key, value) = entry.into_parts();
    assert_eq!(key, b"key");
    assert_eq!(value, 42);
}

#[test]
fn test_range_bound_is_unbounded() {
    assert!(RangeBound::Unbounded.is_unbounded());
    assert!(!RangeBound::Included(b"key").is_unbounded());
    assert!(!RangeBound::Excluded(b"key").is_unbounded());
}

#[test]
fn test_range_bound_key() {
    assert!(RangeBound::Unbounded.key().is_none());
    assert_eq!(RangeBound::Included(b"key").key(), Some(b"key".as_slice()));
    assert_eq!(RangeBound::Excluded(b"key").key(), Some(b"key".as_slice()));
}

// ============================================================================
//  contains_reverse Tests (for DoubleEndedIterator start bound checking)
// ============================================================================

#[test]
fn test_range_bound_contains_reverse_unbounded() {
    // Unbounded contains everything in reverse
    assert!(RangeBound::Unbounded.contains_reverse(b"anything"));
    assert!(RangeBound::Unbounded.contains_reverse(b""));
    assert!(RangeBound::Unbounded.contains_reverse(b"zzz"));
}

#[test]
fn test_range_bound_contains_reverse_included() {
    // Included: key >= bound (reverse direction)
    let included = RangeBound::Included(b"middle");
    assert!(!included.contains_reverse(b"aaa")); // aaa < middle
    assert!(included.contains_reverse(b"middle")); // middle >= middle
    assert!(included.contains_reverse(b"zzz")); // zzz > middle
}

#[test]
fn test_range_bound_contains_reverse_excluded() {
    // Excluded: key > bound (reverse direction)
    let excluded = RangeBound::Excluded(b"middle");
    assert!(!excluded.contains_reverse(b"aaa")); // aaa < middle
    assert!(!excluded.contains_reverse(b"middle")); // middle == middle, excluded
    assert!(excluded.contains_reverse(b"zzz")); // zzz > middle
}

// ============================================================================
//  DoubleEndedIterator Tests
// ============================================================================

#[test]
fn test_rev_empty_tree() {
    let tree = MassTree::<u64>::default();
    let guard = tree.guard();

    let items: Vec<_> = tree.iter(&guard).rev().collect();
    assert!(items.is_empty());
}

#[test]
fn test_rev_single_element() {
    let tree = MassTree::<u64>::default();
    tree.insert(b"only", 42);

    let guard = tree.guard();
    let items: Vec<_> = tree.iter(&guard).rev().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].key(), b"only");
    assert_eq!(*items[0].value(), 42);
}

#[test]
fn test_rev_multiple_elements() {
    let tree = MassTree::<u64>::default();
    tree.insert(b"a", 1);
    tree.insert(b"b", 2);
    tree.insert(b"c", 3);

    let guard = tree.guard();
    let items: Vec<_> = tree.iter(&guard).rev().collect();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].key(), b"c");
    assert_eq!(items[1].key(), b"b");
    assert_eq!(items[2].key(), b"a");
}

#[test]
fn test_rev_keys_are_descending() {
    let tree = MassTree::<u64>::default();
    for i in 0..10u64 {
        tree.insert(format!("key{i:02}").as_bytes(), i);
    }

    let guard = tree.guard();
    let keys: Vec<Vec<u8>> = tree.iter(&guard).rev().map(|e| e.key().to_vec()).collect();

    // Verify descending order
    for window in keys.windows(2) {
        assert!(window[0] > window[1], "Keys should be in descending order");
    }
}

#[test]
fn test_next_and_next_back_meet_in_middle() {
    let tree = MassTree::<u64>::default();
    tree.insert(b"a", 1);
    tree.insert(b"b", 2);
    tree.insert(b"c", 3);
    tree.insert(b"d", 4);

    let guard = tree.guard();
    let mut iter = tree.iter(&guard);

    // Front: a
    let front1 = iter.next().unwrap();
    assert_eq!(front1.key(), b"a");

    // Back: d
    let back1 = iter.next_back().unwrap();
    assert_eq!(back1.key(), b"d");

    // Front: b
    let front2 = iter.next().unwrap();
    assert_eq!(front2.key(), b"b");

    // Back: c
    let back2 = iter.next_back().unwrap();
    assert_eq!(back2.key(), b"c");

    // Should be exhausted now (front and back have met)
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}

#[test]
fn test_alternating_next_next_back() {
    let tree = MassTree::<u64>::default();
    for i in 0..6u64 {
        tree.insert(format!("{}", (b'a' + i as u8) as char).as_bytes(), i);
    }

    let guard = tree.guard();
    let mut iter = tree.iter(&guard);

    // Alternate: next, next_back, next, next_back...
    assert_eq!(iter.next().unwrap().key(), b"a");
    assert_eq!(iter.next_back().unwrap().key(), b"f");
    assert_eq!(iter.next().unwrap().key(), b"b");
    assert_eq!(iter.next_back().unwrap().key(), b"e");
    assert_eq!(iter.next().unwrap().key(), b"c");
    assert_eq!(iter.next_back().unwrap().key(), b"d");

    // Exhausted
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}

#[test]
fn test_forward_only_iterator_supports_reverse_lazily() {
    let tree = MassTree::<u64>::default();
    tree.insert(b"a", 1);
    tree.insert(b"b", 2);
    tree.insert(b"c", 3);
    tree.insert(b"d", 4);

    let guard = tree.guard();
    let mut iter = tree.range_forward(RangeBound::Unbounded, RangeBound::Unbounded, &guard);

    assert_eq!(iter.next().unwrap().key(), b"a");
    assert_eq!(iter.next_back().unwrap().key(), b"d");
    assert_eq!(iter.next_back().unwrap().key(), b"c");
    assert_eq!(iter.next().unwrap().key(), b"b");

    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}

#[test]
fn test_rev_with_longer_keys() {
    let tree = MassTree::<u64>::default();

    // Keys longer than 8 bytes (multi-layer)
    tree.insert(b"prefix_aaa", 1);
    tree.insert(b"prefix_bbb", 2);
    tree.insert(b"prefix_ccc", 3);

    let guard = tree.guard();
    let items: Vec<_> = tree.iter(&guard).rev().collect();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].key(), b"prefix_ccc");
    assert_eq!(items[1].key(), b"prefix_bbb");
    assert_eq!(items[2].key(), b"prefix_aaa");
}

#[test]
fn test_rev_exact_layer_boundary_key_not_skipped() {
    let tree = MassTree::<u64>::default();

    // Exact 24-byte key (3 full 8-byte layers).
    let mut k_exact = vec![0u8; 24];
    k_exact[23] = 163;

    // Longer keys sharing the exact key as prefix (forces layer pointer path).
    let mut k_long1 = k_exact.clone();
    k_long1.extend([129, 173]);

    let mut k_long2 = k_exact.clone();
    k_long2.extend([
        169, 103, 242, 227, 80, 214, 1, 125, 187, 127, 159, 159, 22, 107, 44,
    ]);

    // Immediate predecessor in lexicographic order.
    let mut k_prev = vec![0u8; 24];
    k_prev[23] = 151;
    k_prev.extend([206, 4, 204]);

    for (i, key) in [
        k_prev.clone(),
        k_exact.clone(),
        k_long1.clone(),
        k_long2.clone(),
    ]
    .into_iter()
    .enumerate()
    {
        tree.insert(&key, i as u64);
    }

    assert_eq!(tree.get(&k_prev), Some(0));
    assert_eq!(tree.get(&k_exact), Some(1));
    assert_eq!(tree.get(&k_long1), Some(2));
    assert_eq!(tree.get(&k_long2), Some(3));

    let guard = tree.guard();
    let forward: Vec<Vec<u8>> = tree.iter(&guard).map(|e| e.key().to_vec()).collect();
    assert_eq!(
        forward,
        vec![
            k_prev.clone(),
            k_exact.clone(),
            k_long1.clone(),
            k_long2.clone()
        ]
    );

    let got: Vec<Vec<u8>> = tree.iter(&guard).rev().map(|e| e.key().to_vec()).collect();
    let expected = vec![k_long2, k_long1, k_exact, k_prev];

    assert_eq!(got, expected);
}

#[test]
fn test_rev_boundary_with_included_and_excluded_end() {
    let tree = MassTree::<u64>::default();

    let mut k_exact = vec![0u8; 24];
    k_exact[23] = 163;

    let mut k_long = k_exact.clone();
    k_long.extend([129, 173]);

    let mut k_prev = vec![0u8; 24];
    k_prev[23] = 151;
    k_prev.extend([206, 4, 204]);

    tree.insert(&k_prev, 1);
    tree.insert(&k_exact, 2);
    tree.insert(&k_long, 3);

    let guard = tree.guard();

    let included: Vec<Vec<u8>> = tree
        .range(
            RangeBound::Unbounded,
            RangeBound::Included(k_exact.as_slice()),
            &guard,
        )
        .rev()
        .map(|e| e.key().to_vec())
        .collect();
    assert_eq!(included, vec![k_exact.clone(), k_prev.clone()]);

    let excluded: Vec<Vec<u8>> = tree
        .range(
            RangeBound::Unbounded,
            RangeBound::Excluded(k_exact.as_slice()),
            &guard,
        )
        .rev()
        .map(|e| e.key().to_vec())
        .collect();
    assert_eq!(excluded, vec![k_prev]);
}

#[test]
fn test_rev_consistency_with_forward() {
    let tree = MassTree::<u64>::default();
    for i in 0..20u64 {
        tree.insert(format!("k{i:03}").as_bytes(), i);
    }

    let guard = tree.guard();

    // Collect forward
    let forward: Vec<Vec<u8>> = tree.iter(&guard).map(|e| e.key().to_vec()).collect();

    // Collect reverse
    let mut reverse: Vec<Vec<u8>> = tree.iter(&guard).rev().map(|e| e.key().to_vec()).collect();
    reverse.reverse(); // Reverse it back

    // Should be identical
    assert_eq!(forward, reverse);
}

#[test]
fn test_double_ended_count_small() {
    // Test with exactly one leaf's worth of elements (15)
    let tree = MassTree::<u64>::default();
    for i in 0..15u64 {
        tree.insert(format!("key{i:02}").as_bytes(), i);
    }

    let guard = tree.guard();
    let forward_count = tree.iter(&guard).count();
    let reverse_count = tree.iter(&guard).rev().count();

    assert_eq!(forward_count, 15);
    assert_eq!(reverse_count, 15);
}

#[test]
fn test_double_ended_count_two_leaves() {
    // Test with elements spanning two leaves (20 elements)
    let tree = MassTree::<u64>::default();
    for i in 0..20u64 {
        tree.insert(format!("key{i:02}").as_bytes(), i);
    }

    let guard = tree.guard();
    let forward_count = tree.iter(&guard).count();
    let reverse_count = tree.iter(&guard).rev().count();

    assert_eq!(forward_count, 20);
    assert_eq!(reverse_count, 20, "Reverse scan should traverse all leaves");
}

#[test]
fn test_double_ended_count_50() {
    let tree = MassTree::<u64>::default();
    for i in 0..50u64 {
        tree.insert(format!("key{i:02}").as_bytes(), i);
    }

    let guard = tree.guard();
    let forward_count = tree.iter(&guard).count();
    let reverse_count = tree.iter(&guard).rev().count();

    assert_eq!(forward_count, 50);
    assert_eq!(
        reverse_count, 50,
        "Reverse scan should work with 50 elements"
    );
}

#[test]
fn test_double_ended_count() {
    let tree = MassTree::<u64>::default();
    for i in 0..100u64 {
        tree.insert(format!("key{i:03}").as_bytes(), i);
    }

    let guard = tree.guard();

    // Count using forward iteration
    let forward_count = tree.iter(&guard).count();

    // Count using reverse iteration
    let reverse_count = tree.iter(&guard).rev().count();

    assert_eq!(forward_count, 100);
    assert_eq!(
        reverse_count, 100,
        "Reverse scan should traverse all leaves"
    );
}

#[test]
fn test_rev_after_partial_forward() {
    let tree = MassTree::<u64>::default();
    tree.insert(b"1", 1);
    tree.insert(b"2", 2);
    tree.insert(b"3", 3);
    tree.insert(b"4", 4);
    tree.insert(b"5", 5);

    let guard = tree.guard();
    let mut iter = tree.iter(&guard);

    // Take 2 from front
    assert_eq!(iter.next().unwrap().key(), b"1");
    assert_eq!(iter.next().unwrap().key(), b"2");

    // Now take remaining from back
    let remaining: Vec<_> = iter.rev().collect();
    assert_eq!(remaining.len(), 3);
    assert_eq!(remaining[0].key(), b"5");
    assert_eq!(remaining[1].key(), b"4");
    assert_eq!(remaining[2].key(), b"3");
}

#[test]
fn test_meeting_detection_odd_count() {
    let tree = MassTree::<u64>::default();

    // Odd number of elements
    tree.insert(b"a", 1);
    tree.insert(b"b", 2);
    tree.insert(b"c", 3);

    let guard = tree.guard();
    let mut iter = tree.iter(&guard);

    assert_eq!(iter.next().unwrap().key(), b"a");
    assert_eq!(iter.next_back().unwrap().key(), b"c");
    assert_eq!(iter.next().unwrap().key(), b"b");

    // Middle element consumed, should be exhausted
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}

#[test]
fn test_meeting_detection_even_count() {
    let tree = MassTree::<u64>::default();

    // Even number of elements
    tree.insert(b"a", 1);
    tree.insert(b"b", 2);

    let guard = tree.guard();
    let mut iter = tree.iter(&guard);

    assert_eq!(iter.next().unwrap().key(), b"a");
    assert_eq!(iter.next_back().unwrap().key(), b"b");

    // Both consumed, should be exhausted
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}
