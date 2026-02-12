//! Property-based tests for the `tree` module.
//!
//! These tests verify invariants and properties that should hold for all inputs.
//! Uses differential testing against `BTreeMap` as an oracle.

#![expect(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fail fast in tests"
)]

use masstree::RangeBound;
use masstree::key::MAX_KEY_LENGTH;
use masstree::tree::{MassTree15, MassTree15Inline};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use std::collections::BTreeMap;

/// Max key length for short key tests (single layer, no layer descent).
const SHORT_KEY_LEN: usize = 8;

// ============================================================================
//  Strategies
// ============================================================================

/// Strategy for generating short keys (0-8 bytes, no layer descent needed).
fn short_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=SHORT_KEY_LEN)
}

/// Strategy for generating non-empty short keys (1-8 bytes).
fn short_key_nonempty() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=SHORT_KEY_LEN)
}

/// Strategy for generating long keys (9-64 bytes, requires layer descent).
fn long_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 9..=64)
}

/// Strategy for generating any valid key (0-256 bytes).
#[expect(dead_code, reason = "Available for future tests")]
fn valid_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=MAX_KEY_LENGTH)
}

/// Strategy for generating non-empty short keys (for compatibility with existing tests).
fn valid_key_nonempty() -> impl Strategy<Value = Vec<u8>> {
    short_key_nonempty()
}

/// Strategy for generating a sequence of unique short keys.
fn unique_keys(max_count: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::hash_set(short_key_nonempty(), 0..=max_count)
        .prop_map(|set| set.into_iter().collect())
}

/// Strategy for generating key-value pairs with short keys.
fn key_value_pairs(max_count: usize) -> impl Strategy<Value = Vec<(Vec<u8>, u64)>> {
    prop::collection::vec((short_key(), any::<u64>()), 0..=max_count)
}

/// Operations for random testing.
#[derive(Debug, Clone)]
enum Op {
    Insert(Vec<u8>, u64),
    Get(Vec<u8>),
    Update(Vec<u8>, u64),
    Remove(Vec<u8>),
}

/// Strategy for generating random operations with short keys.
fn operations(max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            3 => (short_key(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
            2 => short_key().prop_map(Op::Get),
            1 => (short_key(), any::<u64>()).prop_map(|(k, v)| Op::Update(k, v)),
            2 => short_key().prop_map(Op::Remove),
        ],
        0..=max_ops,
    )
}

/// Strategy for generating random operations with long keys (multi-layer).
fn operations_long_keys(max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            3 => (long_key(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
            2 => long_key().prop_map(Op::Get),
            1 => (long_key(), any::<u64>()).prop_map(|(k, v)| Op::Update(k, v)),
            2 => long_key().prop_map(Op::Remove),
        ],
        0..=max_ops,
    )
}

/// Strategy for generating very long keys (multiple layer descents).
fn very_long_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 17..=64)
}

/// Strategy for generating unique long keys.
fn unique_long_keys(max_count: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::hash_set(long_key(), 0..=max_count).prop_map(|set| set.into_iter().collect())
}

// ============================================================================
//  Basic Insert/Get Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Every inserted short key should be retrievable.
    #[test]
    fn insert_then_get_returns_value(key in short_key(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&key, value);

        let result = tree.get(&key);
        prop_assert!(result.is_some(), "Key {:?} not found after insert", key);
        prop_assert_eq!(result.unwrap(), value);
    }

    /// get_ref should return borrowed reference to the same value.
    #[test]
    fn get_ref_returns_borrowed_value(key in short_key(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&key, value);

        let guard = tree.guard();
        let result = tree.get_ref(&key, &guard);
        prop_assert!(result.is_some(), "Key {:?} not found via get_ref after insert", key);
        prop_assert_eq!(*result.unwrap(), value);
    }

    /// Inserting duplicate key should return the old value.
    #[test]
    fn insert_duplicate_returns_old_value(key in short_key_nonempty(), v1: u64, v2: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        let old1 = tree.insert(&key, v1);
        prop_assert!(old1.is_none(), "First insert should return None");

        let old2 = tree.insert(&key, v2);
        prop_assert!(old2.is_some(), "Second insert should return old value");
        prop_assert_eq!(old2.unwrap(), v1);

        // Current value should be v2
        prop_assert_eq!(tree.get(&key).unwrap(), v2);
    }

    /// Get on non-existent key returns None.
    #[test]
    fn get_missing_returns_none(
        inserted_key in short_key_nonempty(),
        missing_key in short_key_nonempty(),
        value: u64
    ) {
        prop_assume!(inserted_key != missing_key);

        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&inserted_key, value);

        prop_assert!(tree.get(&missing_key).is_none());
    }

    /// Long keys (requiring layers) should work.
    #[test]
    fn insert_long_key_works(key in long_key(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&key, value);

        let result = tree.get(&key);
        prop_assert!(result.is_some(), "Long key {:?} not found after insert", key);
        prop_assert_eq!(result.unwrap(), value);
    }
}

// ============================================================================
//  Differential Testing Against BTreeMap
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// MassTree should behave identically to BTreeMap for insert/get.
    #[test]
    fn differential_insert_get(pairs in key_value_pairs(100)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (key, value) in pairs {
            // Skip keys that are too long
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            let tree_old = tree.insert(&key, value);
            let oracle_old = oracle.insert(key.clone(), value);

            prop_assert_eq!(
                tree_old,
                oracle_old,
                "Insert mismatch for key {:?}",
                key
            );
        }

        // Verify all keys match
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert!(actual.is_some(), "Key {:?} missing from tree", key);
            prop_assert_eq!(actual.unwrap(), *expected);
        }
    }

    /// Random operation sequences should match BTreeMap behavior.
    #[test]
    fn differential_random_ops(ops in operations(150)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(key, value) | Op::Update(key, value) => {
                    if key.len() > SHORT_KEY_LEN {
                        continue;
                    }

                    let tree_old = tree.insert(&key, value);
                    let oracle_old = oracle.insert(key.clone(), value);

                    prop_assert_eq!(
                        tree_old,
                        oracle_old,
                        "Insert/Update mismatch for key {:?}",
                        key
                    );
                }

                Op::Get(key) => {
                    let tree_val = tree.get(&key);
                    let oracle_val = oracle.get(&key).copied();

                    prop_assert_eq!(
                        tree_val,
                        oracle_val,
                        "Get mismatch for key {:?}",
                        key
                    );
                }

                Op::Remove(key) => {
                    if key.len() > SHORT_KEY_LEN {
                        continue;
                    }

                    let tree_old = tree.remove(&key).unwrap();
                    let oracle_old = oracle.remove(&key);

                    prop_assert_eq!(
                        tree_old,
                        oracle_old,
                        "Remove mismatch for key {:?}",
                        key
                    );
                }
            }
        }

        // Final len check
        prop_assert_eq!(tree.len(), oracle.len(), "Length mismatch");
    }
}

// ============================================================================
//  Split Properties (triggers at WIDTH inserts)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// All keys survive splits intact.
    #[test]
    fn splits_preserve_all_keys(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert all keys
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        // Verify all keys are retrievable
        for (i, key) in keys.iter().enumerate() {
            let result = tree.get(key);
            prop_assert!(
                result.is_some(),
                "Key {:?} (index {}) lost after splits",
                key, i
            );
            prop_assert_eq!(result.unwrap(), i as u64);
        }

        // Verify len matches
        prop_assert_eq!(tree.len(), keys.len());
    }

    /// Sequential ascending inserts work correctly (right-edge splits).
    #[test]
    fn sequential_ascending_inserts(count in 1usize..100) {
        let tree: MassTree15<u64> = MassTree15::new();

        for i in 0..count {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64);
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            let result = tree.get(key.as_bytes());
            prop_assert!(result.is_some(), "Key {} not found", i);
            prop_assert_eq!(result.unwrap(), i as u64);
        }
    }

    /// Sequential descending inserts work correctly (left-edge splits).
    #[test]
    fn sequential_descending_inserts(count in 1usize..100) {
        let tree: MassTree15<u64> = MassTree15::new();

        for i in (0..count).rev() {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64);
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            let result = tree.get(key.as_bytes());
            prop_assert!(result.is_some(), "Key {} not found", i);
            prop_assert_eq!(result.unwrap(), i as u64);
        }
    }

    /// Interleaved inserts (even then odd) work correctly.
    #[test]
    fn interleaved_inserts(count in 1usize..50) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert evens first
        for i in (0..count).filter(|x| x % 2 == 0) {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64);
        }

        // Then odds
        for i in (0..count).filter(|x| x % 2 == 1) {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64);
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            prop_assert_eq!(tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }
}

// ============================================================================
//  Len Property
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// len() should equal the number of unique keys inserted.
    #[test]
    fn len_equals_unique_key_count(pairs in key_value_pairs(100)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut unique_keys: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();

        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            tree.insert(&key, value);
            unique_keys.insert(key);
        }

        prop_assert_eq!(tree.len(), unique_keys.len());
    }

    /// is_empty() should be true only when len() == 0.
    #[test]
    fn is_empty_consistent_with_len(pairs in key_value_pairs(20)) {
        let tree: MassTree15<u64> = MassTree15::new();

        prop_assert!(tree.is_empty());
        prop_assert_eq!(tree.len(), 0);

        let mut count = 0;
        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            // Only count first insert of each key
            if tree.get(&key).is_none() {
                count += 1;
            }

            tree.insert(&key, value);

            prop_assert!(!tree.is_empty());
            prop_assert_eq!(tree.len(), count);
        }
    }
}

// ============================================================================
//  MassTree15Inline Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// MassTree15Inline should behave like MassTree15 but return V directly.
    #[test]
    fn inline_mode_matches_arc_mode(pairs in key_value_pairs(50)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let inline: MassTree15Inline<u64> = MassTree15Inline::new();

        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            let tree_old = tree.insert(&key, value);
            let inline_old = inline.insert(&key, value);

            prop_assert_eq!(tree_old, inline_old, "Insert mismatch for key {:?}", key);
        }

        prop_assert_eq!(tree.len(), inline.len());
        prop_assert_eq!(tree.is_empty(), inline.is_empty());
    }

    /// Inline mode get returns Copy value directly.
    #[test]
    fn inline_get_returns_copy(key in valid_key_nonempty(), value: u64) {
        let inline: MassTree15Inline<u64> = MassTree15Inline::new();
        inline.insert(&key, value);

        let result: Option<u64> = inline.get(&key);
        prop_assert_eq!(result, Some(value));
    }
}

// ============================================================================
//  Smaller WIDTH Properties
// ============================================================================

// Note: WIDTH=3 and WIDTH=5 tests removed as we now only support WIDTH=24

// ============================================================================
//  Edge Cases
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Empty key should work.
    #[test]
    fn empty_key_works(value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(b"", value);
        prop_assert_eq!(tree.get(b"").unwrap(), value);
        prop_assert_eq!(tree.len(), 1);
    }

    /// Max-length key (8 bytes) should work.
    #[test]
    fn max_length_key_works(key in prop::collection::vec(any::<u8>(), 8..=8), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(&key, value);
        prop_assert_eq!(tree.get(&key).unwrap(), value);
    }

    /// Binary keys with null bytes should work.
    #[test]
    fn binary_keys_with_nulls(
        prefix in prop::collection::vec(any::<u8>(), 0..4),
        suffix in prop::collection::vec(any::<u8>(), 0..4),
        value: u64
    ) {
        let mut key = prefix;
        key.push(0x00); // null byte
        key.extend(suffix);
        key.truncate(SHORT_KEY_LEN);

        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&key, value);
        prop_assert_eq!(tree.get(&key).unwrap(), value);
    }

    /// Keys differing only in last byte should be distinct.
    #[test]
    fn keys_differ_in_last_byte(
        prefix in prop::collection::vec(any::<u8>(), 1..7),
        byte1: u8,
        byte2: u8,
        v1: u64,
        v2: u64
    ) {
        prop_assume!(byte1 != byte2);

        let mut key1 = prefix.clone();
        key1.push(byte1);

        let mut key2 = prefix;
        key2.push(byte2);

        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&key1, v1);
        tree.insert(&key2, v2);

        prop_assert_eq!(tree.get(&key1).unwrap(), v1);
        prop_assert_eq!(tree.get(&key2).unwrap(), v2);
        prop_assert_eq!(tree.len(), 2);
    }
}

// ============================================================================
//  Stress Test
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Large number of operations should maintain consistency.
    #[test]
    fn stress_test_many_operations(ops in operations(500)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(key, value) | Op::Update(key, value) => {
                    if key.len() > SHORT_KEY_LEN {
                        continue;
                    }
                    tree.insert(&key, value);
                    oracle.insert(key, value);
                }
                Op::Get(key) => {
                    let tree_val = tree.get(&key);
                    let oracle_val = oracle.get(&key).copied();
                    prop_assert_eq!(tree_val, oracle_val);
                }
                Op::Remove(key) => {
                    if key.len() > SHORT_KEY_LEN {
                        continue;
                    }
                    tree.remove(&key).unwrap();
                    oracle.remove(&key);
                }
            }
        }

        // Final verification
        prop_assert_eq!(tree.len(), oracle.len());

        for (key, expected) in &oracle {
            prop_assert_eq!(tree.get(key).unwrap(), *expected);
        }
    }
}

// ============================================================================
//  Remove Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Removing a key should return the old value and make get return None.
    #[test]
    fn remove_returns_old_value(key in short_key_nonempty(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(&key, value);
        prop_assert_eq!(tree.get(&key).unwrap(), value);

        let removed = tree.remove(&key).unwrap();
        prop_assert!(removed.is_some(), "Remove should return old value");
        prop_assert_eq!(removed.unwrap(), value);

        prop_assert!(tree.get(&key).is_none(), "Key should be gone after remove");
        prop_assert_eq!(tree.len(), 0);
    }

    /// Removing non-existent key returns None.
    #[test]
    fn remove_missing_returns_none(
        inserted_key in short_key_nonempty(),
        missing_key in short_key_nonempty(),
        value: u64
    ) {
        prop_assume!(inserted_key != missing_key);

        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(&inserted_key, value);

        let removed = tree.remove(&missing_key).unwrap();
        prop_assert!(removed.is_none());
        prop_assert_eq!(tree.len(), 1); // Original still there
    }

    /// Insert-remove-insert cycle should work correctly.
    #[test]
    fn insert_remove_insert_cycle(key in short_key_nonempty(), v1: u64, v2: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert
        tree.insert(&key, v1);
        prop_assert_eq!(tree.get(&key).unwrap(), v1);

        // Remove
        let removed = tree.remove(&key).unwrap();
        prop_assert_eq!(removed.unwrap(), v1);
        prop_assert!(tree.get(&key).is_none());

        // Re-insert with different value
        tree.insert(&key, v2);
        prop_assert_eq!(tree.get(&key).unwrap(), v2);
        prop_assert_eq!(tree.len(), 1);
    }

    /// Removing all keys should result in empty tree.
    #[test]
    fn remove_all_keys_makes_empty(keys in unique_keys(30)) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert all
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }
        prop_assert_eq!(tree.len(), keys.len());

        // Remove all
        for key in &keys {
            tree.remove(key).unwrap();
        }

        prop_assert!(tree.is_empty());
        prop_assert_eq!(tree.len(), 0);
    }

    /// Remove with long keys (multiple layers) works correctly.
    #[test]
    fn remove_long_key(key in long_key(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(&key, value);
        prop_assert_eq!(tree.get(&key).unwrap(), value);

        let removed = tree.remove(&key).unwrap();
        prop_assert_eq!(removed.unwrap(), value);
        prop_assert!(tree.get(&key).is_none());
    }

    /// Interleaved insert/remove should maintain consistency.
    #[test]
    fn interleaved_insert_remove(keys in unique_keys(40)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert first half
        for (i, key) in keys.iter().take(keys.len() / 2).enumerate() {
            tree.insert(key, i as u64);
            oracle.insert(key.clone(), i as u64);
        }

        // Remove some from first half
        for key in keys.iter().take(keys.len() / 4) {
            tree.remove(key).unwrap();
            oracle.remove(key);
        }

        // Insert second half
        for (i, key) in keys.iter().skip(keys.len() / 2).enumerate() {
            let val = (i + 100) as u64;
            tree.insert(key, val);
            oracle.insert(key.clone(), val);
        }

        // Verify consistency
        prop_assert_eq!(tree.len(), oracle.len());
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert!(actual.is_some(), "Key {:?} missing", key);
            prop_assert_eq!(actual.unwrap(), *expected);
        }
    }
}

// ============================================================================
//  Range Scan Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// scan_ref should return keys in lexicographic order.
    #[test]
    fn scan_returns_sorted_keys(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let mut scanned: Vec<Vec<u8>> = Vec::new();

        tree.scan_ref(
            RangeBound::Unbounded,
            RangeBound::Unbounded,
            |key: &[u8], _: &u64| {
                scanned.push(key.to_vec());
                true
            },
            &guard,
        );

        // Verify sorted order
        for window in scanned.windows(2) {
            prop_assert!(
                window[0] <= window[1],
                "Keys not in order: {:?} > {:?}",
                window[0], window[1]
            );
        }

        // Should have scanned all keys
        prop_assert_eq!(scanned.len(), keys.len());
    }

    /// scan_ref with range should only return keys in range.
    #[test]
    fn scan_respects_bounds(count in 10usize..50) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert sequential keys
        for i in 0..count {
            let key = format!("{i:04}");
            tree.insert(key.as_bytes(), i as u64);
        }

        let guard = tree.guard();

        // Scan middle range
        let start = format!("{:04}", count / 4);
        let end = format!("{:04}", count * 3 / 4);
        let mut scanned: Vec<(Vec<u8>, u64)> = Vec::new();

        tree.scan_ref(
            RangeBound::Included(start.as_bytes()),
            RangeBound::Excluded(end.as_bytes()),
            |key: &[u8], value: &u64| {
                scanned.push((key.to_vec(), *value));
                true
            },
            &guard,
        );

        // Verify all keys are in range
        for (key, _) in &scanned {
            prop_assert!(key.as_slice() >= start.as_bytes());
            prop_assert!(key.as_slice() < end.as_bytes());
        }
    }

    /// scan_prefix should only return keys with matching prefix.
    #[test]
    fn scan_prefix_filters_correctly(
        prefix in prop::collection::vec(any::<u8>(), 1..4),
        suffixes in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..4), 5..20)
    ) {
        let tree: MassTree15<u64> = MassTree15::new();

        // Insert keys with prefix
        for (i, suffix) in suffixes.iter().enumerate() {
            let mut key = prefix.clone();
            key.extend(suffix);
            key.truncate(SHORT_KEY_LEN);
            tree.insert(&key, i as u64);
        }

        // Insert some keys without prefix
        for i in 0..5 {
            let key = vec![0xFF, i as u8]; // Different prefix
            tree.insert(&key, 100 + i as u64);
        }

        let guard = tree.guard();
        let mut scanned: Vec<Vec<u8>> = Vec::new();

        tree.scan_prefix(&prefix, |key, _| {
            scanned.push(key.to_vec());
            true
        }, &guard);

        // All scanned keys should have the prefix
        for key in &scanned {
            prop_assert!(
                key.starts_with(&prefix),
                "Key {:?} doesn't start with prefix {:?}",
                key, prefix
            );
        }
    }

    /// Early termination in scan should work.
    #[test]
    fn scan_early_termination(count in 20usize..50, stop_at in 1usize..19) {
        let tree: MassTree15<u64> = MassTree15::new();

        for i in 0..count {
            let key = format!("{i:04}");
            tree.insert(key.as_bytes(), i as u64);
        }

        let guard = tree.guard();
        let mut scanned_count: usize = 0;

        tree.scan_ref(
            RangeBound::Unbounded,
            RangeBound::Unbounded,
            |_: &[u8], _: &u64| {
                scanned_count += 1;
                scanned_count < stop_at
            },
            &guard,
        );

        prop_assert_eq!(scanned_count, stop_at);
    }
}

// ============================================================================
//  Iterator Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// iter() should return all keys in sorted order.
    #[test]
    fn iter_returns_sorted(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let mut prev: Option<Vec<u8>> = None;

        for entry in tree.iter(&guard) {
            if let Some(ref p) = prev {
                prop_assert!(
                    p.as_slice() < entry.key(),
                    "Keys not in order: {:?} >= {:?}",
                    p, entry.key()
                );
            }
            prev = Some(entry.key().to_vec());
        }
    }

    /// iter() count should match len().
    #[test]
    fn iter_count_matches_len(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let iter_count = tree.iter(&guard).count();

        prop_assert_eq!(iter_count, tree.len());
        prop_assert_eq!(iter_count, keys.len());
    }

    /// range() should only return keys in range.
    #[test]
    fn range_respects_bounds(count in 10usize..50) {
        let tree: MassTree15<u64> = MassTree15::new();

        for i in 0..count {
            let key = format!("{i:04}");
            tree.insert(key.as_bytes(), i as u64);
        }

        let guard = tree.guard();

        let start = format!("{:04}", count / 4);
        let end = format!("{:04}", count * 3 / 4);

        for entry in tree.range(
            RangeBound::Included(start.as_bytes()),
            RangeBound::Excluded(end.as_bytes()),
            &guard
        ) {
            prop_assert!(entry.key() >= start.as_bytes());
            prop_assert!(entry.key() < end.as_bytes());
        }
    }

    /// DoubleEndedIterator: rev() should return keys in reverse order.
    #[test]
    fn iter_rev_returns_reverse_sorted(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let mut prev: Option<Vec<u8>> = None;

        for entry in tree.iter(&guard).rev() {
            if let Some(ref p) = prev {
                prop_assert!(
                    p.as_slice() > entry.key(),
                    "Reverse keys not in order: {:?} <= {:?}",
                    p, entry.key()
                );
            }
            prev = Some(entry.key().to_vec());
        }
    }

    /// Forward and reverse iteration should return same elements.
    #[test]
    fn iter_forward_reverse_same_elements(keys in unique_keys(30)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();

        let forward: Vec<Vec<u8>> = tree.iter(&guard)
            .map(|e| e.key().to_vec())
            .collect();

        let mut reverse: Vec<Vec<u8>> = tree.iter(&guard)
            .rev()
            .map(|e| e.key().to_vec())
            .collect();
        reverse.reverse();

        prop_assert_eq!(forward, reverse);
    }
}

// ============================================================================
//  MassTree15 Variant Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// MassTree15 should behave identically to MassTree15 for basic ops.
    #[test]
    fn masstree15_matches_oracle(pairs in key_value_pairs(80)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            tree.insert(&key, value);
            oracle.insert(key, value);
        }

        prop_assert_eq!(tree.len(), oracle.len());

        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert!(actual.is_some());
            prop_assert_eq!(actual.unwrap(), *expected);
        }
    }

    /// MassTree15 remove should work correctly.
    #[test]
    fn masstree15_remove_works(keys in unique_keys(40)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        // Remove half
        for key in keys.iter().take(keys.len() / 2) {
            let removed = tree.remove(key).unwrap();
            prop_assert!(removed.is_some());
        }

        prop_assert_eq!(tree.len(), keys.len() - keys.len() / 2);
    }

    /// MassTree15 iteration should work correctly.
    #[test]
    fn masstree15_iter_sorted(keys in unique_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let mut prev: Option<Vec<u8>> = None;

        for entry in tree.iter(&guard) {
            if let Some(ref p) = prev {
                prop_assert!(p.as_slice() < entry.key());
            }
            prev = Some(entry.key().to_vec());
        }
    }
}

// ============================================================================
//  Inline Storage Variant Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// MassTree15Inline should behave identically to oracle.
    #[test]
    fn masstree24_inline_matches_oracle(pairs in key_value_pairs(80)) {
        let tree: MassTree15Inline<u64> = MassTree15Inline::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            tree.insert(&key, value);
            oracle.insert(key, value);
        }

        prop_assert_eq!(tree.len(), oracle.len());

        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert_eq!(actual, Some(*expected));
        }
    }

    /// MassTree15Inline should behave identically to oracle.
    #[test]
    fn masstree15_inline_matches_oracle(pairs in key_value_pairs(80)) {
        let tree: MassTree15Inline<u64> = MassTree15Inline::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (key, value) in pairs {
            if key.len() > SHORT_KEY_LEN {
                continue;
            }

            tree.insert(&key, value);
            oracle.insert(key, value);
        }

        prop_assert_eq!(tree.len(), oracle.len());

        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert_eq!(actual, Some(*expected));
        }
    }

    /// Inline variant remove should work correctly.
    #[test]
    fn inline_remove_works(keys in unique_keys(40)) {
        let tree: MassTree15Inline<u64> = MassTree15Inline::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        for (i, key) in keys.iter().enumerate() {
            let removed = tree.remove(key).unwrap();
            prop_assert_eq!(removed, Some(i as u64));
        }

        prop_assert!(tree.is_empty());
    }

    /// Inline variants should handle update correctly (returning old value).
    #[test]
    fn inline_update_returns_old(key in short_key_nonempty(), v1: u64, v2: u64) {
        let tree: MassTree15Inline<u64> = MassTree15Inline::new();

        let old1 = tree.insert(&key, v1);
        prop_assert_eq!(old1, None);

        let old2 = tree.insert(&key, v2);
        prop_assert_eq!(old2, Some(v1));

        prop_assert_eq!(tree.get(&key), Some(v2));
    }
}

// ============================================================================
//  Multi-Layer Key Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Long keys (9-64 bytes) requiring layer descent should work.
    /// Uses differential testing against BTreeMap to verify correctness.
    ///
    /// The bug was fixed by adding `initializing` parameter to `assign_ksuf`.
    #[test]
    fn multi_layer_insert_get(keys in unique_long_keys(30)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
            oracle.insert(key.clone(), i as u64);
        }

        prop_assert_eq!(tree.len(), oracle.len(), "Length mismatch");

        for (key, expected) in &oracle {
            let result = tree.get(key);
            prop_assert!(
                result.is_some(),
                "Long key {:?} (len={}) not found. Tree len={}, Oracle len={}",
                key, key.len(), tree.len(), oracle.len()
            );
            prop_assert_eq!(result.unwrap(), *expected);
        }
    }

    /// Very long keys (17-64 bytes, 3+ layers) should work.
    #[test]
    fn very_long_keys_work(key in very_long_key(), value: u64) {
        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(&key, value);
        prop_assert_eq!(tree.get(&key).unwrap(), value);

        let removed = tree.remove(&key).unwrap();
        prop_assert_eq!(removed.unwrap(), value);
        prop_assert!(tree.is_empty());
    }

    /// Long key differential testing with insert/remove.
    #[test]
    fn long_key_differential(ops in operations_long_keys(100)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(key, value) | Op::Update(key, value) => {
                    tree.insert(&key, value);
                    oracle.insert(key, value);
                }
                Op::Get(key) => {
                    let tree_val = tree.get(&key);
                    let oracle_val = oracle.get(&key).copied();
                    prop_assert_eq!(tree_val, oracle_val);
                }
                Op::Remove(key) => {
                    tree.remove(&key).unwrap();
                    oracle.remove(&key);
                }
            }
        }

        prop_assert_eq!(tree.len(), oracle.len());
    }

    /// Keys with shared long prefixes should be distinct.
    #[test]
    fn shared_prefix_keys_distinct(
        prefix in prop::collection::vec(any::<u8>(), 16..32),
        suffix1: u8,
        suffix2: u8,
        v1: u64,
        v2: u64
    ) {
        prop_assume!(suffix1 != suffix2);

        let mut key1 = prefix.clone();
        key1.push(suffix1);

        let mut key2 = prefix;
        key2.push(suffix2);

        let tree: MassTree15<u64> = MassTree15::new();

        tree.insert(&key1, v1);
        tree.insert(&key2, v2);

        prop_assert_eq!(tree.get(&key1).unwrap(), v1);
        prop_assert_eq!(tree.get(&key2).unwrap(), v2);
        prop_assert_eq!(tree.len(), 2);
    }

    /// Iteration should work with long keys.
    #[test]
    fn iter_with_long_keys(keys in unique_long_keys(30)) {
        let tree: MassTree15<u64> = MassTree15::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }

        let guard = tree.guard();
        let iter_count = tree.iter(&guard).count();

        prop_assert_eq!(iter_count, keys.len());
    }
}

// ============================================================================
//  Comprehensive Proptest Infrastructure
// ============================================================================

/// Keys 0-64 bytes (all lengths, including multi-layer).
fn any_length_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=64)
}

/// Extended operation enum for stateful oracle testing.
#[derive(Debug, Clone)]
enum FullOp {
    Insert(Vec<u8>, u64),
    Get(Vec<u8>),
    Remove(Vec<u8>),
    ContainsKey(Vec<u8>),
    First,
    Last,
    Len,
    ScanAll,
    IterForward,
    IterReverse,
}

/// Strategy for generating full operation sequences with mixed-length keys.
fn full_operations(max_ops: usize) -> impl Strategy<Value = Vec<FullOp>> {
    prop::collection::vec(
        prop_oneof![
            4 => (any_length_key(), any::<u64>()).prop_map(|(k, v)| FullOp::Insert(k, v)),
            3 => any_length_key().prop_map(FullOp::Get),
            3 => any_length_key().prop_map(FullOp::Remove),
            2 => any_length_key().prop_map(FullOp::ContainsKey),
            1 => Just(FullOp::First),
            1 => Just(FullOp::Last),
            1 => Just(FullOp::Len),
            1 => Just(FullOp::ScanAll),
            1 => Just(FullOp::IterForward),
            1 => Just(FullOp::IterReverse),
        ],
        1..=max_ops,
    )
}

// ============================================================================
//  Verification Helpers
// ============================================================================

/// Verify `first()` and `last()` match oracle.
fn verify_first_last(
    tree: &MassTree15<u64>,
    oracle: &BTreeMap<Vec<u8>, u64>,
) -> Result<(), TestCaseError> {
    let tree_first = tree.first().map(|e| {
        let (k, v) = e.into_parts();
        (k, v)
    });
    let oracle_first = oracle.iter().next().map(|(k, v)| (k.clone(), *v));
    prop_assert_eq!(tree_first, oracle_first, "first() mismatch");

    let tree_last = tree.last().map(|e| {
        let (k, v) = e.into_parts();
        (k, v)
    });
    let oracle_last = oracle.iter().next_back().map(|(k, v)| (k.clone(), *v));
    prop_assert_eq!(tree_last, oracle_last, "last() mismatch");

    Ok(())
}

/// Verify full forward scan matches oracle.
fn verify_full_scan(
    tree: &MassTree15<u64>,
    oracle: &BTreeMap<Vec<u8>, u64>,
) -> Result<(), TestCaseError> {
    let guard = tree.guard();
    let limit = oracle.len() + 1000;
    let mut tree_entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for e in tree.iter(&guard) {
        let (k, v) = e.into_parts();
        tree_entries.push((k, *v));
        if tree_entries.len() > limit {
            prop_assert!(
                false,
                "Forward scan infinite loop: got >{limit} entries (expected {})",
                oracle.len()
            );
        }
    }
    let oracle_entries: Vec<(Vec<u8>, u64)> = oracle.iter().map(|(k, v)| (k.clone(), *v)).collect();
    prop_assert_eq!(tree_entries, oracle_entries, "Full scan mismatch");
    Ok(())
}

/// Verify full reverse iteration matches oracle.
fn verify_reverse_iter(
    tree: &MassTree15<u64>,
    oracle: &BTreeMap<Vec<u8>, u64>,
) -> Result<(), TestCaseError> {
    let guard = tree.guard();
    let limit = oracle.len() + 1000;
    let mut tree_entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for e in tree.iter(&guard).rev() {
        let (k, v) = e.into_parts();
        tree_entries.push((k, *v));
        if tree_entries.len() > limit {
            prop_assert!(
                false,
                "Reverse scan infinite loop: got >{limit} entries (expected {})",
                oracle.len()
            );
        }
    }
    let oracle_entries: Vec<(Vec<u8>, u64)> =
        oracle.iter().rev().map(|(k, v)| (k.clone(), *v)).collect();
    prop_assert_eq!(tree_entries, oracle_entries, "Reverse iter mismatch");
    Ok(())
}

// --------------------------- Stress Tests -----------------------------------

// ============================================================================
//  Test 1: Stateful Oracle — All Operations
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Model-based differential testing with BTreeMap oracle.
    /// Verifies the full invariant set after every mutation.
    #[test]
    fn stateful_oracle_all_ops(ops in full_operations(300)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (i, op) in ops.into_iter().enumerate() {
            match op {
                FullOp::Insert(key, value) => {
                    let tree_old = tree.insert(&key, value);
                    let oracle_old = oracle.insert(key.clone(), value);
                    prop_assert_eq!(
                        tree_old, oracle_old,
                        "Insert mismatch at op {} for key {:?}", i, key
                    );
                    prop_assert_eq!(
                        tree.len(), oracle.len(),
                        "Len mismatch after insert at op {}", i
                    );
                    verify_first_last(&tree, &oracle)?;
                }

                FullOp::Get(key) => {
                    let tree_val = tree.get(&key);
                    let oracle_val = oracle.get(&key).copied();
                    prop_assert_eq!(
                        tree_val, oracle_val,
                        "Get mismatch at op {} for key {:?}", i, key
                    );
                }

                FullOp::Remove(key) => {
                    let tree_old = tree.remove(&key).unwrap();
                    let oracle_old = oracle.remove(&key);
                    prop_assert_eq!(
                        tree_old, oracle_old,
                        "Remove mismatch at op {} for key {:?}", i, key
                    );
                    prop_assert_eq!(
                        tree.len(), oracle.len(),
                        "Len mismatch after remove at op {}", i
                    );
                    verify_first_last(&tree, &oracle)?;
                }

                FullOp::ContainsKey(key) => {
                    let tree_has = tree.contains_key(&key);
                    let oracle_has = oracle.contains_key(&key);
                    prop_assert_eq!(
                        tree_has, oracle_has,
                        "ContainsKey mismatch at op {} for key {:?}", i, key
                    );
                }

                FullOp::First => {
                    let tree_first = tree.first().map(|e| {
                        let (k, v) = e.into_parts();
                        (k, v)
                    });
                    let oracle_first =
                        oracle.iter().next().map(|(k, v)| (k.clone(), *v));
                    prop_assert_eq!(
                        tree_first, oracle_first,
                        "First mismatch at op {}", i
                    );
                }

                FullOp::Last => {
                    let tree_last = tree.last().map(|e| {
                        let (k, v) = e.into_parts();
                        (k, v)
                    });
                    let oracle_last =
                        oracle.iter().next_back().map(|(k, v)| (k.clone(), *v));
                    prop_assert_eq!(
                        tree_last, oracle_last,
                        "Last mismatch at op {}", i
                    );
                }

                FullOp::Len => {
                    prop_assert_eq!(
                        tree.len(), oracle.len(),
                        "Len mismatch at op {}", i
                    );
                }

                FullOp::ScanAll => {
                    verify_full_scan(&tree, &oracle)?;
                }

                FullOp::IterForward => {
                    let guard = tree.guard();
                    let tree_entries: Vec<(Vec<u8>, u64)> = tree
                        .iter(&guard)
                        .map(|e| {
                            let (k, v) = e.into_parts();
                            (k, *v)
                        })
                        .collect();
                    let oracle_entries: Vec<(Vec<u8>, u64)> =
                        oracle.iter().map(|(k, v)| (k.clone(), *v)).collect();
                    prop_assert_eq!(
                        tree_entries, oracle_entries,
                        "IterForward mismatch at op {}", i
                    );
                }

                FullOp::IterReverse => {
                    verify_reverse_iter(&tree, &oracle)?;
                }
            }

            // Every 10th operation: full scan comparison
            if i % 10 == 9 {
                verify_full_scan(&tree, &oracle)?;
            }

            // Every 25th operation: full reverse iter comparison
            if i % 25 == 24 {
                verify_reverse_iter(&tree, &oracle)?;
            }
        }
    }
}

// ============================================================================
//  Test 2: Split-Heavy Sequential Insert Then Remove All
// ============================================================================

/// Generate a sequential key from index, with variable padding for multi-layer coverage.
fn make_sequential_key(i: u64) -> Vec<u8> {
    let mut key = i.to_be_bytes().to_vec();
    let extra = (i % 4) as usize * 8; // 0, 8, 16, 24 extra bytes
    key.resize(key.len() + extra, 0);
    key
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Sequential ascending inserts force continuous right-edge splits.
    /// Then remove ALL keys and verify tree is empty.
    #[test]
    fn split_heavy_sequential_then_remove_all(count in 50usize..=200) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert sequential ascending keys (forces right-edge splits)
        let keys: Vec<Vec<u8>> = (0..count as u64).map(make_sequential_key).collect();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
            oracle.insert(key.clone(), i as u64);
        }

        prop_assert_eq!(tree.len(), count);

        // Verify all present via get
        for (i, key) in keys.iter().enumerate() {
            let got = tree.get(key);
            prop_assert_eq!(got, Some(i as u64), "Key {} missing after inserts", i);
        }

        // Full scan comparison before removal
        verify_full_scan(&tree, &oracle)?;

        // Remove all — scrambled order: evens descending, then odds ascending
        let mut removal_order: Vec<usize> =
            (0..count).rev().filter(|i| i % 2 == 0).collect();
        removal_order.extend((0..count).filter(|i| i % 2 == 1));

        for idx in &removal_order {
            let key = &keys[*idx];
            let removed = tree.remove(key).unwrap();
            prop_assert!(removed.is_some(), "Remove returned None for key {}", idx);
            prop_assert_eq!(removed.unwrap(), *idx as u64);
            oracle.remove(key);
        }

        prop_assert!(tree.is_empty());
        prop_assert_eq!(tree.len(), 0);
        prop_assert!(tree.first().is_none());
        prop_assert!(tree.last().is_none());
    }
}

// ============================================================================
//  Test 3: Insert → Remove → Reinsert Cycle
// ============================================================================

/// Unique keys of any length (1-64 bytes).
fn unique_any_keys(max_count: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::hash_set(prop::collection::vec(any::<u8>(), 1..=64), 1..=max_count)
        .prop_map(|set| set.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Stress insert→remove→reinsert for the same keys, exercising slot reuse.
    #[test]
    fn insert_remove_reinsert_cycle(keys in unique_any_keys(50)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Phase 1: Insert N unique keys
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
            oracle.insert(key.clone(), i as u64);
        }
        prop_assert_eq!(tree.len(), oracle.len());

        // Phase 2: Remove all keys, verify each returns Some(old_value)
        for (i, key) in keys.iter().enumerate() {
            let removed = tree.remove(key).unwrap();
            prop_assert_eq!(
                removed, Some(i as u64),
                "Remove phase: key {:?} (index {}) mismatch", key, i
            );
            oracle.remove(key);
        }
        prop_assert!(tree.is_empty());
        prop_assert!(oracle.is_empty());

        // Phase 3: Reinsert all keys with new values
        for (i, key) in keys.iter().enumerate() {
            let v = (i as u64) + 1000;
            let old = tree.insert(key, v);
            prop_assert!(old.is_none(), "Reinsert should not find old value");
            oracle.insert(key.clone(), v);
        }

        // Verify all present with new values
        verify_full_scan(&tree, &oracle)?;

        // Phase 4: Remove half, reinsert with third values
        let half = keys.len() / 2;
        for key in keys.iter().take(half) {
            tree.remove(key).unwrap();
            oracle.remove(key);
        }
        for (i, key) in keys.iter().take(half).enumerate() {
            let v = (i as u64) + 2000;
            tree.insert(key, v);
            oracle.insert(key.clone(), v);
        }

        // Final full verification
        verify_full_scan(&tree, &oracle)?;
        verify_first_last(&tree, &oracle)?;
        prop_assert_eq!(tree.len(), oracle.len());
    }
}

// ============================================================================
//  Test 4: Shared Prefix Multi-Layer Stress
// ============================================================================

/// Keys with shared prefix (forces deep multi-layer trie descent).
/// Uses vec + dedup instead of `hash_set` for faster proptest generation.
fn shared_prefix_keys_strategy(max_count: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    (
        prop::collection::vec(any::<u8>(), 16..=24),
        prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..=40),
            max_count..=max_count,
        ),
    )
        .prop_map(|(prefix, suffixes)| {
            let mut keys: Vec<Vec<u8>> = suffixes
                .into_iter()
                .map(|s| {
                    let mut key = prefix.clone();
                    key.extend(s);
                    key
                })
                .collect();
            keys.sort();
            keys.dedup();
            keys
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Keys sharing long prefixes force deep trie descent (3+ layers).
    #[test]
    fn shared_prefix_multi_layer_stress(keys in shared_prefix_keys_strategy(50)) {
        let tree: MassTree15<u64> = MassTree15::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert all
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
            oracle.insert(key.clone(), i as u64);
        }
        prop_assert_eq!(tree.len(), oracle.len());

        // Verify all present
        for (i, key) in keys.iter().enumerate() {
            let got = tree.get(key);
            prop_assert_eq!(got, Some(i as u64), "Key index {} missing", i);
        }

        // Full scan comparison
        verify_full_scan(&tree, &oracle)?;

        // Remove half
        let half = keys.len() / 2;
        for key in keys.iter().take(half) {
            tree.remove(key).unwrap();
            oracle.remove(key);
        }

        // Verify removed are gone, remaining are present
        for (i, key) in keys.iter().enumerate() {
            let got = tree.get(key);
            if i < half {
                prop_assert_eq!(got, None, "Key index {} should be removed", i);
            } else {
                prop_assert_eq!(got, Some(i as u64), "Key index {} should remain", i);
            }
        }

        // Full scan and reverse iter comparison
        verify_full_scan(&tree, &oracle)?;
        verify_reverse_iter(&tree, &oracle)?;
        verify_first_last(&tree, &oracle)?;
    }
}
