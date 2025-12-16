//! Property-based tests for the `tree` module.
//!
//! These tests verify invariants and properties that should hold for all inputs.
//! Uses differential testing against `BTreeMap` as an oracle.

#![expect(clippy::unwrap_used, reason = "fail fast in tests")]

use masstree::tree::{InsertError, MAX_INLINE_KEY_LEN, MassTree, MassTreeIndex};
use proptest::prelude::*;
use std::collections::BTreeMap;

// ============================================================================
//  Strategies
// ============================================================================

/// Strategy for generating valid Phase 1 keys (0-8 bytes).
fn valid_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=MAX_INLINE_KEY_LEN)
}

/// Strategy for generating non-empty valid keys (1-8 bytes).
fn valid_key_nonempty() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=MAX_INLINE_KEY_LEN)
}

/// Strategy for generating keys that are too long (9+ bytes).
fn invalid_key_too_long() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), (MAX_INLINE_KEY_LEN + 1)..=32)
}

/// Strategy for generating a sequence of unique keys.
fn unique_keys(max_count: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::hash_set(valid_key_nonempty(), 0..=max_count)
        .prop_map(|set| set.into_iter().collect())
}

/// Strategy for generating key-value pairs.
fn key_value_pairs(max_count: usize) -> impl Strategy<Value = Vec<(Vec<u8>, u64)>> {
    prop::collection::vec((valid_key(), any::<u64>()), 0..=max_count)
}

/// Operations for random testing.
#[derive(Debug, Clone)]
enum Op {
    Insert(Vec<u8>, u64),
    Get(Vec<u8>),
    Update(Vec<u8>, u64),
}

/// Strategy for generating random operations.
fn operations(max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            3 => (valid_key(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
            2 => valid_key().prop_map(Op::Get),
            1 => (valid_key(), any::<u64>()).prop_map(|(k, v)| Op::Update(k, v)),
        ],
        0..=max_ops,
    )
}

// ============================================================================
//  Basic Insert/Get Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Every inserted key should be retrievable.
    #[test]
    fn insert_then_get_returns_value(key in valid_key(), value: u64) {
        let mut tree: MassTree<u64> = MassTree::new();
        tree.insert(&key, value).unwrap();

        let result = tree.get(&key);
        prop_assert!(result.is_some(), "Key {:?} not found after insert", key);
        prop_assert_eq!(*result.unwrap(), value);
    }

    /// Inserting duplicate key should return the old value.
    #[test]
    fn insert_duplicate_returns_old_value(key in valid_key_nonempty(), v1: u64, v2: u64) {
        let mut tree: MassTree<u64> = MassTree::new();

        let old1 = tree.insert(&key, v1).unwrap();
        prop_assert!(old1.is_none(), "First insert should return None");

        let old2 = tree.insert(&key, v2).unwrap();
        prop_assert!(old2.is_some(), "Second insert should return old value");
        prop_assert_eq!(*old2.unwrap(), v1);

        // Current value should be v2
        prop_assert_eq!(*tree.get(&key).unwrap(), v2);
    }

    /// Get on non-existent key returns None.
    #[test]
    fn get_missing_returns_none(
        inserted_key in valid_key_nonempty(),
        missing_key in valid_key_nonempty(),
        value: u64
    ) {
        prop_assume!(inserted_key != missing_key);

        let mut tree: MassTree<u64> = MassTree::new();
        tree.insert(&inserted_key, value).unwrap();

        prop_assert!(tree.get(&missing_key).is_none());
    }

    /// Keys that are too long should be rejected.
    #[test]
    fn long_keys_rejected(key in invalid_key_too_long(), value: u64) {
        let mut tree: MassTree<u64> = MassTree::new();
        let result = tree.insert(&key, value);
        prop_assert!(matches!(result, Err(InsertError::KeyTooLong)));
    }

    /// Get with too-long key returns None (not error).
    #[test]
    fn get_long_key_returns_none(key in invalid_key_too_long()) {
        let tree: MassTree<u64> = MassTree::new();
        prop_assert!(tree.get(&key).is_none());
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
        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for (key, value) in pairs {
            // Skip keys that are too long
            if key.len() > MAX_INLINE_KEY_LEN {
                continue;
            }

            let tree_old = tree.insert(&key, value).unwrap();
            let oracle_old = oracle.insert(key.clone(), value);

            prop_assert_eq!(
                tree_old.map(|arc| *arc),
                oracle_old,
                "Insert mismatch for key {:?}",
                key
            );
        }

        // Verify all keys match
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            prop_assert!(actual.is_some(), "Key {:?} missing from tree", key);
            prop_assert_eq!(*actual.unwrap(), *expected);
        }
    }

    /// Random operation sequences should match BTreeMap behavior.
    #[test]
    fn differential_random_ops(ops in operations(150)) {
        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(key, value) | Op::Update(key, value) => {
                    if key.len() > MAX_INLINE_KEY_LEN {
                        continue;
                    }

                    let tree_old = tree.insert(&key, value).unwrap();
                    let oracle_old = oracle.insert(key.clone(), value);

                    prop_assert_eq!(
                        tree_old.map(|arc| *arc),
                        oracle_old,
                        "Insert/Update mismatch for key {:?}",
                        key
                    );
                }

                Op::Get(key) => {
                    let tree_val = tree.get(&key).map(|arc| *arc);
                    let oracle_val = oracle.get(&key).copied();

                    prop_assert_eq!(
                        tree_val,
                        oracle_val,
                        "Get mismatch for key {:?}",
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
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert all keys
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }

        // Verify all keys are retrievable
        for (i, key) in keys.iter().enumerate() {
            let result = tree.get(key);
            prop_assert!(
                result.is_some(),
                "Key {:?} (index {}) lost after splits",
                key, i
            );
            prop_assert_eq!(*result.unwrap(), i as u64);
        }

        // Verify len matches
        prop_assert_eq!(tree.len(), keys.len());
    }

    /// Sequential ascending inserts work correctly (right-edge splits).
    #[test]
    fn sequential_ascending_inserts(count in 1usize..100) {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in 0..count {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            let result = tree.get(key.as_bytes());
            prop_assert!(result.is_some(), "Key {} not found", i);
            prop_assert_eq!(*result.unwrap(), i as u64);
        }
    }

    /// Sequential descending inserts work correctly (left-edge splits).
    #[test]
    fn sequential_descending_inserts(count in 1usize..100) {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in (0..count).rev() {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            let result = tree.get(key.as_bytes());
            prop_assert!(result.is_some(), "Key {} not found", i);
            prop_assert_eq!(*result.unwrap(), i as u64);
        }
    }

    /// Interleaved inserts (even then odd) work correctly.
    #[test]
    fn interleaved_inserts(count in 1usize..50) {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert evens first
        for i in (0..count).filter(|x| x % 2 == 0) {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Then odds
        for i in (0..count).filter(|x| x % 2 == 1) {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        prop_assert_eq!(tree.len(), count);

        for i in 0..count {
            let key = format!("{i:08}");
            prop_assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
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
        let mut tree: MassTree<u64> = MassTree::new();
        let mut unique_keys: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();

        for (key, value) in pairs {
            if key.len() > MAX_INLINE_KEY_LEN {
                continue;
            }

            tree.insert(&key, value).unwrap();
            unique_keys.insert(key);
        }

        prop_assert_eq!(tree.len(), unique_keys.len());
    }

    /// is_empty() should be true only when len() == 0.
    #[test]
    fn is_empty_consistent_with_len(pairs in key_value_pairs(20)) {
        let mut tree: MassTree<u64> = MassTree::new();

        prop_assert!(tree.is_empty());
        prop_assert_eq!(tree.len(), 0);

        let mut count = 0;
        for (key, value) in pairs {
            if key.len() > MAX_INLINE_KEY_LEN {
                continue;
            }

            // Only count first insert of each key
            if tree.get(&key).is_none() {
                count += 1;
            }

            tree.insert(&key, value).unwrap();

            prop_assert!(!tree.is_empty());
            prop_assert_eq!(tree.len(), count);
        }
    }
}

// ============================================================================
//  MassTreeIndex Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// MassTreeIndex should behave like MassTree but return V directly.
    #[test]
    fn index_mode_matches_arc_mode(pairs in key_value_pairs(50)) {
        let mut tree: MassTree<u64> = MassTree::new();
        let mut index: MassTreeIndex<u64> = MassTreeIndex::new();

        for (key, value) in pairs {
            if key.len() > MAX_INLINE_KEY_LEN {
                continue;
            }

            let tree_old = tree.insert(&key, value).unwrap().map(|arc| *arc);
            let index_old = index.insert(&key, value).unwrap();

            prop_assert_eq!(tree_old, index_old, "Insert mismatch for key {:?}", key);
        }

        prop_assert_eq!(tree.len(), index.len());
        prop_assert_eq!(tree.is_empty(), index.is_empty());
    }

    /// Index mode get returns Copy value directly.
    #[test]
    fn index_get_returns_copy(key in valid_key_nonempty(), value: u64) {
        let mut index: MassTreeIndex<u64> = MassTreeIndex::new();
        index.insert(&key, value).unwrap();

        let result: Option<u64> = index.get(&key);
        prop_assert_eq!(result, Some(value));
    }
}

// ============================================================================
//  Smaller WIDTH Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Tree with WIDTH=3 should work correctly (more splits).
    #[test]
    fn small_width_tree_works(keys in unique_keys(30)) {
        let mut tree: MassTree<u64, 3> = MassTree::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }

        for (i, key) in keys.iter().enumerate() {
            let result = tree.get(key);
            prop_assert!(result.is_some(), "Key {:?} not found in WIDTH=3 tree", key);
            prop_assert_eq!(*result.unwrap(), i as u64);
        }

        prop_assert_eq!(tree.len(), keys.len());
    }

    /// Tree with WIDTH=5 should work correctly.
    #[test]
    fn medium_width_tree_works(keys in unique_keys(40)) {
        let mut tree: MassTree<u64, 5> = MassTree::new();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }

        for (i, key) in keys.iter().enumerate() {
            prop_assert!(tree.get(key).is_some());
            prop_assert_eq!(*tree.get(key).unwrap(), i as u64);
        }

        prop_assert_eq!(tree.len(), keys.len());
    }
}

// ============================================================================
//  Edge Cases
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Empty key should work.
    #[test]
    fn empty_key_works(value: u64) {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"", value).unwrap();
        prop_assert_eq!(*tree.get(b"").unwrap(), value);
        prop_assert_eq!(tree.len(), 1);
    }

    /// Max-length key (8 bytes) should work.
    #[test]
    fn max_length_key_works(key in prop::collection::vec(any::<u8>(), 8..=8), value: u64) {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(&key, value).unwrap();
        prop_assert_eq!(*tree.get(&key).unwrap(), value);
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
        key.truncate(MAX_INLINE_KEY_LEN);

        let mut tree: MassTree<u64> = MassTree::new();
        tree.insert(&key, value).unwrap();
        prop_assert_eq!(*tree.get(&key).unwrap(), value);
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

        let mut tree: MassTree<u64> = MassTree::new();
        tree.insert(&key1, v1).unwrap();
        tree.insert(&key2, v2).unwrap();

        prop_assert_eq!(*tree.get(&key1).unwrap(), v1);
        prop_assert_eq!(*tree.get(&key2).unwrap(), v2);
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
        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(key, value) | Op::Update(key, value) => {
                    if key.len() > MAX_INLINE_KEY_LEN {
                        continue;
                    }
                    tree.insert(&key, value).unwrap();
                    oracle.insert(key, value);
                }
                Op::Get(key) => {
                    let tree_val = tree.get(&key).map(|arc| *arc);
                    let oracle_val = oracle.get(&key).copied();
                    prop_assert_eq!(tree_val, oracle_val);
                }
            }
        }

        // Final verification
        prop_assert_eq!(tree.len(), oracle.len());

        for (key, expected) in &oracle {
            prop_assert_eq!(*tree.get(key).unwrap(), *expected);
        }
    }
}
