#![allow(clippy::indexing_slicing)]
#![allow(unused_variables)]

use super::{InsertError, MassTree, MassTree15, MassTree15Inline, MassTreeGeneric};
use crate::nodeversion::NodeVersion;
use crate::value::LeafValue;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;

// ========================================================================
// Send/Sync Verification
// ========================================================================
//
// MassTree<V> is Send + Sync when V: Send + Sync.
//
// The struct uses:
// - AtomicPtr<u8> for root_ptr (Send + Sync)
// - AtomicUsize for count (Send + Sync)
// - PhantomData<V> which inherits Send/Sync from V
//
// This enables concurrent access via Arc<MassTree<V>>.

fn _assert_send_sync()
where
    MassTree<u64>: Send + Sync,
{
}

// ========================================================================
//  MassTree Basic Tests
// ========================================================================

#[test]
fn test_new_tree_is_empty() {
    let tree: MassTree<u64> = MassTree::new();

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_get_on_empty_tree() {
    let tree: MassTree<u64> = MassTree::new();

    assert!(tree.get(b"hello").is_none());
    assert!(tree.get(b"").is_none());
    assert!(tree.get(b"any key").is_none());
}

#[test]
fn test_contains_key() {
    let tree: MassTree<u64> = MassTree::new();

    // Empty tree
    assert!(!tree.contains_key(b"hello"));
    assert!(!tree.contains_key(b""));

    // Insert and check
    tree.insert(b"hello", 42).unwrap();
    assert!(tree.contains_key(b"hello"));
    assert!(!tree.contains_key(b"world"));

    // Multiple keys
    tree.insert(b"world", 123).unwrap();
    tree.insert(b"foo", 456).unwrap();
    assert!(tree.contains_key(b"hello"));
    assert!(tree.contains_key(b"world"));
    assert!(tree.contains_key(b"foo"));
    assert!(!tree.contains_key(b"bar"));

    // With guard
    let guard = tree.guard();
    assert!(tree.contains_key_with_guard(b"hello", &guard));
    assert!(!tree.contains_key_with_guard(b"missing", &guard));

    // After remove
    tree.remove(b"hello").unwrap();
    assert!(!tree.contains_key(b"hello"));
    assert!(tree.contains_key(b"world"));
}

#[test]
fn test_contains_key_long_keys() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Keys > 8 bytes (multi-layer)
    let long_key = b"this is a very long key that spans multiple layers";
    let another_long = b"this is a very long key that spans multiple layers but different";

    assert!(!tree.contains_key(long_key));

    tree.insert(long_key, 1).unwrap();
    assert!(tree.contains_key(long_key));
    assert!(!tree.contains_key(another_long));

    tree.insert(another_long, 2).unwrap();
    assert!(tree.contains_key(long_key));
    assert!(tree.contains_key(another_long));
}

#[test]
fn test_first_last_empty() {
    let tree: MassTree<u64> = MassTree::new();

    assert!(tree.first().is_none());
    assert!(tree.last().is_none());
}

#[test]
fn test_first_last_single_element() {
    let tree: MassTree<u64> = MassTree::new();
    tree.insert(b"only", 42).unwrap();

    let first = tree.first().unwrap();
    let last = tree.last().unwrap();

    assert_eq!(first.key(), b"only");
    assert_eq!(first.value, 42);
    assert_eq!(last.key(), b"only");
    assert_eq!(last.value, 42);
}

#[test]
fn test_first_last_multiple_elements() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert in non-sorted order
    tree.insert(b"banana", 2).unwrap();
    tree.insert(b"apple", 1).unwrap();
    tree.insert(b"cherry", 3).unwrap();
    tree.insert(b"date", 4).unwrap();

    let first = tree.first().unwrap();
    let last = tree.last().unwrap();

    assert_eq!(first.key(), b"apple");
    assert_eq!(first.value, 1);
    assert_eq!(last.key(), b"date");
    assert_eq!(last.value, 4);

    // With guard
    let guard = tree.guard();
    let first_g = tree.first_with_guard(&guard).unwrap();
    let last_g = tree.last_with_guard(&guard).unwrap();
    assert_eq!(first_g.key(), b"apple");
    assert_eq!(last_g.key(), b"date");
}

#[test]
fn test_first_last_long_keys() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Keys > 8 bytes to test multi-layer behavior
    tree.insert(b"zzz_long_key_that_spans_layers", 3).unwrap();
    tree.insert(b"aaa_long_key_that_spans_layers", 1).unwrap();
    tree.insert(b"mmm_long_key_that_spans_layers", 2).unwrap();

    let first = tree.first().unwrap();
    let last = tree.last().unwrap();

    assert_eq!(first.key(), b"aaa_long_key_that_spans_layers");
    assert_eq!(*first.value, 1);
    assert_eq!(last.key(), b"zzz_long_key_that_spans_layers");
    assert_eq!(*last.value, 3);
}

#[test]
fn test_first_last_after_remove() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"a", 1).unwrap();
    tree.insert(b"b", 2).unwrap();
    tree.insert(b"c", 3).unwrap();

    assert_eq!(tree.first().unwrap().key(), b"a");
    assert_eq!(tree.last().unwrap().key(), b"c");

    // Remove first
    tree.remove(b"a").unwrap();
    assert_eq!(tree.first().unwrap().key(), b"b");
    assert_eq!(tree.last().unwrap().key(), b"c");

    // Remove last
    tree.remove(b"c").unwrap();
    assert_eq!(tree.first().unwrap().key(), b"b");
    assert_eq!(tree.last().unwrap().key(), b"b");

    // Remove remaining
    tree.remove(b"b").unwrap();
    assert!(tree.first().is_none());
    assert!(tree.last().is_none());
}

#[test]
#[expect(clippy::panic)]
#[expect(clippy::too_many_lines)]
#[cfg_attr(miri, ignore)] // Miri struggles with multi-threaded tests
fn concurrent_insert_then_get_does_not_lose_key() {
    // This reproduces the "insert returns Ok(None) but immediate get returns None" bug.
    // If it fails, we also scan the leaf B-link chain to determine whether the key
    // is truly missing from all leaves, or only unreachable via the normal get path.
    #[expect(clippy::cast_ptr_alignment)]
    fn scan_leaf_chain_contains(tree: &MassTree<u64>, ikey: u64) -> bool {
        use crate::internode::InternodeNode;
        use crate::leaf15::LeafNode15;

        let _guard = tree.guard();

        // Descend to the leftmost leaf from the current root.
        let mut node: *mut u8 = tree.root_ptr.load(std::sync::atomic::Ordering::Acquire);
        let mut depth_limit: usize = 0;
        while !node.is_null() {
            depth_limit += 1;
            assert!(depth_limit < 128, "unexpected depth while scanning");

            // SAFETY: `node` is a tree node protected by the guard and root pointer.
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };
            let _ = version.stable();

            if version.is_leaf() {
                break;
            }

            // SAFETY: `!is_leaf()` means this is an internode. Single-threaded test context.
            let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };
            node = unsafe { inode.child_unguarded(0) };
        }

        if node.is_null() {
            return false;
        }

        // SAFETY: node points at a leaf.
        let mut leaf_ptr: *mut LeafNode15<LeafValue<u64>> = node.cast();
        let mut leaf_steps: usize = 0;

        while !leaf_ptr.is_null() {
            leaf_steps += 1;
            assert!(leaf_steps < 100_000, "possible leaf chain cycle");

            // SAFETY: leaf_ptr is protected by the guard and leaf nodes are never freed
            // while guarded.
            let leaf: &LeafNode15<LeafValue<u64>> = unsafe { &*leaf_ptr };
            let _ = leaf.version().stable();

            let perm = leaf.permutation();
            for i in 0..perm.size() {
                let slot = perm.get(i);
                if leaf.ikey(slot) == ikey && leaf.keylenx(slot) == 8 {
                    let ptr = leaf.leaf_value_ptr(slot);
                    if !ptr.is_null() {
                        return true;
                    }
                }
            }

            // SAFETY: Single-threaded test context.
            leaf_ptr = unsafe { leaf.safe_next_unguarded() };
        }

        false
    }

    const NUM_THREADS: usize = 4;
    const OPS_PER_THREAD: usize = 500;

    for run in 0..30 {
        let tree = Arc::new(MassTree::<u64>::new());
        let missing_key = Arc::new(AtomicU64::new(u64::MAX));
        let missing_thread = Arc::new(AtomicUsize::new(usize::MAX));
        let missing_i = Arc::new(AtomicUsize::new(usize::MAX));

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let missing_key = Arc::clone(&missing_key);
                let missing_thread = Arc::clone(&missing_thread);
                let missing_i = Arc::clone(&missing_i);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..OPS_PER_THREAD {
                        let key_val: u64 = (t * 10_000 + i) as u64;
                        let key = key_val.to_be_bytes();

                        let _ = tree.insert_with_guard(&key, key_val, &guard);

                        if tree.get_with_guard(&key, &guard).is_none() {
                            let _ = missing_key.compare_exchange(
                                u64::MAX,
                                key_val,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            );
                            missing_thread.store(t, Ordering::SeqCst);
                            missing_i.store(i, Ordering::SeqCst);
                            break;
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let key_val: u64 = missing_key.load(Ordering::SeqCst);
        if key_val != u64::MAX {
            let t: usize = missing_thread.load(Ordering::SeqCst);
            let i: usize = missing_i.load(Ordering::SeqCst);

            let in_leaf_chain = scan_leaf_chain_contains(&tree, key_val);
            panic!(
                "run {run}: insert/get lost key: t={t} i={i} key=0x{key_val:016x} tree.len()={} scan_leaf_chain_contains={in_leaf_chain}",
                tree.len(),
            );
        }

        assert_eq!(tree.len(), NUM_THREADS * OPS_PER_THREAD);
    }
}

// ========================================================================
//  MassTree15Inline Basic Tests
// ========================================================================

#[test]
fn test_index_new_is_empty() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_index_get_on_empty() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    assert!(tree.get(b"hello").is_none());
}

// ========================================================================
//  Key Handling Tests
// ========================================================================

#[test]
fn test_get_with_various_key_lengths() {
    let tree: MassTree<u64> = MassTree::new();

    // All should return None on empty tree
    assert!(tree.get(b"").is_none());
    assert!(tree.get(b"a").is_none());
    assert!(tree.get(b"ab").is_none());
    assert!(tree.get(b"abc").is_none());
    assert!(tree.get(b"abcd").is_none());
    assert!(tree.get(b"abcde").is_none());
    assert!(tree.get(b"abcdef").is_none());
    assert!(tree.get(b"abcdefg").is_none());
    assert!(tree.get(b"abcdefgh").is_none()); // Exactly 8 bytes
}

#[test]
fn test_get_missing_long_keys() {
    let tree: MassTree<u64> = MassTree::new();

    // Long keys should return None on empty tree (not inserted)
    assert!(tree.get(b"abcdefghi").is_none()); // 9 bytes
    assert!(tree.get(b"hello world").is_none()); // 11 bytes
    assert!(tree.get(b"this is a long key").is_none());
}

// ========================================================================
//  Default Trait Tests
// ========================================================================

#[test]
fn test_default_trait() {
    let tree: MassTree<u64> = MassTree::default();

    assert!(tree.is_empty());
}

#[test]
fn test_index_default_trait() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::default();

    assert!(tree.is_empty());
}

// ========================================================================
//  Get After Insert Tests (using public API)
// ========================================================================

#[test]
fn test_get_after_insert() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"hello", 42).unwrap();

    let result = tree.get(b"hello");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_get_wrong_key_after_insert() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"hello", 42).unwrap();

    assert!(tree.get(b"world").is_none());
    assert!(tree.get(b"hell").is_none());
    assert!(tree.get(b"helloX").is_none());
}

#[test]
fn test_get_multiple_keys() {
    let tree: MassTree<u64> = MassTree::new();

    // Use the insert API to add multiple keys
    tree.insert(b"aaa", 100).unwrap();
    tree.insert(b"bbb", 200).unwrap();
    tree.insert(b"ccc", 300).unwrap();

    // Test retrieval
    assert_eq!(tree.get(b"aaa").unwrap(), 100);
    assert_eq!(tree.get(b"bbb").unwrap(), 200);
    assert_eq!(tree.get(b"ccc").unwrap(), 300);

    // Non-existent keys
    assert!(tree.get(b"ddd").is_none());
    assert!(tree.get(b"aab").is_none());
}

// ========================================================================
//  Index Mode Get Tests (using public API)
// ========================================================================

#[test]
fn test_index_get_after_insert() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    tree.insert(b"test", 999).unwrap();

    // Get returns V directly (not Arc<V>)
    let result = tree.get(b"test");
    assert_eq!(result, Some(999));
}

// ========================================================================
//  Edge Case Tests (using public API)
// ========================================================================

#[test]
fn test_insert_and_get_empty_key() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"", 123).unwrap();
    assert_eq!(tree.get(b"").unwrap(), 123);
}

#[test]
fn test_insert_and_get_max_inline_key() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert 8-byte key (max inline size)
    let key_bytes = b"12345678";
    tree.insert(key_bytes, 888).unwrap();

    assert_eq!(tree.get(key_bytes).unwrap(), 888);
}

#[test]
fn test_insert_and_get_binary_key() {
    let tree: MassTree<u64> = MassTree::new();

    // Binary key with null bytes
    let key_bytes = &[0x00, 0x01, 0x02, 0x00, 0xFF];
    tree.insert(key_bytes, 777).unwrap();

    assert_eq!(tree.get(key_bytes).unwrap(), 777);
}

// ========================================================================
//  Insert Tests
// ========================================================================

#[test]
fn test_insert_into_empty_tree() {
    let tree: MassTree<u64> = MassTree::new();

    let result = tree.insert(b"hello", 42);

    assert!(result.is_ok());
    assert!(result.unwrap().is_none()); // No old value
    assert!(!tree.is_empty());
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_insert_and_get() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"hello", 42).unwrap();

    let value = tree.get(b"hello");
    assert!(value.is_some());
    assert_eq!(value.unwrap(), 42);
}

#[test]
fn test_insert_multiple_keys() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"aaa", 100).unwrap();
    tree.insert(b"bbb", 200).unwrap();
    tree.insert(b"ccc", 300).unwrap();

    assert_eq!(tree.len(), 3);
    assert_eq!(tree.get(b"aaa").unwrap(), 100);
    assert_eq!(tree.get(b"bbb").unwrap(), 200);
    assert_eq!(tree.get(b"ccc").unwrap(), 300);
}

#[test]
fn test_insert_updates_existing() {
    let tree: MassTree<u64> = MassTree::new();

    // First insert
    let old = tree.insert(b"key", 100).unwrap();
    assert!(old.is_none());

    // Update
    let old = tree.insert(b"key", 200).unwrap();
    assert_eq!(old.unwrap(), 100);

    // Verify new value
    assert_eq!(tree.get(b"key").unwrap(), 200);
    assert_eq!(tree.len(), 1); // Still one key
}

#[test]
fn test_insert_returns_old_arc() {
    let tree: MassTree15<String> = MassTree15::new();

    tree.insert(b"key", "first".to_string()).unwrap();

    let old = tree.insert(b"key", "second".to_string()).unwrap();
    assert_eq!(*old.unwrap(), "first");
}

#[test]
fn test_insert_ascending_order() {
    let tree: MassTree<u64> = MassTree::new();

    for i in 0..10 {
        let key = format!("key{i:02}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    assert_eq!(tree.len(), 10);

    // Verify all values
    for i in 0..10 {
        let key = format!("key{i:02}");
        assert_eq!(tree.get(key.as_bytes()).unwrap(), i as u64);
    }
}

#[test]
fn test_insert_descending_order() {
    let tree: MassTree<u64> = MassTree::new();

    for i in (0..10).rev() {
        let key = format!("key{i:02}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    assert_eq!(tree.len(), 10);

    for i in 0..10 {
        let key = format!("key{i:02}");
        assert_eq!(tree.get(key.as_bytes()).unwrap(), i as u64);
    }
}

#[test]
fn test_insert_random_order() {
    let tree: MassTree<u64> = MassTree::new();

    let keys = ["dog", "cat", "bird", "fish", "ant", "bee"];

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    assert_eq!(tree.len(), 6);

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(tree.get(key.as_bytes()).unwrap(), i as u64);
    }
}

#[test]
fn test_insert_triggers_split() {
    let tree: MassTree<u64> = MassTree::new();

    // Fill the leaf (24 slots for WIDTH=24)
    for i in 0..24 {
        let key = format!("{i:02}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    assert_eq!(tree.len(), 24);

    // 25th insert should trigger a split
    let result = tree.insert(b"overflow", 999);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none()); // New key, no old value

    assert_eq!(tree.len(), 25);

    // Verify the new key is accessible
    let value = tree.get(b"overflow");
    assert_eq!(value.unwrap(), 999);

    // Verify all original keys are still accessible
    for i in 0..24 {
        let key = format!("{i:02}");
        let value = tree.get(key.as_bytes());

        assert!(value.is_some(), "Key {key:?} not found after split");
        assert_eq!(value.unwrap(), i as u64);
    }
}

#[test]
fn test_insert_at_capacity() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert exactly WIDTH keys (24 for WIDTH=24)
    for i in 0..24 {
        let key = format!("{i:02}");
        assert!(tree.insert(key.as_bytes(), i as u64).is_ok());
    }

    assert_eq!(tree.len(), 24);
}

#[test]
fn test_insert_error_display() {
    let err = InsertError::LeafFull;
    assert_eq!(format!("{err}"), "leaf node is full");
    //
    // let err = InsertError::AllocationFailed;
    // assert_eq!(format!("{err}"), "memory allocation failed");
}

#[test]
fn test_insert_max_key_length() {
    let tree: MassTree<u64> = MassTree::new();

    // Exactly 8 bytes should work
    let max_key = b"12345678";
    let result = tree.insert(max_key, 42);
    assert!(result.is_ok());
    assert_eq!(tree.get(max_key).unwrap(), 42);
}

#[test]
fn test_slot0_reuse_no_predecessor() {
    let tree: MassTree<u64> = MassTree::new();

    // Root leaf has no predecessor, so slot 0 should be usable
    tree.insert(b"first", 1).unwrap();

    // Verify the value was stored
    assert_eq!(tree.get(b"first").unwrap(), 1);
}

#[test]
fn test_slot0_reuse_same_ikey() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert key with ikey X
    tree.insert(b"aaa", 1).unwrap();

    // Insert another key with same ikey prefix (within 8 bytes)
    // This should allow slot 0 reuse if needed
    tree.insert(b"aab", 2).unwrap();

    assert_eq!(tree.get(b"aaa").unwrap(), 1);
    assert_eq!(tree.get(b"aab").unwrap(), 2);
}

#[test]
fn test_permutation_updates_correctly() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert keys in reverse order
    tree.insert(b"ccc", 3).unwrap();
    tree.insert(b"bbb", 2).unwrap();
    tree.insert(b"aaa", 1).unwrap();

    // Verify size
    assert_eq!(tree.len(), 3);

    // Verify get still works (relies on correct permutation)
    assert_eq!(tree.get(b"aaa").unwrap(), 1);
    assert_eq!(tree.get(b"bbb").unwrap(), 2);
    assert_eq!(tree.get(b"ccc").unwrap(), 3);
}

#[test]
fn test_index_insert_basic() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    let result = tree.insert(b"key", 42);

    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    assert_eq!(tree.get(b"key"), Some(42));
}

#[test]
fn test_index_insert_update() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    tree.insert(b"key", 100).unwrap();

    let old = tree.insert(b"key", 200).unwrap();

    assert_eq!(old, Some(100)); // Old value returned directly (not Arc)
    assert_eq!(tree.get(b"key"), Some(200));
}

#[test]
fn test_insert_empty_key() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"", 42).unwrap();

    assert_eq!(tree.get(b"").unwrap(), 42);
}

#[test]
fn test_insert_max_inline_key() {
    let tree: MassTree<u64> = MassTree::new();

    let key = b"12345678"; // Exactly 8 bytes

    tree.insert(key, 888).unwrap();

    assert_eq!(tree.get(key).unwrap(), 888);
}

#[test]
fn test_insert_binary_keys() {
    let tree: MassTree<u64> = MassTree::new();

    let keys = [
        vec![0x00, 0x01, 0x02],
        vec![0xFF, 0xFE, 0xFD],
        vec![0x00, 0x00, 0x00],
        vec![0xFF, 0xFF, 0xFF],
    ];

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64).unwrap();
    }

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(tree.get(key).unwrap(), i as u64);
    }
}

#[test]
fn test_insert_similar_keys() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"test", 1).unwrap();
    tree.insert(b"test1", 2).unwrap();
    tree.insert(b"test12", 3).unwrap();
    tree.insert(b"tes", 4).unwrap();

    assert_eq!(tree.get(b"test").unwrap(), 1);
    assert_eq!(tree.get(b"test1").unwrap(), 2);
    assert_eq!(tree.get(b"test12").unwrap(), 3);
    assert_eq!(tree.get(b"tes").unwrap(), 4);
}

// ========================================================================
//  Differential Testing
// ========================================================================

#[test]
fn test_differential_small_sequential() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    for i in 0..10 {
        let key = vec![b'a' + i as u8];
        let tree_old = tree.insert(&key, i as u64).unwrap();
        let oracle_old = oracle.insert(key.clone(), i as u64);

        assert_eq!(tree_old, oracle_old, "Insert mismatch for key {key:?}");
    }
}

#[test]
fn test_differential_updates() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    let key = b"key".to_vec();

    for value in [100, 200, 300] {
        let tree_old = tree.insert(&key, value).unwrap();
        let oracle_old = oracle.insert(key.clone(), value);
        assert_eq!(tree_old, oracle_old);
    }

    assert_eq!(tree.get(&key).unwrap(), *oracle.get(&key).unwrap());
}

#[test]
fn test_differential_interleaved() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    tree.insert(b"a", 1).unwrap();
    oracle.insert(b"a".to_vec(), 1);
    assert_eq!(tree.get(b"a"), oracle.get(b"a".as_slice()).copied());

    tree.insert(b"b", 2).unwrap();
    oracle.insert(b"b".to_vec(), 2);
    assert_eq!(tree.get(b"a"), oracle.get(b"a".as_slice()).copied());
    assert_eq!(tree.get(b"b"), oracle.get(b"b".as_slice()).copied());

    tree.insert(b"a", 10).unwrap();
    oracle.insert(b"a".to_vec(), 10);
    assert_eq!(tree.get(b"a"), oracle.get(b"a".as_slice()).copied());
}

#[test]
fn test_differential_get_missing() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    assert_eq!(
        tree.get(b"missing"),
        oracle.get(b"missing".as_slice()).copied()
    );

    tree.insert(b"present", 42).unwrap();

    assert_eq!(tree.get(b"missing"), None);
    assert_eq!(tree.get(b"present").unwrap(), 42);
}

// ========================================================================
//  Split Tests
// ========================================================================

#[test]
fn test_insert_100_keys() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert 100 unique keys (will trigger multiple splits)
    for i in 0..100 {
        let key = format!("{i:08}"); // 8-byte keys
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Verify all keys are retrievable
    for i in 0..100 {
        let key = format!("{i:08}");
        let value = tree.get(key.as_bytes());
        assert!(value.is_some(), "Key {i} not found");
        assert_eq!(value.unwrap(), i as u64);
    }

    // Verify len() works after multiple splits
    assert_eq!(tree.len(), 100);
}

#[test]
fn test_len_multi_level_tree() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert enough keys to create multiple levels
    for i in 0..200 {
        let key = format!("{i:08}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Verify len() returns correct count
    assert_eq!(tree.len(), 200);

    // Insert more and verify again
    for i in 200..500 {
        let key = format!("{i:08}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }
    assert_eq!(tree.len(), 500);
}

#[test]
fn test_insert_1000_sequential() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // Insert 1000 keys sequentially
    for i in 0..1000 {
        let key = format!("{i:08}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
        oracle.insert(key.into_bytes(), i as u64);
    }

    // Differential check: all keys match oracle
    for (key, expected) in &oracle {
        let actual = tree.get(key);
        assert!(actual.is_some(), "Key {key:?} not found in tree");
        assert_eq!(actual.unwrap(), *expected);
    }
}

#[test]
fn test_insert_1000_random() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // Insert 1000 keys in "random" order (deterministic pattern)
    for i in 0..1000 {
        // Mix up the order using modular arithmetic
        let idx = (i * 997) % 1000;
        let key = format!("{idx:08}");
        tree.insert(key.as_bytes(), idx as u64).unwrap();
        oracle.insert(key.into_bytes(), idx as u64);
    }

    // Differential check
    for (key, expected) in &oracle {
        let actual = tree.get(key);
        assert!(actual.is_some(), "Key {key:?} not found");
        assert_eq!(actual.unwrap(), *expected);
    }
}

#[test]
fn test_split_preserves_values() {
    let tree: MassTree15<String> = MassTree15::new();

    // Insert values with distinct content
    let test_values: Vec<(&[u8], String)> = vec![
        (b"aaa", "first value".to_string()),
        (b"bbb", "second value".to_string()),
        (b"ccc", "third value".to_string()),
        (b"ddd", "fourth value".to_string()),
        (b"eee", "fifth value".to_string()),
    ];

    for (key, value) in &test_values {
        tree.insert(key, value.clone()).unwrap();
    }

    // Fill to trigger split
    for i in 0..20 {
        let key = format!("k{i:02}");
        tree.insert(key.as_bytes(), format!("value{i}")).unwrap();
    }

    // Verify original values survived split
    for (key, expected) in &test_values {
        let actual = tree.get(key);
        assert!(actual.is_some(), "Key {key:?} lost after split");
        assert_eq!(*actual.unwrap(), *expected);
    }
}

#[test]
fn test_split_with_updates() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert and immediately update some keys
    for i in 0..30 {
        let key = format!("{i:04}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Update every other key
    for i in (0..30).step_by(2) {
        let key = format!("{i:04}");
        let old = tree.insert(key.as_bytes(), (i + 1000) as u64).unwrap();
        assert_eq!(old.unwrap(), i as u64);
    }

    // Verify
    for i in 0..30 {
        let key = format!("{i:04}");
        let expected = if i % 2 == 0 { i + 1000 } else { i };
        assert_eq!(tree.get(key.as_bytes()).unwrap(), expected as u64);
    }
}

#[test]
fn test_multiple_splits_create_deeper_tree() {
    // With WIDTH=24, multiple splits needed for 100 keys
    let tree: MassTree<u64> = MassTree::new();

    for i in 0..100 {
        let key = format!("{i:08}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Verify all keys accessible
    for i in 0..100 {
        let key = format!("{i:08}");
        assert!(tree.get(key.as_bytes()).is_some(), "Key {i} not found");
    }

    // Verify count
    assert_eq!(tree.len(), 100);
}

#[test]
fn test_split_empty_right_edge_case() {
    // Test the sequential optimization edge case
    let tree: MassTree<u64> = MassTree::new();

    // Fill leaf completely with sequential keys (rightmost optimization)
    for i in 0..24 {
        tree.insert(&[i as u8], i as u64).unwrap();
    }

    // 25th insert at end should trigger split
    tree.insert(&[24u8], 24).unwrap();

    // Verify all keys
    for i in 0..25 {
        assert_eq!(tree.get(&[i as u8]).unwrap(), i as u64);
    }
}

#[test]
fn test_split_index_mode() {
    // Test splits work with MassTree15Inline too
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    for i in 0..50 {
        let key = format!("{i:02}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    for i in 0..50 {
        let key = format!("{i:02}");
        assert_eq!(tree.get(key.as_bytes()), Some(i as u64));
    }
}

#[test]
fn test_split_maintains_key_order() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert in reverse order
    for i in (0..50).rev() {
        let key = format!("{i:02}");
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Keys should still be retrievable
    for i in 0..50 {
        let key = format!("{i:02}");
        let value = tree.get(key.as_bytes());
        assert!(value.is_some(), "Key {i:02} not found");
        assert_eq!(value.unwrap(), i as u64);
    }
}

// ========================================================================
//  Layer Tests
// ========================================================================

#[test]
fn test_layer_creation_same_prefix() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert "hello world!" (12 bytes)
    tree.insert(b"hello world!", 1).unwrap();

    // Insert "hello worm" (10 bytes) - same 8-byte prefix "hello wo"
    tree.insert(b"hello worm", 2).unwrap();

    // Both should be retrievable
    assert_eq!(tree.get(b"hello world!"), Some(1));
    assert_eq!(tree.get(b"hello worm"), Some(2));
}

#[test]
fn test_deep_layer_chain() {
    let tree: MassTree<u64> = MassTree::new();

    // Two 24-byte keys with 16 bytes common prefix
    let key1 = b"aaaaaaaabbbbbbbbXXXXXXXX"; // 24 bytes
    let key2 = b"aaaaaaaabbbbbbbbYYYYYYYY"; // 24 bytes, differs at byte 16

    tree.insert(key1, 1).unwrap();
    tree.insert(key2, 2).unwrap();

    // Should create 2 intermediate twig nodes:
    // Layer 0: ikey="aaaaaaaa" → twig
    // Layer 1: ikey="bbbbbbbb" → twig
    // Layer 2: ikey="XXXXXXXX" and "YYYYYYYY" (different, final leaf)

    assert_eq!(tree.get(key1), Some(1));
    assert_eq!(tree.get(key2), Some(2));
}

#[test]
fn test_layer_with_suffixes() {
    let tree: MassTree<u64> = MassTree::new();

    // Keys that create layers AND have suffixes
    let key1 = b"prefixAArest1"; // 13 bytes
    let key2 = b"prefixAArest2"; // 13 bytes, same first 8, different suffix

    tree.insert(key1, 1).unwrap();
    tree.insert(key2, 2).unwrap();

    assert_eq!(tree.get(key1), Some(1));
    assert_eq!(tree.get(key2), Some(2));
    assert_eq!(tree.get(b"prefixAArest3"), None);
}

#[test]
fn test_insert_into_existing_layer() {
    let tree: MassTree<u64> = MassTree::new();

    // Create a layer
    tree.insert(b"hello world!", 1).unwrap();
    tree.insert(b"hello worm", 2).unwrap();

    // Insert another key with same prefix, different continuation
    tree.insert(b"hello wonder", 3).unwrap();

    assert_eq!(tree.get(b"hello world!"), Some(1));
    assert_eq!(tree.get(b"hello worm"), Some(2));
    assert_eq!(tree.get(b"hello wonder"), Some(3));
}

#[test]
fn test_update_in_layer() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"hello world!", 1).unwrap();
    tree.insert(b"hello worm", 2).unwrap();

    // Update existing key in layer
    let old = tree.insert(b"hello world!", 100).unwrap();
    assert_eq!(old, Some(1));

    assert_eq!(tree.get(b"hello world!"), Some(100));
    assert_eq!(tree.get(b"hello worm"), Some(2));
}

#[test]
fn test_get_nonexistent_in_layer() {
    let tree: MassTree<u64> = MassTree::new();

    tree.insert(b"hello world!", 1).unwrap();
    tree.insert(b"hello worm", 2).unwrap();

    // Same prefix but different suffix that wasn't inserted
    assert_eq!(tree.get(b"hello worst"), None);
    assert_eq!(tree.get(b"hello wo"), None); // Exact 8 bytes (partial match)
}

#[test]
fn test_long_key_insert_and_get() {
    let tree: MassTree<u64> = MassTree::new();

    // Keys longer than 8 bytes without common prefix
    let key1 = b"abcdefghijklmnop"; // 16 bytes
    let key2 = b"zyxwvutsrqponmlk"; // 16 bytes, different prefix

    tree.insert(key1, 100).unwrap();
    tree.insert(key2, 200).unwrap();

    assert_eq!(tree.get(key1), Some(100));
    assert_eq!(tree.get(key2), Some(200));
}

#[test]
fn test_mixed_short_and_long_keys() {
    let tree: MassTree<u64> = MassTree::new();

    // Mix of short (<=8) and long (>8) keys
    tree.insert(b"short", 1).unwrap();
    tree.insert(b"exactly8", 2).unwrap();
    tree.insert(b"this is longer", 3).unwrap();
    tree.insert(b"another long key", 4).unwrap();

    assert_eq!(tree.get(b"short"), Some(1));
    assert_eq!(tree.get(b"exactly8"), Some(2));
    assert_eq!(tree.get(b"this is longer"), Some(3));
    assert_eq!(tree.get(b"another long key"), Some(4));
}

#[test]
fn test_layer_differential_vs_btreemap() {
    use std::collections::BTreeMap;

    let tree: MassTree<u64> = MassTree::new();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // Test with keys that will create layers
    let keys = [
        b"hello world!".to_vec(),
        b"hello worm".to_vec(),
        b"hello wonder".to_vec(),
        b"goodbye world".to_vec(),
        b"goodbye friend".to_vec(),
        b"short".to_vec(),
        b"a".to_vec(),
    ];

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64).unwrap();
        oracle.insert(key.clone(), i as u64);
    }

    // Verify all oracle entries
    for (key, expected) in &oracle {
        let actual = tree.get(key);
        assert_eq!(actual, Some(*expected), "mismatch for key {key:?}");
    }
}

#[test]
fn test_layer_count_tracking() {
    let tree: MassTree<u64> = MassTree::new();

    // Insert keys that create layers
    tree.insert(b"hello world!", 1).unwrap();
    assert_eq!(tree.len(), 1);

    tree.insert(b"hello worm", 2).unwrap();
    assert_eq!(tree.len(), 2);

    tree.insert(b"hello wonder", 3).unwrap();
    assert_eq!(tree.len(), 3);

    // Update shouldn't change count
    tree.insert(b"hello world!", 100).unwrap();
    assert_eq!(tree.len(), 3);
}

// ========================================================================
//  Layer Root Growth Tests
// ========================================================================

#[test]
fn test_layer_growth_beyond_width() {
    // Test that a layer can grow beyond WIDTH entries by promoting to internode
    let tree: MassTree<u64> = MassTree::new();

    // All keys share same 8-byte prefix, forcing them into a layer
    let prefix = b"samepfx!";

    // Insert more than WIDTH (15) keys - will trigger layer root split
    for i in 0..20u64 {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        tree.insert(&key, i).unwrap();
    }

    // All keys must be findable
    for i in 0..20u64 {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        let result = tree.get(&key);
        assert!(result.is_some(), "Key with suffix {i} not found");
        assert_eq!(result.unwrap(), i);
    }
}

#[test]
fn test_layer_root_becomes_internode() {
    // Test with default WIDTH - insert enough keys to trigger layer split
    let tree: MassTree<u64> = MassTree::new();

    // Force layer creation with suffix keys
    tree.insert(b"prefix00suffix_a", 1).unwrap();
    tree.insert(b"prefix00suffix_b", 2).unwrap();

    // These should go into the layer and eventually trigger split
    // With WIDTH=15, we need >15 keys in the layer to trigger split
    for i in 0..20u64 {
        let key = format!("prefix00suffix_{i:02}");
        tree.insert(key.as_bytes(), i + 100).unwrap();
    }

    // Verify initial keys still findable
    assert_eq!(tree.get(b"prefix00suffix_a").unwrap(), 1);
    assert_eq!(tree.get(b"prefix00suffix_b").unwrap(), 2);

    // Verify all numbered keys findable
    for i in 0..20u64 {
        let key = format!("prefix00suffix_{i:02}");
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key {key} not found");
    }
}

#[test]
fn test_layer_split_preserves_all_keys() {
    // Comprehensive test: insert many keys with shared prefix, verify all retrievable
    let tree: MassTree<u64> = MassTree::new();
    let mut keys_and_values: Vec<(Vec<u8>, u64)> = Vec::new();

    // Create keys that will all go into the same layer (share 8-byte prefix)
    for i in 0..50u64 {
        let mut key = b"testpfx!".to_vec(); // Exactly 8 bytes
        key.extend_from_slice(format!("suffix{i:04}").as_bytes());
        keys_and_values.push((key.clone(), i * 10));
        tree.insert(&key, i * 10).unwrap();
    }

    // Verify all keys
    for (key, expected) in &keys_and_values {
        let result = tree.get(key);
        assert!(result.is_some(), "Key {key:?} not found");
        assert_eq!(result.unwrap(), *expected);
    }

    assert_eq!(tree.len(), 50);
}

// ========================================================================
//  MassTreeGeneric Tests
// ========================================================================

#[test]
fn test_masstree_generic_new_is_empty() {
    use crate::alloc15::SeizeAllocator15;
    use crate::leaf15::LeafNode15;

    // Create via with_allocator
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let tree: MassTreeGeneric<
        LeafValue<u64>,
        LeafNode15<LeafValue<u64>>,
        SeizeAllocator15<LeafValue<u64>>,
    > = MassTreeGeneric::with_allocator(alloc);

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_masstree_generic_debug() {
    use crate::alloc15::SeizeAllocator15;
    use crate::leaf15::LeafNode15;

    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let tree: MassTreeGeneric<
        LeafValue<u64>,
        LeafNode15<LeafValue<u64>>,
        SeizeAllocator15<LeafValue<u64>>,
    > = MassTreeGeneric::with_allocator(alloc);

    let debug_str = format!("{tree:?}");
    assert!(debug_str.contains("MassTreeGeneric"));
    assert!(debug_str.contains("width: 15"));
}

#[test]
fn test_masstree_generic_guard() {
    use crate::alloc15::SeizeAllocator15;
    use crate::leaf15::LeafNode15;

    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let tree: MassTreeGeneric<
        LeafValue<u64>,
        LeafNode15<LeafValue<u64>>,
        SeizeAllocator15<LeafValue<u64>>,
    > = MassTreeGeneric::with_allocator(alloc);

    // Just verify guard creation doesn't panic
    let _guard = tree.guard();
}

// ========================================================================
//  MassTreeGeneric Send/Sync Tests
// ========================================================================

fn _assert_masstree_generic_send_sync()
where
    MassTreeGeneric<
        LeafValue<u64>,
        crate::leaf15::LeafNode15<LeafValue<u64>>,
        crate::alloc15::SeizeAllocator15<LeafValue<u64>>,
    >: Send + Sync,
{
}

// ========================================================================
//  MassTree15 Type Alias Tests
// ========================================================================

#[test]
fn test_masstree15_new() {
    let tree: MassTree15<u64> = MassTree15::new();
    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_masstree15_get_empty() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Get on empty tree should return None
    let value = tree.get_with_guard(b"hello", &guard);
    assert!(value.is_none());
}

#[test]
fn test_masstree15_insert_and_get() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert a value
    let result = tree.insert_with_guard(b"hello", 42, &guard);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none()); // No old value

    // Get the value back
    let value = tree.get_with_guard(b"hello", &guard);
    assert!(value.is_some());
    assert_eq!(*value.unwrap(), 42);
}

#[test]
fn test_masstree15_multiple_inserts() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert multiple values (up to WIDTH=24 without split)
    for i in 0..20u64 {
        let key = format!("key{i:02}");
        let result = tree.insert_with_guard(key.as_bytes(), i * 10, &guard);
        assert!(result.is_ok(), "Failed to insert key {i}");
    }

    assert_eq!(tree.len(), 20);

    // Verify all values
    for i in 0..20u64 {
        let key = format!("key{i:02}");
        let value = tree.get_with_guard(key.as_bytes(), &guard);
        assert!(value.is_some(), "Key {i} not found");
        assert_eq!(*value.unwrap(), i * 10);
    }
}

#[test]
fn test_masstree15_update_existing() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert initial value
    tree.insert_with_guard(b"key", 100, &guard).unwrap();
    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 100);

    // Update the value
    let old = tree.insert_with_guard(b"key", 200, &guard).unwrap();
    assert!(old.is_some());
    assert_eq!(*old.unwrap(), 100);

    // Verify updated value
    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 200);

    // Count should still be 1
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_masstree_new() {
    let tree: MassTree<u64> = MassTree::new();
    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_masstree_insert_and_get() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // Insert a value
    let result = tree.insert_with_guard(b"hello", 42, &guard);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    // Get the value back
    let value = tree.get_with_guard(b"hello", &guard);
    assert!(value.is_some());
    assert_eq!(value.unwrap(), 42);
}

#[test]
fn test_masstree15_split_triggers() {
    // Test that inserting more than 24 keys triggers a split
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert 24 keys (should all succeed, filling the root leaf)
    for i in 0..24u64 {
        let key = format!("key{i:02}");
        let result = tree.insert_with_guard(key.as_bytes(), i, &guard);
        assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
    }

    assert_eq!(tree.len(), 24);

    // The 25th key should succeed by triggering a split
    let result = tree.insert_with_guard(b"key24", 24, &guard);
    assert!(
        result.is_ok(),
        "25th insert should succeed with split: {result:?}"
    );
    assert_eq!(tree.len(), 25);

    // Verify we can read the 25th key back
    let value = tree.get_with_guard(b"key24", &guard);
    assert!(value.is_some());
    assert_eq!(*value.unwrap(), 24);

    // Verify all original keys are still readable
    for i in 0..24u64 {
        let key = format!("key{i:02}");
        let value = tree.get_with_guard(key.as_bytes(), &guard);
        assert!(value.is_some(), "key{i:02} not found after split");
        assert_eq!(*value.unwrap(), i);
    }
}

#[test]
fn test_masstree15_many_splits() {
    // Test inserting many more keys that require multiple splits
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert 100 keys (should trigger multiple splits)
    for i in 0..100u64 {
        let key = format!("key{i:03}");
        let result = tree.insert_with_guard(key.as_bytes(), i, &guard);
        assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
    }

    assert_eq!(tree.len(), 100);

    // Verify all keys are readable
    for i in 0..100u64 {
        let key = format!("key{i:03}");
        let value = tree.get_with_guard(key.as_bytes(), &guard);
        assert!(value.is_some(), "key{i:03} not found");
        assert_eq!(*value.unwrap(), i);
    }
}

#[test]
#[cfg(not(miri))]
fn test_masstree15_sequential_u64_limited() {
    // Test inserting keys with u64 sequential pattern
    // Uses 10,000 keys to trigger multiple internode splits
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    const KEY_COUNT: u64 = 10_000;

    for i in 0u64..KEY_COUNT {
        let key = i.to_be_bytes();
        let result = tree.insert_with_guard(&key, i, &guard);
        assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
    }

    assert_eq!(tree.len(), KEY_COUNT as usize);

    // Verify all keys are readable
    for i in 0u64..KEY_COUNT {
        let key = i.to_be_bytes();
        let value = tree.get_with_guard(&key, &guard);
        assert!(value.is_some(), "Key {i} not found in final verification");
        assert_eq!(*value.unwrap(), i);
    }
}

#[test]
#[cfg(not(miri))]
fn test_masstree15_internode_splits_stress() {
    // Stress test to verify multi-level internode splits work
    // Uses 50,000 keys to trigger deep tree with multiple internode levels
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    const KEY_COUNT: u64 = 50_000;

    for i in 0u64..KEY_COUNT {
        let key = i.to_be_bytes();
        let result = tree.insert_with_guard(&key, i, &guard);
        assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
    }

    assert_eq!(tree.len(), KEY_COUNT as usize);

    // Verify a sample of keys (every 100th)
    for i in (0u64..KEY_COUNT).step_by(100) {
        let key = i.to_be_bytes();
        let value = tree.get_with_guard(&key, &guard);
        assert!(value.is_some(), "Key {i} not found");
        assert_eq!(*value.unwrap(), i);
    }
}

/// Regression test for proptest failure: keys = [[52, 0], [52]]
/// Key [52] was being lost.
#[test]
fn test_proptest_regression_two_keys() {
    let tree: MassTree15<u64> = MassTree15::new();

    let key1: &[u8] = &[52, 0];
    let key2: &[u8] = &[52];

    tree.insert(key1, 0).unwrap();
    tree.insert(key2, 1).unwrap();

    // Verify both keys exist
    let result1 = tree.get(key1);
    let result2 = tree.get(key2);

    assert!(result1.is_some(), "Key [52, 0] lost!");
    assert!(result2.is_some(), "Key [52] lost!");
    assert_eq!(*result1.unwrap(), 0);
    assert_eq!(*result2.unwrap(), 1);
    assert_eq!(tree.len(), 2);
}

/// Regression test for proptest failure: 9-byte key of all zeros.
/// Key was being lost after insert.
#[test]
fn test_proptest_regression_nine_zeros() {
    let tree: MassTree15<u64> = MassTree15::new();

    let key: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 0, 0];
    tree.insert(key, 42).unwrap();

    let result = tree.get(key);
    assert!(result.is_some(), "9-byte key not found!");
    assert_eq!(*result.unwrap(), 42);
    assert_eq!(tree.len(), 1);
}

/// Test two suffix keys with same 8-byte prefix.
/// This creates a conflict that should create a layer.
#[test]
fn test_two_suffix_keys_same_prefix() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Two 16-byte keys with same 8-byte prefix "prefix00"
    let key1: &[u8] = b"prefix0000000000";
    let key2: &[u8] = b"prefix0000000001";

    tree.insert(key1, 1).unwrap();
    tree.insert(key2, 2).unwrap();

    // Both should be retrievable
    let result1 = tree.get(key1);
    let result2 = tree.get(key2);

    assert!(result1.is_some(), "key1 not found!");
    assert!(result2.is_some(), "key2 not found!");
    assert_eq!(*result1.unwrap(), 1);
    assert_eq!(*result2.unwrap(), 2);
    assert_eq!(tree.len(), 2);
}

/// Test many suffix keys with same 8-byte prefix.
/// Tests layer handling as keys accumulate.
#[test]
fn test_many_suffix_keys_same_prefix() {
    let tree: MassTree15<u64> = MassTree15::new();

    // 20 keys with same 8-byte prefix "prefix00"
    for i in 0..20u64 {
        let key = format!("prefix00{i:08}"); // 16 bytes each
        tree.insert(key.as_bytes(), i).unwrap();

        // Immediately verify this key
        let result = tree.get(key.as_bytes());
        assert!(
            result.is_some(),
            "Key {i} ('{key}') not found immediately after insert"
        );
        assert_eq!(*result.unwrap(), i);
    }

    // Verify all keys still retrievable at the end
    for i in 0..20u64 {
        let key = format!("prefix00{i:08}");
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key {i} ('{key}') not found at end");
        assert_eq!(*result.unwrap(), i);
    }

    assert_eq!(tree.len(), 20);
}

// ========================================================================
//  MassTree15 Tests (WIDTH=15 variant)
// ========================================================================

#[test]
fn test_masstree15_basic() {
    let tree: MassTree15<u64> = MassTree15::new();

    assert!(tree.is_empty());

    tree.insert(b"hello", 123).unwrap();
    tree.insert(b"world", 456).unwrap();

    assert_eq!(tree.len(), 2);
    assert_eq!(*tree.get(b"hello").unwrap(), 123);
    assert_eq!(*tree.get(b"world").unwrap(), 456);
}

#[test]
fn test_masstree15_splits() {
    // Insert enough keys to trigger splits (WIDTH=15)
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    for i in 0u64..100 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    assert_eq!(tree.len(), 100);

    // Verify all keys
    for i in 0u64..100 {
        let key = i.to_be_bytes();
        let result = tree.get_with_guard(&key, &guard);
        assert!(result.is_some(), "Key {i} not found");
        assert_eq!(*result.unwrap(), i);
    }
}

#[test]
fn test_masstree15_remove() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    for i in 0u64..50 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    // Remove half
    for i in 0u64..25 {
        let key = i.to_be_bytes();
        let result = tree.remove_with_guard(&key, &guard).unwrap();
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), i);
    }

    assert_eq!(tree.len(), 25);

    // Verify remaining
    for i in 25u64..50 {
        let key = i.to_be_bytes();
        assert!(tree.get_with_guard(&key, &guard).is_some());
    }
}

#[test]
fn test_masstree15_inline() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    for i in 0u64..100 {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    assert_eq!(tree.len(), 100);

    for i in 0u64..100 {
        let key = i.to_be_bytes();
        let result = tree.get_with_guard(&key, &guard);
        assert_eq!(result, Some(i));
    }
}

#[test]
fn test_masstree15_concurrent() {
    let tree = Arc::new(MassTree15::<u64>::new());
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0u64..100 {
                    let key = (t * 1000 + i).to_be_bytes();
                    tree.insert_with_guard(&key, t * 1000 + i, &guard).unwrap();
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(tree.len(), 400);
}

// ========================================================================
//  Guard Verification Tests
// ========================================================================

#[test]
fn test_correct_guard_works() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    // Should not panic - guard is from the correct tree
    tree.insert_with_guard(b"key", 42, &guard).unwrap();
    assert_eq!(tree.get(b"key"), Some(Arc::new(42)));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "different collector")]
fn test_wrong_guard_panics_on_insert() {
    let tree1: MassTree15<u64> = MassTree15::new();
    let tree2: MassTree15<u64> = MassTree15::new();
    let wrong_guard = tree2.guard();
    // Should panic in debug builds - guard is from wrong tree
    let _ = tree1.insert_with_guard(b"key", 42, &wrong_guard);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "different collector")]
fn test_wrong_guard_panics_on_remove() {
    let tree1: MassTree15<u64> = MassTree15::new();
    let tree2: MassTree15<u64> = MassTree15::new();
    tree1.insert(b"key", 42).unwrap();
    let wrong_guard = tree2.guard();
    // Should panic in debug builds - guard is from wrong tree
    let _ = tree1.remove_with_guard(b"key", &wrong_guard);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "different collector")]
fn test_wrong_guard_panics_on_iter() {
    let tree1: MassTree15<u64> = MassTree15::new();
    let tree2: MassTree15<u64> = MassTree15::new();
    let wrong_guard = tree2.guard();
    // Should panic in debug builds - guard is from wrong tree
    let _ = tree1.iter(&wrong_guard).count();
}

// ========================================================================
//  Single-Layer Search Optimization Tests
// ========================================================================

/// Test that inline keys work correctly when a layer pointer exists in the leaf.
///
/// This creates a layer pointer by inserting two long keys with the same 8-byte
/// ikey prefix but different suffixes (suffix conflict → layer creation).
/// Then inserts inline keys with the same ikey prefix to verify they are
/// handled correctly by the implicit ordering logic.
#[test]
fn test_inline_key_with_layer_pointer() {
    use crate::RangeBound;

    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // Two keys with same first 8 bytes but different suffixes → creates layer pointer
    // "hello wo" is the shared 8-byte ikey
    tree.insert(b"hello woAAAA", 1).unwrap();
    tree.insert(b"hello woBBBB", 2).unwrap();

    // Now insert inline keys with the same ikey prefix
    // These must be handled correctly despite the layer pointer in the leaf
    tree.insert(b"hello wo", 3).unwrap(); // Exactly 8 bytes
    tree.insert(b"hello w", 4).unwrap(); // 7 bytes
    tree.insert(b"hello", 5).unwrap(); // 5 bytes

    // Verify all keys are retrievable
    assert_eq!(tree.get(b"hello woAAAA"), Some(1));
    assert_eq!(tree.get(b"hello woBBBB"), Some(2));
    assert_eq!(tree.get(b"hello wo"), Some(3));
    assert_eq!(tree.get(b"hello w"), Some(4));
    assert_eq!(tree.get(b"hello"), Some(5));

    // Verify ordering via scan (shorter keys before longer, inline before layer)
    let mut keys: Vec<Vec<u8>> = Vec::new();
    tree.scan(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |k, _| {
            keys.push(k.to_vec());
            true
        },
        &guard,
    );

    // Expected order: "hello" < "hello w" < "hello wo" < "hello woAAAA" < "hello woBBBB"
    assert_eq!(keys.len(), 5);
    assert_eq!(keys[0], b"hello");
    assert_eq!(keys[1], b"hello w");
    assert_eq!(keys[2], b"hello wo");
    assert_eq!(keys[3], b"hello woAAAA");
    assert_eq!(keys[4], b"hello woBBBB");
}

/// Test inline keys that share the same ikey due to zero-padding.
///
/// This exercises the "keylenx ordering decides position" logic that
/// relies on for implicit layer/suffix handling.
#[test]
fn test_inline_keys_same_ikey_different_lengths() {
    use crate::RangeBound;

    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // All these keys have the same ikey (zero-padded to 8 bytes)
    // but different keylenx values (1, 2, 3, 8)
    tree.insert(b"a", 1).unwrap();
    tree.insert(b"aa", 2).unwrap();
    tree.insert(b"aaa", 3).unwrap();
    tree.insert(b"aaaaaaaa", 8).unwrap(); // Exactly 8 bytes

    // Verify retrieval
    assert_eq!(tree.get(b"a"), Some(1));
    assert_eq!(tree.get(b"aa"), Some(2));
    assert_eq!(tree.get(b"aaa"), Some(3));
    assert_eq!(tree.get(b"aaaaaaaa"), Some(8));

    // Verify ordering
    let mut keys: Vec<Vec<u8>> = Vec::new();
    tree.scan(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |k, _| {
            keys.push(k.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(keys.len(), 4);
    assert_eq!(keys[0], b"a");
    assert_eq!(keys[1], b"aa");
    assert_eq!(keys[2], b"aaa");
    assert_eq!(keys[3], b"aaaaaaaa");
}

// ============================================================================
//  FromIterator and Extend Tests
// ============================================================================

#[test]
fn test_from_iterator_basic() {
    let data: Vec<(Vec<u8>, u64)> = vec![
        (b"apple".to_vec(), 1),
        (b"banana".to_vec(), 2),
        (b"cherry".to_vec(), 3),
    ];

    let tree: MassTree<u64> = data.into_iter().collect();

    assert_eq!(tree.len(), 3);
    assert_eq!(tree.get(b"apple"), Some(1));
    assert_eq!(tree.get(b"banana"), Some(2));
    assert_eq!(tree.get(b"cherry"), Some(3));
}

#[test]
fn test_from_iterator_empty() {
    let data: Vec<(Vec<u8>, u64)> = vec![];
    let tree: MassTree<u64> = data.into_iter().collect();

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_from_iterator_string_keys() {
    // Test with String keys (via AsRef<[u8]>)
    let data: Vec<(String, u64)> = vec![("hello".to_string(), 1), ("world".to_string(), 2)];

    let tree: MassTree<u64> = data.into_iter().collect();

    assert_eq!(tree.len(), 2);
    assert_eq!(tree.get(b"hello"), Some(1));
}

#[test]
fn test_from_iterator_byte_literal_keys() {
    // Test with &[u8; N] keys (byte literals) - no .as_slice() needed
    let data = vec![
        (b"alpha".to_vec(), 1u64),
        (b"beta".to_vec(), 2u64),
        (b"gamma".to_vec(), 3u64),
    ];

    let tree: MassTree<u64> = data.into_iter().collect();

    assert_eq!(tree.len(), 3);
    assert_eq!(tree.get(b"alpha"), Some(1));
}

#[test]
fn test_from_iterator_duplicate_keys() {
    // Later values should overwrite earlier ones (upsert semantics)
    let data = vec![
        (b"key".as_slice(), 1u64),
        (b"key".as_slice(), 2u64),
        (b"key".as_slice(), 3u64),
    ];

    let tree: MassTree<u64> = data.into_iter().collect();

    assert_eq!(tree.len(), 1);
    assert_eq!(tree.get(b"key"), Some(3));
}

#[test]
fn test_extend_basic() {
    let mut tree: MassTree<u64> = MassTree::new();
    tree.insert(b"existing", 0).unwrap();

    tree.extend([(b"a", 1u64), (b"b", 2u64), (b"c", 3u64)]);

    assert_eq!(tree.len(), 4);
    assert_eq!(tree.get(b"existing"), Some(0));
    assert_eq!(tree.get(b"a"), Some(1));
    assert_eq!(tree.get(b"b"), Some(2));
    assert_eq!(tree.get(b"c"), Some(3));
}

#[test]
fn test_extend_overwrites_existing() {
    let mut tree: MassTree<u64> = MassTree::new();
    tree.insert(b"key", 100).unwrap();

    tree.extend([(b"key", 200u64)]);

    // Upsert semantics: value should be updated, not duplicated
    assert_eq!(tree.get(b"key"), Some(200));
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_extend_empty_iterator() {
    let mut tree: MassTree<u64> = MassTree::new();
    tree.insert(b"existing", 42).unwrap();

    let empty: Vec<(&[u8], u64)> = vec![];
    tree.extend(empty);

    assert_eq!(tree.len(), 1);
    assert_eq!(tree.get(b"existing"), Some(42));
}

#[test]
fn test_extend_with_vec_keys() {
    let mut tree: MassTree<u64> = MassTree::new();

    tree.extend(vec![(b"key1".to_vec(), 100u64), (b"key2".to_vec(), 200u64)]);

    assert_eq!(tree.len(), 2);
    assert_eq!(tree.get(b"key1"), Some(100));
    assert_eq!(tree.get(b"key2"), Some(200));
}

#[test]
fn test_from_iterator_masstree15() {
    use crate::MassTree15;

    let data = vec![(b"a".to_vec(), 1u64), (b"b".to_vec(), 2u64)];

    let tree: MassTree15<u64> = data.into_iter().collect();
    assert_eq!(tree.len(), 2);
}

#[test]
fn test_from_iterator_inline() {
    use crate::MassTree15Inline;

    let data = vec![(b"x".to_vec(), 10u64), (b"y".to_vec(), 20u64)];

    let tree: MassTree15Inline<u64> = data.into_iter().collect();

    assert_eq!(tree.len(), 2);

    let guard = tree.guard();
    assert_eq!(tree.get_with_guard(b"x", &guard), Some(10));
    assert_eq!(tree.get_with_guard(b"y", &guard), Some(20));
}

#[test]
fn test_extend_inline() {
    use crate::MassTree15Inline;

    let mut tree: MassTree15Inline<u64> = MassTree15Inline::new();

    tree.extend([(b"a", 1u64), (b"b", 2u64)]);

    assert_eq!(tree.len(), 2);
}

#[test]
fn test_from_iterator_large() {
    // Test with many entries to exercise splits
    let tree: MassTree<u64> = (0..1000u64)
        .map(|i| (format!("key{i:04}").into_bytes(), i))
        .collect();

    assert_eq!(tree.len(), 1000);

    // Spot check
    assert_eq!(tree.get(b"key0000"), Some(0));
    assert_eq!(tree.get(b"key0500"), Some(500));
    assert_eq!(tree.get(b"key0999"), Some(999));
}

#[test]
fn test_from_iterator_then_concurrent_access() {
    // Verify tree built via collect() works correctly with concurrent access
    let tree: Arc<MassTree<u64>> = Arc::new(
        (0..100u64)
            .map(|i| (format!("init{i:03}").into_bytes(), i))
            .collect(),
    );

    assert_eq!(tree.len(), 100);

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread inserts 50 unique keys
                for i in 0..50 {
                    let key = format!("t{t}_{i:03}");
                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 1000 + i) as u64, &guard);
                }
                // And reads some original keys
                for i in 0..20 {
                    let key = format!("init{i:03}");
                    let _ = tree.get_with_guard(key.as_bytes(), &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Original 100 + 4 threads * 50 keys = 300
    assert_eq!(tree.len(), 300);

    // Original keys still accessible
    assert_eq!(tree.get(b"init000"), Some(0));
    assert_eq!(tree.get(b"init099"), Some(99));
}
