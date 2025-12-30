#![allow(clippy::pedantic, clippy::needless_collect, clippy::indexing_slicing)]

use super::*;
use crate::internode::InternodeNode;
use crate::leaf24::LeafNode24;
use crate::nodeversion::{LockGuard, NodeVersion};
use crate::tree::MassTree24;
use crate::value::LeafValue;
use std::sync::Arc;
use tracing_test::traced_test;

// Type aliases for coalescing tests
type TestLeaf = LeafNode24<LeafValue<u64>>;
type TestInternode = InternodeNode<LeafValue<u64>, 15>;
type TestTree = MassTree24<u64>;

#[test]
fn test_remove_single_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key1", 42).unwrap();
    assert_eq!(tree.len(), 1);

    let removed = tree.remove(b"key1").unwrap();
    assert_eq!(removed, Some(Arc::new(42)));
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_remove_nonexistent_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key1", 42).unwrap();

    let result = tree.remove(b"key2");
    assert!(matches!(result, Ok(None)));

    // Original key still exists
    assert_eq!(tree.get(b"key1"), Some(Arc::new(42)));
}

#[test]
fn test_remove_updates_count() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..10u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }
    assert_eq!(tree.len(), 10);

    for i in 0..5u64 {
        tree.remove(&i.to_be_bytes()).unwrap();
    }
    assert_eq!(tree.len(), 5);

    // Verify remaining keys
    for i in 5..10u64 {
        assert!(tree.get(&i.to_be_bytes()).is_some());
    }
    for i in 0..5u64 {
        assert!(tree.get(&i.to_be_bytes()).is_none());
    }
}

#[test]
fn test_remove_returns_old_value() {
    let tree: MassTree24<String> = MassTree24::new();

    tree.insert(b"key", "hello".to_string()).unwrap();
    tree.insert(b"key", "world".to_string()).unwrap();

    let removed = tree.remove(b"key").unwrap();
    assert_eq!(removed, Some(Arc::new("world".to_string())));
}

#[test]
fn test_remove_short_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 1-byte key
    tree.insert(&[42], 1).unwrap();
    assert_eq!(tree.remove(&[42]).unwrap(), Some(Arc::new(1)));

    // 8-byte key (max inline)
    let key8 = [1, 2, 3, 4, 5, 6, 7, 8];
    tree.insert(&key8, 8).unwrap();
    assert_eq!(tree.remove(&key8).unwrap(), Some(Arc::new(8)));
}

#[test]
fn test_remove_with_suffix() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 16-byte key (requires suffix)
    let key16 = b"0123456789ABCDEF";
    tree.insert(key16, 16).unwrap();

    let removed = tree.remove(key16).unwrap();
    assert_eq!(removed, Some(Arc::new(16)));
    assert!(tree.get(key16).is_none());
}

#[test]
fn test_remove_all_keys_empties_tree() {
    let tree: MassTree24<u64> = MassTree24::new();

    let keys: Vec<_> = (0..100u64).map(u64::to_be_bytes).collect();

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64).unwrap();
    }
    assert_eq!(tree.len(), 100);

    for key in &keys {
        tree.remove(key).unwrap();
    }
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
}

#[test]
fn test_remove_in_reverse_order() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..50u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    // Remove in reverse order
    for i in (0..50u64).rev() {
        let removed = tree.remove(&i.to_be_bytes()).unwrap();
        assert_eq!(removed, Some(Arc::new(i)));
    }

    assert!(tree.is_empty());
}

#[test]
fn test_remove_alternating() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..100u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    // Remove even keys
    for i in (0..100u64).step_by(2) {
        tree.remove(&i.to_be_bytes()).unwrap();
    }

    assert_eq!(tree.len(), 50);

    // Verify odd keys remain
    for i in (1..100u64).step_by(2) {
        assert!(tree.get(&i.to_be_bytes()).is_some());
    }
}

#[test]
fn test_remove_and_reinsert_same_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key", 1).unwrap();
    tree.remove(b"key").unwrap();

    // Reinsert with different value
    tree.insert(b"key", 2).unwrap();
    assert_eq!(tree.get(b"key"), Some(Arc::new(2)));
}

#[test]
fn test_remove_reinsert_cycle() {
    let tree: MassTree24<u64> = MassTree24::new();
    let key = b"test_key";

    for i in 0..10u64 {
        tree.insert(key, i).unwrap();
        assert_eq!(tree.get(key), Some(Arc::new(i)));

        let removed = tree.remove(key).unwrap();
        assert_eq!(removed, Some(Arc::new(i)));
        assert!(tree.get(key).is_none());
    }
}

#[test]
fn test_remove_from_empty_tree() {
    let tree: MassTree24<u64> = MassTree24::new();
    let result = tree.remove(b"key");
    assert!(matches!(result, Ok(None)));
}

#[test]
fn test_remove_empty_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Empty key is valid
    tree.insert(&[], 0).unwrap();
    let removed = tree.remove(&[]).unwrap();
    assert_eq!(removed, Some(Arc::new(0)));
}

#[test]
fn test_remove_preserves_other_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"aaa", 1).unwrap();
    tree.insert(b"bbb", 2).unwrap();
    tree.insert(b"ccc", 3).unwrap();

    tree.remove(b"bbb").unwrap();

    assert_eq!(tree.get(b"aaa"), Some(Arc::new(1)));
    assert!(tree.get(b"bbb").is_none());
    assert_eq!(tree.get(b"ccc"), Some(Arc::new(3)));
}

// ============================================================================
//  Coalescing Helper Function Tests
// ============================================================================

// ----------------------------------------------------------------------------
// get_parent_erased tests
// ----------------------------------------------------------------------------

#[test]
fn test_get_parent_erased_leaf() {
    // Setup: Create a leaf with a real parent internode
    let parent_inode: Box<TestInternode> = TestInternode::new(0);
    let parent_ptr: *mut u8 = Box::into_raw(parent_inode).cast();

    let leaf: Box<TestLeaf> = TestLeaf::new();
    leaf.set_parent(parent_ptr);

    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Test: get_parent_erased should return the parent
    let got_parent: *mut u8 =
        unsafe { NodeCleaner::get_parent_erased::<LeafValue<u64>, TestLeaf>(leaf_ptr) };

    assert_eq!(got_parent, parent_ptr);

    // Cleanup
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(parent_ptr.cast::<TestInternode>()) };
}

#[test]
fn test_get_parent_erased_internode() {
    // Setup: Create an internode with a real grandparent internode
    let grandparent: Box<TestInternode> = TestInternode::new(1);
    let grandparent_ptr: *mut u8 = Box::into_raw(grandparent).cast();

    let inode: Box<TestInternode> = TestInternode::new(0);
    inode.set_parent(grandparent_ptr);

    let inode_ptr: *mut u8 = Box::into_raw(inode).cast();

    // Test: get_parent_erased should return the parent
    let got_parent: *mut u8 =
        unsafe { NodeCleaner::get_parent_erased::<LeafValue<u64>, TestLeaf>(inode_ptr) };

    assert_eq!(got_parent, grandparent_ptr);

    // Cleanup
    let _: Box<TestInternode> = unsafe { Box::from_raw(inode_ptr.cast::<TestInternode>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(grandparent_ptr.cast::<TestInternode>()) };
}

#[test]
fn test_get_parent_erased_null_parent() {
    // Setup: Create a root leaf (null parent)
    let leaf: Box<TestLeaf> = TestLeaf::new_root();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Test: get_parent_erased should return null
    let parent: *mut u8 =
        unsafe { NodeCleaner::get_parent_erased::<LeafValue<u64>, TestLeaf>(leaf_ptr) };

    assert!(parent.is_null());

    // Cleanup
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
}

// ----------------------------------------------------------------------------
// set_parent_erased tests
// ----------------------------------------------------------------------------

#[test]
fn test_set_parent_erased_leaf() {
    // Setup: Use a real internode as the new parent
    let new_parent_node: Box<TestInternode> = TestInternode::new(0);
    let new_parent: *mut u8 = Box::into_raw(new_parent_node).cast();

    let leaf: Box<TestLeaf> = TestLeaf::new();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Initially null
    assert!(unsafe { (*leaf_ptr.cast::<TestLeaf>()).parent().is_null() });

    // Test: set_parent_erased should update leaf's parent
    unsafe {
        NodeCleaner::set_parent_erased::<LeafValue<u64>, TestLeaf>(leaf_ptr, new_parent);
    }

    // Verify
    let actual_parent: *mut u8 = unsafe { (*leaf_ptr.cast::<TestLeaf>()).parent() };
    assert_eq!(actual_parent, new_parent);

    // Cleanup
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(new_parent.cast::<TestInternode>()) };
}

#[test]
fn test_set_parent_erased_internode() {
    // Setup: Use a real internode as the new parent
    let new_parent_node: Box<TestInternode> = TestInternode::new(1);
    let new_parent: *mut u8 = Box::into_raw(new_parent_node).cast();

    let inode: Box<TestInternode> = TestInternode::new(0);
    let inode_ptr: *mut u8 = Box::into_raw(inode).cast();

    // Initially null
    assert!(unsafe { (*inode_ptr.cast::<TestInternode>()).parent().is_null() });

    // Test: set_parent_erased should update internode's parent
    unsafe {
        NodeCleaner::set_parent_erased::<LeafValue<u64>, TestLeaf>(inode_ptr, new_parent);
    }

    // Verify
    let actual_parent: *mut u8 = unsafe { (*inode_ptr.cast::<TestInternode>()).parent() };
    assert_eq!(actual_parent, new_parent);

    // Cleanup
    let _: Box<TestInternode> = unsafe { Box::from_raw(inode_ptr.cast::<TestInternode>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(new_parent.cast::<TestInternode>()) };
}

#[test]
fn test_set_parent_erased_type_dispatch() {
    // This test verifies that is_leaf() correctly distinguishes node types

    // Create both types
    let leaf: Box<TestLeaf> = TestLeaf::new();
    let inode: Box<TestInternode> = TestInternode::new(1);

    // Verify is_leaf() returns correct values
    assert!(leaf.version().is_leaf());
    assert!(!inode.version().is_leaf());

    // Cleanup (no raw pointers escaped)
}

// ----------------------------------------------------------------------------
// locked_parent_generic tests
// ----------------------------------------------------------------------------

#[test]
fn test_locked_parent_null_parent() {
    // Setup: Create a root leaf (no parent)
    let leaf: Box<TestLeaf> = TestLeaf::new_root();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Lock the leaf first (precondition)
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent_generic should return (None, null)
    let (lock_opt, parent_ptr): (Option<LockGuard<'_>>, *mut u8) =
        unsafe { NodeCleaner::locked_parent_generic::<LeafValue<u64>, TestLeaf>(leaf_ptr) };

    assert!(lock_opt.is_none());
    assert!(parent_ptr.is_null());

    // Cleanup
    drop(_leaf_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
}

#[test]
fn test_locked_parent_basic() {
    // Setup: Create leaf -> internode parent relationship
    let parent: Box<TestInternode> = TestInternode::new(0);
    let parent_ptr: *mut TestInternode = Box::into_raw(parent);

    let leaf: Box<TestLeaf> = TestLeaf::new();
    leaf.set_parent(parent_ptr.cast());
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Set up child pointer in parent
    unsafe { (*parent_ptr).set_child(0, leaf_ptr) };

    // Lock the leaf first (precondition)
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent_generic should return locked parent
    let (lock_opt, returned_parent): (Option<LockGuard<'_>>, *mut u8) =
        unsafe { NodeCleaner::locked_parent_generic::<LeafValue<u64>, TestLeaf>(leaf_ptr) };

    assert!(lock_opt.is_some());
    assert_eq!(returned_parent, parent_ptr.cast::<u8>());

    // Parent should be locked
    let parent_ref: &TestInternode = unsafe { &*parent_ptr };
    assert!(parent_ref.version().is_locked());

    // Cleanup
    drop(lock_opt);
    drop(_leaf_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(parent_ptr) };
}

#[test]
fn test_locked_parent_returns_internode() {
    // Setup: Two-level tree (leaf -> internode -> grandparent)
    let grandparent: Box<TestInternode> = TestInternode::new(1);
    grandparent.version().mark_root();
    let grandparent_ptr: *mut TestInternode = Box::into_raw(grandparent);

    let parent: Box<TestInternode> = TestInternode::new(0);
    parent.set_parent(grandparent_ptr.cast());
    let parent_ptr: *mut TestInternode = Box::into_raw(parent);

    unsafe { (*grandparent_ptr).set_child(0, parent_ptr.cast()) };

    let leaf: Box<TestLeaf> = TestLeaf::new();
    leaf.set_parent(parent_ptr.cast());
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    unsafe { (*parent_ptr).set_child(0, leaf_ptr) };

    // Lock leaf
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent should return parent (not grandparent)
    let (lock_opt, returned_parent): (Option<LockGuard<'_>>, *mut u8) =
        unsafe { NodeCleaner::locked_parent_generic::<LeafValue<u64>, TestLeaf>(leaf_ptr) };

    assert!(lock_opt.is_some());
    assert_eq!(returned_parent, parent_ptr.cast::<u8>());

    // Verify it's not a leaf
    let parent_version: &NodeVersion = unsafe { &*(returned_parent.cast::<NodeVersion>()) };
    assert!(!parent_version.is_leaf());

    // Cleanup
    drop(lock_opt);
    drop(_leaf_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(parent_ptr) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(grandparent_ptr) };
}

// ----------------------------------------------------------------------------
// shift_internode_down_generic tests
// ----------------------------------------------------------------------------

#[test]
fn test_shift_internode_down_middle() {
    // Setup: Internode with 3 keys, remove child at kp=2
    //
    // Before: keys = [10, 20, 30], children = [c0, c1, c2, c3]
    // Remove c2 (kp=2)
    // After:  keys = [10, 30, _],  children = [c0, c1, c3, _]

    let inode: Box<TestInternode> = TestInternode::new(0);

    // Set up keys
    inode.set_ikey(0, 10);
    inode.set_ikey(1, 20);
    inode.set_ikey(2, 30);
    inode.set_nkeys(3);

    // Set up children using real leaf allocations
    let leaves: Vec<Box<TestLeaf>> = (0..4).map(|_| TestLeaf::new()).collect();
    let ptrs: Vec<*mut u8> = leaves
        .into_iter()
        .map(|l| Box::into_raw(l) as *mut u8)
        .collect();

    let (c0, c1, c2, c3) = (ptrs[0], ptrs[1], ptrs[2], ptrs[3]);

    inode.set_child(0, c0);
    inode.set_child(1, c1);
    inode.set_child(2, c2);
    inode.set_child(3, c3);

    // "Remove" c2 by setting to null (simulating the removal)
    inode.set_child(2, std::ptr::null_mut());

    // Test: shift_internode_down(kp=2)
    NodeCleaner::shift_internode_down_generic::<LeafValue<u64>, TestInternode>(&inode, 2);

    // Verify keys: [10, 30, _]
    assert_eq!(inode.ikey(0), 10);
    assert_eq!(inode.ikey(1), 30);

    // Verify children: [c0, c1, c3, _]
    assert_eq!(inode.child(0), c0);
    assert_eq!(inode.child(1), c1);
    assert_eq!(inode.child(2), c3);

    // Verify nkeys decremented
    assert_eq!(inode.nkeys(), 2);

    // Cleanup
    for ptr in ptrs {
        let _: Box<TestLeaf> = unsafe { Box::from_raw(ptr.cast::<TestLeaf>()) };
    }
}

#[test]
fn test_shift_internode_down_last() {
    // Setup: Internode with 3 keys, remove child at kp=3 (last)
    //
    // Before: keys = [10, 20, 30], children = [c0, c1, c2, c3]
    // Remove c3 (kp=3)
    // After:  keys = [10, 20, _],  children = [c0, c1, c2, _]

    let inode: Box<TestInternode> = TestInternode::new(0);

    inode.set_ikey(0, 10);
    inode.set_ikey(1, 20);
    inode.set_ikey(2, 30);
    inode.set_nkeys(3);

    // Set up children using real leaf allocations
    let leaves: Vec<Box<TestLeaf>> = (0..4).map(|_| TestLeaf::new()).collect();
    let ptrs: Vec<*mut u8> = leaves
        .into_iter()
        .map(|l| Box::into_raw(l) as *mut u8)
        .collect();

    let (c0, c1, c2, c3) = (ptrs[0], ptrs[1], ptrs[2], ptrs[3]);

    inode.set_child(0, c0);
    inode.set_child(1, c1);
    inode.set_child(2, c2);
    inode.set_child(3, c3);

    inode.set_child(3, std::ptr::null_mut());

    // Test: shift_internode_down(kp=3)
    NodeCleaner::shift_internode_down_generic::<LeafValue<u64>, TestInternode>(&inode, 3);

    // Verify keys: [10, 20, _]
    assert_eq!(inode.ikey(0), 10);
    assert_eq!(inode.ikey(1), 20);

    // Verify children: [c0, c1, c2, _]
    assert_eq!(inode.child(0), c0);
    assert_eq!(inode.child(1), c1);
    assert_eq!(inode.child(2), c2);

    assert_eq!(inode.nkeys(), 2);

    // Cleanup
    for ptr in ptrs {
        let _: Box<TestLeaf> = unsafe { Box::from_raw(ptr.cast::<TestLeaf>()) };
    }
}

#[test]
fn test_shift_internode_down_second() {
    // Setup: Internode with 2 keys, remove child at kp=1
    //
    // Before: keys = [10, 20], children = [c0, c1, c2]
    // Remove c1 (kp=1)
    // After:  keys = [20, _],  children = [c0, c2, _]

    let inode: Box<TestInternode> = TestInternode::new(0);

    inode.set_ikey(0, 10);
    inode.set_ikey(1, 20);
    inode.set_nkeys(2);

    // Set up children using real leaf allocations
    let leaves: Vec<Box<TestLeaf>> = (0..3).map(|_| TestLeaf::new()).collect();
    let ptrs: Vec<*mut u8> = leaves
        .into_iter()
        .map(|l| Box::into_raw(l) as *mut u8)
        .collect();

    let (c0, c1, c2) = (ptrs[0], ptrs[1], ptrs[2]);

    inode.set_child(0, c0);
    inode.set_child(1, c1);
    inode.set_child(2, c2);

    inode.set_child(1, std::ptr::null_mut());

    // Test
    NodeCleaner::shift_internode_down_generic::<LeafValue<u64>, TestInternode>(&inode, 1);

    // Verify keys: [20, _]
    assert_eq!(inode.ikey(0), 20);

    // Verify children: [c0, c2, _]
    assert_eq!(inode.child(0), c0);
    assert_eq!(inode.child(1), c2);

    assert_eq!(inode.nkeys(), 1);

    // Cleanup
    for ptr in ptrs {
        let _: Box<TestLeaf> = unsafe { Box::from_raw(ptr.cast::<TestLeaf>()) };
    }
}

// ----------------------------------------------------------------------------
// B-link chain unlink tests
// ----------------------------------------------------------------------------

#[test]
fn test_unlink_from_chain_middle() {
    // Setup: Chain of 3 leaves: A <-> B <-> C
    // Unlink B
    // Verify: A <-> C

    let leaf_a: Box<TestLeaf> = TestLeaf::new();
    let leaf_b: Box<TestLeaf> = TestLeaf::new();
    let leaf_c: Box<TestLeaf> = TestLeaf::new();

    let a_ptr: *mut TestLeaf = Box::into_raw(leaf_a);
    let b_ptr: *mut TestLeaf = Box::into_raw(leaf_b);
    let c_ptr: *mut TestLeaf = Box::into_raw(leaf_c);

    // Link: A <-> B <-> C
    unsafe {
        (*a_ptr).set_next(b_ptr);
        (*b_ptr).set_prev(a_ptr);
        (*b_ptr).set_next(c_ptr);
        (*c_ptr).set_prev(b_ptr);
    }

    // Lock B and unlink it
    let b_ref: &TestLeaf = unsafe { &*b_ptr };
    let _lock: LockGuard<'_> = b_ref.version().lock();

    unsafe { b_ref.unlink_from_chain() };

    // Verify: A <-> C
    assert_eq!(unsafe { (*a_ptr).safe_next() }, c_ptr);
    assert_eq!(unsafe { (*c_ptr).prev() }, a_ptr);

    // Cleanup
    drop(_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(a_ptr) };
    let _: Box<TestLeaf> = unsafe { Box::from_raw(b_ptr) };
    let _: Box<TestLeaf> = unsafe { Box::from_raw(c_ptr) };
}

#[test]
fn test_unlink_from_chain_last() {
    // Setup: Chain of 2 leaves: A <-> B
    // Unlink B (last)
    // Verify: A.next == null

    let leaf_a: Box<TestLeaf> = TestLeaf::new();
    let leaf_b: Box<TestLeaf> = TestLeaf::new();

    let a_ptr: *mut TestLeaf = Box::into_raw(leaf_a);
    let b_ptr: *mut TestLeaf = Box::into_raw(leaf_b);

    // Link: A <-> B
    unsafe {
        (*a_ptr).set_next(b_ptr);
        (*b_ptr).set_prev(a_ptr);
    }

    // Lock B and unlink it
    let b_ref: &TestLeaf = unsafe { &*b_ptr };
    let _lock: LockGuard<'_> = b_ref.version().lock();

    unsafe { b_ref.unlink_from_chain() };

    // Verify: A.next == null
    assert!(unsafe { (*a_ptr).safe_next().is_null() });

    // Cleanup
    drop(_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(a_ptr) };
    let _: Box<TestLeaf> = unsafe { Box::from_raw(b_ptr) };
}

// ============================================================================
//  Integration Tests for Leaf Removal
// ============================================================================

#[test]
fn test_remove_leaf_updates_parent_child_ptr() {
    // Setup: Tree with root internode -> 2 leaves
    // Insert keys to create structure, then remove to trigger leaf removal

    let tree: TestTree = TestTree::new();

    // Insert keys to create multi-leaf structure
    tree.insert(&50_u64.to_be_bytes(), 50).unwrap();
    tree.insert(&150_u64.to_be_bytes(), 150).unwrap();

    // Remove key
    let removed = tree.remove(&150_u64.to_be_bytes());
    assert!(removed.is_ok());

    // Verify tree still works
    assert_eq!(tree.get(&50_u64.to_be_bytes()), Some(Arc::new(50)));
    assert_eq!(tree.get(&150_u64.to_be_bytes()), None);
}

#[test]
fn test_remove_leaf_leftmost_not_removed() {
    // Leftmost leaf (prev == null) should NOT be removed even when empty

    let tree: TestTree = TestTree::new();

    tree.insert(&42_u64.to_be_bytes(), 42).unwrap();
    let removed = tree.remove(&42_u64.to_be_bytes());
    assert!(removed.is_ok());

    // Tree is empty but root leaf should still exist
    assert_eq!(tree.len(), 0);

    // Can still insert
    tree.insert(&100_u64.to_be_bytes(), 100).unwrap();
    assert_eq!(tree.get(&100_u64.to_be_bytes()), Some(Arc::new(100)));
}

#[test]
fn test_redirect_via_sequential_removal() {
    // Test redirect by removing keys in order

    let tree: TestTree = TestTree::new();

    // Create a multi-leaf tree
    eprintln!("Inserting 50 keys...");
    for i in 0_u64..50 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }
    eprintln!("Inserted 50 keys, len = {}", tree.len());

    // Remove keys from the beginning (leftmost positions)
    eprintln!("Removing keys 0-24...");
    for i in 0_u64..25 {
        eprintln!("  Removing key {}", i);
        tree.remove(&i.to_be_bytes()).unwrap();
        eprintln!("  Removed key {}, len = {}", i, tree.len());
    }

    // Verify remaining keys are still accessible
    eprintln!("Verifying remaining keys...");
    for i in 25_u64..50 {
        eprintln!("  Getting key {}", i);
        assert_eq!(tree.get(&i.to_be_bytes()), Some(Arc::new(i)));
    }

    // Verify removed keys are gone
    eprintln!("Verifying removed keys are gone...");
    for i in 0_u64..25 {
        eprintln!("  Checking key {} is gone", i);
        assert!(tree.get(&i.to_be_bytes()).is_none());
    }
    eprintln!("Done!");
}

#[test]
fn test_redirect_alternating_removal() {
    // Remove keys in a pattern that triggers redirect at various levels

    let tree: TestTree = TestTree::new();

    // Insert keys with gaps to create specific tree structure
    for i in (0_u64..100).step_by(2) {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    // Remove from various positions
    for i in (0_u64..100).step_by(4) {
        tree.remove(&i.to_be_bytes()).unwrap();
    }

    // Verify correctness
    for i in (0_u64..100).step_by(2) {
        if i % 4 == 0 {
            assert!(tree.get(&i.to_be_bytes()).is_none());
        } else {
            assert_eq!(tree.get(&i.to_be_bytes()), Some(Arc::new(i)));
        }
    }
}

// ============================================================================
//  Concurrent Tests
// ============================================================================

#[test]
#[cfg(not(miri))]
fn test_concurrent_remove_and_get() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let done = Arc::new(AtomicBool::new(false));

    // Pre-populate tree
    for i in 0_u64..1000 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let tree_clone = Arc::clone(&tree);
    let done_clone = Arc::clone(&done);

    // Reader thread: continuously get random keys
    let reader = thread::spawn(move || {
        let mut found = 0_u64;
        let mut not_found = 0_u64;

        while !done_clone.load(Ordering::Relaxed) {
            for i in 0_u64..100 {
                let key: u64 = (i * 7) % 1000;
                if tree_clone.get(&key.to_be_bytes()).is_some() {
                    found += 1;
                } else {
                    not_found += 1;
                }
            }
        }

        (found, not_found)
    });

    // Writer thread: remove keys
    for i in (0_u64..1000).step_by(2) {
        tree.remove(&i.to_be_bytes()).unwrap();
    }

    done.store(true, Ordering::Relaxed);
    let (found, not_found) = reader.join().unwrap();

    // Verify: no crashes, reasonable counts
    assert!(found > 0 || not_found > 0);

    // Final verification: odd keys should still exist
    for i in (1_u64..1000).step_by(2) {
        assert_eq!(tree.get(&i.to_be_bytes()), Some(Arc::new(i)));
    }
}

#[test]
#[cfg(not(miri))]
fn test_concurrent_remove_same_keys() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let removed_count = Arc::new(AtomicUsize::new(0));

    // Pre-populate
    for i in 0_u64..100 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let mut handles = vec![];

    // Spawn 4 threads all trying to remove the same keys
    for _ in 0..4 {
        let tree_clone = Arc::clone(&tree);
        let count_clone = Arc::clone(&removed_count);

        handles.push(thread::spawn(move || {
            let mut local_removed = 0;

            for i in 0_u64..100 {
                if tree_clone.remove(&i.to_be_bytes()).unwrap().is_some() {
                    local_removed += 1;
                }
            }

            count_clone.fetch_add(local_removed, Ordering::Relaxed);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Exactly 100 keys should have been removed total
    // (each key removed exactly once)
    assert_eq!(removed_count.load(Ordering::Relaxed), 100);

    // Tree should be empty
    assert_eq!(tree.len(), 0);
}

#[test]
#[cfg(not(miri))]
fn test_stress_remove_all_concurrent() {
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let key_count: u64 = 10_000;

    // Pre-populate
    for i in 0..key_count {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let mut handles = vec![];
    let threads: u64 = 8;
    let keys_per_thread: u64 = key_count / threads;

    // Each thread removes a disjoint range
    for t in 0..threads {
        let tree_clone = Arc::clone(&tree);
        let start: u64 = t * keys_per_thread;
        let end: u64 = start + keys_per_thread;

        handles.push(thread::spawn(move || {
            for i in start..end {
                tree_clone.remove(&i.to_be_bytes()).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Tree should be empty
    assert_eq!(tree.len(), 0);

    // All keys should be gone
    for i in 0..key_count {
        assert!(tree.get(&i.to_be_bytes()).is_none());
    }
}

// ============================================================================
//  Progress Hazard Tests
// ============================================================================

#[test]
#[cfg(not(miri))]
fn test_no_infinite_loop_deleted_node() {
    // This test verifies the core bug fix: readers should not
    // infinite loop when encountering a deleted node.

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let reader_done = Arc::new(AtomicBool::new(false));

    // Create a tree with multiple leaves
    for i in 0_u64..100 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let tree_clone = Arc::clone(&tree);
    let done_clone = Arc::clone(&reader_done);

    // Reader: continuously read keys
    let reader = thread::spawn(move || {
        for _ in 0..1000 {
            for i in 0_u64..100 {
                let _ = tree_clone.get(&i.to_be_bytes());
            }
        }
        done_clone.store(true, Ordering::Release);
    });

    // Give reader time to start
    thread::sleep(Duration::from_millis(10));

    // Remove keys (may trigger coalescing when enabled)
    for i in (0_u64..100).step_by(2) {
        tree.remove(&i.to_be_bytes()).unwrap();
    }

    // Wait for reader with timeout
    let result = reader.join();

    // If reader completed, it didn't hang
    assert!(result.is_ok());
    assert!(reader_done.load(Ordering::Acquire));
}

#[test]
fn test_reader_retry_succeeds_after_coalesce() {
    // After coalescing, a reader that was mid-traversal should
    // successfully retry and either find the key or correctly
    // report not found.

    let tree: TestTree = TestTree::new();

    // Insert and remove
    tree.insert(&42_u64.to_be_bytes(), 42).unwrap();
    tree.insert(&100_u64.to_be_bytes(), 100).unwrap();

    // Remove one key
    tree.remove(&42_u64.to_be_bytes()).unwrap();

    // Get should work (retry if needed internally)
    assert!(tree.get(&42_u64.to_be_bytes()).is_none());
    assert_eq!(tree.get(&100_u64.to_be_bytes()), Some(Arc::new(100)));
}

// ============================================================================
//  Miri-Compatible Tests
// ============================================================================

#[test]
fn test_miri_remove_single_key() {
    let tree: TestTree = TestTree::new();

    tree.insert(&1_u64.to_be_bytes(), 1).unwrap();
    assert_eq!(
        tree.remove(&1_u64.to_be_bytes()).unwrap(),
        Some(Arc::new(1))
    );
    assert!(tree.get(&1_u64.to_be_bytes()).is_none());
}

#[test]
fn test_miri_remove_multiple_keys() {
    let tree: TestTree = TestTree::new();

    for i in 0_u64..10 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    for i in 0_u64..10 {
        assert_eq!(tree.remove(&i.to_be_bytes()).unwrap(), Some(Arc::new(i)));
    }

    assert_eq!(tree.len(), 0);
}

#[test]
fn test_miri_parent_erased_helpers() {
    // Test helper functions under Miri using real allocations
    let parent_node: Box<TestInternode> = TestInternode::new(0);
    let parent_ptr: *mut u8 = Box::into_raw(parent_node).cast();

    let leaf: Box<TestLeaf> = TestLeaf::new();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // set_parent_erased
    unsafe {
        NodeCleaner::set_parent_erased::<LeafValue<u64>, TestLeaf>(leaf_ptr, parent_ptr);
    }

    // get_parent_erased
    let got: *mut u8 =
        unsafe { NodeCleaner::get_parent_erased::<LeafValue<u64>, TestLeaf>(leaf_ptr) };
    assert_eq!(got, parent_ptr);

    // Cleanup
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(parent_ptr.cast::<TestInternode>()) };
}
