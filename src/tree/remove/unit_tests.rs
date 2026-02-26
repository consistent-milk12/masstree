#![allow(
    clippy::panic,
    clippy::pedantic,
    clippy::needless_collect,
    clippy::indexing_slicing
)]

use super::{LockedParentResult, NodeCleaner};
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::nodeversion::{LockGuard, NodeVersion};
use crate::policy::{BoxPolicy, ValuePtr};
use crate::tree::MassTree15;

use std::ptr as StdPtr;
use std::sync::Arc;

/// Helper macro: compare `Option<V>` with `Option<V>`.
///
/// Previously handled `ValuePtr<T>` → `T` conversion via `Deref`.
/// Now that auto-guard methods return owned values directly, this is a
/// simple equality check (kept for call-site compatibility).
macro_rules! assert_val_eq {
    ($got:expr, $expected:expr) => {
        assert_eq!($got, $expected);
    };
    ($got:expr, $expected:expr, $($arg:tt)*) => {
        assert_eq!($got, $expected, $($arg)*);
    };
}

// Type aliases for coalescing tests
type TestLeaf = LeafNode15<BoxPolicy<u64>>;
type TestInternode = InternodeNode;
type TestTree = MassTree15<u64>;

#[test]
fn test_remove_single_key() {
    let tree: MassTree15<u64> = MassTree15::new();

    tree.insert(b"key1", 42);
    assert_eq!(tree.len(), 1);

    let removed = tree.remove(b"key1").unwrap();
    assert_val_eq!(removed, Some(42));
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_remove_nonexistent_key() {
    let tree: MassTree15<u64> = MassTree15::new();

    tree.insert(b"key1", 42);

    let result = tree.remove(b"key2");
    assert!(matches!(result, Ok(None)));

    // Original key still exists
    assert_val_eq!(tree.get(b"key1"), Some(42));
}

#[test]
fn test_remove_updates_count() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..10u64 {
        tree.insert(&i.to_be_bytes(), i);
    }
    assert_eq!(tree.len(), 10);

    for i in 0..5u64 {
        let _ = tree.remove(&i.to_be_bytes());
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
    let tree: MassTree15<String> = MassTree15::new();

    tree.insert(b"key", "hello".to_string());
    tree.insert(b"key", "world".to_string());

    let removed = tree.remove(b"key").unwrap();
    assert_val_eq!(removed, Some("world".to_string()));
}

#[test]
fn test_remove_short_key() {
    let tree: MassTree15<u64> = MassTree15::new();

    // 1-byte key
    tree.insert(&[42], 1);
    assert_val_eq!(tree.remove(&[42]).unwrap(), Some(1));

    // 8-byte key (max inline)
    let key8 = [1, 2, 3, 4, 5, 6, 7, 8];
    tree.insert(&key8, 8);
    assert_val_eq!(tree.remove(&key8).unwrap(), Some(8));
}

#[test]
fn test_remove_with_suffix() {
    let tree: MassTree15<u64> = MassTree15::new();

    // 16-byte key (requires suffix)
    let key16 = b"0123456789ABCDEF";
    tree.insert(key16, 16);

    let removed = tree.remove(key16).unwrap();
    assert_val_eq!(removed, Some(16));
    assert!(tree.get(key16).is_none());
}

#[test]
fn test_remove_all_keys_empties_tree() {
    let tree: MassTree15<u64> = MassTree15::new();

    let keys: Vec<_> = (0..100u64).map(u64::to_be_bytes).collect();

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64);
    }
    assert_eq!(tree.len(), 100);

    for key in &keys {
        let _ = tree.remove(key);
    }
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
}

#[test]
fn test_remove_in_reverse_order() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..50u64 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove in reverse order
    for i in (0..50u64).rev() {
        let removed = tree.remove(&i.to_be_bytes()).unwrap();
        assert_val_eq!(removed, Some(i));
    }

    assert!(tree.is_empty());
}

#[test]
fn test_remove_alternating() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove even keys
    for i in (0..100u64).step_by(2) {
        let _ = tree.remove(&i.to_be_bytes());
    }

    assert_eq!(tree.len(), 50);

    // Verify odd keys remain
    for i in (1..100u64).step_by(2) {
        assert!(tree.get(&i.to_be_bytes()).is_some());
    }
}

#[test]
fn test_remove_and_reinsert_same_key() {
    let tree: MassTree15<u64> = MassTree15::new();

    tree.insert(b"key", 1);
    let _ = tree.remove(b"key");

    // Reinsert with different value
    tree.insert(b"key", 2);
    assert_val_eq!(tree.get(b"key"), Some(2));
}

#[test]
fn test_remove_reinsert_cycle() {
    let tree: MassTree15<u64> = MassTree15::new();
    let key = b"test_key";

    for i in 0..10u64 {
        tree.insert(key, i);
        assert_val_eq!(tree.get(key), Some(i));

        let removed = tree.remove(key).unwrap();
        assert_val_eq!(removed, Some(i));
        assert!(tree.get(key).is_none());
    }
}

#[test]
fn test_remove_from_empty_tree() {
    let tree: MassTree15<u64> = MassTree15::new();
    let result = tree.remove(b"key");
    assert!(matches!(result, Ok(None)));
}

#[test]
fn test_remove_empty_key() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Empty key is valid
    tree.insert(&[], 0);
    let removed = tree.remove(&[]).unwrap();
    assert_val_eq!(removed, Some(0));
}

#[test]
fn test_remove_preserves_other_keys() {
    let tree: MassTree15<u64> = MassTree15::new();

    tree.insert(b"aaa", 1);
    tree.insert(b"bbb", 2);
    tree.insert(b"ccc", 3);

    let _ = tree.remove(b"bbb");

    assert_val_eq!(tree.get(b"aaa"), Some(1));
    assert!(tree.get(b"bbb").is_none());
    assert_val_eq!(tree.get(b"ccc"), Some(3));
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

    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
    leaf.set_parent(parent_ptr);

    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Test: get_parent_erased should return the parent
    let got_parent: *mut u8 = unsafe { NodeCleaner::get_parent_erased::<BoxPolicy<u64>>(leaf_ptr) };

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
        unsafe { NodeCleaner::get_parent_erased::<BoxPolicy<u64>>(inode_ptr) };

    assert_eq!(got_parent, grandparent_ptr);

    // Cleanup
    let _: Box<TestInternode> = unsafe { Box::from_raw(inode_ptr.cast::<TestInternode>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(grandparent_ptr.cast::<TestInternode>()) };
}

#[test]
fn test_get_parent_erased_null_parent() {
    // Setup: Create a root leaf (null parent)
    let leaf: Box<TestLeaf> = TestLeaf::new_root_boxed();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Test: get_parent_erased should return null
    let parent: *mut u8 = unsafe { NodeCleaner::get_parent_erased::<BoxPolicy<u64>>(leaf_ptr) };

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

    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Initially null
    assert!(unsafe { (*leaf_ptr.cast::<TestLeaf>()).parent_unguarded().is_null() });

    // Test: set_parent_erased should update leaf's parent
    unsafe {
        NodeCleaner::set_parent_erased::<BoxPolicy<u64>>(leaf_ptr, new_parent);
    }

    // Verify
    let actual_parent: *mut u8 = unsafe { (*leaf_ptr.cast::<TestLeaf>()).parent_unguarded() };
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
    assert!(unsafe {
        (*inode_ptr.cast::<TestInternode>())
            .parent_unguarded()
            .is_null()
    });

    // Test: set_parent_erased should update internode's parent
    unsafe {
        NodeCleaner::set_parent_erased::<BoxPolicy<u64>>(inode_ptr, new_parent);
    }

    // Verify
    let actual_parent: *mut u8 = unsafe { (*inode_ptr.cast::<TestInternode>()).parent_unguarded() };
    assert_eq!(actual_parent, new_parent);

    // Cleanup
    let _: Box<TestInternode> = unsafe { Box::from_raw(inode_ptr.cast::<TestInternode>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(new_parent.cast::<TestInternode>()) };
}

#[test]
fn test_set_parent_erased_type_dispatch() {
    // This test verifies that is_leaf() correctly distinguishes node types

    // Create both types
    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
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
    let leaf: Box<TestLeaf> = TestLeaf::new_root_boxed();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Lock the leaf first (precondition)
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent_generic should return NoParent for root leaf
    let result: LockedParentResult<'_> =
        unsafe { NodeCleaner::locked_parent_generic::<BoxPolicy<u64>>(leaf_ptr) };

    assert!(matches!(result, LockedParentResult::NoParent));

    // Cleanup
    drop(_leaf_lock);
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
}

#[test]
fn test_locked_parent_basic() {
    // Setup: Create leaf -> internode parent relationship
    let parent: Box<TestInternode> = TestInternode::new(0);
    let parent_ptr: *mut TestInternode = Box::into_raw(parent);

    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
    leaf.set_parent(parent_ptr.cast());
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // Set up child pointer in parent
    unsafe { (*parent_ptr).set_child(0, leaf_ptr) };

    // Lock the leaf first (precondition)
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent_generic should return locked parent
    let result: LockedParentResult<'_> =
        unsafe { NodeCleaner::locked_parent_generic::<BoxPolicy<u64>>(leaf_ptr) };

    let (lock, returned_parent) = match result {
        LockedParentResult::Locked(l, p) => (l, p),
        _ => panic!("Expected Locked result"),
    };

    assert_eq!(returned_parent, parent_ptr.cast::<u8>());

    // Parent should be locked
    let parent_ref: &TestInternode = unsafe { &*parent_ptr };
    assert!(parent_ref.version().is_locked());

    // Cleanup
    drop(lock);
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

    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
    leaf.set_parent(parent_ptr.cast());
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    unsafe { (*parent_ptr).set_child(0, leaf_ptr) };

    // Lock leaf
    let leaf_ref: &TestLeaf = unsafe { &*leaf_ptr.cast::<TestLeaf>() };
    let _leaf_lock: LockGuard<'_> = leaf_ref.version().lock();

    // Test: locked_parent should return parent (not grandparent)
    let result: LockedParentResult<'_> =
        unsafe { NodeCleaner::locked_parent_generic::<BoxPolicy<u64>>(leaf_ptr) };

    let (lock, returned_parent) = match result {
        LockedParentResult::Locked(l, p) => (l, p),
        _ => panic!("Expected Locked result"),
    };

    assert_eq!(returned_parent, parent_ptr.cast::<u8>());

    // Verify it's not a leaf
    let parent_version: &NodeVersion = unsafe { &*(returned_parent.cast::<NodeVersion>()) };
    assert!(!parent_version.is_leaf());

    // Cleanup
    drop(lock);
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
    let leaves: Vec<Box<TestLeaf>> = (0..4).map(|_| TestLeaf::new_boxed()).collect();
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
    inode.set_child(2, StdPtr::null_mut());

    // Test: shift_internode_down(kp=2)
    NodeCleaner::shift_internode_down_generic::<TestInternode>(&inode, 2);

    // Verify keys: [10, 30, _]
    assert_eq!(inode.ikey(0), 10);
    assert_eq!(inode.ikey(1), 30);

    // Verify children: [c0, c1, c3, _]
    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { inode.child_unguarded(0) }, c0);
    assert_eq!(unsafe { inode.child_unguarded(1) }, c1);
    assert_eq!(unsafe { inode.child_unguarded(2) }, c3);

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
    let leaves: Vec<Box<TestLeaf>> = (0..4).map(|_| TestLeaf::new_boxed()).collect();
    let ptrs: Vec<*mut u8> = leaves
        .into_iter()
        .map(|l| Box::into_raw(l) as *mut u8)
        .collect();

    let (c0, c1, c2, c3) = (ptrs[0], ptrs[1], ptrs[2], ptrs[3]);

    inode.set_child(0, c0);
    inode.set_child(1, c1);
    inode.set_child(2, c2);
    inode.set_child(3, c3);

    inode.set_child(3, StdPtr::null_mut());

    // Test: shift_internode_down(kp=3)
    NodeCleaner::shift_internode_down_generic::<TestInternode>(&inode, 3);

    // Verify keys: [10, 20, _]
    assert_eq!(inode.ikey(0), 10);
    assert_eq!(inode.ikey(1), 20);

    // Verify children: [c0, c1, c2, _]
    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { inode.child_unguarded(0) }, c0);
    assert_eq!(unsafe { inode.child_unguarded(1) }, c1);
    assert_eq!(unsafe { inode.child_unguarded(2) }, c2);

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
    let leaves: Vec<Box<TestLeaf>> = (0..3).map(|_| TestLeaf::new_boxed()).collect();
    let ptrs: Vec<*mut u8> = leaves
        .into_iter()
        .map(|l| Box::into_raw(l) as *mut u8)
        .collect();

    let (c0, c1, c2) = (ptrs[0], ptrs[1], ptrs[2]);

    inode.set_child(0, c0);
    inode.set_child(1, c1);
    inode.set_child(2, c2);

    inode.set_child(1, StdPtr::null_mut());

    // Test
    NodeCleaner::shift_internode_down_generic::<TestInternode>(&inode, 1);

    // Verify keys: [20, _]
    assert_eq!(inode.ikey(0), 20);

    // Verify children: [c0, c2, _]
    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { inode.child_unguarded(0) }, c0);
    assert_eq!(unsafe { inode.child_unguarded(1) }, c2);

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

    let leaf_a: Box<TestLeaf> = TestLeaf::new_boxed();
    let leaf_b: Box<TestLeaf> = TestLeaf::new_boxed();
    let leaf_c: Box<TestLeaf> = TestLeaf::new_boxed();

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
    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { (*a_ptr).safe_next_unguarded() }, c_ptr);
    assert_eq!(unsafe { (*c_ptr).prev_unguarded() }, a_ptr);

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

    let leaf_a: Box<TestLeaf> = TestLeaf::new_boxed();
    let leaf_b: Box<TestLeaf> = TestLeaf::new_boxed();

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
    // SAFETY: Single-threaded test context.
    assert!(unsafe { (*a_ptr).safe_next_unguarded().is_null() });

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
    tree.insert(&50_u64.to_be_bytes(), 50);
    tree.insert(&150_u64.to_be_bytes(), 150);

    // Remove key
    let removed = tree.remove(&150_u64.to_be_bytes());
    assert!(removed.is_ok());

    // Verify tree still works
    assert_val_eq!(tree.get(&50_u64.to_be_bytes()), Some(50));
    assert_eq!(tree.get(&150_u64.to_be_bytes()), None);
}

#[test]
fn test_remove_leaf_leftmost_not_removed() {
    // Leftmost leaf (prev == null) should NOT be removed even when empty

    let tree: TestTree = TestTree::new();

    tree.insert(&42_u64.to_be_bytes(), 42);
    let removed = tree.remove(&42_u64.to_be_bytes());
    assert!(removed.is_ok());

    // Tree is empty but root leaf should still exist
    assert_eq!(tree.len(), 0);

    // Can still insert
    tree.insert(&100_u64.to_be_bytes(), 100);
    assert_val_eq!(tree.get(&100_u64.to_be_bytes()), Some(100));
}

#[test]
fn test_redirect_via_sequential_removal() {
    // Test redirect by removing keys in order

    let tree: TestTree = TestTree::new();

    // Create a multi-leaf tree
    eprintln!("Inserting 50 keys...");
    for i in 0_u64..50 {
        tree.insert(&i.to_be_bytes(), i);
    }
    eprintln!("Inserted 50 keys, len = {}", tree.len());

    // Remove keys from the beginning (leftmost positions)
    eprintln!("Removing keys 0-24...");
    for i in 0_u64..25 {
        eprintln!("  Removing key {}", i);
        let _ = tree.remove(&i.to_be_bytes());
        eprintln!("  Removed key {}, len = {}", i, tree.len());
    }

    // Verify remaining keys are still accessible
    eprintln!("Verifying remaining keys...");
    for i in 25_u64..50 {
        eprintln!("  Getting key {}", i);
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
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
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove from various positions
    for i in (0_u64..100).step_by(4) {
        let _ = tree.remove(&i.to_be_bytes());
    }

    // Verify correctness
    for i in (0_u64..100).step_by(2) {
        if i % 4 == 0 {
            assert!(tree.get(&i.to_be_bytes()).is_none());
        } else {
            assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
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
        tree.insert(&i.to_be_bytes(), i);
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

    // Let reader thread start before we begin removing
    thread::sleep(std::time::Duration::from_millis(1));

    // Writer thread: remove keys
    for i in (0_u64..1000).step_by(2) {
        let _ = tree.remove(&i.to_be_bytes());
    }

    done.store(true, Ordering::Relaxed);
    let (found, not_found) = reader.join().unwrap();

    // Verify: no crashes, reasonable counts
    assert!(found > 0 || not_found > 0);

    // Final verification: odd keys should still exist
    for i in (1_u64..1000).step_by(2) {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
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
        tree.insert(&i.to_be_bytes(), i);
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
        let _ = h.join();
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
        tree.insert(&i.to_be_bytes(), i);
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
                let _ = tree_clone.remove(&i.to_be_bytes());
            }
        }));
    }

    for h in handles {
        let _ = h.join();
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
        tree.insert(&i.to_be_bytes(), i);
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
        let _ = tree.remove(&i.to_be_bytes());
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
    tree.insert(&42_u64.to_be_bytes(), 42);
    tree.insert(&100_u64.to_be_bytes(), 100);

    // Remove one key
    let _ = tree.remove(&42_u64.to_be_bytes());

    // Get should work (retry if needed internally)
    assert!(tree.get(&42_u64.to_be_bytes()).is_none());
    assert_val_eq!(tree.get(&100_u64.to_be_bytes()), Some(100));
}

// ============================================================================
//  Miri-Compatible Tests
// ============================================================================

#[test]
fn test_miri_remove_single_key() {
    let tree: TestTree = TestTree::new();

    tree.insert(&1_u64.to_be_bytes(), 1);
    assert_val_eq!(tree.remove(&1_u64.to_be_bytes()).unwrap(), Some(1));
    assert!(tree.get(&1_u64.to_be_bytes()).is_none());
}

#[test]
fn test_miri_remove_multiple_keys() {
    let tree: TestTree = TestTree::new();

    for i in 0_u64..10 {
        tree.insert(&i.to_be_bytes(), i);
    }

    for i in 0_u64..10 {
        assert_val_eq!(tree.remove(&i.to_be_bytes()).unwrap(), Some(i));
    }

    assert_eq!(tree.len(), 0);
}

#[test]
fn test_miri_parent_erased_helpers() {
    // Test helper functions under Miri using real allocations
    let parent_node: Box<TestInternode> = TestInternode::new(0);
    let parent_ptr: *mut u8 = Box::into_raw(parent_node).cast();

    let leaf: Box<TestLeaf> = TestLeaf::new_boxed();
    let leaf_ptr: *mut u8 = Box::into_raw(leaf).cast();

    // set_parent_erased
    unsafe {
        NodeCleaner::set_parent_erased::<BoxPolicy<u64>>(leaf_ptr, parent_ptr);
    }

    // get_parent_erased
    let got: *mut u8 = unsafe { NodeCleaner::get_parent_erased::<BoxPolicy<u64>>(leaf_ptr) };
    assert_eq!(got, parent_ptr);

    // Cleanup
    let _: Box<TestLeaf> = unsafe { Box::from_raw(leaf_ptr.cast::<TestLeaf>()) };
    let _: Box<TestInternode> = unsafe { Box::from_raw(parent_ptr.cast::<TestInternode>()) };
}

// ============================================================================
//  Coalesce Safety Tests
// ============================================================================

/// Test that process_coalesce doesn't cause infinite loops or panics.
#[test]
fn test_coalesce_safety_no_infinite_loop() {
    let tree: TestTree = TestTree::new();

    // Insert enough keys to create multiple leaves
    for i in 0_u64..50 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove all keys to create empty leaves
    for i in 0_u64..50 {
        let _ = tree.remove(&i.to_be_bytes());
    }

    // Process coalesce - this should complete without hanging
    let guard = tree.guard();
    let processed = tree.process_coalesce(&guard);

    // We should have processed some entries
    assert!(
        processed > 0,
        "Expected some coalesce entries to be processed"
    );

    // Tree should now be empty
    assert_eq!(tree.len(), 0);

    // Insert new keys - this should work correctly
    // (traversal through deleted nodes should follow B-links)
    for i in 100_u64..110 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Verify new keys are accessible
    for i in 100_u64..110 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
}

/// Test concurrent coalesce with reads doesn't hang.
#[test]
#[cfg(not(miri))]
fn test_coalesce_concurrent_with_reads() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let test_complete = Arc::new(AtomicBool::new(false));

    // Insert keys
    for i in 0_u64..100 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove some keys to create empty leaves
    for i in 0_u64..50 {
        let _ = tree.remove(&i.to_be_bytes());
    }

    let tree_reader = Arc::clone(&tree);
    let complete_reader = Arc::clone(&test_complete);

    // Reader thread - continuously reads
    let reader = thread::spawn(move || {
        while !complete_reader.load(Ordering::Acquire) {
            for i in 50_u64..100 {
                let _ = tree_reader.get(&i.to_be_bytes());
            }
        }
    });

    // Run coalesce in a thread with timeout to detect hangs
    let tree_coalesce = Arc::clone(&tree);
    let coalesce_result = thread::spawn(move || {
        let guard = tree_coalesce.guard();
        tree_coalesce.process_coalesce(&guard)
    });

    // Wait for coalesce with timeout
    let result = coalesce_result.join();

    // Signal reader to stop
    test_complete.store(true, Ordering::Release);

    // Wait for reader
    let _ = reader.join();

    // Verify coalesce completed successfully
    assert!(result.is_ok(), "Coalesce should not panic");

    // Verify remaining keys are still accessible
    for i in 50_u64..100 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
}

/// Test that insert works correctly when encountering deleted nodes.
#[test]
fn test_insert_through_deleted_nodes() {
    let tree: TestTree = TestTree::new();

    // Create a tree with keys that will span multiple leaves
    for i in 0_u64..30 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove middle keys to create empty leaves in the middle
    for i in 10_u64..20 {
        let _ = tree.remove(&i.to_be_bytes());
    }

    // Process coalesce to mark those leaves as deleted
    let guard = tree.guard();
    let _ = tree.process_coalesce(&guard);

    // Insert new keys that might traverse through deleted nodes
    for i in 10_u64..20 {
        tree.insert(&i.to_be_bytes(), i * 10);
    }

    // Verify all keys
    for i in 0_u64..10 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
    for i in 10_u64..20 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i * 10));
    }
    for i in 20_u64..30 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
}

/// Test multiple coalesce cycles don't accumulate issues.
///
/// This verifies that parent cleanup works correctly and doesn't
/// leave orphaned pointers or cause memory issues over time.
#[test]
fn test_coalesce_multiple_cycles() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    for cycle in 0..5 {
        let base: u64 = cycle * 100;

        // Insert keys
        for i in 0_u64..50 {
            tree.insert(&(base + i).to_be_bytes(), base + i);
        }

        // Remove all keys
        for i in 0_u64..50 {
            let _ = tree.remove(&(base + i).to_be_bytes());
        }

        // Process coalesce
        let processed = tree.process_coalesce(&guard);
        assert!(processed > 0, "Cycle {cycle}: should process some entries");

        // Verify tree is empty after each cycle
        assert_eq!(tree.len(), 0, "Cycle {cycle}: tree should be empty");

        // Verify pending coalesce is zero after processing
        assert_eq!(
            tree.pending_coalesce(),
            0,
            "Cycle {cycle}: pending coalesce should be 0"
        );
    }
}

/// Test that leftmost leaf is preserved during coalesce.
///
/// The leftmost leaf cannot be removed because B-link traversal
/// requires it as an anchor point. This test verifies the leftmost
/// check works correctly.
#[test]
fn test_coalesce_preserves_leftmost_leaf() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Insert and remove keys - the leftmost leaf should remain
    for i in 0_u64..10 {
        tree.insert(&i.to_be_bytes(), i);
    }

    for i in 0_u64..10 {
        let _ = tree.remove(&i.to_be_bytes());
    }

    // Process coalesce
    let _ = tree.process_coalesce(&guard);

    // Even after coalesce, we should be able to insert new keys
    // (the leftmost leaf is still there as a valid root)
    for i in 0_u64..10 {
        tree.insert(&i.to_be_bytes(), i * 2);
    }

    // Verify keys
    for i in 0_u64..10 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i * 2));
    }
}

/// Test coalesce with interleaved operations.
///
/// This simulates a more realistic workload where inserts, removes,
/// and coalescing happen in an interleaved fashion.
#[test]
fn test_coalesce_interleaved_operations() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Phase 1: Insert initial keys
    for i in 0_u64..100 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Phase 2: Remove some, coalesce, insert new
    for i in 0_u64..25 {
        let _ = tree.remove(&i.to_be_bytes());
    }
    let _ = tree.process_coalesce(&guard);

    // Insert in the "gap"
    for i in 0_u64..25 {
        tree.insert(&i.to_be_bytes(), i + 1000);
    }

    // Phase 3: Remove different keys, coalesce again
    for i in 50_u64..75 {
        let _ = tree.remove(&i.to_be_bytes());
    }
    let _ = tree.process_coalesce(&guard);

    // Insert again
    for i in 50_u64..75 {
        tree.insert(&i.to_be_bytes(), i + 2000);
    }

    // Verify all keys have correct values
    for i in 0_u64..25 {
        assert_val_eq!(
            tree.get(&i.to_be_bytes()),
            Some(i + 1000),
            "Key {i} should have value {}",
            i + 1000
        );
    }
    for i in 25_u64..50 {
        assert_val_eq!(
            tree.get(&i.to_be_bytes()),
            Some(i),
            "Key {i} should have original value"
        );
    }
    for i in 50_u64..75 {
        assert_val_eq!(
            tree.get(&i.to_be_bytes()),
            Some(i + 2000),
            "Key {i} should have value {}",
            i + 2000
        );
    }
    for i in 75_u64..100 {
        assert_val_eq!(
            tree.get(&i.to_be_bytes()),
            Some(i),
            "Key {i} should have original value"
        );
    }
}

/// Test coalesce batch processing.
///
/// Verifies that process_coalesce_batch correctly limits the
/// number of entries processed.
#[test]
#[expect(clippy::panic)]
fn test_coalesce_batch_processing() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Insert enough keys to create many empty leaves
    for i in 0_u64..200 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove all to queue many entries
    for i in 0_u64..200 {
        let _ = tree.remove(&i.to_be_bytes());
    }

    let initial_pending = tree.pending_coalesce();
    assert!(initial_pending > 0, "Should have pending coalesce entries");

    // Process in batches
    let mut total_processed: usize = 0;
    let batch_limit: usize = 5;

    while tree.pending_coalesce() > 0 {
        let processed = tree.process_coalesce_batch(&guard, batch_limit);
        total_processed += processed;

        // Each batch should process at most the limit
        // (could be less if entries are re-queued)
        assert!(
            processed <= batch_limit,
            "Batch processed {processed}, expected <= {batch_limit}"
        );

        // Prevent infinite loop in test
        if total_processed > initial_pending * 3 {
            panic!("Too many iterations, possible infinite loop");
        }
    }

    assert_eq!(
        tree.pending_coalesce(),
        0,
        "All entries should be processed"
    );
}

/// Test concurrent insert/remove with coalesce.
///
/// Stress tests the synchronization between normal operations
/// and background coalescing.
#[test]
#[cfg(not(miri))]
fn test_coalesce_concurrent_with_writers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Pre-populate
    for i in 0_u64..100 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Writer thread - continuously insert and remove
    let tree_writer = Arc::clone(&tree);
    let stop_writer = Arc::clone(&stop_flag);
    let writer = thread::spawn(move || {
        let mut counter: u64 = 1000;
        while !stop_writer.load(Ordering::Acquire) {
            // Insert
            let key = counter;
            let _ = tree_writer.insert(&key.to_be_bytes(), key);

            // Remove a random-ish key
            let remove_key = (counter % 200) + 100;
            tree_writer.remove(&remove_key.to_be_bytes()).ok();

            counter += 1;
            if counter > 10000 {
                counter = 1000;
            }
        }
    });

    // Coalesce thread
    let tree_coalesce = Arc::clone(&tree);
    let stop_coalesce = Arc::clone(&stop_flag);
    let coalescer = thread::spawn(move || {
        let mut cycles: usize = 0;
        while !stop_coalesce.load(Ordering::Acquire) {
            let guard = tree_coalesce.guard();
            let _ = tree_coalesce.process_coalesce(&guard);
            cycles += 1;

            // Small yield
            thread::yield_now();
        }
        cycles
    });

    // Let it run for a bit
    thread::sleep(Duration::from_millis(100));

    // Stop threads
    stop_flag.store(true, Ordering::Release);

    let writer_result = writer.join();
    let coalesce_result = coalescer.join();

    assert!(writer_result.is_ok(), "Writer should not panic");
    assert!(coalesce_result.is_ok(), "Coalescer should not panic");

    let cycles = coalesce_result.unwrap();
    assert!(cycles > 0, "Should have run some coalesce cycles");
}

/// Test that empty tree coalesce is a no-op.
#[test]
fn test_coalesce_empty_tree() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Empty tree should have nothing to coalesce
    assert_eq!(tree.pending_coalesce(), 0);
    let processed = tree.process_coalesce(&guard);
    assert_eq!(processed, 0, "Empty tree should process 0 entries");
}

/// Test coalesce with range scans.
///
/// Verifies that range iteration works correctly after coalescing.
#[test]
fn test_coalesce_with_range_scan() {
    use crate::RangeBound;

    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Insert keys with gaps
    for i in (0_u64..100).step_by(2) {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Remove some keys
    for i in (20_u64..40).step_by(2) {
        let _ = tree.remove(&i.to_be_bytes());
    }

    // Coalesce
    let _ = tree.process_coalesce(&guard);

    // Range scan should work correctly
    let mut found: Vec<u64> = Vec::new();
    tree.scan(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |k: &[u8], v: ValuePtr<u64>| {
            let key = u64::from_be_bytes(k.try_into().unwrap());
            found.push(key);
            assert_eq!(*v, key, "Value should match key");
            true
        },
        &guard,
    );

    // Verify we got the expected keys
    let expected: Vec<u64> = (0_u64..100)
        .step_by(2)
        .filter(|&i| !(20..40).contains(&i))
        .collect();
    assert_eq!(found, expected, "Range scan should return correct keys");
}

/// Stress test: rapid insert-remove-coalesce cycles.
#[test]
fn test_coalesce_stress_rapid_cycles() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    for cycle in 0_u64..20 {
        // Insert
        for i in 0_u64..20 {
            tree.insert(&(cycle * 100 + i).to_be_bytes(), i);
        }

        // Remove
        for i in 0_u64..20 {
            let _ = tree.remove(&(cycle * 100 + i).to_be_bytes());
        }

        // Coalesce immediately
        tree.process_coalesce(&guard);
    }

    // Tree should be empty and healthy
    assert_eq!(tree.len(), 0);
    assert_eq!(tree.pending_coalesce(), 0);

    // Should still work for new insertions
    for i in 0_u64..50 {
        tree.insert(&i.to_be_bytes(), i);
    }

    for i in 0_u64..50 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
}

// ============================================================================
//  gc_layer Tests (Sublayer Cleanup)
// ============================================================================

/// Test basic gc_layer: create sublayer, remove all keys, verify cleanup.
#[test]
fn test_gc_layer_basic_sublayer_cleanup() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create a sublayer by inserting keys with shared 8-byte prefix
    // Keys: "prefix00" + "A", "prefix00" + "B" share the first 8 bytes
    let key1 = b"prefix00A";
    let key2 = b"prefix00B";

    tree.insert(key1, 1);
    tree.insert(key2, 2);
    assert_eq!(tree.len(), 2);

    // Verify both keys exist
    assert_val_eq!(tree.get(key1), Some(1));
    assert_val_eq!(tree.get(key2), Some(2));

    // Remove all keys from the sublayer
    assert_val_eq!(tree.remove(key1).unwrap(), Some(1));
    assert_val_eq!(tree.remove(key2).unwrap(), Some(2));
    assert_eq!(tree.len(), 0);

    // Process coalesce - should trigger gc_layer for the empty sublayer
    let processed = tree.process_coalesce(&guard);
    assert!(processed > 0, "Should process the empty sublayer");

    // Tree should still be functional - insert new keys
    tree.insert(key1, 10);
    assert_val_eq!(tree.get(key1), Some(10));
}

/// Test gc_layer with multiple sublayers.
#[test]
fn test_gc_layer_multiple_sublayers() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create multiple sublayers with different prefixes
    let prefixes = [b"aaaaaaaa", b"bbbbbbbb", b"cccccccc"];
    let suffixes = [b"1", b"2", b"3"];

    // Insert keys into each sublayer
    for prefix in &prefixes {
        for (i, suffix) in suffixes.iter().enumerate() {
            let mut key = Vec::with_capacity(9);
            key.extend_from_slice(*prefix);
            key.extend_from_slice(*suffix);
            tree.insert(&key, i as u64);
        }
    }
    assert_eq!(tree.len(), 9);

    // Remove all keys from sublayer "aaaaaaaa"
    for suffix in &suffixes {
        let mut key = Vec::with_capacity(9);
        key.extend_from_slice(b"aaaaaaaa");
        key.extend_from_slice(*suffix);
        let _ = tree.remove(&key);
    }
    assert_eq!(tree.len(), 6);

    // Process coalesce - should gc the empty sublayer
    tree.process_coalesce(&guard);

    // Other sublayers should still work
    assert_val_eq!(tree.get(b"bbbbbbbb1"), Some(0));
    assert_val_eq!(tree.get(b"cccccccc2"), Some(1));

    // Can reuse the cleaned-up prefix
    tree.insert(b"aaaaaaaaX", 99);
    assert_val_eq!(tree.get(b"aaaaaaaaX"), Some(99));
}

/// Test gc_layer with deep layer chains (multiple levels of sublayers).
#[test]
fn test_gc_layer_deep_chain() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create a chain of sublayers:
    // Level 0: 8 bytes "level000"
    // Level 1: 16 bytes "level000level001"
    // Level 2: 24 bytes "level000level001level002"
    let key_l2_a = b"level000level001level002A";
    let key_l2_b = b"level000level001level002B";

    tree.insert(key_l2_a, 1);
    tree.insert(key_l2_b, 2);
    assert_eq!(tree.len(), 2);

    // Remove one key - sublayer should NOT be gc'd yet
    let _ = tree.remove(key_l2_a);
    assert_eq!(tree.len(), 1);
    tree.process_coalesce(&guard);

    // Remaining key should still exist
    assert_val_eq!(tree.get(key_l2_b), Some(2));

    // Remove the last key - now sublayer should be gc'd
    let _ = tree.remove(key_l2_b);
    assert_eq!(tree.len(), 0);
    tree.process_coalesce(&guard);

    // Tree should be empty but functional
    tree.insert(key_l2_a, 100);
    assert_val_eq!(tree.get(key_l2_a), Some(100));
}

/// Test gc_layer doesn't affect sibling sublayers.
#[test]
fn test_gc_layer_preserves_siblings() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create two sublayers under the same parent leaf
    // Parent has slots for both "prefix_A" and "prefix_B" layer pointers
    let key_a1 = b"prefix_Akey1";
    let key_a2 = b"prefix_Akey2";
    let key_b1 = b"prefix_Bkey1";
    let key_b2 = b"prefix_Bkey2";

    tree.insert(key_a1, 1);
    tree.insert(key_a2, 2);
    tree.insert(key_b1, 3);
    tree.insert(key_b2, 4);
    assert_eq!(tree.len(), 4);

    // Remove all keys from sublayer A
    let _ = tree.remove(key_a1);
    let _ = tree.remove(key_a2);
    assert_eq!(tree.len(), 2);

    // Process coalesce - should gc sublayer A but not B
    tree.process_coalesce(&guard);

    // Sublayer B should be unaffected
    assert_val_eq!(tree.get(key_b1), Some(3));
    assert_val_eq!(tree.get(key_b2), Some(4));

    // Can insert new keys into the cleaned-up sublayer A
    tree.insert(key_a1, 10);
    assert_val_eq!(tree.get(key_a1), Some(10));
}

/// Test gc_layer with concurrent reads.
#[test]
#[cfg_attr(miri, ignore)] // Gets stuck in Miri due to thread scheduling complexity
fn test_gc_layer_concurrent_reads() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let done = Arc::new(AtomicBool::new(false));

    // Create sublayer
    let key1 = b"sublayer0key1xxx";
    let key2 = b"sublayer0key2xxx";
    tree.insert(key1, 1);
    tree.insert(key2, 2);

    // Also insert some non-sublayer keys for readers to find
    for i in 0_u64..10 {
        tree.insert(&i.to_be_bytes(), i);
    }

    let tree_reader = Arc::clone(&tree);
    let done_reader = Arc::clone(&done);

    // Reader thread continuously reads
    let reader = thread::spawn(move || {
        while !done_reader.load(Ordering::Acquire) {
            for i in 0_u64..10 {
                let _ = tree_reader.get(&i.to_be_bytes());
            }
            // Also try to read from the sublayer (may or may not exist)
            let _ = tree_reader.get(b"sublayer0key1xxx");
        }
    });

    // Main thread: remove sublayer keys and gc
    let _ = tree.remove(key1);
    let _ = tree.remove(key2);

    let guard = tree.guard();
    tree.process_coalesce(&guard);

    // Signal reader to stop
    done.store(true, Ordering::Release);
    #[expect(clippy::expect_used, reason = "test code - panicking is appropriate")]
    reader.join().expect("Reader thread panicked");

    // Tree should be consistent
    for i in 0_u64..10 {
        assert_val_eq!(tree.get(&i.to_be_bytes()), Some(i));
    }
}

/// Test that gc_layer handles the case where parent slot changed concurrently.
///
/// This is hard to test deterministically, but we can at least verify
/// the code path doesn't crash when the slot has changed.
#[test]
fn test_gc_layer_slot_changed() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create sublayer
    let key1 = b"changedXXkey1";
    let key2 = b"changedXXkey2";
    tree.insert(key1, 1);
    tree.insert(key2, 2);

    // Remove one key
    let _ = tree.remove(key1);

    // Remove second key - sublayer becomes empty
    let _ = tree.remove(key2);

    // Before coalesce runs, insert a new key with the same prefix
    // This might reuse the sublayer or create a new one
    tree.insert(b"changedXXnewkey", 99);

    // Coalesce should handle this gracefully
    // (the old sublayer entry may be stale)
    tree.process_coalesce(&guard);

    // New key should be accessible
    assert_val_eq!(tree.get(b"changedXXnewkey"), Some(99));
}

/// Stress test: rapid sublayer create-remove-gc cycles.
#[test]
fn test_gc_layer_stress() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    for cycle in 0_u32..50 {
        // Create unique sublayer for this cycle
        let prefix = format!("cyc{cycle:05}");
        let key1 = format!("{prefix}key1");
        let key2 = format!("{prefix}key2");
        let key3 = format!("{prefix}key3");

        // Insert
        tree.insert(key1.as_bytes(), cycle as u64);
        tree.insert(key2.as_bytes(), cycle as u64 + 1);
        tree.insert(key3.as_bytes(), cycle as u64 + 2);

        // Remove all
        let _ = tree.remove(key1.as_bytes());
        let _ = tree.remove(key2.as_bytes());
        let _ = tree.remove(key3.as_bytes());

        // Coalesce every 5 cycles
        if cycle % 5 == 4 {
            tree.process_coalesce(&guard);
        }
    }

    // Final coalesce
    tree.process_coalesce(&guard);

    // Tree should be empty and healthy
    assert_eq!(tree.len(), 0);

    // Should work for new insertions
    tree.insert(b"finaltest!", 12345);
    assert_val_eq!(tree.get(b"finaltest!"), Some(12345));
}

// ============================================================================
//  gc_layer: bounded spin and re-queue path tests
// ============================================================================

/// Test that gc_layer re-queues entries when the parent leaf is locked by a
/// concurrent writer. The coalesce must eventually succeed once the writer
/// releases the lock.
#[test]
#[cfg_attr(miri, ignore)]
fn test_gc_layer_requeue_under_parent_contention() {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());

    // Create a sublayer: keys share "parentXX" prefix, forcing a trie layer.
    let key_a = b"parentXXchild_a!";
    let key_b = b"parentXXchild_b!";
    tree.insert(key_a, 1);
    tree.insert(key_b, 2);

    // Also insert siblings in the same parent leaf under a different ikey
    // so the parent leaf stays interesting.
    let sibling = b"siblingZ";
    tree.insert(sibling, 99);

    // Remove both sublayer keys to make the sublayer empty.
    let _ = tree.remove(key_a);
    let _ = tree.remove(key_b);

    // Now the coalesce queue has an entry that will trigger gc_layer.
    assert!(tree.pending_coalesce() > 0);

    let barrier = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));
    let writer_ops = Arc::new(AtomicUsize::new(0));

    // Spawn a writer that hammers the parent leaf's sibling key, creating
    // lock contention on the parent leaf that gc_layer needs to acquire.
    let tree_w = Arc::clone(&tree);
    let barrier_w = Arc::clone(&barrier);
    let done_w = Arc::clone(&done);
    let ops_w = Arc::clone(&writer_ops);

    let writer = thread::spawn(move || {
        barrier_w.wait();
        let mut i: u64 = 1000;
        while !done_w.load(Ordering::Acquire) {
            tree_w.insert(sibling, i);
            i += 1;
            ops_w.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Sync so the writer is actively contending before we coalesce.
    barrier.wait();

    // Let the writer build up some contention.
    thread::sleep(Duration::from_millis(1));

    // Process coalesce. gc_layer may re-queue on first attempt(s) due to the
    // writer holding the parent lock, but must eventually succeed.
    let guard = tree.guard();
    let mut total_processed = 0;
    for _ in 0..100 {
        total_processed += tree.process_coalesce(&guard);
        if tree.pending_coalesce() == 0 {
            break;
        }
        thread::yield_now();
    }

    done.store(true, Ordering::Release);
    #[expect(clippy::expect_used, reason = "test code")]
    writer.join().expect("writer panicked");

    assert!(
        total_processed > 0,
        "coalesce should have processed the gc_layer entry"
    );

    // Sibling must survive the sublayer cleanup.
    assert!(tree.get(sibling).is_some());

    // Tree is still functional after gc_layer.
    tree.insert(key_a, 42);
    assert_val_eq!(tree.get(key_a), Some(42));
}

/// Test that gc_layer correctly preserves and re-queues the full layer
/// context chain when it fails to acquire the parent lock.
#[test]
fn test_gc_layer_requeue_preserves_context_chain() {
    let tree: TestTree = TestTree::new();
    let guard = tree.guard();

    // Create a deep chain: 3 layers of sublayers (24+ byte keys).
    // When the deepest sublayer is emptied, gc_layer gets a context chain
    // of length >= 2.
    let key1 = b"layer000layer001deep_key_A_here";
    let key2 = b"layer000layer001deep_key_B_here";

    tree.insert(key1, 1);
    tree.insert(key2, 2);
    assert_eq!(tree.len(), 2);

    // Remove both keys, emptying the deepest sublayer.
    let _ = tree.remove(key1);
    let _ = tree.remove(key2);
    assert_eq!(tree.len(), 0);

    // Process coalesce one entry at a time to observe re-queue behavior.
    // Even if gc_layer re-queues internally, repeated processing must
    // eventually drain the queue.
    let mut rounds = 0;
    while tree.pending_coalesce() > 0 && rounds < 50 {
        tree.process_coalesce_batch(&guard, 1);
        rounds += 1;
    }

    assert_eq!(
        tree.pending_coalesce(),
        0,
        "coalesce queue should be drained after sufficient rounds"
    );

    // Tree must remain functional.
    tree.insert(key1, 100);
    assert_val_eq!(tree.get(key1), Some(100));
}

/// Test that entries are dropped (not re-queued forever) after exceeding the
/// maximum re-queue count.
#[test]
#[cfg_attr(miri, ignore)]
fn test_gc_layer_max_requeue_drop() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());

    // Create sublayer.
    let key1 = b"requeueXchild01!";
    let key2 = b"requeueXchild02!";
    tree.insert(key1, 1);
    tree.insert(key2, 2);

    // Keep a sibling in the parent to prevent the parent from becoming empty.
    let anchor = b"requeueY";
    tree.insert(anchor, 999);

    let _ = tree.remove(key1);
    let _ = tree.remove(key2);

    assert!(tree.pending_coalesce() > 0);

    let done = Arc::new(AtomicBool::new(false));
    let tree_w = Arc::clone(&tree);
    let done_w = Arc::clone(&done);

    // Writer that continuously locks the parent leaf via insert on the
    // anchor key, making gc_layer's try_lock fail repeatedly.
    let writer = thread::spawn(move || {
        let mut i: u64 = 0;
        while !done_w.load(Ordering::Acquire) {
            tree_w.insert(anchor, i);
            i += 1;
        }
    });

    thread::sleep(Duration::from_millis(2));

    // Process coalesce many times. Even if gc_layer keeps re-queuing
    // due to contention, the MAX_REQUEUE_COUNT limit ensures entries
    // are eventually dropped.
    let guard = tree.guard();
    for _ in 0..200 {
        tree.process_coalesce(&guard);
    }

    done.store(true, Ordering::Release);
    #[expect(clippy::expect_used, reason = "test code")]
    writer.join().expect("writer panicked");

    // Queue must be empty: either gc_layer succeeded or the entry was
    // dropped after MAX_REQUEUE_COUNT.
    assert_eq!(
        tree.pending_coalesce(),
        0,
        "queue should be drained (either processed or dropped)"
    );

    // Anchor key must survive.
    assert!(tree.get(anchor).is_some());
}

/// Verify that concurrent gc_layer calls do not corrupt the tree.
/// Each thread operates on its own distinct prefix to avoid the known
/// shared-parent race.
#[test]
#[cfg_attr(miri, ignore)]
fn test_gc_layer_concurrent_insert_remove_cycle() {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());
    let barrier = Arc::new(Barrier::new(5));
    let done = Arc::new(AtomicBool::new(false));

    // 4 worker threads, each with a DISTINCT 8-byte prefix (no shared parent).
    let handles: Vec<_> = (0..4)
        .map(|tid: u32| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let done = Arc::clone(&done);

            thread::spawn(move || {
                let prefix = format!("pfx{tid:05}");
                let suffix_a = format!("keyA!!!!");
                let suffix_b = format!("keyB!!!!");
                let key_a: Vec<u8> = [prefix.as_bytes(), suffix_a.as_bytes()].concat();
                let key_b: Vec<u8> = [prefix.as_bytes(), suffix_b.as_bytes()].concat();

                barrier.wait();

                let mut cycle: u64 = 0;
                while !done.load(Ordering::Acquire) {
                    tree.insert(&key_a, cycle);
                    tree.insert(&key_b, cycle + 1);

                    let _ = tree.remove(&key_a);
                    let _ = tree.remove(&key_b);

                    let guard = tree.guard();
                    tree.process_coalesce(&guard);

                    cycle += 2;
                }
            })
        })
        .collect();

    barrier.wait();
    thread::sleep(std::time::Duration::from_millis(100));
    done.store(true, Ordering::Release);

    for h in handles {
        #[expect(clippy::expect_used, reason = "test code")]
        h.join().expect("worker panicked");
    }

    let guard = tree.guard();
    tree.process_coalesce(&guard);

    // Tree must be consistent after concurrent gc_layer cycles.
    tree.insert(b"pfx00000verify!", 1);
    assert_val_eq!(tree.get(b"pfx00000verify!"), Some(1));
}

/// Verify that gc_layer does not prevent concurrent progress on the same
/// parent leaf. A reader thread must be able to complete reads while gc_layer
/// is actively running on a sibling sublayer.
#[test]
#[cfg_attr(miri, ignore)]
fn test_gc_layer_does_not_block_concurrent_reads() {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    let tree: Arc<TestTree> = Arc::new(TestTree::new());

    // Build a sublayer with keys that gc_layer will target.
    let drain_a = b"blockXXXchild_a!";
    let drain_b = b"blockXXXchild_b!";
    tree.insert(drain_a, 1);
    tree.insert(drain_b, 2);

    // Sibling under same parent, different ikey. Reader reads this.
    let probe = b"blockYYY";
    tree.insert(probe, 100);

    let barrier = Arc::new(Barrier::new(3));
    let done = Arc::new(AtomicBool::new(false));
    let reader_completed = Arc::new(AtomicU64::new(0));

    // Reader: continuously reads the probe key.
    let tree_r = Arc::clone(&tree);
    let barrier_r = Arc::clone(&barrier);
    let done_r = Arc::clone(&done);
    let completed = Arc::clone(&reader_completed);

    let reader = thread::spawn(move || {
        barrier_r.wait();
        while !done_r.load(Ordering::Acquire) {
            let _ = tree_r.get(probe);
            completed.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Coalescer: repeatedly creates and destroys sublayers, forcing gc_layer.
    let tree_c = Arc::clone(&tree);
    let barrier_c = Arc::clone(&barrier);
    let done_c = Arc::clone(&done);

    let coalescer = thread::spawn(move || {
        barrier_c.wait();
        let mut cycle: u64 = 0;
        while !done_c.load(Ordering::Acquire) {
            let _ = tree_c.remove(drain_a);
            let _ = tree_c.remove(drain_b);

            let guard = tree_c.guard();
            tree_c.process_coalesce(&guard);

            tree_c.insert(drain_a, cycle);
            tree_c.insert(drain_b, cycle + 1);
            cycle += 2;
        }
    });

    barrier.wait();
    thread::sleep(Duration::from_millis(100));
    done.store(true, Ordering::Release);

    #[expect(clippy::expect_used, reason = "test code")]
    reader.join().expect("reader panicked");
    #[expect(clippy::expect_used, reason = "test code")]
    coalescer.join().expect("coalescer panicked");

    let reads = reader_completed.load(Ordering::Relaxed);

    // The reader must have completed a meaningful number of reads.
    // If gc_layer were blocking for extended periods (e.g. yielding under
    // lock), the reader would be starved. Even in debug mode, 100ms should
    // allow thousands of reads.
    assert!(
        reads > 100,
        "reader only completed {reads} reads in 100ms, \
         gc_layer may be holding locks too long"
    );

    // Probe key must still be readable.
    assert_val_eq!(tree.get(probe), Some(100));
}
