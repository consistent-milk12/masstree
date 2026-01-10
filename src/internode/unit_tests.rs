#![expect(clippy::indexing_slicing)]

use super::{InternodeNode, Ordering};
use std::ptr as StdPtr;

#[test]
fn test_new_internode() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    assert!(!node.version().is_leaf());
    assert!(!node.version().is_root());
    assert_eq!(node.nkeys(), 0);
    assert_eq!(node.height(), 0);
    assert!(node.is_empty());
    assert!(!node.is_full());
    assert!(node.children_are_leaves());
    assert!(unsafe { node.parent_unguarded() }.is_null());
}

#[test]
fn test_new_root() {
    let node: Box<InternodeNode> = InternodeNode::new_root(1);

    assert!(!node.version().is_leaf());
    assert!(node.version().is_root());
    assert_eq!(node.height(), 1);
    assert!(!node.children_are_leaves());
}

#[test]
fn test_key_accessors() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    node.set_ikey(0, 0x1000_0000_0000_0000);
    node.set_ikey(1, 0x2000_0000_0000_0000);
    node.set_ikey(2, 0x3000_0000_0000_0000);
    node.set_nkeys(3);

    assert_eq!(node.ikey(0), 0x1000_0000_0000_0000);
    assert_eq!(node.ikey(1), 0x2000_0000_0000_0000);
    assert_eq!(node.ikey(2), 0x3000_0000_0000_0000);
    assert_eq!(node.size(), 3);
}

#[test]
fn test_child_accessors() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    let fake_child0: *mut u8 = StdPtr::without_provenance_mut(0x1000);
    let fake_child1: *mut u8 = StdPtr::without_provenance_mut(0x2000);
    let fake_child2: *mut u8 = StdPtr::without_provenance_mut(0x3000);

    node.set_child(0, fake_child0);
    node.set_child(1, fake_child1);
    node.set_child(2, fake_child2);

    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { node.child_unguarded(0) }, fake_child0);
    assert_eq!(unsafe { node.child_unguarded(1) }, fake_child1);
    assert_eq!(unsafe { node.child_unguarded(2) }, fake_child2);
}

#[test]
fn test_assign() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    let left_child: *mut u8 = StdPtr::without_provenance_mut(0x1000);
    let right_child: *mut u8 = StdPtr::without_provenance_mut(0x2000);

    // Set left child first
    node.set_child(0, left_child);

    // Assign key and right child
    node.assign(0, 0xABCD_0000_0000_0000, right_child);
    node.set_nkeys(1);

    assert_eq!(node.ikey(0), 0xABCD_0000_0000_0000);
    // SAFETY: Single-threaded test context.
    assert_eq!(unsafe { node.child_unguarded(0) }, left_child);
    assert_eq!(unsafe { node.child_unguarded(1) }, right_child);
    assert_eq!(node.size(), 1);
}

#[test]
fn test_inc_nkeys() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    assert_eq!(node.nkeys(), 0);

    node.inc_nkeys();
    assert_eq!(node.nkeys(), 1);

    node.inc_nkeys();
    assert_eq!(node.nkeys(), 2);
}

#[test]
fn test_is_full() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    assert!(!node.is_full());

    node.set_nkeys(15);
    assert!(node.is_full());
}

#[test]
fn test_parent_accessors() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let mut parent: Box<InternodeNode> = InternodeNode::new(1);

    let parent_ptr: *mut InternodeNode = parent.as_mut() as *mut InternodeNode;

    // set_parent takes *mut u8, so cast the pointer
    node.set_parent(parent_ptr.cast::<u8>());
    assert_eq!(unsafe { node.parent_unguarded() }, parent_ptr.cast::<u8>());
}

#[test]
fn test_compare_key() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    node.set_ikey(0, 0x5000_0000_0000_0000);
    node.set_nkeys(1);

    assert_eq!(node.compare_key(0x3000_0000_0000_0000, 0), Ordering::Less);
    assert_eq!(node.compare_key(0x5000_0000_0000_0000, 0), Ordering::Equal);
    assert_eq!(
        node.compare_key(0x7000_0000_0000_0000, 0),
        Ordering::Greater
    );
}

#[test]
fn test_invariants_valid() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Set up correctly sorted keys
    node.set_ikey(0, 0x1000_0000_0000_0000);
    node.set_ikey(1, 0x2000_0000_0000_0000);
    node.set_ikey(2, 0x3000_0000_0000_0000);
    node.set_nkeys(3);

    // Should not panic
    node.debug_assert_invariants();
}

#[test]
#[should_panic(expected = "keys not in ascending order")]
#[cfg(debug_assertions)]
fn test_invariant_unsorted_keys() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Set up unsorted keys
    node.set_ikey(0, 0x3000_0000_0000_0000);
    node.set_ikey(1, 0x1000_0000_0000_0000); // Wrong order!
    node.set_nkeys(2);

    node.debug_assert_invariants(); // Should panic
}

// ========================================================================
//  find_insert_position tests (binary search verification)
// ========================================================================

#[test]
fn test_find_insert_position_empty() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    // Empty node: any key goes at position 0
    assert_eq!(node.find_insert_position(0x1000), 0);
    assert_eq!(node.find_insert_position(0), 0);
    assert_eq!(node.find_insert_position(u64::MAX), 0);
}

#[test]
fn test_find_insert_position_single_key() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    node.set_ikey(0, 100);
    node.set_nkeys(1);

    // Key < existing: goes before
    assert_eq!(node.find_insert_position(50), 0);
    // Key == existing: goes at same position
    assert_eq!(node.find_insert_position(100), 0);
    // Key > existing: goes after
    assert_eq!(node.find_insert_position(150), 1);
}

#[test]
fn test_find_insert_position_multiple_keys() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Set up keys: 10, 20, 30, 40, 50
    node.set_ikey(0, 10);
    node.set_ikey(1, 20);
    node.set_ikey(2, 30);
    node.set_ikey(3, 40);
    node.set_ikey(4, 50);
    node.set_nkeys(5);

    // Before all
    assert_eq!(node.find_insert_position(5), 0);
    // Equal to first
    assert_eq!(node.find_insert_position(10), 0);
    // Between first and second
    assert_eq!(node.find_insert_position(15), 1);
    // Equal to middle
    assert_eq!(node.find_insert_position(30), 2);
    // Between 30 and 40
    assert_eq!(node.find_insert_position(35), 3);
    // Equal to last
    assert_eq!(node.find_insert_position(50), 4);
    // After all
    assert_eq!(node.find_insert_position(100), 5);
}

#[test]
fn test_find_insert_position_full_node() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Fill with keys 10, 20, 30, ..., 150
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 10);
    }
    node.set_nkeys(15);

    // Verify binary search works for all positions
    assert_eq!(node.find_insert_position(5), 0); // Before first
    assert_eq!(node.find_insert_position(10), 0); // Equal to first
    assert_eq!(node.find_insert_position(75), 7); // Mid-range
    assert_eq!(node.find_insert_position(80), 7); // Equal to key[7]
    assert_eq!(node.find_insert_position(145), 14); // Between 140 and 150
    assert_eq!(node.find_insert_position(150), 14); // Equal to last
    assert_eq!(node.find_insert_position(200), 15); // After all
}

// ========================================================================
//  Split edge case tests
// ========================================================================

#[test]
fn test_split_insert_at_position_0() {
    // Test splitting when the new key goes at position 0 (smallest)
    // Use height=0 so split_into treats children as leaves (doesn't dereference them)
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode> = InternodeNode::new(0);

    // Fill the node with keys 20, 30, 40, ..., 160 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 2) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode = new_right.as_mut();
    let new_child: *mut u8 = StdPtr::without_provenance_mut(0xABCD);

    // Insert key 10 at position 0 (smallest)
    let (popup_key, insert_went_left) =
        node.split_into(&mut new_right, new_right_ptr, 0, 10, new_child);

    // Insert at position 0 < mid(8), so it goes left
    assert!(insert_went_left, "Insert at position 0 should go left");

    // Verify popup key is reasonable (should be one of the keys around mid)
    assert!(popup_key > 0, "Popup key should be non-zero");
}

#[test]
fn test_split_insert_at_width() {
    // Test splitting when the new key goes at position WIDTH (largest)
    // Use height=0 so split_into treats children as leaves (doesn't dereference them)
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode> = InternodeNode::new(0);

    // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode = new_right.as_mut();
    let new_child: *mut u8 = StdPtr::without_provenance_mut(0xABCD);

    // Insert key 200 at position 15 (largest, after all existing)
    let (popup_key, insert_went_left) =
        node.split_into(&mut new_right, new_right_ptr, 15, 200, new_child);

    // Insert at position 15 > mid(8), so it goes right
    assert!(
        !insert_went_left,
        "Insert at position WIDTH should go right"
    );

    // Verify popup key is reasonable
    assert!(popup_key > 0, "Popup key should be non-zero");
}

#[test]
fn test_split_insert_at_midpoint() {
    // Test splitting when the new key goes at the midpoint (becomes popup)
    // Use height=0 so split_into treats children as leaves (doesn't dereference them)
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode> = InternodeNode::new(0);

    // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode = new_right.as_mut();
    let new_child: *mut u8 = StdPtr::without_provenance_mut(0xABCD);

    // mid = ceil(15/2) = 8
    // Insert key 85 at position 8 (the midpoint)
    let (popup_key, insert_went_left) =
        node.split_into(&mut new_right, new_right_ptr, 8, 85, new_child);

    // When insert_pos == mid, the insert key becomes the popup key
    assert_eq!(popup_key, 85, "Insert at midpoint should become popup key");

    // insert_went_left behavior at midpoint depends on implementation details.
    // The key observation is that when insert_pos == mid, the inserted key
    // becomes the popup key, so it doesn't go to either sibling.
    // We just verify the function completed successfully.
    let _ = insert_went_left;
}

// ========================================================================
//  load_all_ikeys tests
// ========================================================================

#[test]
fn test_load_all_ikeys_empty() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    // Even with 0 keys, load_all_ikeys returns WIDTH values (uninitialized slots are 0)
    let ikeys = node.load_all_ikeys();
    assert_eq!(ikeys.len(), 15);
    // All should be zero since node is empty
    for key in ikeys {
        assert_eq!(key, 0);
    }
}

#[test]
fn test_load_all_ikeys_partial() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Set 5 keys
    node.set_ikey(0, 100);
    node.set_ikey(1, 200);
    node.set_ikey(2, 300);
    node.set_ikey(3, 400);
    node.set_ikey(4, 500);
    node.set_nkeys(5);

    let ikeys = node.load_all_ikeys();

    // First 5 should match what we set
    assert_eq!(ikeys[0], 100);
    assert_eq!(ikeys[1], 200);
    assert_eq!(ikeys[2], 300);
    assert_eq!(ikeys[3], 400);
    assert_eq!(ikeys[4], 500);

    // Remaining should be zero (uninitialized)
    (5..15).for_each(|i| {
        assert_eq!(ikeys[i], 0);
    });
}

#[test]
fn test_load_all_ikeys_full() {
    let node: Box<InternodeNode> = InternodeNode::new(0);

    // Fill all 15 keys
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 1000);
    }
    node.set_nkeys(15);

    let ikeys = node.load_all_ikeys();

    (0..15).for_each(|i| {
        assert_eq!(ikeys[i], (i as u64 + 1) * 1000);
    });
}

// ========================================================================
//  shift_from tests
// ========================================================================

#[test]
fn test_shift_from_basic() {
    // Test shifting entries from one internode to another
    let src: Box<InternodeNode> = InternodeNode::new(0);
    let dst: Box<InternodeNode> = InternodeNode::new(0);

    // Set up source with 5 keys and 6 children
    src.set_ikey(0, 100);
    src.set_ikey(1, 200);
    src.set_ikey(2, 300);
    src.set_ikey(3, 400);
    src.set_ikey(4, 500);
    src.set_nkeys(5);

    // Set up fake children
    for i in 0..6 {
        src.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }

    // Shift 3 entries starting at src position 1 to dst position 0
    // This should copy: ikey[1], ikey[2], ikey[3] and child[2], child[3], child[4]
    // SAFETY: Single-threaded test context.
    unsafe { dst.shift_from(0, &src, 1, 3) };
    dst.set_nkeys(3);

    // Verify keys were copied
    assert_eq!(dst.ikey(0), 200);
    assert_eq!(dst.ikey(1), 300);
    assert_eq!(dst.ikey(2), 400);

    // Verify children were copied (shift_from copies child[src_pos + 1 + i])
    // SAFETY: Single-threaded test context.
    assert_eq!(
        unsafe { dst.child_unguarded(1) },
        StdPtr::without_provenance_mut(3 * 0x1000)
    ); // src.child(2)
    assert_eq!(
        unsafe { dst.child_unguarded(2) },
        StdPtr::without_provenance_mut(4 * 0x1000)
    ); // src.child(3)
    assert_eq!(
        unsafe { dst.child_unguarded(3) },
        StdPtr::without_provenance_mut(5 * 0x1000)
    ); // src.child(4)
}

#[test]
fn test_shift_from_to_different_position() {
    let src: Box<InternodeNode> = InternodeNode::new(0);
    let dst: Box<InternodeNode> = InternodeNode::new(0);

    // Set up source
    src.set_ikey(0, 10);
    src.set_ikey(1, 20);
    src.set_ikey(2, 30);
    src.set_nkeys(3);

    for i in 0..4 {
        src.set_child(i, StdPtr::without_provenance_mut((i + 10) * 0x100));
    }

    // Pre-fill dst with some data at position 0
    dst.set_ikey(0, 5);
    dst.set_child(0, StdPtr::without_provenance_mut(0x1));
    dst.set_child(1, StdPtr::without_provenance_mut(0x2));

    // Shift 2 entries from src position 1 to dst position 1
    // This copies ikey[1], ikey[2] and child[2], child[3]
    // SAFETY: Single-threaded test context.
    unsafe { dst.shift_from(1, &src, 1, 2) };
    dst.set_nkeys(3);

    // Original data should be preserved
    assert_eq!(dst.ikey(0), 5);
    // SAFETY: Single-threaded test context.
    assert_eq!(
        unsafe { dst.child_unguarded(0) },
        StdPtr::without_provenance_mut(0x1)
    );

    // Shifted data should be at position 1+
    assert_eq!(dst.ikey(1), 20);
    assert_eq!(dst.ikey(2), 30);
    // SAFETY: Single-threaded test context.
    assert_eq!(
        unsafe { dst.child_unguarded(2) },
        StdPtr::without_provenance_mut(12 * 0x100)
    ); // src.child(2)
    assert_eq!(
        unsafe { dst.child_unguarded(3) },
        StdPtr::without_provenance_mut(13 * 0x100)
    ); // src.child(3)
}

#[test]
fn test_shift_from_zero_count() {
    let src: Box<InternodeNode> = InternodeNode::new(0);
    let dst: Box<InternodeNode> = InternodeNode::new(0);

    src.set_ikey(0, 100);
    src.set_nkeys(1);

    // dst has some existing data
    dst.set_ikey(0, 999);
    dst.set_nkeys(1);

    // Shift 0 entries - should be a no-op
    // SAFETY: Single-threaded test context.
    unsafe { dst.shift_from(0, &src, 0, 0) };

    // dst should be unchanged
    assert_eq!(dst.ikey(0), 999);
    assert_eq!(dst.nkeys(), 1);
}

// ========================================================================
//  Depth Prefetch Tests
// ========================================================================

#[test]
fn test_depth_prefetch_methods() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let collector = seize::Collector::new();
    let guard = collector.enter();

    // Set up a child pointer
    let child_box: Box<InternodeNode> = InternodeNode::new(0);
    let child: *mut InternodeNode = Box::into_raw(child_box);
    node.set_child(0, child.cast());

    // Test all prefetch variants return same pointer
    let ptr1: *mut u8 = node.child(0, &guard);
    let ptr2: *mut u8 = node.child_with_prefetch(0, 5, &guard);
    let ptr3: *mut u8 = node.child_with_depth_prefetch(0, &guard);
    let ptr4: *mut u8 = node.child_with_full_prefetch(0, &guard);

    assert_eq!(ptr1, ptr2);
    assert_eq!(ptr2, ptr3);
    assert_eq!(ptr3, ptr4);

    // Cleanup: reclaim the boxed node
    unsafe {
        let _ = Box::from_raw(child);
    }
}

#[test]
fn test_prefetch_null_child() {
    let node: Box<InternodeNode> = InternodeNode::new(0);
    let collector = seize::Collector::new();
    let guard = collector.enter();

    // Prefetch of null pointer should not crash
    let ptr: *mut u8 = node.child_with_depth_prefetch(0, &guard);
    assert!(ptr.is_null());

    let ptr2: *mut u8 = node.child_with_full_prefetch(0, &guard);
    assert!(ptr2.is_null());
}
