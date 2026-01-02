use super::*;

#[test]
fn test_new_internode() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    assert!(!node.version().is_leaf());
    assert!(!node.version().is_root());
    assert_eq!(node.nkeys(), 0);
    assert_eq!(node.height(), 0);
    assert!(node.is_empty());
    assert!(!node.is_full());
    assert!(node.children_are_leaves());
    assert!(node.parent().is_null());
}

#[test]
fn test_new_root() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new_root(1);

    assert!(!node.version().is_leaf());
    assert!(node.version().is_root());
    assert_eq!(node.height(), 1);
    assert!(!node.children_are_leaves());
}

#[test]
fn test_key_accessors() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    let fake_child0: *mut u8 = StdPtr::without_provenance_mut(0x1000);
    let fake_child1: *mut u8 = StdPtr::without_provenance_mut(0x2000);
    let fake_child2: *mut u8 = StdPtr::without_provenance_mut(0x3000);

    node.set_child(0, fake_child0);
    node.set_child(1, fake_child1);
    node.set_child(2, fake_child2);

    assert_eq!(node.child(0), fake_child0);
    assert_eq!(node.child(1), fake_child1);
    assert_eq!(node.child(2), fake_child2);
}

#[test]
fn test_assign() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    let left_child: *mut u8 = StdPtr::without_provenance_mut(0x1000);
    let right_child: *mut u8 = StdPtr::without_provenance_mut(0x2000);

    // Set left child first
    node.set_child(0, left_child);

    // Assign key and right child
    node.assign(0, 0xABCD_0000_0000_0000, right_child);
    node.set_nkeys(1);

    assert_eq!(node.ikey(0), 0xABCD_0000_0000_0000);
    assert_eq!(node.child(0), left_child);
    assert_eq!(node.child(1), right_child);
    assert_eq!(node.size(), 1);
}

#[test]
fn test_inc_nkeys() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    assert_eq!(node.nkeys(), 0);

    node.inc_nkeys();
    assert_eq!(node.nkeys(), 1);

    node.inc_nkeys();
    assert_eq!(node.nkeys(), 2);
}

#[test]
fn test_is_full() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    assert!(!node.is_full());

    node.set_nkeys(15);
    assert!(node.is_full());
}

#[test]
fn test_parent_accessors() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
    let mut parent: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(1);

    let parent_ptr: *mut InternodeNode<LeafValue<u64>> =
        parent.as_mut() as *mut InternodeNode<LeafValue<u64>>;

    // set_parent takes *mut u8, so cast the pointer
    node.set_parent(parent_ptr.cast::<u8>());
    assert_eq!(node.parent(), parent_ptr.cast::<u8>());
}

#[test]
fn test_compare_key() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
    // Empty node: any key goes at position 0
    assert_eq!(node.find_insert_position(0x1000), 0);
    assert_eq!(node.find_insert_position(0), 0);
    assert_eq!(node.find_insert_position(u64::MAX), 0);
}

#[test]
fn test_find_insert_position_single_key() {
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    // Fill the node with keys 20, 30, 40, ..., 160 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 2) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode<LeafValue<u64>> = new_right.as_mut();
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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode<LeafValue<u64>> = new_right.as_mut();
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
    let node: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);
    let mut new_right: Box<InternodeNode<LeafValue<u64>>> = InternodeNode::new(0);

    // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
    for i in 0..15 {
        node.set_ikey(i, (i as u64 + 1) * 10);
        node.set_child(i, StdPtr::without_provenance_mut((i + 1) * 0x1000));
    }
    node.set_child(15, StdPtr::without_provenance_mut(16 * 0x1000));
    node.set_nkeys(15);

    let new_right_ptr: *mut InternodeNode<LeafValue<u64>> = new_right.as_mut();
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

