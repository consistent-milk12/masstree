use super::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode24, WIDTH_24};
use crate::value::LeafValue;
use std::ptr as StdPtr;

#[test]
fn test_new_leaf24_is_empty() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    assert!(leaf.is_empty());
    assert_eq!(leaf.size(), 0);
    assert!(!leaf.is_full());
}

#[test]
fn test_leaf24_permutation_basic() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let perm = leaf.permutation();
    assert_eq!(perm.size(), 0);

    // Insert one key
    let mut new_perm = perm;
    let slot = new_perm.insert_from_back(0);
    leaf.set_permutation(new_perm);

    assert_eq!(leaf.size(), 1);
    assert_eq!(leaf.permutation().get(0), slot);
}

#[test]
fn test_leaf24_ikey_operations() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    leaf.set_ikey(0, 12345);
    assert_eq!(leaf.ikey(0), 12345);
    assert_eq!(leaf.ikey_bound(), 12345);
}

#[test]
fn test_leaf24_keylenx_operations() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    leaf.set_keylenx(5, 8);
    assert_eq!(leaf.keylenx(5), 8);
    assert!(!leaf.is_layer(5));
    assert!(!leaf.has_ksuf(5));

    leaf.set_keylenx(10, LAYER_KEYLENX);
    assert!(leaf.is_layer(10));

    leaf.set_keylenx(15, KSUF_KEYLENX);
    assert!(leaf.has_ksuf(15));
}

#[test]
fn test_leaf24_full_at_24_slots() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let mut perm = leaf.permutation();

    for i in 0..WIDTH_24 {
        let _slot = perm.insert_from_back(i);
    }
    leaf.set_permutation(perm);

    assert!(leaf.is_full());
    assert_eq!(leaf.size(), WIDTH_24);
}

#[test]
fn test_leaf24_linking() {
    let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

    let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);

    leaf1.set_next(leaf2_ptr);
    assert_eq!(unsafe { leaf1.safe_next_unguarded() }, leaf2_ptr);
    assert!(!leaf1.next_is_marked());

    leaf1.mark_next();
    assert!(leaf1.next_is_marked());
    assert_eq!(unsafe { leaf1.safe_next_unguarded() }, leaf2_ptr);

    leaf1.unmark_next();
    assert!(!leaf1.next_is_marked());

    // Cleanup
    // SAFETY: leaf2_ptr was created by Box::into_raw above and not yet freed
    let _ = unsafe { Box::from_raw(leaf2_ptr) };
}

#[test]
fn test_leaf24_parent() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    assert!(unsafe { leaf.parent_unguarded() }.is_null());

    let dummy: u64 = 0xDEAD_BEEF;
    let dummy_ptr: *mut u8 = StdPtr::from_ref(&dummy).cast_mut().cast();
    leaf.set_parent(dummy_ptr);
    assert_eq!(unsafe { leaf.parent_unguarded() }, dummy_ptr);
}

#[test]
fn test_lock_next_null() {
    // lock_next on a leaf with NULL next should return NULL and mark the next pointer.
    // This prevents two concurrent splits both observing NULL and racing to publish
    // different siblings (orphaning one).
    let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    assert!(unsafe { leaf.safe_next_unguarded() }.is_null());

    let result = leaf.lock_next();
    assert!(result.is_null());
    // After lock_next with NULL, next is marked (but safe_next still reads as NULL)
    assert!(leaf.next_is_marked());
    assert!(unsafe { leaf.safe_next_unguarded() }.is_null());

    // Unmark should restore a clean NULL next
    leaf.unmark_next();
    assert!(!leaf.next_is_marked());
}

#[test]
fn test_lock_next_marks_and_unmarks() {
    // lock_next should mark the next pointer, returned value is unmarked
    let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

    let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);
    leaf1.set_next(leaf2_ptr);

    // Before lock_next: not marked
    assert!(!leaf1.next_is_marked());

    // Call lock_next
    let result = leaf1.lock_next();

    // Returned value should be the unmarked pointer
    assert_eq!(result, leaf2_ptr);
    // But the next pointer should now be marked
    assert!(leaf1.next_is_marked());
    // safe_next still returns the unmarked pointer
    assert_eq!(unsafe { leaf1.safe_next_unguarded() }, leaf2_ptr);

    // "Unlock" by storing the unmarked pointer
    leaf1.set_next(leaf2_ptr);
    assert!(!leaf1.next_is_marked());

    // Cleanup
    // SAFETY: leaf2_ptr was created by Box::into_raw above and not yet freed
    let _ = unsafe { Box::from_raw(leaf2_ptr) };
}

#[test]
#[expect(clippy::similar_names)]
fn test_link_sibling_preserves_existing_next() {
    // Set up chain: A -> B
    // After link A -> C: should have A -> C -> B with correct prev pointers
    use crate::leaf_trait::TreeLeafNode;

    let leaf_a: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf_b: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf_c: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

    let leaf_a_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_a);
    let leaf_b_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_b);
    let leaf_c_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_c);

    // Reconstruct references (SAFETY: Pointers are valid from Box::into_raw above)
    let leaf_a: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_a_ptr }; // SAFETY: valid ptr
    let leaf_b: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_b_ptr }; // SAFETY: valid ptr
    let leaf_c: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_c_ptr }; // SAFETY: valid ptr

    // Initial chain: A -> B
    leaf_a.set_next(leaf_b_ptr);
    leaf_b.set_prev(leaf_a_ptr);

    // Link C after A: A -> C -> B
    // SAFETY: All pointers are valid and properly initialized
    unsafe { TreeLeafNode::<LeafValue<u64>>::link_sibling(leaf_a, leaf_c_ptr) };

    // Verify chain structure
    assert_eq!(unsafe { leaf_a.safe_next_unguarded() }, leaf_c_ptr, "A.next should be C");
    assert_eq!(unsafe { leaf_c.safe_next_unguarded() }, leaf_b_ptr, "C.next should be B");
    assert_eq!(unsafe { leaf_c.prev_unguarded() }, leaf_a_ptr, "C.prev should be A");
    assert_eq!(unsafe { leaf_b.prev_unguarded() }, leaf_c_ptr, "B.prev should be C");

    // Verify not marked after link
    assert!(!leaf_a.next_is_marked());
    assert!(!leaf_c.next_is_marked());

    // Cleanup (SAFETY: Pointers from Box::into_raw above, not yet freed)
    let _ = unsafe { Box::from_raw(leaf_a_ptr) }; // SAFETY: valid ptr
    let _ = unsafe { Box::from_raw(leaf_b_ptr) }; // SAFETY: valid ptr
    let _ = unsafe { Box::from_raw(leaf_c_ptr) }; // SAFETY: valid ptr
}

#[test]
#[expect(clippy::similar_names)]
fn test_link_sibling_null_next() {
    // Link sibling when next is NULL: A -> NULL becomes A -> C -> NULL
    use crate::leaf_trait::TreeLeafNode;

    let leaf_a: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf_c: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

    let leaf_a_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_a);
    let leaf_c_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_c);

    // SAFETY: Pointers are valid, created by Box::into_raw above
    let leaf_a: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_a_ptr };
    let leaf_c: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_c_ptr };

    // A has no next
    assert!(unsafe { leaf_a.safe_next_unguarded() }.is_null());

    // Link C after A
    // SAFETY: All pointers are valid and properly initialized
    unsafe { TreeLeafNode::<LeafValue<u64>>::link_sibling(leaf_a, leaf_c_ptr) };

    // Verify chain
    assert_eq!(unsafe { leaf_a.safe_next_unguarded() }, leaf_c_ptr, "A.next should be C");
    assert!(unsafe { leaf_c.safe_next_unguarded() }.is_null(), "C.next should be NULL");
    assert_eq!(unsafe { leaf_c.prev_unguarded() }, leaf_a_ptr, "C.prev should be A");
    assert!(!leaf_a.next_is_marked());

    // Cleanup
    // SAFETY: Pointers were created by Box::into_raw above and not yet freed
    let _ = unsafe { Box::from_raw(leaf_a_ptr) };
    let _ = unsafe { Box::from_raw(leaf_c_ptr) };
}

#[test]
#[expect(clippy::unwrap_used)]
fn test_cas_next() {
    let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
    let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

    let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);

    // CAS from NULL to leaf2_ptr should succeed
    let result = leaf1.cas_next(StdPtr::null_mut(), leaf2_ptr);
    assert!(result.is_ok());
    assert_eq!(unsafe { leaf1.safe_next_unguarded() }, leaf2_ptr);

    // CAS with wrong expected should fail
    let result = leaf1.cas_next(StdPtr::null_mut(), StdPtr::null_mut());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), leaf2_ptr);

    // Cleanup
    // SAFETY: leaf2_ptr was created by Box::into_raw above and not yet freed
    let _ = unsafe { Box::from_raw(leaf2_ptr) };
}
