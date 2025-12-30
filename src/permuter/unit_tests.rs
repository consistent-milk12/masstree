use super::*;

// ==================== Basic Tests ====================

#[test]
fn test_empty_permuter() {
    let p: Permuter<15> = Permuter::empty();
    assert_eq!(p.size(), 0);

    // Initial value: positions in reverse order, back() returns 0
    assert_eq!(p.back(), 0);
    assert_eq!(p.value(), 0x0123_4567_89AB_CDE0);
}

#[test]
fn test_default_is_empty() {
    let p: Permuter<15> = Permuter::default();

    assert_eq!(p.size(), 0);
    assert_eq!(p.value(), Permuter::<15>::empty().value());
}

// ==================== make_sorted Tests ====================

#[test]
fn test_make_sorted_full() {
    // make_sorted(WIDTH) should give position i → slot i (sorted order)
    let p: Permuter<15> = Permuter::make_sorted(15);
    assert_eq!(p.size(), 15);

    // Verify position i maps to slot i
    for i in 0..15 {
        assert_eq!(p.get(i), i, "position {i} should map to slot {i}");
    }

    // Expected value: 0xEDCBA9876543210F (sorted_value | 15)
    assert_eq!(p.value(), 0xEDCB_A987_6543_210F);
}

#[test]
fn test_make_sorted_partial() {
    let p: Permuter<15> = Permuter::make_sorted(3);
    assert_eq!(p.size(), 3);

    // Positions 0..3 are sorted (slot i at position i)
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);
    assert_eq!(p.get(2), 2);

    // back() should return 3 (next slot to allocate)
    assert_eq!(p.back(), 3);
}

#[test]
fn test_make_sorted_zero() {
    let p: Permuter<15> = Permuter::make_sorted(0);
    assert_eq!(p.size(), 0);

    // back() should return 0 (first slot to allocate)
    assert_eq!(p.back(), 0);
}

#[test]
fn test_make_sorted_one() {
    let p: Permuter<15> = Permuter::make_sorted(1);
    assert_eq!(p.size(), 1);
    assert_eq!(p.get(0), 0);

    // back() should return 1 (next slot to allocate)
    assert_eq!(p.back(), 1);
}

// ==================== insert_from_back Tests ====================

#[test]
fn test_insert_from_back() {
    let mut p: Permuter<15> = Permuter::empty();
    assert_eq!(p.size(), 0);

    // Insert at position 0
    let slot0: usize = p.insert_from_back(0);
    assert_eq!(slot0, 0); // First slot allocated is 0
    assert_eq!(p.size(), 1);
    assert_eq!(p.get(0), 0);

    // Insert at position 0 again (shifts previous to position 1)
    let slot1: usize = p.insert_from_back(0);
    assert_eq!(slot1, 1); // Second slot allocated is 1
    assert_eq!(p.size(), 2);
    assert_eq!(p.get(0), 1); // New slot at position 0
    assert_eq!(p.get(1), 0); // Old slot shifted to position 1

    // Insert at position 1 (between the two)
    let slot2: usize = p.insert_from_back(1);
    assert_eq!(slot2, 2);
    assert_eq!(p.size(), 3);
    assert_eq!(p.get(0), 1);
    assert_eq!(p.get(1), 2); // New slot at position 1
    assert_eq!(p.get(2), 0); // Old position 1 shifted to position 2
}

#[test]
fn test_insert_from_back_at_end() {
    let mut p: Permuter<15> = Permuter::empty();

    // Insert at end positions
    let slot0: usize = p.insert_from_back(0);
    let slot1: usize = p.insert_from_back(1); // Insert at end
    let slot2: usize = p.insert_from_back(2); // Insert at end

    assert_eq!(p.size(), 3);
    assert_eq!(p.get(0), slot0);
    assert_eq!(p.get(1), slot1);
    assert_eq!(p.get(2), slot2);
}

#[test]
fn test_insert_fill_to_capacity() {
    let mut p: Permuter<15> = Permuter::empty();

    // Fill the permuter completely
    for i in 0..15 {
        let slot = p.insert_from_back(i);
        assert_eq!(slot, i);
    }

    assert_eq!(p.size(), 15);

    // Verify all positions
    for i in 0..15 {
        assert_eq!(p.get(i), i);
    }
}

// ==================== remove_to_back Tests ====================

#[test]
fn test_remove_to_back() {
    // Start with a sorted permuter of size 5
    let mut p: Permuter<15> = Permuter::make_sorted(5);

    // Positions: 0→0, 1→1, 2→2, 3→3, 4→4
    assert_eq!(p.size(), 5);

    // Remove position 2 (slot 2)
    p.remove_to_back(2);
    assert_eq!(p.size(), 4);

    // Positions 0, 1 unchanged
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);

    // Position 2 now has what was at position 3
    assert_eq!(p.get(2), 3);

    // Position 3 now has what was at position 4
    assert_eq!(p.get(3), 4);

    // back() should return the removed slot (2)
    assert_eq!(p.back(), 2);
}

#[test]
fn test_remove_to_back_first() {
    let mut p: Permuter<15> = Permuter::make_sorted(3);
    // Positions: 0→0, 1→1, 2→2

    // Remove position 0 (slot 0)
    p.remove_to_back(0);
    assert_eq!(p.size(), 2);

    // Position 0 now has slot 1
    assert_eq!(p.get(0), 1);

    // Position 1 now has slot 2
    assert_eq!(p.get(1), 2);

    // back() should return the removed slot (0)
    assert_eq!(p.back(), 0);
}

#[test]
fn test_remove_to_back_last() {
    let mut p: Permuter<15> = Permuter::make_sorted(3);
    // Positions: 0→0, 1→1, 2→2

    // Remove position 2 (the last in-use position)
    p.remove_to_back(2);
    assert_eq!(p.size(), 2);

    // Positions 0, 1 unchanged
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);

    // back() should return the removed slot (2)
    assert_eq!(p.back(), 2);
}

// ==================== remove Tests ====================

#[test]
fn test_remove_middle() {
    let mut p: Permuter<15> = Permuter::make_sorted(5);
    // Positions: 0→0, 1→1, 2→2, 3→3, 4→4

    // Remove position 2
    p.remove(2);
    assert_eq!(p.size(), 4);

    // Positions 0, 1 unchanged
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);

    // Positions 2, 3 shifted down
    assert_eq!(p.get(2), 3);
    assert_eq!(p.get(3), 4);

    // Position 4 (first free) should have the removed slot
    assert_eq!(p.get(4), 2);
}

#[test]
fn test_remove_first() {
    let mut p: Permuter<15> = Permuter::make_sorted(3);

    p.remove(0);
    assert_eq!(p.size(), 2);

    assert_eq!(p.get(0), 1);
    assert_eq!(p.get(1), 2);
    assert_eq!(p.get(2), 0); // Removed slot at position 2 (size)
}

#[test]
fn test_remove_last() {
    let mut p: Permuter<15> = Permuter::make_sorted(3);

    // Remove the last element (fast path)
    p.remove(2);
    assert_eq!(p.size(), 2);

    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);

    // Position 2 still has slot 2 (unchanged by fast path)
    assert_eq!(p.get(2), 2);
}

// ==================== exchange Tests ====================

#[test]
fn test_exchange_basic() {
    let mut p: Permuter<15> = Permuter::make_sorted(5);
    // Positions: 0→0, 1→1, 2→2, 3→3, 4→4

    // Exchange positions 1 and 3
    p.exchange(1, 3);

    assert_eq!(p.size(), 5); // Size unchanged
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 3); // Swapped
    assert_eq!(p.get(2), 2);
    assert_eq!(p.get(3), 1); // Swapped
    assert_eq!(p.get(4), 4);
}

#[test]
fn test_exchange_same() {
    let mut p: Permuter<15> = Permuter::make_sorted(5);
    let original: u64 = p.value();

    // Exchange same position (no-op)
    p.exchange(2, 2);

    assert_eq!(p.value(), original);
}

#[test]
fn test_exchange_adjacent() {
    let mut p: Permuter<15> = Permuter::make_sorted(5);

    // Exchange adjacent positions
    p.exchange(2, 3);

    assert_eq!(p.get(2), 3);
    assert_eq!(p.get(3), 2);
}

#[test]
fn test_exchange_first_last() {
    let mut p: Permuter<15> = Permuter::make_sorted(15);

    // Exchange first and last positions
    p.exchange(0, 14);

    assert_eq!(p.get(0), 14);
    assert_eq!(p.get(14), 0);
}

// ==================== rotate Tests ====================

#[test]
fn test_rotate_basic() {
    let mut p: Permuter<7> = Permuter::make_sorted(7);
    // Positions: 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6

    // Rotate starting at position 2 by 2 positions
    p.rotate(2, 4);

    // Positions 0, 1 unchanged
    assert_eq!(p.get(0), 0);
    assert_eq!(p.get(1), 1);

    // Positions 2+ are rotated
    // The rotation shifts by (j-i) = 2 positions
}

#[test]
fn test_rotate_no_op() {
    let mut p: Permuter<15> = Permuter::make_sorted(5);
    let original: u64 = p.value();

    // rotate(i, i) is a no-op
    p.rotate(2, 2);
    assert_eq!(p.value(), original);

    // rotate(WIDTH, j) is a no-op
    p.rotate(15, 15);
    assert_eq!(p.value(), original);
}

// ==================== Roundtrip Tests ====================

#[test]
fn test_insert_remove_roundtrip() {
    let mut p: Permuter<15> = Permuter::empty();

    // Insert 5 elements
    for i in 0..5 {
        let _ = p.insert_from_back(i);
    }
    assert_eq!(p.size(), 5);

    // Remove them one by one using remove_to_back
    for _ in 0..5 {
        p.remove_to_back(0);
    }
    assert_eq!(p.size(), 0);

    // Should still have valid invariants
    p.debug_assert_valid();
}

#[test]
fn test_insert_remove_roundtrip_alt() {
    let mut p: Permuter<15> = Permuter::empty();

    // Insert 5 elements
    for i in 0..5 {
        let _ = p.insert_from_back(i);
    }

    // Remove them using remove() (different from remove_to_back)
    for _ in 0..5 {
        p.remove(0);
    }
    assert_eq!(p.size(), 0);

    p.debug_assert_valid();
}

// ==================== WIDTH Variant Tests ====================

#[test]
fn test_compact_permuter() {
    // Test WIDTH=7 variant
    let mut p: Permuter<7> = Permuter::empty();
    assert_eq!(p.size(), 0);
    assert_eq!(p.back(), 0);

    let slot: usize = p.insert_from_back(0);
    assert_eq!(slot, 0);
    assert_eq!(p.size(), 1);

    let sorted: Permuter<7> = Permuter::make_sorted(7);
    assert_eq!(sorted.size(), 7);

    for i in 0..7 {
        assert_eq!(sorted.get(i), i);
    }
}

#[test]
fn test_width_3_permuter() {
    let mut p: Permuter<3> = Permuter::empty();
    assert_eq!(p.size(), 0);

    let _ = p.insert_from_back(0);
    let _ = p.insert_from_back(0);
    let _ = p.insert_from_back(0);

    assert_eq!(p.size(), 3);
    p.debug_assert_valid();
}

// ==================== Edge Case Tests ====================

#[test]
fn test_set_size() {
    let mut p: Permuter<15> = Permuter::make_sorted(10);
    assert_eq!(p.size(), 10);

    p.set_size(5);
    assert_eq!(p.size(), 5);

    p.set_size(0);
    assert_eq!(p.size(), 0);
}

#[test]
fn test_value_accessor() {
    let p: Permuter<15> = Permuter::empty();
    let v: u64 = p.value();

    // Create another permuter with same value
    let p2: Permuter<15> = Permuter::empty();
    assert_eq!(p.value(), p2.value());
    assert_eq!(v, 0x0123_4567_89AB_CDE0);
}

#[test]
fn test_clone_and_eq() {
    let p1: Permuter<15> = Permuter::make_sorted(5);
    let p2: Permuter = p1; // Copy

    assert_eq!(p1, p2);
    assert_eq!(p1.value(), p2.value());
}

// ==================== insert_from_back_immutable Tests ====================

#[test]
fn test_insert_from_back_immutable_basic() {
    let p: Permuter<15> = Permuter::empty();
    assert_eq!(p.size(), 0);

    // Insert at position 0 (immutable)
    let (new_p, slot) = p.insert_from_back_immutable(0);

    // Original unchanged
    assert_eq!(p.size(), 0);

    // New permuter has the insert
    assert_eq!(new_p.size(), 1);
    assert_eq!(slot, 0);
    assert_eq!(new_p.get(0), 0);
}

#[test]
fn test_insert_from_back_immutable_matches_mutable() {
    // Verify immutable version produces same result as mutable version
    let original: Permuter<15> = Permuter::make_sorted(5);

    // Mutable insert
    let mut mutable = original;
    let mutable_slot = mutable.insert_from_back(2);

    // Immutable insert
    let (immutable, immutable_slot) = original.insert_from_back_immutable(2);

    // Should produce identical results
    assert_eq!(mutable_slot, immutable_slot);
    assert_eq!(mutable.value(), immutable.value());
    assert_eq!(mutable.size(), immutable.size());

    for i in 0..mutable.size() {
        assert_eq!(mutable.get(i), immutable.get(i));
    }
}

#[test]
fn test_insert_from_back_immutable_chain() {
    // Test chaining multiple immutable inserts
    let p0: Permuter<15> = Permuter::empty();

    let (p1, slot0) = p0.insert_from_back_immutable(0);
    let (p2, slot1) = p1.insert_from_back_immutable(0);
    let (p3, slot2) = p2.insert_from_back_immutable(1);

    // Original unchanged
    assert_eq!(p0.size(), 0);

    // Each step added one
    assert_eq!(p1.size(), 1);
    assert_eq!(p2.size(), 2);
    assert_eq!(p3.size(), 3);

    // Slots allocated in order
    assert_eq!(slot0, 0);
    assert_eq!(slot1, 1);
    assert_eq!(slot2, 2);

    // Final permuter has correct structure
    assert_eq!(p3.get(0), 1); // slot1 at position 0
    assert_eq!(p3.get(1), 2); // slot2 at position 1
    assert_eq!(p3.get(2), 0); // slot0 at position 2 (shifted)
}

#[test]
fn test_insert_from_back_immutable_for_cas() {
    // Simulate CAS usage: compute new value, compare old
    let current: Permuter<15> = Permuter::make_sorted(3);
    let current_value = current.value();

    let (new_perm, slot) = current.insert_from_back_immutable(1);

    // Simulate CAS: old value should match current
    assert_eq!(current.value(), current_value);

    // New value is different
    assert_ne!(new_perm.value(), current_value);

    // Could now do: compare_exchange(current_value, new_perm.value())
    assert_eq!(slot, 3); // Next slot after sorted(3)
    assert_eq!(new_perm.size(), 4);
}

#[test]
fn test_make_sorted_via_trait() {
    fn check_trait<P: TreePermutation>() {
        let p = P::make_sorted(2);

        assert_eq!(p.size(), 2);
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);
    }

    check_trait::<Permuter<15>>();
}
