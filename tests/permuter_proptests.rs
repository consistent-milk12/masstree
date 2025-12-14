//! Property-based tests for the `permuter` module.
//!
//! These tests verify invariants and properties that should hold for all inputs.
//! The permuter is a critical data structure that must maintain strict invariants.

use madtree::permuter::{Permuter, MAX_WIDTH};
use proptest::prelude::*;

// ============================================================================
//  Strategies
// ============================================================================

/// Strategy for generating a sequence of insert positions.
fn insert_sequence(max_inserts: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(0usize..=14, 0..=max_inserts).prop_map(|positions| {
        // Adjust positions to be valid for each insert
        let mut result = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            // Position must be <= current size
            result.push(pos % (i + 1));
        }
        result
    })
}

// ============================================================================
//  Bijection Property (Core Invariant)
// ============================================================================

/// Helper to verify the bijection property: all slots 0..WIDTH appear exactly once.
fn verify_bijection<const WIDTH: usize>(perm: &Permuter<WIDTH>) -> bool {
    let mut seen = [false; MAX_WIDTH];

    for i in 0..WIDTH {
        let slot = perm.get(i);
        if slot >= WIDTH || seen[slot] {
            return false;
        }
        seen[slot] = true;
    }

    // All slots should be seen exactly once
    seen[..WIDTH].iter().all(|&s| s)
}

proptest! {
    /// Empty permuter should maintain bijection property.
    #[test]
    fn empty_maintains_bijection(_dummy: u8) {
        let perm: Permuter<15> = Permuter::empty();
        prop_assert!(verify_bijection(&perm));
    }

    /// make_sorted should maintain bijection property.
    #[test]
    fn make_sorted_maintains_bijection(n in 0usize..=15) {
        let perm: Permuter<15> = Permuter::make_sorted(n);
        prop_assert!(verify_bijection(&perm));
    }

    /// insert_from_back should maintain bijection property.
    #[test]
    fn insert_maintains_bijection(positions in insert_sequence(15)) {
        let mut perm: Permuter<15> = Permuter::empty();

        for (i, &pos) in positions.iter().enumerate() {
            if i >= 15 {
                break;
            }
            let _ = perm.insert_from_back(pos);
            prop_assert!(verify_bijection(&perm), "Bijection broken after insert {} at pos {}", i, pos);
        }
    }

    /// remove_to_back should maintain bijection property.
    #[test]
    fn remove_to_back_maintains_bijection(
        initial_size in 1usize..=15,
        remove_indices in prop::collection::vec(0usize..15, 1..=15)
    ) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);

        for &idx in &remove_indices {
            if perm.size() == 0 {
                break;
            }
            let valid_idx = idx % perm.size();
            perm.remove_to_back(valid_idx);
            prop_assert!(verify_bijection(&perm), "Bijection broken after remove_to_back({})", valid_idx);
        }
    }

    /// remove should maintain bijection property.
    #[test]
    fn remove_maintains_bijection(
        initial_size in 1usize..=15,
        remove_indices in prop::collection::vec(0usize..15, 1..=15)
    ) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);

        for &idx in &remove_indices {
            if perm.size() == 0 {
                break;
            }
            let valid_idx = idx % perm.size();
            perm.remove(valid_idx);
            prop_assert!(verify_bijection(&perm), "Bijection broken after remove({})", valid_idx);
        }
    }

    /// exchange should maintain bijection property.
    #[test]
    fn exchange_maintains_bijection(
        initial_size in 1usize..=15,
        i in 0usize..15,
        j in 0usize..15
    ) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        perm.exchange(i, j);
        prop_assert!(verify_bijection(&perm));
    }

    /// rotate should maintain bijection property.
    #[test]
    fn rotate_maintains_bijection(
        initial_size in 1usize..=15,
        i in 0usize..=15,
        j in 0usize..=15
    ) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        let j = j.min(15);
        perm.rotate(i, j);
        prop_assert!(verify_bijection(&perm));
    }
}

// ============================================================================
//  Size Properties
// ============================================================================

proptest! {
    /// Empty permuter should have size 0.
    #[test]
    fn empty_has_size_zero(_dummy: u8) {
        let perm: Permuter<15> = Permuter::empty();
        prop_assert_eq!(perm.size(), 0);
    }

    /// make_sorted(n) should have size n.
    #[test]
    fn make_sorted_has_correct_size(n in 0usize..=15) {
        let perm: Permuter<15> = Permuter::make_sorted(n);
        prop_assert_eq!(perm.size(), n);
    }

    /// insert_from_back increments size by 1.
    #[test]
    fn insert_increments_size(positions in insert_sequence(14)) {
        let mut perm: Permuter<15> = Permuter::empty();

        for (expected_size, &pos) in positions.iter().enumerate() {
            if expected_size >= 15 {
                break;
            }
            prop_assert_eq!(perm.size(), expected_size);
            let _ = perm.insert_from_back(pos);
            prop_assert_eq!(perm.size(), expected_size + 1);
        }
    }

    /// remove_to_back decrements size by 1.
    #[test]
    fn remove_to_back_decrements_size(initial_size in 1usize..=15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);

        for expected_size in (0..initial_size).rev() {
            perm.remove_to_back(0);
            prop_assert_eq!(perm.size(), expected_size);
        }
    }

    /// remove decrements size by 1.
    #[test]
    fn remove_decrements_size(initial_size in 1usize..=15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);

        for expected_size in (0..initial_size).rev() {
            perm.remove(0);
            prop_assert_eq!(perm.size(), expected_size);
        }
    }

    /// exchange does not change size.
    #[test]
    fn exchange_preserves_size(initial_size in 0usize..=15, i in 0usize..15, j in 0usize..15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let size_before = perm.size();
        perm.exchange(i, j);
        prop_assert_eq!(perm.size(), size_before);
    }

    /// rotate does not change size.
    #[test]
    fn rotate_preserves_size(initial_size in 0usize..=15, i in 0usize..=15, j in 0usize..=15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        let j = j.min(15);
        let size_before = perm.size();
        perm.rotate(i, j);
        prop_assert_eq!(perm.size(), size_before);
    }

    /// set_size should update size correctly.
    #[test]
    fn set_size_works(initial_size in 0usize..=15, new_size in 0usize..=15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        perm.set_size(new_size);
        prop_assert_eq!(perm.size(), new_size);
    }
}

// ============================================================================
//  Insert/Remove Roundtrip Properties
// ============================================================================

proptest! {
    /// insert_from_back followed by remove_to_back at same position restores original state.
    #[test]
    fn insert_remove_to_back_roundtrip(
        initial_size in 0usize..14,
        insert_pos in 0usize..15
    ) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let insert_pos = insert_pos % (initial_size + 1);

        let slot = perm.insert_from_back(insert_pos);
        perm.remove_to_back(insert_pos);

        // Size should be restored
        prop_assert_eq!(perm.size(), initial_size);

        // The slot we inserted should now be at back()
        prop_assert_eq!(perm.back(), slot);
    }

    /// After inserting at position 0, get(0) should return the newly allocated slot.
    #[test]
    fn insert_at_zero_puts_slot_at_position_zero(initial_size in 0usize..15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let slot = perm.insert_from_back(0);
        prop_assert_eq!(perm.get(0), slot);
    }

    /// After inserting at position n (end), get(n) should return the newly allocated slot.
    #[test]
    fn insert_at_end_puts_slot_at_end(initial_size in 0usize..15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let slot = perm.insert_from_back(initial_size);
        prop_assert_eq!(perm.get(initial_size), slot);
    }

    /// Inserting n elements should allocate slots 0, 1, 2, ..., n-1 (in some order).
    #[test]
    fn insert_allocates_sequential_slots(n in 1usize..=15) {
        let mut perm: Permuter<15> = Permuter::empty();
        let mut allocated = Vec::with_capacity(n);

        for i in 0..n {
            let slot = perm.insert_from_back(i);
            allocated.push(slot);
        }

        // All allocated slots should be unique
        allocated.sort();
        for (i, &slot) in allocated.iter().enumerate() {
            prop_assert_eq!(slot, i, "Expected sequential slots");
        }
    }
}

// ============================================================================
//  Exchange Properties
// ============================================================================

proptest! {
    /// exchange(i, j) followed by exchange(i, j) should restore original state.
    #[test]
    fn exchange_is_involution(initial_size in 1usize..=15, i in 0usize..15, j in 0usize..15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let original_value = perm.value();

        perm.exchange(i, j);
        perm.exchange(i, j);

        prop_assert_eq!(perm.value(), original_value);
    }

    /// exchange(i, i) should be a no-op.
    #[test]
    fn exchange_same_is_noop(initial_size in 0usize..=15, i in 0usize..15) {
        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let original_value = perm.value();

        perm.exchange(i, i);

        prop_assert_eq!(perm.value(), original_value);
    }

    /// exchange(i, j) should actually swap the values at positions i and j.
    #[test]
    fn exchange_swaps_values(initial_size in 2usize..=15, i in 0usize..15, j in 0usize..15) {
        let i = i % initial_size;
        let j = j % initial_size;

        let mut perm: Permuter<15> = Permuter::make_sorted(initial_size);
        let val_i_before = perm.get(i);
        let val_j_before = perm.get(j);

        perm.exchange(i, j);

        prop_assert_eq!(perm.get(i), val_j_before);
        prop_assert_eq!(perm.get(j), val_i_before);
    }
}

// ============================================================================
//  make_sorted Properties
// ============================================================================

proptest! {
    /// make_sorted(n) should have position i map to slot i for i in 0..n.
    #[test]
    fn make_sorted_is_identity_mapping(n in 0usize..=15) {
        let perm: Permuter<15> = Permuter::make_sorted(n);

        for i in 0..n {
            prop_assert_eq!(perm.get(i), i, "make_sorted({}) should have get({}) == {}", n, i, i);
        }
    }

    /// make_sorted(n) should have back() return n (next slot to allocate).
    #[test]
    fn make_sorted_back_returns_n(n in 0usize..14) {
        let perm: Permuter<15> = Permuter::make_sorted(n);
        prop_assert_eq!(perm.back(), n);
    }

    /// make_sorted(WIDTH) is a special case - fully sorted.
    #[test]
    fn make_sorted_full_is_fully_sorted(_dummy: u8) {
        let perm: Permuter<15> = Permuter::make_sorted(15);
        prop_assert_eq!(perm.size(), 15);

        for i in 0..15 {
            prop_assert_eq!(perm.get(i), i);
        }
    }
}

// ============================================================================
//  Different WIDTH Tests
// ============================================================================

proptest! {
    /// WIDTH=7 permuter should work correctly.
    #[test]
    fn width_7_basic_operations(n in 0usize..=7) {
        let mut perm: Permuter<7> = Permuter::empty();

        // Insert n elements
        for i in 0..n {
            let _ = perm.insert_from_back(i);
        }

        prop_assert_eq!(perm.size(), n);
        prop_assert!(verify_bijection_generic::<7>(&perm));
    }

    /// WIDTH=3 permuter should work correctly.
    #[test]
    fn width_3_basic_operations(n in 0usize..=3) {
        let mut perm: Permuter<3> = Permuter::empty();

        // Insert n elements
        for i in 0..n {
            let _ = perm.insert_from_back(i);
        }

        prop_assert_eq!(perm.size(), n);
        prop_assert!(verify_bijection_generic::<3>(&perm));
    }

    /// WIDTH=1 permuter should work correctly.
    #[test]
    fn width_1_basic_operations(_dummy: u8) {
        let mut perm: Permuter<1> = Permuter::empty();
        prop_assert_eq!(perm.size(), 0);

        let slot = perm.insert_from_back(0);
        prop_assert_eq!(slot, 0);
        prop_assert_eq!(perm.size(), 1);
        prop_assert_eq!(perm.get(0), 0);

        perm.remove(0);
        prop_assert_eq!(perm.size(), 0);
    }
}

/// Generic bijection verification for any WIDTH.
fn verify_bijection_generic<const WIDTH: usize>(perm: &Permuter<WIDTH>) -> bool {
    let mut seen = [false; MAX_WIDTH];

    for i in 0..WIDTH {
        let slot = perm.get(i);
        if slot >= WIDTH || seen[slot] {
            return false;
        }
        seen[slot] = true;
    }

    seen[..WIDTH].iter().all(|&s| s)
}

// ============================================================================
//  Value/Copy Properties
// ============================================================================

proptest! {
    /// Clone/copy should produce identical permuter.
    #[test]
    fn copy_produces_identical(n in 0usize..=15) {
        let perm1: Permuter<15> = Permuter::make_sorted(n);
        let perm2 = perm1;  // Copy

        prop_assert_eq!(perm1.value(), perm2.value());
        prop_assert_eq!(perm1.size(), perm2.size());

        for i in 0..15 {
            prop_assert_eq!(perm1.get(i), perm2.get(i));
        }
    }

    /// Default should equal empty.
    #[test]
    fn default_equals_empty(_dummy: u8) {
        let perm1: Permuter<15> = Permuter::default();
        let perm2: Permuter<15> = Permuter::empty();

        prop_assert_eq!(perm1.value(), perm2.value());
    }

    /// PartialEq should work correctly.
    #[test]
    fn equality_works(n in 0usize..=15) {
        let perm1: Permuter<15> = Permuter::make_sorted(n);
        let perm2: Permuter<15> = Permuter::make_sorted(n);

        prop_assert_eq!(perm1, perm2);
    }
}

// ============================================================================
//  Stress Tests
// ============================================================================

/// Operation type for stress testing.
#[derive(Debug, Clone)]
enum Op {
    Insert(usize),
    Remove(usize),
    RemoveToBack(usize),
    Exchange(usize, usize),
    Rotate(usize, usize),
}

/// Strategy to generate a random operation.
fn random_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0usize..16).prop_map(Op::Insert),
        (0usize..16).prop_map(Op::Remove),
        (0usize..16).prop_map(Op::RemoveToBack),
        (0usize..15, 0usize..15).prop_map(|(i, j)| Op::Exchange(i, j)),
        (0usize..16, 0usize..16).prop_map(|(i, j)| Op::Rotate(i.min(j), i.max(j).min(15))),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Random sequence of operations should maintain invariants.
    #[test]
    fn random_operations_maintain_invariants(
        ops in prop::collection::vec(random_op(), 1..100)
    ) {
        let mut perm: Permuter<15> = Permuter::empty();

        for op in ops {
            match op {
                Op::Insert(pos) if perm.size() < 15 => {
                    let pos = pos % (perm.size() + 1);
                    let _ = perm.insert_from_back(pos);
                }
                Op::Remove(pos) if perm.size() > 0 => {
                    let pos = pos % perm.size();
                    perm.remove(pos);
                }
                Op::RemoveToBack(pos) if perm.size() > 0 => {
                    let pos = pos % perm.size();
                    perm.remove_to_back(pos);
                }
                Op::Exchange(i, j) => {
                    perm.exchange(i, j);
                }
                Op::Rotate(i, j) => {
                    perm.rotate(i, j);
                }
                _ => {
                    // Skip invalid operations (e.g., remove when empty)
                }
            }

            // Verify invariants after each operation
            prop_assert!(verify_bijection(&perm), "Bijection broken after operation {:?}", op);
            prop_assert!(perm.size() <= 15, "Size exceeded WIDTH");
        }
    }
}
