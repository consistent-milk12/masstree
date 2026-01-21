use super::{
    KeyIndexPosition, Permuter, lower_bound_by, lower_bound_linear_by, upper_bound_by,
    upper_bound_linear_by,
};

// Helper to create a sorted permutation
fn sorted_perm<const W: usize>(size: usize) -> Permuter<W> {
    Permuter::make_sorted(size)
}

// ========================================================================
//  KeyIndexPosition Tests
// ========================================================================

#[test]
fn test_key_index_position_found() {
    let pos = KeyIndexPosition::found(3, 5);

    assert!(pos.is_found());
    assert_eq!(pos.i, 3);
    assert_eq!(pos.p, 5);
    assert_eq!(pos.slot(), 5);
    assert_eq!(pos.try_slot(), Some(5));
}

#[test]
fn test_key_index_position_not_found() {
    let pos = KeyIndexPosition::not_found(7);

    assert!(!pos.is_found());
    assert_eq!(pos.i, 7);
    assert_eq!(pos.p, KeyIndexPosition::NOT_FOUND);
    assert_eq!(pos.try_slot(), None);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "slot() called on not-found")]
fn test_key_index_position_slot_panics() {
    let pos = KeyIndexPosition::not_found(0);
    let _ = pos.slot();
}

// ========================================================================
//  Generic Binary Search Tests
// ========================================================================

#[test]
fn test_lower_bound_empty() {
    let perm: Permuter<15> = sorted_perm(0);
    let keys: [u64; 0] = [];

    let pos: KeyIndexPosition = lower_bound_by(0, perm, |slot| 100u64.cmp(&keys[slot]));

    assert!(!pos.is_found());
    assert_eq!(pos.i, 0);
}

#[test]
fn test_lower_bound_single_less() {
    let perm: Permuter<15> = sorted_perm(1);
    let keys: [u64; 1] = [50];

    let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 25u64.cmp(&keys[slot]));

    assert!(!pos.is_found());
    assert_eq!(pos.i, 0); // Insert before the only key
}

#[test]
fn test_lower_bound_single_equal() {
    let perm: Permuter<15> = sorted_perm(1);
    let keys: [u64; 1] = [50];

    let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 50u64.cmp(&keys[slot]));

    assert!(pos.is_found());
    assert_eq!(pos.i, 0);
    assert_eq!(pos.slot(), 0);
}

#[test]
fn test_lower_bound_single_greater() {
    let perm: Permuter<15> = sorted_perm(1);
    let keys: [u64; 1] = [50];

    let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 75u64.cmp(&keys[slot]));

    assert!(!pos.is_found());
    assert_eq!(pos.i, 1); // Insert after the only key
}

#[test]
fn test_lower_bound_multiple_exact_match() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    // Find middle element
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot| 30u64.cmp(&keys[slot]));

    assert!(pos.is_found());
    assert_eq!(pos.i, 2);
    assert_eq!(pos.slot(), 2);
}

#[test]
fn test_lower_bound_multiple_not_found() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    // Search for value between 20 and 30
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot| 25u64.cmp(&keys[slot]));

    assert!(!pos.is_found());
    assert_eq!(pos.i, 2); // Would insert at position 2 (before 30)
}

#[test]
fn test_upper_bound_empty() {
    let perm: Permuter<15> = sorted_perm(0);
    let keys: [u64; 0] = [];

    let idx: usize = upper_bound_by(0, perm, |slot| 100u64.cmp(&keys[slot]));

    assert_eq!(idx, 0);
}

#[test]
fn test_upper_bound_exact_match() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    // Exact match at index 2 (key 30) returns 3 (right child)
    let idx: usize = upper_bound_by(5, perm, |slot| 30u64.cmp(&keys[slot]));

    assert_eq!(idx, 3);
}

#[test]
fn test_upper_bound_between_keys() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    // Search for 25 (between 20 and 30) returns 2
    let idx: usize = upper_bound_by(5, perm, |slot| 25u64.cmp(&keys[slot]));

    assert_eq!(idx, 2);
}

#[test]
fn test_upper_bound_less_than_all() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    let idx: usize = upper_bound_by(5, perm, |slot| 5u64.cmp(&keys[slot]));

    assert_eq!(idx, 0); // Route to leftmost child
}

#[test]
fn test_upper_bound_greater_than_all() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    let idx: usize = upper_bound_by(5, perm, |slot| 100u64.cmp(&keys[slot]));

    assert_eq!(idx, 5); // Route to rightmost child
}

// ========================================================================
//  Linear Search Tests (verify same results as binary)
// ========================================================================

#[test]
fn test_linear_matches_binary() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    for search in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] {
        let binary: KeyIndexPosition = lower_bound_by(5, perm, |slot| search.cmp(&keys[slot]));
        let linear: KeyIndexPosition =
            lower_bound_linear_by(5, perm, |slot| search.cmp(&keys[slot]));

        assert_eq!(
            binary, linear,
            "Mismatch for search key {search}: binary={binary:?}, linear={linear:?}"
        );
    }
}

#[test]
fn test_linear_upper_bound_matches_binary() {
    let perm: Permuter<15> = sorted_perm(5);
    let keys: [u64; 5] = [10, 20, 30, 40, 50];

    for search in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] {
        let binary: usize = upper_bound_by(5, perm, |slot: usize| search.cmp(&keys[slot]));
        let linear: usize = upper_bound_linear_by(5, perm, |slot: usize| search.cmp(&keys[slot]));

        assert_eq!(
            binary, linear,
            "Upper bound mismatch for search key {search}: binary={binary}, linear={linear}"
        );
    }
}

// ========================================================================
//  Permutation-Aware Tests
// ========================================================================

#[test]
fn test_lower_bound_with_permutation() {
    // Test with make_sorted which creates an identity permutation.
    // Physical slots: [10, 20, 30, 40, 50] (already sorted)
    // Permutation: identity [0, 1, 2, 3, 4]
    // Logical order: [10, 20, 30, 40, 50]

    let keys: [u64; 5] = [10, 20, 30, 40, 50]; // Physical order (sorted)

    // Create sorted permutation
    let perm: Permuter<15> = Permuter::make_sorted(5);

    // Search for 30: logical position 2, physical slot 2
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 30u64.cmp(&keys[slot]));

    assert!(pos.is_found());
    assert_eq!(pos.i, 2); // Logical position
    assert_eq!(pos.slot(), 2); // Physical slot (same as logical for identity permutation)

    // Search for 10: logical position 0, physical slot 0
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 10u64.cmp(&keys[slot]));
    assert!(pos.is_found());
    assert_eq!(pos.i, 0);
    assert_eq!(pos.slot(), 0);

    // Search for 50: logical position 4, physical slot 4
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 50u64.cmp(&keys[slot]));
    assert!(pos.is_found());
    assert_eq!(pos.i, 4);
    assert_eq!(pos.slot(), 4);

    // Search for non-existent key (25): should be not found, insert at position 2
    let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 25u64.cmp(&keys[slot]));
    assert!(!pos.is_found());
    assert_eq!(pos.i, 2); // Would insert before 30
}
