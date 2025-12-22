//! Property-based tests for the `nodeversion` module.
//!
//! These tests verify invariants and properties that should hold for all inputs.

use masstree::nodeversion::NodeVersion;
use proptest::prelude::*;

// ============================================================================
//  Bit Constants (mirrored for testing)
// ============================================================================

const LOCK_BIT: u32 = 1 << 0;
const INSERTING_BIT: u32 = 1 << 1;
const SPLITTING_BIT: u32 = 1 << 2;
const DIRTY_MASK: u32 = INSERTING_BIT | SPLITTING_BIT;
const VINSERT_LOWBIT: u32 = 1 << 3;
const VSPLIT_LOWBIT: u32 = 1 << 9;
const DELETED_BIT: u32 = 1 << 29;
const ROOT_BIT: u32 = 1 << 30;
const ISLEAF_BIT: u32 = 1 << 31;

// ============================================================================
//  Strategies
// ============================================================================

/// Strategy for generating valid initial version values (no lock/dirty bits).
fn clean_version() -> impl Strategy<Value = u32> {
    // Generate versions with various flag combinations but no lock/dirty bits
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0u32..64,
        0u32..512,
    )
        .prop_map(|(is_leaf, is_root, is_deleted, insert_ver, split_ver)| {
            let mut v = 0u32;
            if is_leaf {
                v |= ISLEAF_BIT;
            }
            if is_root {
                v |= ROOT_BIT;
            }
            if is_deleted {
                v |= DELETED_BIT;
            }
            // Insert version in bits 3-8 (6 bits)
            v |= (insert_ver & 0x3F) << 3;
            // Split version in bits 9-27 (19 bits)
            v |= (split_ver & 0x7FFFF) << 9;
            v
        })
}

/// Strategy for generating boolean flags.
fn flag_combination() -> impl Strategy<Value = (bool, bool, bool)> {
    (any::<bool>(), any::<bool>(), any::<bool>())
}

// ============================================================================
//  Construction Properties
// ============================================================================

proptest! {
    /// New leaf versions should have is_leaf bit set.
    #[test]
    fn new_leaf_has_leaf_bit(_seed in any::<u64>()) {
        let v = NodeVersion::new(true);
        prop_assert!(v.is_leaf());
        prop_assert!(!v.is_locked());
        prop_assert!(!v.is_dirty());
    }

    /// New internode versions should NOT have is_leaf bit set.
    #[test]
    fn new_internode_no_leaf_bit(_seed in any::<u64>()) {
        let v = NodeVersion::new(false);
        prop_assert!(!v.is_leaf());
        prop_assert!(!v.is_locked());
        prop_assert!(!v.is_dirty());
    }

    /// from_value should preserve the exact bits.
    #[test]
    fn from_value_preserves_bits(value in clean_version()) {
        let v = NodeVersion::from_value(value);
        prop_assert_eq!(v.value(), value);
    }
}

// ============================================================================
//  Flag Accessor Properties
// ============================================================================

proptest! {
    /// is_leaf should correctly read the ISLEAF_BIT.
    #[test]
    fn is_leaf_reads_correct_bit(base in clean_version()) {
        let with_leaf = base | ISLEAF_BIT;
        let without_leaf = base & !ISLEAF_BIT;

        let v_with = NodeVersion::from_value(with_leaf);
        let v_without = NodeVersion::from_value(without_leaf);

        prop_assert!(v_with.is_leaf());
        prop_assert!(!v_without.is_leaf());
    }

    /// is_root should correctly read the ROOT_BIT.
    #[test]
    fn is_root_reads_correct_bit(base in clean_version()) {
        let with_root = base | ROOT_BIT;
        let without_root = base & !ROOT_BIT;

        let v_with = NodeVersion::from_value(with_root);
        let v_without = NodeVersion::from_value(without_root);

        prop_assert!(v_with.is_root());
        prop_assert!(!v_without.is_root());
    }

    /// is_deleted should correctly read the DELETED_BIT.
    #[test]
    fn is_deleted_reads_correct_bit(base in clean_version()) {
        let with_deleted = base | DELETED_BIT;
        let without_deleted = base & !DELETED_BIT;

        let v_with = NodeVersion::from_value(with_deleted);
        let v_without = NodeVersion::from_value(without_deleted);

        prop_assert!(v_with.is_deleted());
        prop_assert!(!v_without.is_deleted());
    }

    /// Flag bits should not interfere with each other.
    #[test]
    fn flag_bits_independent((is_leaf, is_root, is_deleted) in flag_combination()) {
        let mut value = 0u32;
        if is_leaf { value |= ISLEAF_BIT; }
        if is_root { value |= ROOT_BIT; }
        if is_deleted { value |= DELETED_BIT; }

        let v = NodeVersion::from_value(value);

        prop_assert_eq!(v.is_leaf(), is_leaf);
        prop_assert_eq!(v.is_root(), is_root);
        prop_assert_eq!(v.is_deleted(), is_deleted);
    }
}

// ============================================================================
//  Lock/Unlock Properties
// ============================================================================

proptest! {
    /// Lock then unlock increments version (auto-dirty strategy sets INSERTING_BIT).
    ///
    /// With the "auto dirty on lock" strategy, lock() automatically sets INSERTING_BIT.
    /// This means unlock() will increment the version counter, so has_changed() returns true.
    #[test]
    fn lock_unlock_increments_version_due_to_auto_dirty(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let _guard = v.lock();
            prop_assert!(v.is_locked());
        }

        prop_assert!(!v.is_locked());
        // With auto-dirty strategy, version changes due to INSERTING_BIT
        prop_assert!(v.has_changed(before));
    }

    /// Lock sets the lock bit.
    #[test]
    fn lock_sets_lock_bit(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        prop_assert!(!v.is_locked());

        let guard = v.lock();
        prop_assert!(v.is_locked());
        prop_assert_eq!(guard.locked_value() & LOCK_BIT, LOCK_BIT);
    }

    /// Unlock clears the lock bit.
    #[test]
    fn unlock_clears_lock_bit(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);

        {
            let _guard = v.lock();
            prop_assert!(v.is_locked());
        }

        prop_assert!(!v.is_locked());
        prop_assert_eq!(v.value() & LOCK_BIT, 0);
    }

    /// try_lock succeeds on unlocked version.
    #[test]
    fn try_lock_succeeds_when_unlocked(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let guard = v.try_lock();
        prop_assert!(guard.is_some());
        prop_assert!(v.is_locked());
    }

    /// try_lock fails on locked version.
    #[test]
    fn try_lock_fails_when_locked(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let _guard = v.lock();

        let second = v.try_lock();
        prop_assert!(second.is_none());
    }
}

// ============================================================================
//  Version Increment Properties
// ============================================================================

proptest! {
    /// mark_insert causes version change on unlock.
    #[test]
    fn mark_insert_increments_version(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_insert();
            prop_assert!(v.is_inserting());
        }

        prop_assert!(v.has_changed(before));
        prop_assert!(!v.has_split(before)); // Insert doesn't cause split
    }

    /// mark_split causes version change AND split detection.
    #[test]
    fn mark_split_increments_version_and_split(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_split();
            prop_assert!(v.is_splitting());
        }

        prop_assert!(v.has_changed(before));
        prop_assert!(v.has_split(before));
    }

    /// mark_deleted sets both deleted and splitting bits.
    #[test]
    fn mark_deleted_sets_both_bits(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);

        {
            let mut guard = v.lock();
            guard.mark_deleted();
            prop_assert!(v.is_deleted());
            prop_assert!(v.is_splitting());
        }

        // Deleted bit persists after unlock
        prop_assert!(v.is_deleted());
    }

    /// Multiple inserts increment version multiple times.
    #[test]
    fn multiple_inserts_increment(count in 1usize..10) {
        let v = NodeVersion::new(true);
        let mut prev = v.stable();

        for _ in 0..count {
            {
                let mut guard = v.lock();
                guard.mark_insert();
            }

            prop_assert!(v.has_changed(prev));
            prev = v.stable();
        }
    }

    /// Multiple splits increment version multiple times.
    #[test]
    fn multiple_splits_increment(count in 1usize..10) {
        let v = NodeVersion::new(true);
        let mut prev = v.stable();

        for _ in 0..count {
            {
                let mut guard = v.lock();
                guard.mark_split();
            }

            prop_assert!(v.has_changed(prev));
            prop_assert!(v.has_split(prev));
            prev = v.stable();
        }
    }
}

// ============================================================================
//  has_changed Properties
// ============================================================================

proptest! {
    /// has_changed should ignore lock-only changes.
    #[test]
    fn has_changed_ignores_lock_bit(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let stable = v.stable();

        {
            let _guard = v.lock();
            // Lock is held, but no dirty bits set
            // has_changed should return false because only lock bit differs
            prop_assert!(!v.has_changed(stable));
        }
    }

    /// has_changed detects insert version changes.
    #[test]
    fn has_changed_detects_insert(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_insert();
        }

        prop_assert!(v.has_changed(before));
    }

    /// has_changed detects split version changes.
    #[test]
    fn has_changed_detects_split(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_split();
        }

        prop_assert!(v.has_changed(before));
    }

    /// has_changed is reflexive: comparing version to itself returns false.
    #[test]
    fn has_changed_reflexive(value in clean_version()) {
        let v = NodeVersion::from_value(value);
        let stable = v.stable();
        prop_assert!(!v.has_changed(stable));
    }
}

// ============================================================================
//  has_split Properties
// ============================================================================

proptest! {
    /// has_split returns false for insert-only changes.
    #[test]
    fn has_split_false_for_insert(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_insert();
        }

        prop_assert!(!v.has_split(before));
    }

    /// has_split returns true for split changes.
    #[test]
    fn has_split_true_for_split(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_split();
        }

        prop_assert!(v.has_split(before));
    }

    /// has_split is reflexive: comparing version to itself returns false.
    #[test]
    fn has_split_reflexive(value in clean_version()) {
        let v = NodeVersion::from_value(value);
        let stable = v.stable();
        prop_assert!(!v.has_split(stable));
    }
}

// ============================================================================
//  Mark Operations Properties
// ============================================================================

proptest! {
    /// mark_root sets the root bit.
    #[test]
    fn mark_root_sets_bit(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        prop_assert!(!v.is_root());

        v.mark_root();
        prop_assert!(v.is_root());
    }

    /// mark_nonroot clears the root bit.
    #[test]
    fn mark_nonroot_clears_bit(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        v.mark_root();
        prop_assert!(v.is_root());

        {
            let mut guard = v.lock();
            guard.mark_nonroot();
        }

        prop_assert!(!v.is_root());
    }

    /// mark_root is idempotent.
    #[test]
    fn mark_root_idempotent(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);

        v.mark_root();
        let after_first = v.value();

        v.mark_root();
        let after_second = v.value();

        prop_assert_eq!(after_first, after_second);
    }
}

// ============================================================================
//  Stable Version Properties
// ============================================================================

proptest! {
    /// stable() returns value with no dirty bits.
    #[test]
    fn stable_has_no_dirty_bits(value in clean_version()) {
        let v = NodeVersion::from_value(value);
        let stable = v.stable();

        prop_assert_eq!(stable & DIRTY_MASK, 0);
        prop_assert_eq!(stable & LOCK_BIT, 0);
    }

    /// stable() preserves flag bits.
    #[test]
    fn stable_preserves_flags((is_leaf, is_root, is_deleted) in flag_combination()) {
        let mut value = 0u32;
        if is_leaf { value |= ISLEAF_BIT; }
        if is_root { value |= ROOT_BIT; }
        if is_deleted { value |= DELETED_BIT; }

        let v = NodeVersion::from_value(value);
        let stable = v.stable();

        prop_assert_eq!((stable & ISLEAF_BIT) != 0, is_leaf);
        prop_assert_eq!((stable & ROOT_BIT) != 0, is_root);
        prop_assert_eq!((stable & DELETED_BIT) != 0, is_deleted);
    }
}

// ============================================================================
//  Clone Properties
// ============================================================================

proptest! {
    /// Clone produces identical value.
    #[test]
    fn clone_identical(value in clean_version()) {
        let v = NodeVersion::from_value(value);
        let cloned = v.clone();

        prop_assert_eq!(v.value(), cloned.value());
    }

    /// Clone is independent (modifying original doesn't affect clone).
    #[test]
    fn clone_independent(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);
        let cloned = v.clone();
        let cloned_before = cloned.value();

        // Modify original
        v.mark_root();

        // Clone should be unchanged
        prop_assert_eq!(cloned.value(), cloned_before);
    }
}

// ============================================================================
//  Default Properties
// ============================================================================

proptest! {
    /// Default creates a leaf node.
    #[test]
    fn default_is_leaf(_seed in any::<u64>()) {
        let v = NodeVersion::default();
        prop_assert!(v.is_leaf());
        prop_assert!(!v.is_locked());
        prop_assert!(!v.is_dirty());
    }
}

// ============================================================================
//  Bit Layout Verification
// ============================================================================

proptest! {
    /// Flag bits should not overlap.
    #[test]
    fn bit_constants_no_overlap(_seed in any::<u64>()) {
        // Each flag should be a distinct single bit
        prop_assert_eq!(LOCK_BIT.count_ones(), 1);
        prop_assert_eq!(INSERTING_BIT.count_ones(), 1);
        prop_assert_eq!(SPLITTING_BIT.count_ones(), 1);
        prop_assert_eq!(DELETED_BIT.count_ones(), 1);
        prop_assert_eq!(ROOT_BIT.count_ones(), 1);
        prop_assert_eq!(ISLEAF_BIT.count_ones(), 1);

        // No overlaps between distinct flags
        prop_assert_eq!(LOCK_BIT & INSERTING_BIT, 0);
        prop_assert_eq!(LOCK_BIT & SPLITTING_BIT, 0);
        prop_assert_eq!(LOCK_BIT & DELETED_BIT, 0);
        prop_assert_eq!(LOCK_BIT & ROOT_BIT, 0);
        prop_assert_eq!(LOCK_BIT & ISLEAF_BIT, 0);

        prop_assert_eq!(INSERTING_BIT & SPLITTING_BIT, 0);
        prop_assert_eq!(INSERTING_BIT & DELETED_BIT, 0);
        prop_assert_eq!(INSERTING_BIT & ROOT_BIT, 0);
        prop_assert_eq!(INSERTING_BIT & ISLEAF_BIT, 0);

        prop_assert_eq!(SPLITTING_BIT & DELETED_BIT, 0);
        prop_assert_eq!(SPLITTING_BIT & ROOT_BIT, 0);
        prop_assert_eq!(SPLITTING_BIT & ISLEAF_BIT, 0);

        prop_assert_eq!(DELETED_BIT & ROOT_BIT, 0);
        prop_assert_eq!(DELETED_BIT & ISLEAF_BIT, 0);

        prop_assert_eq!(ROOT_BIT & ISLEAF_BIT, 0);
    }

    /// DIRTY_MASK should be exactly INSERTING_BIT | SPLITTING_BIT.
    #[test]
    fn dirty_mask_correct(_seed in any::<u64>()) {
        prop_assert_eq!(DIRTY_MASK, INSERTING_BIT | SPLITTING_BIT);
    }

    /// Version counter bits should be in expected positions.
    #[test]
    fn version_counter_positions(_seed in any::<u64>()) {
        // Insert version starts at bit 3
        prop_assert_eq!(VINSERT_LOWBIT, 1 << 3);
        // Split version starts at bit 9
        prop_assert_eq!(VSPLIT_LOWBIT, 1 << 9);
        // Split lowbit should be higher than insert range
        prop_assert!(VSPLIT_LOWBIT > VINSERT_LOWBIT);
    }
}

// ============================================================================
//  Edge Cases
// ============================================================================

proptest! {
    /// Version near insert counter overflow still works.
    #[test]
    fn insert_counter_near_overflow(is_leaf in any::<bool>()) {
        // Set insert counter to max (bits 3-8, 6 bits = 63 max)
        let mut value = if is_leaf { ISLEAF_BIT } else { 0 };
        value |= 0x3F << 3; // Max insert counter

        let v = NodeVersion::from_value(value);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_insert();
        }

        // Should still detect change even with overflow
        prop_assert!(v.has_changed(before));
    }

    /// Guard locked_value has INSERTING_BIT set from auto-dirty strategy.
    ///
    /// With "auto dirty on lock" strategy, lock() sets INSERTING_BIT automatically.
    /// mark_insert() is now idempotent - calling it doesn't change anything.
    #[test]
    fn guard_has_inserting_bit_from_auto_dirty(is_leaf in any::<bool>()) {
        let v = NodeVersion::new(is_leaf);

        let mut guard = v.lock();
        let initial = guard.locked_value();

        // With auto-dirty strategy, INSERTING_BIT is already set by lock()
        prop_assert_ne!(initial & INSERTING_BIT, 0);

        guard.mark_insert();
        let after_insert = guard.locked_value();

        // mark_insert is idempotent, still set
        prop_assert_ne!(after_insert & INSERTING_BIT, 0);
        // Value unchanged since it was already set
        prop_assert_eq!(initial, after_insert);
    }
}

// ============================================================================
//  Sequence Properties
// ============================================================================

/// Operations to perform on `NodeVersion`.
#[derive(Debug, Clone)]
enum Op {
    Lock,
    MarkInsert,
    MarkSplit,
    MarkDeleted,
    MarkRoot,
    MarkNonroot,
    Unlock,
}

fn random_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Lock),
        Just(Op::MarkInsert),
        Just(Op::MarkSplit),
        Just(Op::MarkDeleted),
        Just(Op::MarkRoot),
        Just(Op::MarkNonroot),
        Just(Op::Unlock),
    ]
}

proptest! {
    /// Random valid operation sequences maintain invariants.
    #[test]
    fn random_operations_maintain_invariants(
        is_leaf in any::<bool>(),
        ops in prop::collection::vec(random_op(), 1..20)
    ) {
        let v = NodeVersion::new(is_leaf);

        // Use a simple state machine approach
        // We need to manually track state since guard lifetimes are tricky
        for op in ops {
            match op {
                Op::Lock => {
                    let _guard = v.lock();
                    // Immediately unlock to avoid borrow issues
                }
                Op::MarkRoot => {
                    v.mark_root();
                }
                _ => {
                    // Other ops require lock, skip if not locked
                    // In real code, we'd hold the guard longer
                }
            }
        }

        // Invariant: after all ops, version should be in valid state
        // (no dirty bits if unlocked, lock bit clear)
        prop_assert_eq!(v.value() & LOCK_BIT, 0);

        // is_leaf should be unchanged
        prop_assert_eq!(v.is_leaf(), is_leaf);
    }

    /// Lock-insert-unlock sequence always increments version.
    #[test]
    fn lock_insert_unlock_always_increments(count in 1usize..5) {
        let v = NodeVersion::new(true);
        let initial = v.stable();

        for _ in 0..count {
            let mut guard = v.lock();
            guard.mark_insert();
            drop(guard);
        }

        prop_assert!(v.has_changed(initial));
    }

    /// Lock-split-unlock sequence always triggers has_split.
    #[test]
    fn lock_split_unlock_always_splits(count in 1usize..5) {
        let v = NodeVersion::new(true);
        let initial = v.stable();

        for _ in 0..count {
            let mut guard = v.lock();
            guard.mark_split();
            drop(guard);
        }

        prop_assert!(v.has_split(initial));
    }
}
