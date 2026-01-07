use super::*;

// ========================================================================
// !Send/!Sync Verification
// ========================================================================
//
// LockGuard uses PhantomData<*mut ()> to be !Send and !Sync.
// Raw pointers (*mut T, *const T) are neither Send nor Sync in Rust,
// and PhantomData<T> inherits the auto-traits of T.
//
// To verify this works, uncomment the following and observe the compile error:
//
// ```
// fn require_send<T: Send>() {}
// fn require_sync<T: Sync>() {}
//
// fn test_would_fail() {
//     require_send::<LockGuard<'static>>();  // ERROR: LockGuard is !Send
//     require_sync::<LockGuard<'static>>();  // ERROR: LockGuard is !Sync
// }
// ```

#[test]
fn test_new_leaf() {
    let v = NodeVersion::new(true);
    assert!(v.is_leaf());
    assert!(!v.is_root());
    assert!(!v.is_deleted());
    assert!(!v.is_locked());
    assert!(!v.is_dirty());
}

#[test]
fn test_new_internode() {
    let v = NodeVersion::new(false);
    assert!(!v.is_leaf());
    assert!(!v.is_root());
    assert!(!v.is_locked());
}

#[test]
fn test_lock_unlock_roundtrip() {
    let v = NodeVersion::new(true);
    let stable_before: u32 = v.stable();

    {
        let guard: LockGuard<'_> = v.lock();
        assert!(v.is_locked());
        // With "always dirty on lock" strategy, INSERTING_BIT is set automatically
        assert_eq!(guard.locked_value() & LOCK_BIT, LOCK_BIT);
        assert_eq!(guard.locked_value() & INSERTING_BIT, INSERTING_BIT);

        // Guard drops here, releasing lock
    }

    assert!(!v.is_locked());

    // With "always dirty on lock" strategy, version ALWAYS increments on unlock
    // because INSERTING_BIT is set automatically.
    assert!(v.has_changed(stable_before));
}

#[test]
fn test_try_lock() {
    let v = NodeVersion::new(true);

    // First try_lock succeeds
    let guard: Option<LockGuard<'_>> = v.try_lock();
    assert!(guard.is_some());
    assert!(v.is_locked());

    // Second try_lock fails (lock is held)
    let second: Option<LockGuard<'_>> = v.try_lock();
    assert!(second.is_none());

    // Drop guard to release lock
    drop(guard);
    assert!(!v.is_locked());
}

#[test]
fn test_version_increment_on_insert() {
    let v: NodeVersion = NodeVersion::new(true);
    let stable_before: u32 = v.stable();

    {
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_insert();

        assert!(v.is_inserting());
        // Guard drops, lock released, version incremented
    }

    // Version should have changed (insert counter incremented)
    assert!(v.has_changed(stable_before));

    // But no split occurred
    assert!(!v.has_split(stable_before));
}

#[test]
fn test_version_increment_on_split() {
    let v: NodeVersion = NodeVersion::new(true);
    let stable_before: u32 = v.stable();

    {
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_split();

        assert!(v.is_splitting());
        // Guard drops, lock released, version incremented
    }

    // Both changed and split should be true
    assert!(v.has_changed(stable_before));
    assert!(v.has_split(stable_before));
}

#[test]
fn test_version_always_increments_with_auto_dirty() {
    // With "always dirty on lock" strategy, version ALWAYS increments
    // because INSERTING_BIT is set automatically on lock().
    let v: NodeVersion = NodeVersion::new(true);
    let stable_before: u32 = v.stable();

    {
        // Lock sets INSERTING_BIT automatically
        let _guard: LockGuard<'_> = v.lock();
        // INSERTING_BIT is set, so version will increment on drop
    }

    // Version SHOULD have changed (auto-dirty strategy)
    assert!(v.has_changed(stable_before));
}

#[test]
fn test_mark_root() {
    let v = NodeVersion::new(true);
    assert!(!v.is_root());

    v.mark_root();
    assert!(v.is_root());
}

#[test]
fn test_mark_deleted() {
    let v = NodeVersion::new(true);

    {
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_deleted();

        assert!(v.is_deleted());
        assert!(v.is_splitting()); // Deleted also sets splitting
        // Guard drops here
    }

    assert!(v.is_deleted()); // Deleted bit persists
}

#[test]
fn test_mark_nonroot() {
    let v: NodeVersion = NodeVersion::new(true);
    v.mark_root();

    assert!(v.is_root());

    {
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_nonroot();

        assert!(!v.is_root());
        // Guard drops here
    }
}

#[test]
fn test_has_changed_ignores_lock_bit() {
    let v = NodeVersion::new(true);
    let stable: u32 = v.stable();

    {
        let _guard: LockGuard<'_> = v.lock();

        // Even though lock bit changed, has_changed checks for version changes.
        // Since we haven't set dirty bits, the "version" hasn't changed.
        // has_changed returns (old ^ new) > LOCK_BIT
        // If only lock bit changed, XOR = 1, which is NOT > 1, so returns false.
        // This is correct: lock-only change is not a "version change".
        assert!(
            !v.has_changed(stable),
            "lock bit alone should not trigger has_changed"
        );

        // Guard drops here
    }
}

#[test]
fn test_version_counter_wraparound() {
    // Create a version near the insert counter maximum
    let near_max: u32 = ISLEAF_BIT | ((VSPLIT_LOWBIT - VINSERT_LOWBIT) - VINSERT_LOWBIT);
    let v = NodeVersion::from_value(near_max);

    let stable_before: u32 = v.stable();

    {
        // Do an insert - this should increment and potentially overflow into split bits
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_insert();
        // Guard drops here
    }

    // Version should have changed
    assert!(v.has_changed(stable_before));
}

#[test]
fn test_stable_returns_clean_version() {
    let v = NodeVersion::new(true);
    let stable: u32 = v.stable();

    // Stable version should have no dirty bits
    assert_eq!(stable & DIRTY_MASK, 0);
    assert_eq!(stable & LOCK_BIT, 0);
}

#[test]
fn test_flag_combinations() {
    let v = NodeVersion::new(true);
    v.mark_root();

    {
        let mut guard: LockGuard<'_> = v.lock();
        guard.mark_deleted();

        // Check all flags
        assert!(v.is_leaf());
        assert!(v.is_root()); // Root persists through delete
        assert!(v.is_deleted());
        assert!(v.is_locked());
        assert!(v.is_splitting()); // Set by mark_deleted
        // Guard drops here
    }
}

// =======================================================================
// Type-State Pattern Tests
// =======================================================================

#[test]
fn test_guard_unlocks_on_drop() {
    let v = NodeVersion::new(true);

    let guard: LockGuard<'_> = v.lock();
    assert!(v.is_locked());

    drop(guard);
    assert!(!v.is_locked());
}

#[test]
fn test_guard_locked_value() {
    let v = NodeVersion::new(true);
    let initial: u32 = v.value();

    let guard: LockGuard<'_> = v.lock();
    // With "always dirty on lock" strategy, INSERTING_BIT is set automatically
    assert_eq!(guard.locked_value(), initial | LOCK_BIT | INSERTING_BIT);
}

#[test]
fn test_guard_mark_insert_is_idempotent() {
    // With "always dirty on lock" strategy, INSERTING_BIT is already set.
    // mark_insert() should be idempotent (no-op if already set).
    let v: NodeVersion = NodeVersion::new(true);

    let mut guard: LockGuard<'_> = v.lock();
    let initial_locked: u32 = guard.locked_value();

    // INSERTING_BIT is already set by lock()
    assert_ne!(initial_locked & INSERTING_BIT, 0);

    guard.mark_insert();

    // Guard's locked_value should be unchanged (idempotent)
    assert_eq!(guard.locked_value(), initial_locked);
}

// =======================================================================
// Version Wraparound Stress Tests
// =======================================================================

#[test]
fn test_insert_counter_wraparound_stress() {
    // The insert counter is 6 bits (bits 3-8), so it wraps after 64 increments.
    // This test verifies the counter wraps correctly without corrupting other bits.
    let v = NodeVersion::new(true);
    v.mark_root();

    // Do 100 lock/unlock cycles (more than 64 to trigger wraparound)
    for i in 0..100 {
        let stable_before = v.stable();

        {
            let _guard = v.lock();
            // INSERTING_BIT set automatically, version increments on drop
        }

        // Version should always change after unlock
        assert!(
            v.has_changed(stable_before),
            "Version should change after unlock (iteration {i})"
        );

        // Flags should be preserved through wraparound
        assert!(v.is_leaf(), "is_leaf should persist through wraparound");
        assert!(v.is_root(), "is_root should persist through wraparound");
        assert!(!v.is_deleted(), "is_deleted should stay false");
    }
}

#[test]
fn test_split_counter_wraparound() {
    // The split counter is 19 bits (bits 9-27), wrapping after ~500K splits.
    // We can't test full wraparound, but we can verify it increments correctly.
    let v = NodeVersion::new(true);

    let mut last_value = v.stable();

    for _ in 0..10 {
        {
            let mut guard = v.lock();
            guard.mark_split();
        }

        let new_value = v.stable();

        // Split counter should have incremented (bits 9+)
        assert!(
            v.has_split(last_value),
            "has_split should detect split counter change"
        );

        last_value = new_value;
    }
}

#[test]
fn test_has_split_no_compiler_fence() {
    // Test that has_split_no_compiler_fence works correctly (same logic, no fence)
    let v = NodeVersion::new(true);
    let before = v.stable();

    {
        let mut guard = v.lock();
        guard.mark_split();
    }

    // has_split_no_compiler_fence should detect the change
    assert!(v.has_split_no_compiler_fence(before));

    // And should match has_split
    assert_eq!(v.has_split(before), v.has_split_no_compiler_fence(before));
}

// =======================================================================
// SingleThreadedNodeVersion Tests
// =======================================================================

#[test]
fn test_single_threaded_basic() {
    let mut v = SingleThreadedNodeVersion::new(true);

    assert!(v.is_leaf());
    assert!(!v.is_root());
    assert!(!v.is_deleted());

    v.mark_root();
    assert!(v.is_root());

    v.mark_nonroot();
    assert!(!v.is_root());
}

#[test]
fn test_single_threaded_lock_unlock() {
    let mut v = SingleThreadedNodeVersion::new(true);
    let stable_before = v.stable();

    {
        let mut guard = v.lock();
        guard.mark_insert();
        // Guard drops, version increments
    }

    // Version should have changed
    assert!(v.has_changed(stable_before));
}

#[test]
fn test_single_threaded_split() {
    let mut v = SingleThreadedNodeVersion::new(true);
    let stable_before = v.stable();

    {
        let mut guard = v.lock();
        guard.mark_split();
    }

    assert!(v.has_split(stable_before));
}

#[test]
fn test_single_threaded_deleted() {
    let mut v = SingleThreadedNodeVersion::new(true);

    {
        let mut guard = v.lock();
        guard.mark_deleted();
    }

    assert!(v.is_deleted());
}

// =======================================================================
// Help-Along Protocol Tests
// =======================================================================

#[test]
fn test_new_for_split() {
    let source = NodeVersion::new(true);
    let _guard = source.lock();

    let split_version = NodeVersion::new_for_split(&source);

    // Should be locked with splitting bit
    assert!(split_version.is_split_locked());
    assert!(split_version.is_leaf());
    assert!(!split_version.is_root());

    // Should have LOCK_BIT and SPLITTING_BIT set
    let value = split_version.value();
    assert!((value & LOCK_BIT) != 0, "LOCK_BIT should be set");
    assert!((value & SPLITTING_BIT) != 0, "SPLITTING_BIT should be set");
    assert!((value & ISLEAF_BIT) != 0, "ISLEAF_BIT should be preserved");
}

#[test]
fn test_unlock_for_split() {
    let source = NodeVersion::new(true);
    let _guard = source.lock();

    let split_version = NodeVersion::new_for_split(&source);
    assert!(split_version.is_split_locked());

    // Simulate setting parent pointer (would normally happen in propagate_split)

    split_version.unlock_for_split();

    // Should now be unlocked
    assert!(!split_version.is_locked());
    assert!(!split_version.is_splitting());
    assert!(!split_version.is_split_locked());

    // stable() should return immediately (no dirty bits)
    let v = split_version.stable();
    assert!((v & DIRTY_MASK) == 0);
}

#[test]
fn test_split_version_blocks_stable() {
    // This test verifies that a split-locked version blocks stable()
    // until unlock_for_split() is called.

    // Create a split-locked version directly
    let split_version = NodeVersion::from_value(ISLEAF_BIT | LOCK_BIT | SPLITTING_BIT);

    // Verify it has the expected bits set
    assert!(split_version.is_split_locked());
    assert!(split_version.is_dirty());

    // stable() would spin here, so we just verify the dirty check
    let value = split_version.value();
    assert!(
        (value & DIRTY_MASK) != 0,
        "Split-locked version should have dirty bits set"
    );

    // After unlock, stable() should work
    split_version.unlock_for_split();
    let stable = split_version.stable();
    assert!((stable & DIRTY_MASK) == 0);
}

#[test]
fn test_new_for_split_preserves_isleaf() {
    // Test with leaf node
    let leaf_source = NodeVersion::new(true);
    let guard1 = leaf_source.lock();
    let split_leaf = NodeVersion::new_for_split(&leaf_source);
    assert!(split_leaf.is_leaf());
    drop(guard1);

    // Test with internode
    let inode_source = NodeVersion::new(false);
    let _guard2 = inode_source.lock();
    let split_inode = NodeVersion::new_for_split(&inode_source);
    assert!(!split_inode.is_leaf());
}

#[test]
fn test_unlock_for_split_increments_split_counter() {
    let source = NodeVersion::new(true);
    let _guard = source.lock();

    let split_version = NodeVersion::new_for_split(&source);
    let before = split_version.value();

    split_version.unlock_for_split();
    let after = split_version.value();

    // Split counter should have incremented (bits 9+)
    // The split counter is in the upper bits, masked by SPLIT_UNLOCK_MASK
    assert!(
        after != before,
        "Version should change after unlock_for_split"
    );
    assert!(
        (after & DIRTY_MASK) == 0,
        "Dirty bits should be cleared after unlock"
    );
    assert!(
        (after & LOCK_BIT) == 0,
        "Lock bit should be cleared after unlock"
    );
}

// =======================================================================
// Non-Spinning Version Acquisition Tests
// =======================================================================

#[test]
fn test_try_stable_clean_node() {
    let v = NodeVersion::new(true);
    // Clean node should return Some
    let result = v.try_stable();
    assert!(result.is_some());
    // Returned value should have no dirty bits
    assert_eq!(result.unwrap() & DIRTY_MASK, 0);
}

#[test]
fn test_try_stable_equals_stable_on_clean() {
    let v = NodeVersion::new(true);
    // On clean node, try_stable() should return same value as stable()
    let try_result = v.try_stable().unwrap();
    let stable_result = v.stable();
    assert_eq!(try_result, stable_result);
}

#[test]
fn test_acquire_raw_clean_node() {
    let v = NodeVersion::new(true);
    let raw = v.acquire_raw();
    // Clean node should not have dirty bits
    assert!(!NodeVersion::is_dirty_value(raw));
}

#[test]
fn test_is_dirty_value_static() {
    // Test the static helper with various bit patterns
    // DIRTY_MASK = INSERTING_BIT | SPLITTING_BIT (not LOCK_BIT)
    assert!(!NodeVersion::is_dirty_value(0));
    assert!(!NodeVersion::is_dirty_value(ISLEAF_BIT));
    assert!(!NodeVersion::is_dirty_value(LOCK_BIT)); // Lock alone is not "dirty"
    assert!(NodeVersion::is_dirty_value(INSERTING_BIT));
    assert!(NodeVersion::is_dirty_value(SPLITTING_BIT));
    assert!(NodeVersion::is_dirty_value(INSERTING_BIT | SPLITTING_BIT));
    assert!(NodeVersion::is_dirty_value(LOCK_BIT | INSERTING_BIT)); // Dirty due to INSERTING
}

#[test]
fn test_stable_yield_clean_node() {
    let v = NodeVersion::new(true);
    // Clean node should return immediately
    let result = v.stable_yield();
    // Should have no dirty bits
    assert_eq!(result & DIRTY_MASK, 0);
    // Should match stable()
    assert_eq!(result, v.stable());
}

#[test]
fn test_acquire_raw_vs_stable_equivalence_when_clean() {
    // When node is clean, acquire_raw and stable should return same value
    let v = NodeVersion::new(true);
    v.mark_root();

    let raw = v.acquire_raw();
    let stable = v.stable();

    // Both should be clean
    assert!(!NodeVersion::is_dirty_value(raw));
    assert_eq!(raw & DIRTY_MASK, 0);

    // Values should match
    assert_eq!(raw, stable);
}
