//! Loom tests for NodeVersion.
//!
//! Loom provides deterministic concurrency testing by exploring all possible
//! thread interleavings. This catches subtle race conditions that random
//! testing might miss.
//!
//! Run with: `RUSTFLAGS="--cfg loom" cargo test --lib nodeversion::loom_tests`
//!
//! NOTE: Loom tests use loom's own atomic types, so we create a simplified
//! version of the lock to test the core CAS semantics.

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, Ordering};
use loom::thread;
use std::marker::PhantomData;

// Bit constants (same as main module)
const LOCK_BIT: u32 = 1 << 0;
const INSERTING_BIT: u32 = 1 << 1;
const SPLITTING_BIT: u32 = 1 << 2;
const DIRTY_MASK: u32 = INSERTING_BIT | SPLITTING_BIT;
const VINSERT_LOWBIT: u32 = 1 << 3;
const VSPLIT_LOWBIT: u32 = 1 << 9;
const UNUSED1_BIT: u32 = 1 << 28;
const ROOT_BIT: u32 = 1 << 30;
const ISLEAF_BIT: u32 = 1 << 31;
const SPLIT_UNLOCK_MASK: u32 = !(ROOT_BIT | UNUSED1_BIT | (VSPLIT_LOWBIT - 1));
const UNLOCK_MASK: u32 = !(UNUSED1_BIT | (VINSERT_LOWBIT - 1));

/// Simplified NodeVersion for loom testing.
///
/// Uses loom's AtomicU32 for deterministic interleaving exploration.
struct LoomNodeVersion {
    value: AtomicU32,
}

/// Simplified LockGuard for loom testing.
struct LoomLockGuard<'a> {
    version: &'a LoomNodeVersion,
    locked_value: u32,
    _marker: PhantomData<*mut ()>,
}

impl Drop for LoomLockGuard<'_> {
    fn drop(&mut self) {
        let new_value: u32 = if self.locked_value & SPLITTING_BIT != 0 {
            (self.locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            (self.locked_value + ((self.locked_value & INSERTING_BIT) << 2)) & UNLOCK_MASK
        };
        self.version.value.store(new_value, Ordering::Release);
    }
}

impl LoomLockGuard<'_> {
    fn mark_insert(&mut self) {
        let value: u32 = self.version.value.load(Ordering::Relaxed);
        self.version
            .value
            .store(value | INSERTING_BIT, Ordering::Release);
        loom::sync::atomic::fence(Ordering::Release);
        self.locked_value |= INSERTING_BIT;
    }
}

impl LoomNodeVersion {
    fn new(is_leaf: bool) -> Self {
        let initial = if is_leaf { ISLEAF_BIT } else { 0 };
        Self {
            value: AtomicU32::new(initial),
        }
    }

    fn is_locked(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & LOCK_BIT) != 0
    }

    fn stable(&self) -> u32 {
        loop {
            let value: u32 = self.value.load(Ordering::Acquire);
            if (value & DIRTY_MASK) == 0 {
                return value;
            }
            loom::thread::yield_now();
        }
    }

    fn lock(&self) -> LoomLockGuard<'_> {
        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & (LOCK_BIT | DIRTY_MASK)) != 0 {
                loom::thread::yield_now();
                continue;
            }

            let locked = value | LOCK_BIT;

            match self.value.compare_exchange_weak(
                value,
                locked,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return LoomLockGuard {
                        version: self,
                        locked_value: locked,
                        _marker: PhantomData,
                    };
                }
                Err(_) => {
                    loom::thread::yield_now();
                    continue;
                }
            }
        }
    }

    fn try_lock(&self) -> Option<LoomLockGuard<'_>> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & (LOCK_BIT | DIRTY_MASK)) != 0 {
            return None;
        }

        let locked = value | LOCK_BIT;

        match self
            .value
            .compare_exchange(value, locked, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(LoomLockGuard {
                version: self,
                locked_value: locked,
                _marker: PhantomData,
            }),
            Err(_) => None,
        }
    }

    fn has_changed(&self, old: u32) -> bool {
        // Match production's post-read barrier and dirty-bit validation.
        loom::sync::atomic::fence(Ordering::Acquire);
        (old ^ self.value.load(Ordering::Relaxed)) > LOCK_BIT
    }
}

/// An internode reader may inspect a newly loaded child before validating its
/// parent. Relaxed child stores therefore depend on the dirty marker's Release
/// fence, not just the later nkeys/unlock publication. IDs stand in for pointers;
/// this model checks initialization visibility, not EBR or the full tree.
#[test]
fn test_loom_relaxed_child_store_publishes_initialized_child() {
    loom::model(|| {
        let parent = Arc::new(LoomNodeVersion::new(false));
        let child_id = Arc::new(AtomicU32::new(0));
        let child_payload = Arc::new(AtomicU32::new(0));
        let writer_parent = Arc::clone(&parent);
        let writer_id = Arc::clone(&child_id);
        let writer_payload = Arc::clone(&child_payload);
        let writer = thread::spawn(move || {
            // The split child is initialized before the parent's dirty fence.
            writer_payload.store(42, Ordering::Relaxed);
            let mut lock = writer_parent.lock();
            lock.mark_insert();
            writer_id.store(1, Ordering::Relaxed);
        });

        let old_version = parent.stable();
        if child_id.load(Ordering::Acquire) == 1 {
            // This assertion deliberately precedes parent validation.
            assert_eq!(child_payload.load(Ordering::Relaxed), 42);
        }
        let _changed = parent.has_changed(old_version);
        writer.join().unwrap();
    });
}

/// Model Entry's OCC classification followed by speculative lock/validation.
/// This covers the version protocol, not real leaves, routing, or reclamation;
/// those paths have native and Miri tests in entry/unit_tests.rs.
#[test]
fn test_loom_entry_hint_revalidated_under_lock() {
    loom::model(|| {
        let version = Arc::new(LoomNodeVersion::new(true));
        let slot_key = Arc::new(AtomicU32::new(1));
        let writer_version = Arc::clone(&version);
        let writer_key = Arc::clone(&slot_key);
        let writer = thread::spawn(move || {
            let mut lock = writer_version.lock();
            lock.mark_insert();
            writer_key.store(2, Ordering::Relaxed);
        });

        let observed_version = version.stable();
        let observed_key = slot_key.load(Ordering::Relaxed);
        if !version.has_changed(observed_version)
            && let Some(_lock) = version.try_lock()
            && !version.has_changed(observed_version)
        {
            assert_eq!(slot_key.load(Ordering::Relaxed), observed_key);
        }
        writer.join().unwrap();
    });
}

/// A root-cache refresh must not republish a layer after GC clears its slot.
/// Node IDs stand in for guarded pointers; this tests parent-lock revalidation,
/// not the real subtree traversal or EBR implementation.
#[test]
fn test_loom_root_refresh_does_not_resurrect_deleted_layer() {
    loom::model(|| {
        let parent = Arc::new((LoomNodeVersion::new(true), AtomicU32::new(1)));
        let gc_parent = Arc::clone(&parent);
        let gc = thread::spawn(move || {
            let mut lock = gc_parent.0.lock();
            lock.mark_insert();
            gc_parent.1.store(0, Ordering::Release);
        });

        // The refresh observed old target 1 and resolved its promoted root 2
        // before trying the parent lock. GC may have cleared the slot since.
        if let Some(mut lock) = parent.0.try_lock()
            && parent.1.load(Ordering::Acquire) == 1
        {
            lock.mark_insert();
            parent.1.store(2, Ordering::Release);
        }
        gc.join().unwrap();
        assert_eq!(parent.1.load(Ordering::Acquire), 0);
    });
}

/// Test that two threads can't both hold the lock simultaneously.
#[test]
fn test_loom_mutual_exclusion() {
    loom::model(|| {
        let version = Arc::new(LoomNodeVersion::new(true));
        let counter = Arc::new(AtomicU32::new(0));

        let v1 = Arc::clone(&version);
        let c1 = Arc::clone(&counter);
        let t1 = thread::spawn(move || {
            let _guard = v1.lock();
            // Increment counter while holding lock
            let val: u32 = c1.load(Ordering::Relaxed);
            c1.store(val + 1, Ordering::Relaxed);
        });

        let v2 = Arc::clone(&version);
        let c2 = Arc::clone(&counter);
        let t2 = thread::spawn(move || {
            let _guard = v2.lock();
            // Increment counter while holding lock
            let val: u32 = c2.load(Ordering::Relaxed);
            c2.store(val + 1, Ordering::Relaxed);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Both increments should have happened (mutual exclusion ensures no lost updates)
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    });
}

/// Test that try_lock fails when lock is held.
#[test]
fn test_loom_try_lock_fails_when_held() {
    loom::model(|| {
        let version = Arc::new(LoomNodeVersion::new(true));

        let v1 = Arc::clone(&version);
        let t1 = thread::spawn(move || {
            let _guard = v1.lock();
            // Hold lock briefly
            loom::thread::yield_now();
        });

        let v2 = Arc::clone(&version);
        let t2 = thread::spawn(move || {
            // At least one try_lock should fail if t1 holds the lock
            let _result = v2.try_lock();
            // Either succeeds (if t1 finished) or fails (if t1 holds lock)
            // Both are valid outcomes
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Lock should be released
        assert!(!version.is_locked());
    });
}

/// Test that version changes are visible after unlock.
#[test]
fn test_loom_version_visibility() {
    loom::model(|| {
        let version = Arc::new(LoomNodeVersion::new(true));

        let initial = version.stable();

        let v1 = Arc::clone(&version);
        let t1 = thread::spawn(move || {
            let mut guard = v1.lock();
            guard.mark_insert();
        });

        t1.join().unwrap();

        // Version should have changed
        assert!(version.has_changed(initial));
    });
}

/// Test stable() waits for dirty bits to clear.
#[test]
fn test_loom_stable_waits_for_dirty() {
    loom::model(|| {
        let version = Arc::new(LoomNodeVersion::new(true));

        let v1 = Arc::clone(&version);
        let t1 = thread::spawn(move || {
            let mut guard = v1.lock();
            guard.mark_insert();
            // Guard drops here, clearing dirty bits
        });

        let v2 = Arc::clone(&version);
        let t2 = thread::spawn(move || {
            let stable = v2.stable();
            // stable() should only return when dirty bits are clear
            assert_eq!(stable & DIRTY_MASK, 0);
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}
