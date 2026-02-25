//! Filepath: src/nodeversion.rs
//!
//! Node version for optimistic concurrency control.
//!
//! [`NodeVersion`] combines lock state, version counters, and metadata flags
//! in a single `u32`. Readers use optimistic validation, writers acquire locks.
//!
//! # Synchronization
//!
//! - **OCC (this module)**: version counters detect structural changes;
//!   dirty flags signal in-flight modifications.
//!   Readers: `stable()` -> read -> `has_changed()`.
//!   Writers: `lock()` -> `mark_insert()` -> modify -> drop guard.
//!
//! - **Permutation (`leaf15.rs`)**: Release store is the linearization point
//!   for new slots; readers Acquire-load permutation to see slot writes.
//!
//! # Type-State Pattern
//!
//! [`LockGuard`] provides compile-time proof that the lock is held.
//! Auto-unlocks on drop (panic-safe).

use std::hint as StdHint;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic as StdAtomic;
use std::sync::atomic::{AtomicU32, Ordering, fence};
use std::thread as StdThread;
use std::time::{Duration, Instant};

mod bounded;

// ============================================================================
//  Bit Constants
// ============================================================================

const LOCK_BIT: u32 = 1 << 0;
const INSERTING_BIT: u32 = 1 << 1;
const SPLITTING_BIT: u32 = 1 << 2;
const DIRTY_MASK: u32 = INSERTING_BIT | SPLITTING_BIT;
const VINSERT_LOWBIT: u32 = 1 << 3;
const VSPLIT_LOWBIT: u32 = 1 << 9;
const UNUSED1_BIT: u32 = 1 << 28;
const DELETED_BIT: u32 = 1 << 29;
const ROOT_BIT: u32 = 1 << 30;
const ISLEAF_BIT: u32 = 1 << 31;

/// Mask for unlock after split: clears root, unused, and version bits below vsplit.
const SPLIT_UNLOCK_MASK: u32 = !(ROOT_BIT | UNUSED1_BIT | (VSPLIT_LOWBIT - 1));

/// Mask for unlock after insert: clears unused and version bits below vinsert.
const UNLOCK_MASK: u32 = !(UNUSED1_BIT | (VINSERT_LOWBIT - 1));

// ============================================================================
//  Backoff (for spin loops)
// ============================================================================

/// Exponential backoff
struct Backoff {
    count: u32,
}

impl Backoff {
    #[inline(always)]
    const fn new() -> Self {
        Self { count: 0 }
    }

    /// Spin for `count+1` iterations with CPU pause hints, then double count.
    #[inline(always)]
    fn spin(&mut self) {
        for _ in 0..=self.count {
            StdHint::spin_loop();
        }

        StdAtomic::compiler_fence(Ordering::SeqCst);

        // Double count, cap at 15: 0 -> 1 -> 3 -> 7 -> 15 -> 15
        self.count = ((self.count << 1) | 1) & 15;
    }
}

// ============================================================================
//  NodeVersion
// ============================================================================

/// A versioned lock for tree nodes.
///
/// # Layout
/// Bit 31: `is_leaf` | Bit 30: `root` | Bit 29: `deleted` | BITS 9-27: `split_version`
/// Bits 3-8: `insert_version` | Bit 2: `splitting` | Bit 1: `inserting` | Bit 0: `locked`
#[derive(Debug)]
pub struct NodeVersion {
    value: AtomicU32,
}

// ============================================================================
//  LockGuard (Type-State Pattern)
// ============================================================================

/// Zero-sized proof that a lock is held.
///
/// Cannot be constructed except by [`NodeVersion::lock()`]. The lock is
/// automatically released on drop, even during unwinding.
///
/// `!Send + !Sync` via `PhantomData<*mut ()>`.
#[derive(Debug)]
#[must_use = "releasing a lock without using the guard is a logic error"]
pub struct LockGuard<'a> {
    version: *const NodeVersion,
    locked_value: u32,

    _lifetime: PhantomData<&'a NodeVersion>,
    _marker: PhantomData<*mut ()>,
}

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        // Version counter increment depends on dirty bits:
        // - Splitting: increment vsplit, clear all dirty/lock bits
        // - Inserting: increment vinsert, clear inserting/lock bits
        // - Neither (validation-only lock): no version increment
        let new_value: u32 = if self.locked_value & SPLITTING_BIT != 0 {
            (self.locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            // (inserting << 2) == vinsert_lowbit when INSERTING_BIT is set, 0 otherwise.
            (self.locked_value + ((self.locked_value & INSERTING_BIT) << 2)) & UNLOCK_MASK
        };

        // SAFETY: Guard lifetime tied to NodeVersion; nodes freed via deferred reclamation.
        unsafe { (*self.version).value.store(new_value, Ordering::Release) };
    }
}

impl LockGuard<'_> {
    #[inline(always)]
    const fn version(&self) -> &NodeVersion {
        // SAFETY: Pointer valid for guard's lifetime (PhantomData<&'a NodeVersion>).
        // Nodes freed via deferred reclamation, never while locked.
        unsafe { &*self.version }
    }

    /// Get the locked version value.
    #[must_use]
    #[inline(always)]
    pub const fn locked_value(&self) -> u32 {
        self.locked_value
    }

    /// Mark node as being inserted into. Sets INSERTING dirty bit.
    /// Version counter increments on unlock. Idempotent.
    ///
    /// ORDERING: Release store + Acquire fence ensures readers see the dirty bit
    /// before any subsequent modifications become visible.
    #[inline(always)]
    pub fn mark_insert(&mut self) {
        if (self.locked_value & INSERTING_BIT) != 0 {
            return;
        }

        let value: u32 = self.version().value.load(Ordering::Relaxed);
        self.version()
            .value
            .store(value | INSERTING_BIT, Ordering::Release);
        fence(Ordering::Acquire);
        self.locked_value |= INSERTING_BIT;
    }

    /// Mark node as being split. Sets SPLITTING dirty bit.
    /// Split version counter increments on unlock.
    ///
    /// Must be called explicitly (not all inserts cause splits).
    #[inline(always)]
    pub fn mark_split(&mut self) {
        let value: u32 = self.version().value.load(Ordering::Relaxed);
        self.version()
            .value
            .store(value | SPLITTING_BIT, Ordering::Release);
        fence(Ordering::Acquire);
        self.locked_value |= SPLITTING_BIT;
    }

    /// Mark node as deleted. Also sets `SPLITTING_BIT` to bump version on unlock.
    #[inline(always)]
    pub fn mark_deleted(&mut self) {
        let value: u32 = self.version().value.load(Ordering::Relaxed);
        let new_value: u32 = value | DELETED_BIT | SPLITTING_BIT;
        self.version().value.store(new_value, Ordering::Release);
        fence(Ordering::Acquire);
        self.locked_value = new_value;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub fn mark_nonroot(&mut self) {
        let value: u32 = self.version().value.load(Ordering::Relaxed);
        self.version()
            .value
            .store(value & !ROOT_BIT, Ordering::Release);
        self.locked_value &= !ROOT_BIT;
    }
}

impl NodeVersion {
    /// Create a new node version.
    #[must_use]
    #[inline(always)]
    pub const fn new(is_leaf: bool) -> Self {
        let initial: u32 = if is_leaf { ISLEAF_BIT } else { 0 };
        Self {
            value: AtomicU32::new(initial),
        }
    }

    /// Create from raw value. **Testing only.**
    #[must_use]
    #[inline(always)]
    pub const fn from_value(value: u32) -> Self {
        Self {
            value: AtomicU32::new(value),
        }
    }

    // ========================================================================
    //  Flag Accessors
    // ========================================================================

    /// Check if this is a leaf node.
    #[must_use]
    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ISLEAF_BIT) != 0
    }

    /// Check if this is a root node.
    #[must_use]
    #[inline(always)]
    pub fn is_root(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ROOT_BIT) != 0
    }

    /// Check if this node is logically deleted.
    #[must_use]
    #[inline(always)]
    pub fn is_deleted(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DELETED_BIT) != 0
    }

    /// Static check on an already-loaded version value.
    #[must_use]
    #[inline(always)]
    pub const fn is_deleted_version(version: u32) -> bool {
        (version & DELETED_BIT) != 0
    }

    /// Check if this node is locked.
    #[must_use]
    #[inline(always)]
    pub fn is_locked(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & LOCK_BIT) != 0
    }

    /// True if version counters are zero (freshly allocated, not yet linked into tree).
    #[must_use]
    #[inline(always)]
    pub fn is_unpublished(&self) -> bool {
        let v = self.value.load(Ordering::Relaxed);
        let mask = ISLEAF_BIT | ROOT_BIT;
        (v & !mask) == 0
    }

    /// Check if this node is being inserted into.
    #[must_use]
    #[inline(always)]
    pub fn is_inserting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & INSERTING_BIT) != 0
    }

    /// Check if this node is being split.
    #[must_use]
    #[inline(always)]
    pub fn is_splitting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & SPLITTING_BIT) != 0
    }

    /// Check if any dirty bit set (inserting or splitting).
    #[must_use]
    #[inline(always)]
    pub fn is_dirty(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DIRTY_MASK) != 0
    }

    /// Get the raw version value.
    #[must_use]
    #[inline(always)]
    pub fn value(&self) -> u32 {
        self.value.load(Ordering::Relaxed)
    }

    /// Raw pointer for APIs that lock nodes via pointers (e.g., `PropagationContext`).
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const Self {
        StdPtr::from_ref(self)
    }

    // ========================================================================
    //  Stable Version (for optimistic reads)
    // ========================================================================

    /// Get a stable version for optimistic reading.
    ///
    /// Spins while dirty bits are set, then returns a clean version.
    /// Validate with [`has_changed()`](Self::has_changed) after reading.
    ///
    /// Uses Relaxed loads during spinning, Acquire fence on success.
    #[must_use]
    #[inline(always)]
    pub fn stable(&self) -> u32 {
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & DIRTY_MASK) == 0 {
            StdAtomic::fence(Ordering::Acquire);
            return value;
        }

        self.stable_contended()
    }

    /// Slow path: hybrid spin + yield to reduce convoy effects.
    #[cold]
    #[inline(never)]
    fn stable_contended(&self) -> u32 {
        const SPINS_BEFORE_YIELD: u32 = 2;

        let mut backoff = Backoff::new();
        let mut spin_count: u32 = 0;

        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & DIRTY_MASK) == 0 {
                StdAtomic::fence(Ordering::Acquire);
                return value;
            }

            spin_count += 1;

            if spin_count < SPINS_BEFORE_YIELD {
                backoff.spin();
            } else {
                StdThread::yield_now();
                spin_count = 0;
                backoff = Backoff::new();
            }
        }
    }

    /// Load version without spinning on dirty bits.
    ///
    /// Returns immediately even if dirty. Use when you want to detect
    /// concurrent modification and retry at a higher level.
    #[must_use]
    #[inline(always)]
    pub fn acquire_raw(&self) -> u32 {
        self.value.load(Ordering::Acquire)
    }

    /// Static check: does this version value have dirty bits set?
    #[must_use]
    #[inline(always)]
    pub const fn is_dirty_value(v: u32) -> bool {
        (v & DIRTY_MASK) != 0
    }

    /// Non-spinning stable: returns `Some(version)` if clean, `None` if dirty.
    #[must_use]
    #[inline(always)]
    pub fn try_stable(&self) -> Option<u32> {
        let value: u32 = self.value.load(Ordering::Acquire);

        if (value & DIRTY_MASK) == 0 {
            Some(value)
        } else {
            None
        }
    }

    /// Variant of [`stable()`](Self::stable) that yields instead of exponential backoff.
    #[must_use]
    pub fn stable_yield(&self) -> u32 {
        const SPINS_BEFORE_YIELD: u32 = 2;
        let mut spin_count: u32 = 0;

        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & DIRTY_MASK) == 0 {
                StdAtomic::fence(Ordering::Acquire);
                return value;
            }

            spin_count += 1;

            if spin_count < SPINS_BEFORE_YIELD {
                for _ in 0..spin_count {
                    StdHint::spin_loop();
                }
            } else {
                StdThread::yield_now();
                spin_count = 0;
            }
        }
    }

    /// Check if the version has changed since `old`.
    ///
    /// Returns true if any version counter bits changed (ignoring lock/inserting bits).
    ///
    /// # C++ Divergence (INTENTIONAL)
    ///
    /// C++ uses `(x.v_ ^ v_) > lock_bit` (ignores bit 0).
    /// We use `> (LOCK_BIT | INSERTING_BIT)` (ignores bits 0-1).
    ///
    /// This is safe because version COUNTERS (VINSERT/VSPLIT) are the source of truth.
    /// `INSERTING_BIT` is only a progress indicator set while modifications are in-flight
    /// (not yet visible). If writer acquires lock after `stable()` but before this check,
    /// no data has been modified yet so returning false is correct. If writer has released,
    /// the version counter increment is detected.
    ///
    /// # Compiler Fence
    ///
    /// Critical: prevents compiler from reordering field reads past the version check.
    #[must_use]
    #[inline(always)]
    pub fn has_changed(&self, old: u32) -> bool {
        StdAtomic::compiler_fence(Ordering::Acquire);
        (old ^ self.value.load(Ordering::Relaxed)) > (LOCK_BIT | INSERTING_BIT)
    }

    /// Check if a split has occurred since `old`.
    ///
    /// Includes compiler fence. See [`has_changed()`](Self::has_changed).
    #[must_use]
    #[inline(always)]
    pub fn has_split(&self, old: u32) -> bool {
        StdAtomic::compiler_fence(Ordering::Acquire);
        (old ^ self.value.load(Ordering::Relaxed)) >= VSPLIT_LOWBIT
    }

    /// Like [`has_split()`](Self::has_split) but without compiler fence.
    /// Caller must ensure prior reads are already ordered (e.g., after Acquire load).
    #[must_use]
    #[inline(always)]
    pub fn has_split_no_compiler_fence(&self, old: u32) -> bool {
        (old ^ self.value.load(Ordering::Acquire)) >= VSPLIT_LOWBIT
    }

    /// Stronger than [`has_changed()`](Self::has_changed): also returns true if
    /// dirty bits are set (modification in progress).
    ///
    /// Used by CAS inserts to avoid racing with locked splits.
    #[must_use]
    #[inline(always)]
    pub fn has_changed_or_locked(&self, old: u32) -> bool {
        StdAtomic::compiler_fence(Ordering::Acquire);

        let current: u32 = self.value.load(Ordering::Acquire);

        if (old ^ current) > (LOCK_BIT | INSERTING_BIT) {
            return true;
        }

        (current & DIRTY_MASK) != 0
    }

    // ========================================================================
    //  Lock Operations
    // ========================================================================

    /// Acquire the lock with exponential backoff. Returns a guard.
    ///
    /// Only waits for `LOCK_BIT` (not dirty bits) - caller validates version after.
    /// Only sets `LOCK_BIT`, `INSERTING_BIT` deferred to `mark_insert()` to avoid
    /// making `stable()` callers spin for the entire lock duration.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    #[inline(always)]
    pub fn lock(&self) -> LockGuard<'_> {
        // Fast path: try immediate acquire.
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & LOCK_BIT) == 0 {
            let locked: u32 = value | LOCK_BIT;

            if self
                .value
                .compare_exchange_weak(value, locked, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return LockGuard {
                    version: StdPtr::from_ref(self),
                    locked_value: locked,
                    _lifetime: PhantomData,
                    _marker: PhantomData,
                };
            }
        }

        self.lock_contended()
    }

    #[cold]
    #[inline(never)]
    fn lock_contended(&self) -> LockGuard<'_> {
        let mut backoff = Backoff::new();
        let mut value: u32 = self.value.load(Ordering::Relaxed);

        loop {
            if (value & LOCK_BIT) == 0 {
                let locked: u32 = value | LOCK_BIT;

                match self.value.compare_exchange_weak(
                    value,
                    locked,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return LockGuard {
                            version: StdPtr::from_ref(self),
                            locked_value: locked,
                            _lifetime: PhantomData,
                            _marker: PhantomData,
                        };
                    }
                    Err(v) => {
                        value = v;
                        continue; // Retry immediately with fresh value
                    }
                }
            }

            backoff.spin();
            value = self.value.load(Ordering::Relaxed);
        }
    }

    /// Try to acquire the lock without blocking.
    #[must_use]
    #[inline(always)]
    pub fn try_lock(&self) -> Option<LockGuard<'_>> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & LOCK_BIT) != 0 {
            return None;
        }

        let locked: u32 = value | LOCK_BIT;

        match self
            .value
            .compare_exchange(value, locked, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(LockGuard {
                version: StdPtr::from_ref(self),
                locked_value: locked,
                _lifetime: PhantomData,
                _marker: PhantomData,
            }),

            Err(_) => None,
        }
    }

    /// Try to acquire the lock within `timeout`.
    ///
    /// ```rust
    /// use std::time::Duration;
    /// use masstree::NodeVersion;
    ///
    /// let version = NodeVersion::new(true);
    /// if let Some(_guard) = version.try_lock_for(Duration::from_millis(100)) {
    ///     // Lock acquired within 100ms
    /// }
    /// ```
    #[must_use]
    pub fn try_lock_for(&self, timeout: Duration) -> Option<LockGuard<'_>> {
        const ATTEMPTS_BEFORE_TIME_CHECK: u32 = 8;

        let deadline = Instant::now() + timeout;
        let mut backoff = Backoff::new();

        loop {
            for _ in 0..ATTEMPTS_BEFORE_TIME_CHECK {
                if let Some(guard) = self.try_lock() {
                    return Some(guard);
                }
                backoff.spin();
            }

            if Instant::now() >= deadline {
                return None;
            }
        }
    }

    /// Acquire lock with yield-on-contention instead of exponential backoff.
    ///
    /// Better for split propagation (longer critical sections, high thread counts).
    /// Use [`lock_bounded()`](Self::lock_bounded) for leaf-level inserts.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock_with_yield(&self) -> LockGuard<'_> {
        const SPINS_BEFORE_YIELD: u32 = 2;

        let mut spin_count: u32 = 0;

        loop {
            if let Some(guard) = self.try_lock() {
                return guard;
            }

            spin_count += 1;

            if spin_count < SPINS_BEFORE_YIELD {
                for _ in 0..spin_count {
                    StdHint::spin_loop();
                }
            } else {
                StdThread::yield_now();
                spin_count = 0;
            }
        }
    }

    // ========================================================================
    //  Non-Locking Operations
    // ========================================================================

    /// Mark as root. Does not require lock.
    #[inline(always)]
    pub fn mark_root(&self) {
        self.value.fetch_or(ROOT_BIT, Ordering::Release);
    }

    /// Clear root bit. Used when layer root is demoted.
    #[inline(always)]
    pub fn mark_nonroot(&self) {
        self.value.fetch_and(!ROOT_BIT, Ordering::Release);
    }

    // ========================================================================
    //  Split-Locked Node Creation
    // ========================================================================

    /// Create a version for a split sibling: locked + `SPLITTING_BIT` + same ISLEAF,
    /// zero version counters.
    ///
    /// Caller must ensure source is locked and `unlock_for_split()` is called
    /// exactly once after the parent pointer is set.
    #[must_use]
    #[inline(always)]
    pub fn new_for_split(source: &Self) -> Self {
        let source_value: u32 = source.value.load(Ordering::Relaxed);
        debug_assert!(
            (source_value & LOCK_BIT) != 0,
            "new_for_split: source must be locked"
        );

        let new_value: u32 = (source_value & ISLEAF_BIT) | LOCK_BIT | SPLITTING_BIT;

        Self {
            value: AtomicU32::new(new_value),
        }
    }

    /// Unlock a node created with `new_for_split`. Increments split version counter.
    ///
    /// Compiler fence ensures all prior writes (parent pointer, data) complete
    /// before the Release store makes the unlock visible.
    #[inline(always)]
    pub fn unlock_for_split(&self) {
        let locked_value = self.value.load(Ordering::Relaxed);

        debug_assert!(
            (locked_value & LOCK_BIT) != 0,
            "unlock_for_split: node must be locked, got value={locked_value:#010x}"
        );

        debug_assert!(
            (locked_value & SPLITTING_BIT) != 0,
            "unlock_for_split: node must have SPLITTING_BIT, got value={locked_value:#010x}"
        );

        let new_value = (locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK;

        StdAtomic::compiler_fence(Ordering::SeqCst);
        self.value.store(new_value, Ordering::Release);
    }

    /// True if `LOCK_BIT` and `SPLITTING_BIT` are both set. For debugging.
    #[must_use]
    #[inline(always)]
    pub fn is_split_locked(&self) -> bool {
        let value = self.value.load(Ordering::Relaxed);
        (value & (LOCK_BIT | SPLITTING_BIT)) == (LOCK_BIT | SPLITTING_BIT)
    }
}

impl Clone for NodeVersion {
    fn clone(&self) -> Self {
        Self {
            value: AtomicU32::new(self.value.load(Ordering::Relaxed)),
        }
    }
}

impl Default for NodeVersion {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod unit_tests;

#[cfg(test)]
#[cfg(not(miri))]
mod concurrent_tests;

#[cfg(all(test, loom, not(miri)))]
mod loom_tests;
