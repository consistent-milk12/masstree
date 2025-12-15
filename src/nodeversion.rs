//! Filepath: src/nodeversion.rs
//!
//! Node version for optimistic concurrency control.
//!
//! [`NodeVersion`] combines lock state, version counters, and metadata flags
//! in a single `u32`. Readers use optimistic validation, writers acquire locks.
//!
//! # Concurrency Model
//! 1. Readers: Call `stable()` to get version, perform read, call `has_changed()`
//! 2. Writers: Call `lock()` to get a [`LockGuard`], modify node, let guard drop.
//!
//! # Type-State Pattern
//! The [`LockGuard`] type provides compile-time verification that the lock is held.
//! Operations that require the lock take `&mut LockGuard` as proof. The guard
//! automatically unlocks on drop (panic-safe).
//!
//! ```rust,ignore
//! let mut guard = version.lock();
//! guard.mark_insert();
//! // Lock released when guard drops
//! ```

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
//  Bit Constants (matching C++ nodeversion_parameters<uint32_t>)
// ============================================================================

/// Lock bit: node is locked for modification.
const LOCK_BIT: u32 = 1 << 0;

/// Inserting bit: node is being inserted into.
const INSERTING_BIT: u32 = 1 << 1;

/// Splitting bit: node is being split.
const SPLITTING_BIT: u32 = 1 << 2;

/// Dirty mask: either inserting or splitting.
const DIRTY_MASK: u32 = INSERTING_BIT | SPLITTING_BIT;

/// Low bit of insert version counter.
const VINSERT_LOWBIT: u32 = 1 << 3;

/// Low bit of split version counter.
const VSPLIT_LOWBIT: u32 = 1 << 9;

/// Unused bit (reserved).
const UNUSED1_BIT: u32 = 1 << 28;

/// Deleted bit: node is logically deleted.
const DELETED_BIT: u32 = 1 << 29;

/// Root bit: node is a tree root.
const ROOT_BIT: u32 = 1 << 30;

/// Is-leaf bit: node is a leaf (vs internode).
const ISLEAF_BIT: u32 = 1 << 31;

/// Mask for unlock after split: clears root, unused, and version bits below vsplit.
const SPLIT_UNLOCK_MASK: u32 = !(ROOT_BIT | UNUSED1_BIT | (VSPLIT_LOWBIT - 1));

/// Mask for unlock after insert: clears unused and version bits below vinsert.
const UNLOCK_MASK: u32 = !(UNUSED1_BIT | (VINSERT_LOWBIT - 1));

// ============================================================================
//  NodeVersion
// ============================================================================

/// A versioned lock for tree nodes.
///
/// Combines lock state, dirty flags, version counters, and node metadata.
///
/// # Layout
/// Bit 31: `is_leaf` | Bit 30: `root` | Bit 29: `deleted` | BITS 9-27: `split_version`
/// Bits 3-8: `insert_version` | Bit 2: `splitting` | Bit 1: `inserting` | Bit 0: `locked`
///
/// # Example
///
/// ```rust
/// use masstree::nodeversion::NodeVersion;
///
/// // Create a leaf node version
/// let v = NodeVersion::new(true);
///
/// assert!(v.is_leaf());
/// assert!(!v.is_locked());
/// ```
#[derive(Debug)]
pub struct NodeVersion {
    value: AtomicU32,
}

// ============================================================================
//  LockGuard (Type-State Pattern)
// ============================================================================

/// Zero-sized proof that a lock is held.
///
/// Cannot be constructed except by calling [`NodeVersion::lock()`].
/// Operations that require the lock take `&mut LockGuard` as proof.
/// The lock is automatically released when the guard drops.
///
/// # Panic Safety
/// The guard releases the lock on drop, even during unwinding. This ensures
/// the lock is never held after a panic.
///
/// # Thread Safety
/// Guards are `!Send` and `!Sync` via `PhantomData<*mut ()>` to prevent them
/// from crossing thread boundaries.
///
/// We use `PhantomData`<*mut ()> which makes the type !Send + !Sync because
/// raw pointers are neither Send nor Sync. This is the standard stable Rust
/// pattern for preventing types from being transferred across threads, at least
/// until 1.92.0.
///
/// NOTE: This is sufficient for our use case. The guard holds a reference to
/// `NodeVersion` which already prevents the guard from outliving the version.
#[derive(Debug)]
#[must_use = "releasing a lock without using the guard is a logic error"]
pub struct LockGuard<'a> {
    version: &'a NodeVersion,
    locked_value: u32,

    // PhantomData<*mut ()> makes this type !Send + !Sync (these are still nightly features)
    _marker: PhantomData<*mut ()>,
}

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        // Automatically unlock on drop, even during panic.
        // Version counter increment depend on dirty bits:
        // - If splitting: increment split counter, clear all dirty/lock bits
        // - If inserting: increment insert counter, clear inserting/lock bits
        let new_value: u32 = if self.locked_value & SPLITTING_BIT != 0 {
            (self.locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            // The expression `(inserting << 2)` equals `vinsert_lowbit` when inserting
            (self.locked_value + ((self.locked_value & INSERTING_BIT) << 2)) & UNLOCK_MASK
        };

        self.version.value.store(new_value, Ordering::Release);
    }
}

impl LockGuard<'_> {
    /// Get the locked version value.
    #[inline]
    #[must_use]
    pub const fn locked_value(&self) -> u32 {
        self.locked_value
    }

    /// Mark the node as being inserted into
    ///
    /// Sets the inserting dirty bit. Version counter will increment on unlock.
    ///
    /// # Memory Ordering (planned after single-threaded works)
    /// The C++ reference uses an acquire fence AFTER setting the dirty bit:
    ///
    /// ```cpp
    /// void mark_insert() {
    ///     v_ |= P::inserting_bit;
    ///     acquire_fence();
    /// }
    /// ```
    ///
    /// This ensures subsequent structural modifications cannot be reordered
    /// before the dirty bit becomes visible. As I am still in early stages
    /// and only working on single-threaded cases, the current approach is sufficient.
    /// The concurrent mode will probably require `std::sync::atomic::fence(Ordering::Acquire)`,
    /// after the store.
    #[inline]
    pub fn mark_insert(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);
        self.version
            .value
            .store(value | INSERTING_BIT, Ordering::Release);

        //  TODO: Add `std::sync::atomic::fence(Ordering::Acquire)` after single-threaded impl works.
        self.locked_value |= INSERTING_BIT;
    }

    /// Mark the node as being split.
    ///
    /// Sets the splitting dirty bit. Version counter will increment on unlock.
    ///
    /// # Memory Ordering (After single-threaded works)
    /// Same as `mark_insert()`: The C++ reference uses an acquire fence after
    /// setting the dirty bit. See `mark_insert` documentation for details.
    #[inline]
    pub fn mark_split(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);
        self.version
            .value
            .store(value | SPLITTING_BIT, Ordering::Release);

        //  TODO: Add `std::sync::atomic::fence(Ordering::Acquire)` after single-threaded impl works.
        self.locked_value |= SPLITTING_BIT;
    }

    /// Mark the node as deleted.
    ///
    /// Also sets the splitting bit to bump version on unlock.
    #[inline]
    pub fn mark_deleted(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);
        let new_value: u32 = value | DELETED_BIT | SPLITTING_BIT;

        self.version.value.store(new_value, Ordering::Release);
        self.locked_value = new_value;
    }

    /// Clear the root bit.
    #[inline]
    pub fn mark_nonroot(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);

        self.version
            .value
            .store(value & !ROOT_BIT, Ordering::Release);
        self.locked_value &= !ROOT_BIT;
    }
}

impl NodeVersion {
    /// Create a new node version.
    ///
    /// # Arguments
    /// - `is_leaf` - true for leaf nodes, false for internodes
    #[must_use]
    pub const fn new(is_leaf: bool) -> Self {
        let initial: u32 = if is_leaf { ISLEAF_BIT } else { 0 };

        Self {
            value: AtomicU32::new(initial),
        }
    }

    /// Create a node version from a raw value.
    ///
    ///  WARN: ONLY FOR TESTING.
    #[must_use]
    pub const fn from_value(value: u32) -> Self {
        Self {
            value: AtomicU32::new(value),
        }
    }

    // ========================================================================
    //  Flag Accessors
    // ========================================================================

    /// Check if this is a leaf node.
    #[inline]
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ISLEAF_BIT) != 0
    }

    /// Check if this is a root node.
    #[inline]
    #[must_use]
    pub fn is_root(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ROOT_BIT) != 0
    }

    /// Check if this node is logically deleted.
    #[inline]
    #[must_use]
    pub fn is_deleted(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DELETED_BIT) != 0
    }

    /// Check if this node is locked.
    #[inline]
    #[must_use]
    pub fn is_locked(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & LOCK_BIT) != 0
    }

    /// Check if this node is being inserted into.
    #[inline]
    #[must_use]
    pub fn is_inserting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & INSERTING_BIT) != 0
    }

    /// Check if this node is being split.
    #[inline]
    #[must_use]
    pub fn is_splitting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & SPLITTING_BIT) != 0
    }

    /// Check if any dirty bit set (inserting or splitting).
    #[inline]
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DIRTY_MASK) != 0
    }

    /// Get the raw version value.
    #[inline]
    #[must_use]
    pub fn value(&self) -> u32 {
        self.value.load(Ordering::Relaxed)
    }

    // ========================================================================
    // Stable Version (for optimistic reads)
    // ========================================================================

    /// Get a stable version value for optimistic reading.
    ///
    /// In concurrent mode, this spins until dirty bits are clear.
    /// In a single-threaded mode, this just returns the current value.
    ///
    /// # Returns
    /// A version value with no dirty bits set. Use with `has_changed()` after reading.
    #[inline]
    #[must_use]
    pub fn stable(&self) -> u32 {
        // For now, as we are only single-threaded, no spinning is needed.
        // Later add: spin while (v & DIRTY_MASK) != 0
        let value: u32 = self.value.load(Ordering::Acquire);

        debug_assert!(
            (value & DIRTY_MASK) == 0,
            "stable() called while dirty (single-threaded mode)"
        );

        value
    }

    /// Check if the version has changed since `old`.
    ///
    /// Returns true if any version relevant bits changed (ignoring lock bit).
    ///
    /// # Implementation Note
    /// Uses `> LOCK_BIT` (i.e., `> 1`) because XOR of only the lock bit equals 1,
    /// which is NOT > 1, so lock-only changes return false.
    #[inline]
    #[must_use]
    pub fn has_changed(&self, old: u32) -> bool {
        // XOR the versions, change = differing bits above LOCK_BIT
        (old ^ self.value.load(Ordering::Acquire)) > LOCK_BIT
    }

    /// Check if a split has occurred since `old`.
    ///
    /// Returns true if the split version counter changed.
    #[inline]
    #[must_use]
    pub fn has_split(&self, old: u32) -> bool {
        (old ^ self.value.load(Ordering::Acquire)) >= VSPLIT_LOWBIT
    }

    // ========================================================================
    // Lock Operations (Type-State Pattern)
    // ========================================================================

    /// Acquire the lock and return a guard.
    ///
    /// The returned [`LockGuard`] proves the lock is held. Operations that
    /// require the lock take `&mut LockGuard` as proof. The lock is auto
    /// released when the guard drops.
    ///
    /// # Panics
    /// Panics in debug mode if already locked (currently).
    /// In concurrent mode, this will spin instead.
    ///
    /// # Note
    /// Current implementation uses load-then-store which is NOT atomic.
    /// For concurrent use, this must be replaced with a `compare_exchange_weak`
    /// loop and proper `Acquire` ordering on success.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock(&self) -> LockGuard<'_> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        debug_assert!(
            (value & LOCK_BIT) == 0,
            "lock() called on already locked node"
        );
        debug_assert!((value & DIRTY_MASK) == 0, "lock called on dirty node");

        let locked: u32 = value | LOCK_BIT;
        self.value.store(locked, Ordering::Release);

        LockGuard {
            version: self,
            locked_value: locked,
            _marker: PhantomData,
        }
    }

    /// Try to acquire the lock without blocking.
    ///
    /// # Returns
    /// `Some(guard)` if lock acquired, `None` if already locked or dirty.
    ///
    /// # Note
    /// Current implementation uses load-then-store which is NOT atomic.
    /// For concurrent use, this must be replaced with `compare_exchange_weak`
    /// and proper `Acquire` ordering on success.
    #[must_use]
    pub fn try_lock(&self) -> Option<LockGuard<'_>> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & (LOCK_BIT | DIRTY_MASK)) != 0 {
            return None;
        }

        let locked: u32 = value | LOCK_BIT;
        self.value.store(locked, Ordering::Release);

        Some(LockGuard {
            version: self,
            locked_value: locked,
            _marker: PhantomData,
        })
    }

    // ========================================================================
    // Non-Locking Operations
    // ========================================================================

    /// Mark the node as a root.
    ///
    /// Does not require the lock. Used during tree initialization.
    pub fn mark_root(&self) {
        let value: u32 = self.value.load(Ordering::Relaxed);

        self.value.store(value | ROOT_BIT, Ordering::Release);
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
    /// Creates a new leaf node version.
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(guard.locked_value() & LOCK_BIT, LOCK_BIT);

            // Guard drops here, releasing lock
        }

        assert!(!v.is_locked());

        // No dirty bits set, so version should be unchanged
        assert!(!v.has_changed(stable_before));
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
    fn test_no_version_increment_without_dirty() {
        let v: NodeVersion = NodeVersion::new(true);
        let stable_before: u32 = v.stable();

        {
            // Lock and let drop without setting dirty bits
            let _guard: LockGuard<'_> = v.lock();
        }

        // Version should NOT have changed
        assert!(!v.has_changed(stable_before));
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
        assert_eq!(guard.locked_value(), initial | LOCK_BIT);
    }

    #[test]
    fn test_guard_mark_updates_locked_value() {
        let v: NodeVersion = NodeVersion::new(true);

        let mut guard: LockGuard<'_> = v.lock();
        let initial_locked: u32 = guard.locked_value();

        assert_eq!(initial_locked & INSERTING_BIT, 0);

        guard.mark_insert();

        // Guard's locked_value is updated
        assert_ne!(guard.locked_value() & INSERTING_BIT, 0);
    }
}
