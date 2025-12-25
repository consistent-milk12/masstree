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
use std::sync::atomic::compiler_fence;
use std::sync::atomic::{AtomicU32, Ordering, fence};
use std::time::{Duration, Instant};

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
//  Backoff (for spin loops)
// ============================================================================

/// Exponential backoff for spin loops.
///
/// Matches C++ `backoff_fence_function` from `reference/compiler.hh:133-143`.
/// Each call to `spin()` executes `count+1` pause instructions, then doubles
/// the count (capped at 15).
///
/// Sequence: 0 → 1 → 3 → 7 → 15 (capped)
struct Backoff {
    count: u32,
}

impl Backoff {
    /// Create a new backoff with count = 0.
    #[inline(always)]
    const fn new() -> Self {
        Self { count: 0 }
    }

    /// Spin for `count+1` iterations using CPU pause hints, then increase count.
    ///
    /// Uses `std::hint::spin_loop()` which maps to the x86 `PAUSE` instruction,
    /// improving performance on hyper-threaded CPUs by hinting that we're in
    /// a spin-wait loop.
    fn spin(&mut self) {
        for _ in 0..=self.count {
            std::hint::spin_loop();
        }
        // Double count, cap at 15: 0 -> 1 -> 3 -> 7 -> 15 -> 15
        self.count = ((self.count << 1) | 1) & 15;
    }
}

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
        // Version counter increment depends on dirty bits:
        // - If splitting: increment split counter, clear all dirty/lock bits
        // - If inserting: increment insert counter, clear inserting/lock bits
        //
        // With current strategy, INSERTING_BIT is always set (unless SPLITTING_BIT was set),
        // so the version counter is always incremented on unlock.
        let new_value: u32 = if self.locked_value & SPLITTING_BIT != 0 {
            (self.locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            // The expression `(inserting << 2)` equals `vinsert_lowbit` when inserting
            // Currently, INSERTING_BIT is always 1 here, so version increments
            (self.locked_value + ((self.locked_value & INSERTING_BIT) << 2)) & UNLOCK_MASK
        };

        self.version.value.store(new_value, Ordering::Release);
    }
}

impl LockGuard<'_> {
    /// Get the locked version value.
    #[must_use]
    #[inline(always)]
    pub const fn locked_value(&self) -> u32 {
        self.locked_value
    }

    /// Mark the node as being inserted into.
    ///
    /// NOTE: With auto dirty on lock strategy, the [`INSERTING_BIT`] is already set
    /// by `lock()`. This method is kept for:
    /// 1. Explicit documentation of intent in calling code
    /// 2. Updating `locked_value` tracking if needed
    /// 3. Backward compatibility with code that calls this explicitly
    ///
    /// This method is idempotent, calling it multiple times has no additional effect.
    #[inline]
    pub fn mark_insert(&mut self) {
        if (self.locked_value & INSERTING_BIT) == 0 {
            // This shouldn't happen with the always dirty on lock strategy
            // we are currently going for. But still handle it gracefully.
            let value: u32 = self.version.value.load(Ordering::Relaxed);

            self.version
                .value
                .store(value | INSERTING_BIT, Ordering::Release);

            fence(Ordering::Acquire);

            self.locked_value |= INSERTING_BIT;
        }

        // If already set, this is a no-op (idempotent)
    }

    /// Mark the node as being split.
    ///
    /// Sets the splitting dirty bit. Version counter will increment on unlock.
    ///
    /// NOTE: Must be called explicitly
    ///
    /// Unlike `mark_insert()` (which is now auto set due to new strategy), `mark_split()`
    /// must be called explicitly before split operations. This is because:
    /// 1. Not all inserts require splits
    /// 2. The [`SPLITTING_BIT`] affects version increment logic differently
    /// 3. Split operations need the split version counter incremented
    ///
    /// # Memory Ordering
    /// Uses [`Ordering::Release`] followed by [`Ordering::Acquire`] fence.
    #[inline]
    pub fn mark_split(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);

        self.version
            .value
            .store(value | SPLITTING_BIT, Ordering::Release);

        // Acquire fence ensures subsequent structural modifications
        // cannot be reordered before the dirty bit becomes visible.
        fence(Ordering::Acquire);

        self.locked_value |= SPLITTING_BIT;
    }

    /// Mark the node as deleted.
    ///
    /// Also sets the splitting bit to bump version on unlock.
    ///
    /// # Memory Ordering
    /// Same as [`mark_insert()`]: Release store followed by Acquire fence.
    #[inline(always)]
    pub fn mark_deleted(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version.value.load(Ordering::Relaxed);
        let new_value: u32 = value | DELETED_BIT | SPLITTING_BIT;

        self.version.value.store(new_value, Ordering::Release);

        // Acquire fence ensures subsequent structural modifications
        // cannot be reordered before the dirty bit becomes visible.
        fence(Ordering::Acquire);

        self.locked_value = new_value;
    }

    /// Clear the root bit.
    #[inline(always)]
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
    #[inline(always)]
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

    /// Check if this node is locked.
    #[must_use]
    #[inline(always)]
    pub fn is_locked(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & LOCK_BIT) != 0
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

    // ========================================================================
    // Stable Version (for optimistic reads)
    // ========================================================================

    /// Get a stable version value for optimistic reading.
    ///
    /// Spins while dirty bits (inserting or splitting) are set, then returns
    /// a version with no dirty bits. Use with [`has_changed()`] after reading
    /// to detect concurrent modifications.
    ///
    /// # Memory Ordering
    /// Uses `Relaxed` loads during spinning for efficiency (especially on ARM),
    /// then issues an `Acquire` fence only on success. This is equivalent to
    /// `Acquire` on every load on x86, but saves ~1 cycle per spin on ARM.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:36-48` - `stable()` template method
    ///
    /// # Returns
    /// A version value with no dirty bits set.
    #[must_use]
    pub fn stable(&self) -> u32 {
        let mut backoff = Backoff::new();
        #[cfg(feature = "tracing")]
        let start = Instant::now();
        #[cfg(feature = "tracing")]
        let mut spins: u32 = 0;

        loop {
            // Use Relaxed ordering for spin loop efficiency (saves ~1 cycle/spin on ARM).
            // We only need Acquire semantics when we actually succeed.
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & DIRTY_MASK) == 0 {
                // Upgrade to an Acquire load on the success path.
                //
                // This avoids relying on fence+relaxed subtleties and ensures we establish
                // a proper synchronizes-with relationship with the writer's Release store
                // in `LockGuard::drop()` on the same atomic.
                let acquired: u32 = self.value.load(Ordering::Acquire);
                if (acquired & DIRTY_MASK) != 0 || acquired != value {
                    backoff.spin();
                    #[cfg(feature = "tracing")]
                    {
                        spins += 1;
                    }
                    continue;
                }

                #[expect(clippy::cast_possible_truncation)]
                #[cfg(feature = "tracing")]
                {
                    let elapsed = start.elapsed();
                    if elapsed > Duration::from_millis(1) || spins > 100 {
                        tracing::warn!(
                            version_ptr = ?std::ptr::from_ref(self),
                            elapsed_us = elapsed.as_micros() as u64,
                            spins = spins,
                            final_value = format_args!("{:#010x}", value),
                            "SLOW_STABLE: stable() took >1ms waiting for dirty bits to clear"
                        );
                    }
                }

                return value;
            }

            backoff.spin();
            #[cfg(feature = "tracing")]
            {
                spins += 1;
            }
        }
    }

    /// Check if the version has changed since `old`.
    ///
    /// Returns true if any version relevant bits changed (ignoring lock bit).
    ///
    /// # Implementation Note
    /// Uses `> LOCK_BIT` (i.e., `> 1`) because XOR of only the lock bit equals 1,
    /// which is NOT > 1, so lock-only changes return false.
    ///
    /// # Compiler Fence Requirement
    ///
    /// This method includes a **compiler fence** before the version load.
    /// This is critical for correctness: the optimistic read protocol is
    /// "read fields → validate version". Without the fence, the compiler
    /// could reorder field reads to occur AFTER the version check, defeating
    /// the validation.
    ///
    /// ## C++ Reference
    ///
    /// The C++ `nodeversion.hh:72-74` uses `fence()` (compiler barrier):
    /// ```cpp
    /// bool has_changed(nodeversion x) const {
    ///     fence();  // compiler barrier from compiler.hh:77
    ///     return (x.v_ ^ v_) > lock_bit;
    /// }
    /// ```
    ///
    /// ## Why Acquire Alone Is Insufficient
    ///
    /// `Ordering::Acquire` on the load only prevents reordering of operations
    /// AFTER the load. It does NOT prevent the compiler from moving reads
    /// that occurred BEFORE the `has_changed()` call to occur after it.
    ///
    /// The compiler fence ensures all prior reads (slot data) complete before
    /// we load the version for validation.
    #[must_use]
    #[inline(always)]
    pub fn has_changed(&self, old: u32) -> bool {
        // Compiler fence: ensures all prior reads complete before version check.
        // This matches C++ fence() in nodeversion.hh:72.
        compiler_fence(Ordering::SeqCst);

        // XOR the versions, change = differing bits above LOCK_BIT | INSERTING_BIT.
        //
        // With the "always dirty on lock" strategy, we set INSERTING_BIT when acquiring
        // the lock. This means `old` (from stable()) won't have INSERTING_BIT, but
        // `current` (after we lock) will. We must ignore this difference, otherwise
        // has_changed() always returns true after we acquire the lock.
        //
        // LOCK_BIT = 1, INSERTING_BIT = 2, so LOCK_BIT | INSERTING_BIT = 3.
        // We check if any bits above bit 1 (i.e., bits 2+) differ.
        (old ^ self.value.load(Ordering::Acquire)) > (LOCK_BIT | INSERTING_BIT)
    }

    /// Check if a split has occurred since `old`.
    ///
    /// Returns true if the split version counter changed.
    ///
    /// Uses the same compiler fence as `has_changed()` for correctness.
    /// See [`has_changed`] for the full explanation.
    #[must_use]
    #[inline(always)]
    pub fn has_split(&self, old: u32) -> bool {
        // Compiler fence: ensures all prior reads complete before version check.
        compiler_fence(Ordering::SeqCst);

        (old ^ self.value.load(Ordering::Acquire)) >= VSPLIT_LOWBIT
    }

    /// Check if a split has occurred since `old`, without a fence.
    ///
    /// This is a faster variant of [`has_split`] that omits the compiler fence.
    /// Use this only when you've already issued a fence (e.g., after an Acquire load).
    ///
    /// # Safety (Logical)
    /// The caller must ensure that all reads that need to be validated have
    /// already been completed and are visible before calling this method.
    /// Typically this means you've already done an Acquire load or fence.
    ///
    /// # Reference
    /// C++ `nodeversion.hh` has `simple_has_split()` for this purpose.
    #[must_use]
    #[inline(always)]
    pub fn simple_has_split(&self, old: u32) -> bool {
        (old ^ self.value.load(Ordering::Acquire)) >= VSPLIT_LOWBIT
    }

    /// Check if the version has changed OR if a modification is in progress.
    ///
    /// This is a stronger check than [`Self::has_changed`] for CAS operations.
    /// It returns true if:
    /// - The version counter has changed (same as `has_changed`), OR
    /// - The node is currently being modified ([`INSERTING_BIT`] or [`SPLITTING_BIT`] set)
    ///
    /// CAS inserts should use this instead of `has_changed` to avoid racing
    /// with locked splits. The race scenario:
    /// 1. CAS insert reads version V via `stable()` (no dirty bits)
    /// 2. Locked thread acquires lock, sets [`INSERTING_BIT`]
    /// 3. CAS insert checks `has_changed(V)` - returns false (ignores [`INSERTING_BIT`])
    /// 4. CAS insert proceeds, racing with the split
    ///
    /// By checking [`INSERTING_BIT`] directly, we catch this race.
    #[must_use]
    #[inline(always)]
    pub fn has_changed_or_locked(&self, old: u32) -> bool {
        compiler_fence(Ordering::SeqCst);

        let current: u32 = self.value.load(Ordering::Acquire);

        // Check if version changed (ignoring lock/dirty bits)
        if (old ^ current) > (LOCK_BIT | INSERTING_BIT) {
            return true;
        }

        // Check if modification in progress (INSERTING_BIT or SPLITTING_BIT set)
        // This catches the race where we got a stable version but then a lock was acquired
        if (current & DIRTY_MASK) != 0 {
            return true;
        }

        false
    }

    // ========================================================================
    // Lock Operations (Type-State Pattern)
    // ========================================================================

    /// Acquire the lock and return a guard.
    ///
    /// Strategy: Always-Dirty-On-Lock
    /// This implementation automatically sets the [`INSERTING_BIT`] when acquiring the lock.
    /// This eliminates the race window between lock acquisition and explicit dirty marking,
    /// ensuring that CAS insert threads always wait for locked writers to complete.
    ///
    /// 1. `stable()` spins until `DIRTY_MASK == 0` (includes `INSERTING_BIT`)
    /// 2. `lock()` atomically sets `LOCK_BIT | INSERTING_BIT`
    /// 3. Therefore, any thread calling `stable()` will wait for the lock holder
    /// 4. This eliminates the window where a locked writer hasn't called `mark_insert()` yet
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful CAS to synchronize with the
    /// `Release` store in [`Drop::drop`] of the previous lock holder.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:87-109` - `lock()` template method
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock(&self) -> LockGuard<'_> {
        let mut backoff: Backoff = Backoff::new();

        #[cfg(feature = "tracing")]
        let start = Instant::now();

        #[cfg(feature = "tracing")]
        let mut spins: u32 = 0;

        #[cfg(feature = "tracing")]
        let mut contended = false;

        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            // Must wait for both lock bit and dirty bits to clear.
            // This ensures we don't acquire while another writer is active.
            if (value & (LOCK_BIT | DIRTY_MASK)) == 0 {
                // STRATEGY: Set LOCK_BIT and INSERTING_BIT atomically.
                // This ensures CAS insert threads (which call stable()) will wait for us.
                //
                // The INSERTING_BIT is set here rather than in a seprate mark_insert() call.
                // This eliminates the race window between lock acquisition and dirty marking.
                let locked: u32 = value | LOCK_BIT | INSERTING_BIT;

                // CAS to acquire lock with auto-dirty.
                // Acquire on success ensures we see all prior writes from previous holder.
                // Relaxed on failure is fine, we'll retry.
                if self
                    .value
                    .compare_exchange_weak(value, locked, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    #[expect(clippy::cast_possible_truncation)]
                    #[cfg(feature = "tracing")]
                    {
                        let elapsed = start.elapsed();
                        if elapsed > Duration::from_millis(1) || spins > 100 {
                            tracing::warn!(
                                version_ptr = ?std::ptr::from_ref(self),
                                elapsed_us = elapsed.as_micros() as u64,
                                spins = spins,
                                contended = contended,
                                "SLOW_LOCK: lock() took >1ms"
                            );
                        }
                    }

                    return LockGuard {
                        version: self,
                        // locked_value now includes INSERTING_BIT
                        locked_value: locked,
                        _marker: PhantomData,
                    };
                }
            } else {
                #[cfg(feature = "tracing")]
                {
                    contended = true;
                }
            }

            backoff.spin();
            #[cfg(feature = "tracing")]
            {
                spins += 1;
            }
        }
    }

    /// Try to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was acquired, `None` if the lock
    /// is held or dirty bits are set.
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful CAS.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:111-127` - `try_lock()` template method
    #[must_use]
    pub fn try_lock(&self) -> Option<LockGuard<'_>> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        // Fail fast if locked or dirty.
        if (value & (LOCK_BIT | DIRTY_MASK)) != 0 {
            return None;
        }

        // Set both LOCK_BIT and INSERTING_BIT atomically (same as lock()).
        let locked: u32 = value | LOCK_BIT | INSERTING_BIT;

        // Single CAS attempt (use strong CAS for single-shot).
        match self
            .value
            .compare_exchange(value, locked, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(LockGuard {
                version: self,
                locked_value: locked,
                _marker: PhantomData,
            }),

            Err(_) => None,
        }
    }

    /// Try to acquire the lock with a timeout.
    ///
    /// Returns `Some(guard)` if the lock was acquired within `timeout`,
    /// `None` if the timeout expired.
    ///
    /// # Use Cases
    /// - Deadlock detection in tests
    /// - Bounded wait times in production
    ///
    /// # Example
    /// ```rust,ignore
    /// use std::time::Duration;
    /// use masstree::nodeversion::NodeVersion;
    ///
    /// let version = NodeVersion::new(true);
    /// if let Some(guard) = version.try_lock_for(Duration::from_millis(100)) {
    ///     // Lock acquired within 100ms
    /// } else {
    ///     // Timeout expired, lock not acquired
    /// }
    /// ```
    #[must_use]
    pub fn try_lock_for(&self, timeout: Duration) -> Option<LockGuard<'_>> {
        let deadline = Instant::now() + timeout;
        let mut backoff = Backoff::new();

        loop {
            // Try to acquire.
            if let Some(guard) = self.try_lock() {
                return Some(guard);
            }

            // Check timeout.
            if Instant::now() >= deadline {
                return None;
            }

            backoff.spin();
        }
    }

    /// Acquire the lock using try-lock with yield.
    ///
    /// Unlike [`lock()`] which spins with exponential backoff, this method
    /// yields the CPU to other threads when the lock is contended. This is
    /// more efficient for lock convoy situations where multiple threads are
    /// waiting on the same lock.
    ///
    /// # Algorithm
    ///
    /// 1. Try to acquire the lock with `try_lock()`
    /// 2. If failed, do a small number of spin-loop hints
    /// 3. Then yield the CPU with `thread::yield_now()`
    /// 4. Repeat until lock acquired
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful lock acquisition.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock_with_yield(&self) -> LockGuard<'_> {
        const SPINS_BEFORE_YIELD: u32 = 4;

        #[cfg(feature = "tracing")]
        let start = Instant::now();

        #[cfg(feature = "tracing")]
        let mut total_spins: u32 = 0;

        #[cfg(feature = "tracing")]
        let mut yields: u32 = 0;

        let mut spin_count: u32 = 0;

        loop {
            // Try to acquire the lock
            if let Some(guard) = self.try_lock() {
                #[cfg(feature = "tracing")]
                #[expect(clippy::cast_possible_truncation)]
                {
                    let elapsed = start.elapsed();
                    if elapsed > Duration::from_millis(1) || total_spins > 100 || yields > 10 {
                        tracing::warn!(
                            version_ptr = ?std::ptr::from_ref(self),
                            elapsed_us = elapsed.as_micros() as u64,
                            total_spins = total_spins,
                            yields = yields,
                            "SLOW_LOCK_YIELD: lock_with_yield() took >1ms"
                        );
                    }
                }
                return guard;
            }

            spin_count += 1;
            #[cfg(feature = "tracing")]
            {
                total_spins += 1;
            }

            if spin_count < SPINS_BEFORE_YIELD {
                // Brief spin before yielding
                for _ in 0..spin_count {
                    std::hint::spin_loop();
                }
            } else {
                // Yield CPU to other threads - reduces lock convoy
                std::thread::yield_now();
                spin_count = 0; // Reset for next cycle
                #[cfg(feature = "tracing")]
                {
                    yields += 1;
                }
            }
        }
    }

    // ========================================================================
    // Non-Locking Operations
    // ========================================================================

    /// Mark the node as a root.
    ///
    /// Does not require the lock. Used during tree initialization.
    ///
    /// # Implementation Note
    /// Uses `fetch_or` for atomic read-modify-write. The previous implementation
    /// used separate load/store which could lose concurrent modifications.
    #[inline(always)]
    pub fn mark_root(&self) {
        self.value.fetch_or(ROOT_BIT, Ordering::Release);
    }

    /// Clear the root bit.
    ///
    /// Called when a layer root leaf is demoted (layer root split).
    ///
    /// # Implementation Note
    /// Uses `fetch_and` for atomic read-modify-write. The previous implementation
    /// used separate load/store which could lose concurrent modifications.
    #[inline(always)]
    pub fn mark_nonroot(&self) {
        self.value.fetch_and(!ROOT_BIT, Ordering::Release);
    }

    // ========================================================================
    // Split-Locked Node Creation (Help-Along Protocol)
    // ========================================================================

    /// Create a new node version for a split sibling.
    ///
    /// The new version is:
    /// - Locked ([`LOCK_BIT`] set)
    /// - Marked as splitting ([`SPLITTING_BIT`] set)
    /// - Has the same [`ISLEAF_BIT`] as the source
    /// - Has zeroed version counters (fresh node)
    ///
    /// This is used during splits to create a right sibling that starts locked.
    /// The sibling remains locked until its parent pointer is set, preventing
    /// other threads from trying to split it while parent is NULL.
    ///
    /// # C++ Reference
    ///
    /// Matches `child->assign_version(*n_)` in `masstree_split.hh:198`.
    /// However, we use [`SPLITTING_BIT`] instead of copying [`INSERTING_BIT`] because
    /// the right sibling's unlock should increment the split counter.
    ///
    /// # Safety Considerations
    ///
    /// The caller must ensure:
    /// 1. The source node is locked
    /// 2. `unlock_for_split()` will be called on this node exactly once
    /// 3. The new node is not visible to other threads until after `link_sibling()`
    ///
    /// # Memory Ordering
    ///
    /// Uses Relaxed ordering because the new node is not yet visible to other threads.
    /// The fence in `link_sibling()` establishes visibility.
    #[must_use]
    pub fn new_for_split(source: &Self) -> Self {
        let source_value = source.value.load(Ordering::Relaxed);
        debug_assert!(
            (source_value & LOCK_BIT) != 0,
            "new_for_split: source must be locked"
        );

        // New version has:
        // - ISLEAF_BIT from source (preserved)
        // - LOCK_BIT (locked)
        // - SPLITTING_BIT (will increment split counter on unlock)
        // - Zero version counters (fresh node)
        //
        // We deliberately use SPLITTING_BIT (not INSERTING_BIT) because:
        // 1. This is a split operation
        // 2. unlock_for_split should increment vsplit, not vinsert
        // 3. SPLIT_UNLOCK_MASK clears ROOT_BIT which is correct for split children
        let new_value = (source_value & ISLEAF_BIT) | LOCK_BIT | SPLITTING_BIT;

        Self {
            value: AtomicU32::new(new_value),
        }
    }

    /// Unlock a node that was created with `new_for_split`.
    ///
    /// This performs a split unlock (increments split version counter).
    /// Must be called exactly once after the node's parent pointer is set.
    ///
    /// # C++ Reference
    ///
    /// Matches the unlock in `masstree_split.hh:280`: `child->unlock()`.
    /// The C++ version uses the hand-over-hand pattern where the child
    /// is unlocked after the parent insert completes.
    ///
    /// # Memory Ordering
    ///
    /// Issues a compiler fence before the store to ensure all prior writes
    /// (parent pointer, data) are complete before the unlock is visible.
    /// Uses Release ordering on the store to synchronize with readers'
    /// Acquire loads in `stable()` and `has_changed()`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the node is locked with [`SPLITTING_BIT`] set.
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

        // Compute unlocked value: increment split counter, clear dirty/lock/root bits
        // This matches the SPLITTING_BIT branch in LockGuard::drop
        let new_value = (locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK;

        // Compiler fence: ensures all prior writes are ordered before version store.
        // This is critical - parent pointer and all data must be visible to readers
        // before we unlock. Without this, a reader could see the unlocked version
        // but read stale/missing parent pointer.
        compiler_fence(Ordering::SeqCst);

        // Release store: synchronizes with Acquire loads in stable()/has_changed()
        self.value.store(new_value, Ordering::Release);
    }

    /// Check if this node was created for a split and hasn't been unlocked yet.
    ///
    /// Returns true if [`LOCK_BIT`] and [`SPLITTING_BIT`] are both set.
    /// Used for debugging and assertions.
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
    /// Creates a new leaf node version.
    fn default() -> Self {
        Self::new(true)
    }
}

// ============================================================================
//  SingleThreadedNodeVersion (for benchmarks)
// ============================================================================

/// A single-threaded node version that skips synchronization.
///
/// This is useful for single-threaded benchmarks where you want to measure
/// the overhead of the data structure without synchronization costs.
///
/// All operations return immediately without any atomic operations or fences.
/// This is NOT thread-safe and must only be used in single-threaded contexts.
///
/// # Reference
/// C++ `nodeversion.hh` has `singlethreaded_nodeversion` for this purpose.
#[derive(Debug, Clone)]
pub struct SingleThreadedNodeVersion {
    value: u32,
}

/// A no-op lock guard for single-threaded usage.
///
/// Does nothing on drop since there's no actual lock to release.
#[derive(Debug)]
#[must_use = "releasing a lock without using the guard is a logic error"]
pub struct SingleThreadedLockGuard<'a> {
    version: &'a mut SingleThreadedNodeVersion,
}

impl Drop for SingleThreadedLockGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        // Same logic as the real LockGuard drop:
        // - If splitting: increment split counter, clear dirty/lock bits
        // - If inserting: increment insert counter, clear inserting/lock bits
        let value = self.version.value;
        self.version.value = if (value & SPLITTING_BIT) != 0 {
            (value.wrapping_add(VSPLIT_LOWBIT)) & SPLIT_UNLOCK_MASK
        } else {
            value.wrapping_add(VINSERT_LOWBIT) & UNLOCK_MASK
        };
    }
}

impl SingleThreadedLockGuard<'_> {
    /// Mark the node as being inserted into (no-op, for API compatibility).
    #[inline(always)]
    pub const fn mark_insert(&mut self) {
        // No-op - version increments on drop anyway
    }

    /// Mark the node as being split.
    #[inline(always)]
    pub const fn mark_split(&mut self) {
        self.version.value |= SPLITTING_BIT;
    }

    /// Mark the node as deleted.
    #[inline(always)]
    pub const fn mark_deleted(&mut self) {
        self.version.value |= DELETED_BIT | SPLITTING_BIT;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub const fn mark_nonroot(&mut self) {
        self.version.value &= !ROOT_BIT;
    }
}

impl SingleThreadedNodeVersion {
    /// Create a new single-threaded node version.
    #[must_use]
    #[inline(always)]
    pub const fn new(is_leaf: bool) -> Self {
        let initial: u32 = if is_leaf { ISLEAF_BIT } else { 0 };
        Self { value: initial }
    }

    /// Check if this is a leaf node.
    #[must_use]
    #[inline(always)]
    pub const fn is_leaf(&self) -> bool {
        (self.value & ISLEAF_BIT) != 0
    }

    /// Check if this is a root node.
    #[must_use]
    #[inline(always)]
    pub const fn is_root(&self) -> bool {
        (self.value & ROOT_BIT) != 0
    }

    /// Check if this node is logically deleted.
    #[must_use]
    #[inline(always)]
    pub const fn is_deleted(&self) -> bool {
        (self.value & DELETED_BIT) != 0
    }

    /// Get a stable version (returns immediately in single-threaded mode).
    #[must_use]
    #[inline(always)]
    pub const fn stable(&self) -> u32 {
        self.value
    }

    /// Check if the version has changed since `old`.
    #[must_use]
    #[inline(always)]
    pub const fn has_changed(&self, old: u32) -> bool {
        (old ^ self.value) > (LOCK_BIT | INSERTING_BIT)
    }

    /// Check if a split has occurred since `old`.
    #[must_use]
    #[inline(always)]
    pub const fn has_split(&self, old: u32) -> bool {
        (old ^ self.value) >= VSPLIT_LOWBIT
    }

    /// Acquire the "lock" (no-op, returns guard immediately).
    #[inline(always)]
    pub const fn lock(&mut self) -> SingleThreadedLockGuard<'_> {
        SingleThreadedLockGuard { version: self }
    }

    /// Mark the node as a root.
    #[inline(always)]
    pub const fn mark_root(&mut self) {
        self.value |= ROOT_BIT;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub const fn mark_nonroot(&mut self) {
        self.value &= !ROOT_BIT;
    }
}

impl Default for SingleThreadedNodeVersion {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
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
    fn test_simple_has_split_no_fence() {
        // Test that simple_has_split works correctly (same logic, no fence)
        let v = NodeVersion::new(true);
        let before = v.stable();

        {
            let mut guard = v.lock();
            guard.mark_split();
        }

        // simple_has_split should detect the change
        assert!(v.simple_has_split(before));

        // And should match has_split
        assert_eq!(v.has_split(before), v.simple_has_split(before));
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
}

// Concurrent tests live in a submodule to keep this file lean.
// Guarded with `#[cfg(not(miri))]` because Miri doesn't support multi-threading well.
#[cfg(test)]
#[cfg(not(miri))]
mod concurrent_tests;

// Loom tests for deterministic concurrency verification.
#[cfg(all(test, loom, not(miri)))]
mod loom_tests;
