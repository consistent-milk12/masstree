use std::array as StdArray;
use std::sync::atomic::{Ordering as AtomicOrdering, fence};

use seize::Guard;

use crate::{
    internode::{InternodeNode, WIDTH},
    nodeversion::NodeVersion,
    ordering::{READ_ORD, RELAXED, WRITE_ORD},
    prefetch::prefetch_read,
};

impl InternodeNode {
    // ========================================================================
    //  Version Accessors
    // ========================================================================

    /// Get a reference to the node's version.
    #[must_use]
    #[inline(always)]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the node's version.
    #[inline(always)]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ========================================================================
    //  Key Accessors
    // ========================================================================

    /// Get the number of keys in this internode.
    #[must_use]
    #[inline(always)]
    pub fn nkeys(&self) -> usize {
        self.nkeys.load(READ_ORD) as usize
    }

    /// Get the number of keys as usize (convenience method).
    #[must_use]
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.nkeys()
    }

    /// Check if the internode has no keys.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nkeys.load(READ_ORD) == 0
    }

    /// Check if the internode is full.
    #[must_use]
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.nkeys.load(READ_ORD) as usize >= WIDTH
    }

    /// Get the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn ikey(&self, i: usize) -> u64 {
        debug_assert!(i < WIDTH, "ikey: index {i} out of bounds (WIDTH={WIDTH})");
        self.ikey0[i].load(READ_ORD)
    }

    /// Batch load all ikeys into a local array.
    ///
    /// This allows the search loop to operate on a local array without
    /// per-element atomic load overhead. The compiler can optimize the
    /// sequential loads and the subsequent search operates on cached data.
    ///
    /// # Note
    ///
    /// This method is NOT used by `upper_bound_internode_generic` because
    /// internode search benefits from early exit (most lookups find their
    /// key in the first few comparisons). Batch loading all 15 keys upfront
    /// wastes work for the common case.
    ///
    /// This method may be useful for other scenarios (e.g., serialization,
    /// debugging, or operations that need all keys).
    ///
    /// # Memory Ordering
    ///
    /// Uses Relaxed loads for all 15 keys followed by a single Acquire fence.
    /// This is more efficient than 15 individual Acquire loads while providing
    /// the same ordering guarantee: all loads complete before the fence, and
    /// the fence synchronizes with Release stores from writers.
    ///
    /// # Returns
    ///
    /// An array of all WIDTH ikeys. Only the first `nkeys` are valid.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing)]
    pub fn load_all_ikeys(&self) -> [u64; WIDTH] {
        // Use Relaxed loads - ordering is established by the fence below
        let ikeys: [u64; WIDTH] = StdArray::from_fn(|i| self.ikey0[i].load(RELAXED));

        // Single Acquire fence ensures all loads above complete before we return,
        // and synchronizes with Release stores from writers. This is equivalent to
        // 15 individual Acquire loads but more efficient (1 fence vs 15 barriers).
        fence(AtomicOrdering::Acquire);

        ikeys
    }

    /// Set the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn set_ikey(&self, i: usize, ikey: u64) {
        debug_assert!(i < WIDTH, "set_ikey: index {i} out of bounds");
        self.ikey0[i].store(ikey, WRITE_ORD);
    }

    /// Set key using Relaxed ordering (for internal shifting during insert).
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by caller")]
    pub(super) fn set_ikey_relaxed(&self, i: usize, ikey: u64) {
        self.ikey0[i].store(ikey, RELAXED);
    }

    /// Get the tree height.
    ///
    /// - `height = 0` means children are leaves
    /// - `height > 0` means children are internodes
    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> u32 {
        self.height as u32
    }

    /// Check if children are leaves (height == 0).
    #[must_use]
    #[inline(always)]
    pub const fn children_are_leaves(&self) -> bool {
        self.height == 0
    }

    // ========================================================================
    //  Child Accessors
    // ========================================================================

    /// Get the child pointer at the given index.
    ///
    /// Valid indices are `0..=nkeys` (one more child than keys).
    /// Index 15 (WIDTH) returns the rightmost child.
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    ///
    /// # Panics
    /// Panics in debug mode if `i > WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; i <= WIDTH (16 children)"
    )]
    pub fn child(&self, i: usize, guard: &impl Guard) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child: index {i} out of bounds");
        guard.protect(&self.child[i], READ_ORD)
    }

    /// Get the child pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the child pointer's target won't be retired during use.
    /// Valid when:
    /// - Called during `Drop` (no concurrent access)
    /// - Called in teardown after `reclaim_all()`
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; i <= WIDTH (16 children)"
    )]
    pub unsafe fn child_unguarded(&self, i: usize) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child_unguarded: index {i} out of bounds");
        self.child[i].load(READ_ORD)
    }

    /// Get the child pointer with prefetch hint for the next likely child.
    ///
    /// This is used in the optimized traversal path to hide memory latency.
    /// Prefetches both the next child node (`child[i+1]`) and its key array
    /// (cache line 1 at offset 64) while returning `child[i]`.
    ///
    /// Uses guard protection for the returned child pointer. The speculative
    /// prefetch does NOT use protection (it's just a hint, never dereferenced).
    ///
    /// # Arguments
    /// * `i` - Child index to return
    /// * `nkeys` - Current number of keys (to avoid prefetching beyond valid children)
    /// * `guard` - Guard for protected load
    ///
    /// # Prefetch Strategy
    ///
    /// When descending to `child[i]`, we speculatively prefetch `child[i+1]`:
    /// - Offset 0: Node header + first 6 ikeys (cache line 0)
    /// - Offset 64: Remaining ikeys (cache line 1)
    ///
    /// Prefetching null pointers is harmless on x86/ARM (becomes a no-op).
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn child_with_prefetch(&self, i: usize, nkeys: usize, guard: &impl Guard) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child_with_prefetch: index out of bounds");

        // Protected load for the child we're actually returning
        let ptr = guard.protect(&self.child[i], READ_ORD);

        // Speculatively prefetch next child's node header + ikey array
        // Note: This prefetch is just a hint - we never dereference next_child_ptr
        // directly, so it doesn't need protection.
        if i < nkeys {
            let next_child_ptr = self.child[i + 1].load(RELAXED);

            // SAFETY: Prefetch instructions are safe even with null/invalid pointers.
            // On x86/ARM, prefetching an invalid address is a no-op (silently ignored).
            // The CPU's prefetch unit handles TLB misses gracefully.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Prefetch cache line 0: header + ikey0[0..5]
                std::arch::x86_64::_mm_prefetch(
                    next_child_ptr.cast::<i8>(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
                // Prefetch cache line 1: ikey0[6..13]
                std::arch::x86_64::_mm_prefetch(
                    next_child_ptr.cast::<i8>().wrapping_add(64),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            // SAFETY: Same as x86_64 - prefetch of invalid addresses is harmless.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                // Prefetch cache line 0: header + ikey0[0..5]
                std::arch::aarch64::_prefetch(
                    next_child_ptr.cast::<i8>(),
                    std::arch::aarch64::_PREFETCH_READ,
                    std::arch::aarch64::_PREFETCH_LOCALITY3,
                );

                // Prefetch cache line 1: ikey0[6..13]
                std::arch::aarch64::_prefetch(
                    next_child_ptr.cast::<i8>().wrapping_add(64),
                    std::arch::aarch64::_PREFETCH_READ,
                    std::arch::aarch64::_PREFETCH_LOCALITY3,
                );
            }
        }

        ptr
    }

    /// Get the child pointer with depth-first prefetch for tree descent.
    ///
    /// Unlike [`Self::child_with_prefetch`] which prefetches the next sibling,
    /// this method prefetches the target child's internal data (header + keys).
    /// Use this variant in descent paths where you're going down the tree,
    /// not sideways through siblings.
    ///
    /// # Prefetch Strategy
    ///
    /// Prefetches two cache lines of the target child node:
    /// - CL 0 (offset 0): Header (16B) + ikey0[0..=5] (48B)
    /// - CL 1 (offset 64): ikey0[6..=13] (64B)
    ///
    /// This covers 14 of 15 keys, hiding memory latency for the next level's search.
    ///
    /// # When to Use
    ///
    /// - **Descent (point lookup)**: Use `child_with_depth_prefetch`
    /// - **Scan (sibling traversal)**: Use `child_with_prefetch`
    ///
    /// # Arguments
    ///
    /// * `i` - Child index to return
    /// * `guard` - Guard for protected load
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn child_with_depth_prefetch(&self, i: usize, guard: &impl Guard) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child_with_depth_prefetch: index out of bounds");

        // Protected load for the child we're actually returning
        let ptr: *mut u8 = guard.protect(&self.child[i], READ_ORD);

        // Prefetch the target child's header and key array
        Self::prefetch_child_internal(ptr);

        ptr
    }

    /// Get the child pointer with full prefetch for tree descent.
    ///
    /// Prefetches all three cache lines containing keys (CL 0, 1, 2).
    /// Use when nodes are expected to be full or nearly full.
    ///
    /// # Arguments
    ///
    /// * `i` - Child index to return
    /// * `guard` - Guard for protected load
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn child_with_full_prefetch(&self, i: usize, guard: &impl Guard) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child_with_full_prefetch: index out of bounds");

        let ptr: *mut u8 = guard.protect(&self.child[i], READ_ORD);

        Self::prefetch_child_full(ptr);

        ptr
    }

    /// Prefetch a child node's header and first two key cache lines.
    ///
    /// This is a static helper that can be called without a guard,
    /// enabling use in trait implementations.
    ///
    /// # Safety Note
    ///
    /// Prefetch is a performance hint only; it must not be relied upon for
    /// correctness. This helper is safe and cheap on all targets.
    #[inline(always)]
    pub(super) fn prefetch_child_internal(ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }

        // Prefetch CL 0 (offset 0) and CL 1 (offset 64) of the child node.
        //
        // Note: use `wrapping_add` so this remains purely "address arithmetic"
        // (no in-bounds requirement), and avoid integer casts for provenance.
        prefetch_read(ptr);
        prefetch_read(ptr.wrapping_add(64));
    }

    /// Prefetch a child node's header and all three key cache lines.
    #[inline(always)]
    pub(super) fn prefetch_child_full(ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }

        prefetch_read(ptr);
        prefetch_read(ptr.wrapping_add(64));
        prefetch_read(ptr.wrapping_add(128));
    }

    /// Set the child pointer at the given index.
    ///
    /// Valid indices are `0..=WIDTH` (16 children for 15 keys).
    ///
    /// # Panics
    /// Panics in debug mode if `i > WIDTH`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; i <= WIDTH (16 children)"
    )]
    pub fn set_child(&self, i: usize, child: *mut u8) {
        debug_assert!(i <= WIDTH, "set_child: index {i} out of bounds");
        self.child[i].store(child, WRITE_ORD);
    }

    /// Assign a key and its right child at position `p`.
    ///
    /// Following the C++ pattern:
    /// - `ikey[p] = ikey`
    /// - `child[p + 1] = right_child`
    ///
    /// The left child (`child[p]`) must already be set.
    ///
    /// # Panics
    /// Panics in debug mode if `p >= WIDTH`.
    #[inline(always)]
    pub fn assign(&self, p: usize, ikey: u64, right_child: *mut u8) {
        debug_assert!(p < WIDTH, "assign: position {p} out of bounds");

        self.set_ikey(p, ikey);
        self.set_child(p + 1, right_child);
    }

    /// Set the number of keys.
    ///
    /// # Panics
    /// Panics in debug mode if `n > WIDTH`.
    #[inline(always)]
    pub fn set_nkeys(&self, n: u8) {
        debug_assert!((n as usize) <= WIDTH, "set_nkeys: count {n} out of bounds");
        self.nkeys.store(n, WRITE_ORD);
    }

    /// Increment the number of keys by 1.
    ///
    /// # Precondition
    ///
    /// Caller must hold the node lock. This is a load-then-store operation,
    /// not an atomic increment, because only one writer can modify nkeys
    /// while the lock is held.
    ///
    /// # Panics
    /// Panics in debug mode if already at WIDTH.
    #[inline(always)]
    pub fn inc_nkeys(&self) {
        let current: u8 = self.nkeys.load(RELAXED);
        debug_assert!((current as usize) < WIDTH, "inc_nkeys: would overflow");
        self.nkeys.store(current.wrapping_add(1), WRITE_ORD);
    }
}
