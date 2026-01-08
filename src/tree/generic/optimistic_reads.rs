//! ========================================================================
//!  Generic Optimistic Read Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - Linear search (predictable branches, cache-friendly)
//! - Unified implementation via closure for value extraction

use std::ptr as StdPtr;

use super::{
    Key, LayerCapableLeaf, LocalGuard, MassTreeGeneric, NodeAllocatorGeneric, NodeVersion,
    ValueSlot,
};

use crate::leaf_trait::TreePermutation;
use crate::leaf24::KSUF_KEYLENX;
use crate::leaf24::LAYER_KEYLENX;
use crate::link::Linker;
use crate::ref_value_slot::RefValueSlot;
use crate::value::traits::LeafValueLoad;

mod get_guarded;

// ============================================================================
//  LookupResult - Search outcome enum
// ============================================================================

/// Result of searching a leaf node for a key.
///
/// This enum captures the three possible outcomes without interpreting
/// the pointer until after version validation.
enum LookupResult {
    /// Found a terminal value at the given slot index.
    /// The `keylenx` confirms it's a value (< [`LAYER_KEYLENX`])
    ValueSlot(usize),

    /// Found a layer pointer. Need to descend into sublayer.
    /// Still returns the raw pointer since layer pointers are always real pointers.
    Layer(*mut u8),

    /// Key not found in this leaf.
    NotFound,
}

// ============================================================================
//  Search Helpers (Hot Path)
// ============================================================================

/// Search a leaf for a key in multi-layer mode (keys > 8 bytes).
///
/// Handles:
/// - Suffix comparison for keys with same 8-byte prefix
/// - Layer pointer detection for descent
///
/// Optimized with loop unrolling (3 at a time).
///
/// Uses Relaxed ordering for ikey loads after the initial Acquire on permutation.
/// This is safe because:
/// 1. `permutation()` uses Acquire ordering, synchronizing with writer's Release
/// 2. OCC version validation at the end catches any races
///
/// Uses `#[inline]` - medium-sized function with loop unrolling; let compiler
/// decide based on call-site context to avoid I-cache pressure.
#[inline]
#[expect(clippy::collapsible_if, reason = "Leads to unusual regressions?!")]
fn search_leaf_multi_layer<S, L>(leaf: &L, key: &Key<'_>) -> LookupResult
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    // Acquire ordering on permutation synchronizes with writer's Release fence
    let perm = leaf.permutation();
    let size = perm.size();
    let target_ikey: u64 = key.ikey();

    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
    let search_keylenx: u8 = if key.has_suffix() {
        KSUF_KEYLENX
    } else {
        key.current_len() as u8
    };

    // OPTIMIZATION: Fast path flag - if search key has no suffix, skip suffix/layer checks
    // For inline keys (≤8 bytes), we don't need to compare suffixes or check for layer pointers
    let needs_suffix_check: bool = key.has_suffix();

    let mut i: usize = 0;

    // Unrolled loop: process 3 slots per iteration
    // Speculative batch load: load all slots and ikeys upfront for better ILP
    // Use Relaxed ordering - synchronization already established by permutation load
    while i + 3 <= size {
        // Batch load slots (bit extraction only, no memory access)
        let s0: usize = perm.get(i);
        let s1: usize = perm.get(i + 1);
        let s2: usize = perm.get(i + 2);

        // Batch load ikeys with Relaxed ordering (safe after permutation Acquire)
        let ikey0: u64 = leaf.ikey_relaxed(s0);
        let ikey1: u64 = leaf.ikey_relaxed(s1);
        let ikey2: u64 = leaf.ikey_relaxed(s2);

        // Now check sequentially with early exit
        if ikey0 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s0, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        if ikey1 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s1, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        if ikey2 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s2, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        i += 3;
    }

    // Handle remainder (0-2 elements)
    while i < size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, slot, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        i += 1;
    }

    LookupResult::NotFound
}

/// Optimized slot match check with suffix-check bypass.
///
/// When `needs_suffix_check` is false (inline keys ≤8 bytes), skips:
/// - Suffix comparison (no suffix exists)
/// - Layer pointer detection (inline keys can't be layer pointers)
#[inline(always)]
fn check_slot_match<S, L>(
    leaf: &L,
    slot: usize,
    search_keylenx: u8,
    key: &Key<'_>,
    needs_suffix_check: bool,
) -> Option<LookupResult>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let slot_keylenx: u8 = leaf.keylenx(slot);
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

    // Null pointer means slot is being modified - skip
    if slot_ptr.is_null() {
        return None;
    }

    if slot_keylenx == search_keylenx {
        // Potential exact match
        // OPTIMIZATION: Only check suffix if the search key has one
        if needs_suffix_check
            && slot_keylenx == KSUF_KEYLENX
            && !leaf.ksuf_equals(slot, key.suffix())
        {
            return None;
        }

        // Return slot index, not pointer
        return Some(LookupResult::ValueSlot(slot));
    }

    // OPTIMIZATION: Layer pointer check only relevant if search key has suffix
    if needs_suffix_check && slot_keylenx >= LAYER_KEYLENX {
        // Layer pointer, still return the actual pointer for descent
        return Some(LookupResult::Layer(slot_ptr));
    }

    None
}

// ============================================================================
//  Helper Functions
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Handle version change during optimistic read.
    ///
    /// Called when version validation fails. Follows B-link chain if split
    /// occurred, otherwise returns new version for retry.
    ///
    /// Returns `(new_leaf_ptr, should_restart_leaf_loop)`:
    /// - If leaf changed: `(new_ptr, true)`
    /// - If same leaf, new version: `(same_ptr, false)` with updated version
    #[cold]
    #[inline(never)]
    fn handle_version_change(
        &self,
        leaf: &L,
        key: &Key<'_>,
        version: u32,
        guard: &LocalGuard<'_>,
    ) -> (*mut L, u32, bool) {
        let (advanced, new_version) = self.advance_to_key_generic(leaf, key, version, guard);

        if StdPtr::eq(advanced, leaf) {
            // Same leaf, new version - retry search
            (StdPtr::from_ref(leaf).cast_mut(), new_version, false)
        } else {
            // Different leaf - search there
            (StdPtr::from_ref(advanced).cast_mut(), new_version, true)
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "method signature kept for API consistency"
    )]
    fn check_blink_chain(&self, leaf: &L, target_ikey: u64) -> Option<*mut L> {
        let next_raw: *mut L = leaf.next_raw();

        // If leaf is deleted, we gotta follow the next ptr (unmarked) to find our key.
        // C++ ref: masstree_struct.hh:704 - "while (likely(!v.deleted()) && ..."
        // When deleted, the next ptr is marked but still valid.
        if leaf.version().is_deleted() {
            let next_ptr: *mut L = Linker::unmark_ptr(next_raw);

            if !next_ptr.is_null() {
                return Some(next_ptr);
            }

            // Deleted leaf with no successor, key can not exist
            return None;
        }

        // Normal case: follow B-link if key >= next leaf's bound
        // only follow unmarked ptr's (marked = being unlinked)
        let next_ptr: *mut L = Linker::unmark_ptr(next_raw);

        if !next_ptr.is_null() && !Linker::is_marked(next_raw) {
            // SAFETY: next_ptr is valid (protected by guard in caller)
            let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };

            if target_ikey >= next_bound {
                return Some(next_ptr);
            }
        }

        None
    }

    /// Check if sublayer is deleted before descending.
    ///
    /// Returns `true` if sublayer is valid, `false` if deleted (key not found).
    ///
    /// This check runs on every layer descent (hot path). Only finding a deleted
    /// sublayer is rare. The function is tiny (pointer cast + load), so inlining
    /// is always beneficial.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn check_sublayer_valid(&self, layer_ptr: *mut u8) -> bool {
        // SAFETY: ptr is non-null (came from valid slot) and protected by guard.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked")]
        let sublayer_version: &NodeVersion = unsafe { &*layer_ptr.cast::<NodeVersion>() };

        !sublayer_version.is_deleted()
    }
}

// ============================================================================
//  Public API
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Get a value by key.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`get_with_guard`](Self::get_with_guard) to amortize guard creation cost.
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<S::Output> {
        let guard = self.guard();
        self.get_with_guard(key, &guard)
    }

    /// Unified get implementation.
    ///
    /// Both `get_with_guard` and `get_ref` delegate to this function.
    /// The `extract` closure handles the difference in return type.
    ///
    /// # Type Parameters
    ///
    /// * `R` - Return type (`S::Output` or `&'g S::Value`)
    /// * `F` - Closure that extracts the value from a raw pointer
    #[inline(always)]
    fn get_impl<R, F>(&self, key: &mut Key<'_>, guard: &LocalGuard<'_>, extract: F) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        // Detect single-layer mode: key <= 8 bytes means no suffix, no layer descent
        // This enables a completely inline fast path without enum overhead
        if !key.has_suffix() {
            return self.get_impl_single_layer(key, guard, extract);
        }

        // Multi-layer path for keys > 8 bytes
        self.get_impl_multi_layer(key, guard, extract)
    }

    /// Handle landing on a deleted leaf during point read.
    ///
    /// Called when we detect `is_deleted()` at the start of leaf processing.
    /// Follows B-link chain to find the correct successor leaf.
    ///
    /// Returns the next valid leaf pointer, or restarts from root if no successor.
    #[cold]
    #[inline(never)]
    fn handle_deleted_leaf(
        &self,
        leaf: &L,
        layer_root: *const u8,
        key: &Key<'_>,
        is_sublayer: bool,
        guard: &LocalGuard<'_>,
    ) -> *mut L {
        // Try to follow B-link to successor
        let next_raw: *mut L = leaf.next_raw();
        let next_ptr: *mut L = Linker::unmark_ptr(next_raw);

        if !next_ptr.is_null() {
            // Follow to successor leaf
            return next_ptr;
        }

        // No successor mean we must restart from root
        // This can happen if the deleted leaf was the last in its layer
        self.reach_leaf_concurrent_generic(layer_root, key, is_sublayer, guard)
    }

    #[inline(always)]
    fn get_impl_single_layer<R, F>(
        &self,
        key: &Key<'_>,
        guard: &LocalGuard<'_>,
        extract: F,
    ) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        let layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // Traverse to leaf
        let mut leaf_ptr: *mut L =
            self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

        'leaf_loop: loop {
            // SAFETY: leaf_ptr protected by guard
            let leaf: &L = unsafe { &*leaf_ptr };

            // If we landed on a deleted leaf, follow B-link to successor.
            // This can happen during concurrent coalesce operations.
            if leaf.version().is_deleted() {
                leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            // Prefetch ikey cache lines while waiting for stable version.
            leaf.prefetch_for_search();

            // OPTIMIZATION: Use try_stable() to avoid spinning on locked leaf.
            // If leaf is locked, opportunistically check B-link chain - under
            // high contention, our key may have moved to a sibling leaf.
            let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                v
            } else {
                // Leaf is locked - check if key might be in sibling
                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                // No B-link escape route, must wait for lock
                leaf.version().stable()
            };

            'search_loop: loop {
                let perm = leaf.permutation();
                let size: usize = perm.size();
                let mut found_ptr: *mut u8 = StdPtr::null_mut();

                // Simple linear search - let LLVM decide optimal unrolling.
                // LLVM SHOULD auto unroll with #[inline(always)], and speculative batch
                // loads waste work on early matches.
                for i in 0..size {
                    let slot: usize = perm.get(i);

                    if (leaf.ikey(slot) == target_ikey) && (leaf.keylenx(slot) == search_keylenx) {
                        let ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                        if !ptr.is_null() {
                            found_ptr = ptr;

                            break;
                        }
                    }
                }

                // Version validation after all reads
                if leaf.version().has_changed(version) {
                    let (advanced, new_version) =
                        self.advance_to_key_generic(leaf, key, version, guard);

                    if !StdPtr::eq(advanced, leaf) {
                        leaf_ptr = StdPtr::from_ref(advanced).cast_mut();

                        continue 'leaf_loop;
                    }

                    version = new_version;

                    continue 'search_loop;
                }

                if !found_ptr.is_null() {
                    return Some(extract(found_ptr));
                }

                // Not found, check dirty or B-link (also handles deleted via check_blink_chain)
                if leaf.version().is_dirty() {
                    version = leaf.version().stable();

                    continue 'search_loop;
                }

                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                return None;
            }
        }
    }

    /// Multi-layer path for keys > 8 bytes.
    ///
    /// Handles layer descent, suffix matching, and complex key structures.
    #[inline(always)]
    fn get_impl_multi_layer<R, F>(
        &self,
        key: &mut Key<'_>,
        guard: &LocalGuard<'_>,
        extract: F,
    ) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            layer_root = self.maybe_parent_generic(layer_root);

            let mut leaf_ptr: *mut L =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            'leaf_loop: loop {
                let leaf: &L = unsafe { &*leaf_ptr };

                // If we landed on a deleted leaf, follow B-link to successor.
                // This can happen during concurrent coalesce operations.
                if leaf.version().is_deleted() {
                    leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, in_sublayer, guard);

                    continue 'leaf_loop;
                }

                // Prefetch ikey cache lines while waiting for stable version.
                leaf.prefetch_for_search();

                // OPTIMIZATION: Use try_stable() to avoid spinning on locked leaf.
                let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                    v
                } else {
                    let target_ikey: u64 = key.ikey();

                    if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                        leaf_ptr = next_ptr;

                        continue 'leaf_loop;
                    }

                    leaf.version().stable()
                };

                'search_loop: loop {
                    // Check for gc'd sublayer
                    if leaf.deleted_layer() {
                        key.unshift_all();
                        layer_root = self.load_root_ptr_generic(guard);
                        in_sublayer = false;

                        continue 'layer_loop;
                    }

                    let target_ikey: u64 = key.ikey();
                    let result: LookupResult = search_leaf_multi_layer::<S, L>(leaf, key);

                    if leaf.version().has_changed(version) {
                        let (new_ptr, new_version, changed_leaf) =
                            self.handle_version_change(leaf, key, version, guard);

                        if changed_leaf {
                            leaf_ptr = new_ptr;

                            continue 'leaf_loop;
                        }

                        version = new_version;

                        continue 'search_loop;
                    }

                    match result {
                        LookupResult::ValueSlot(slot) => {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                            return Some(extract(ptr));
                        }

                        LookupResult::Layer(ptr) => {
                            if !self.check_sublayer_valid(ptr) {
                                return None;
                            }

                            key.shift();
                            layer_root = ptr;
                            in_sublayer = true;

                            continue 'layer_loop;
                        }

                        LookupResult::NotFound => {
                            if leaf.version().is_dirty() {
                                version = leaf.version().stable();

                                continue 'search_loop;
                            }

                            // check_blink_chain now also handles deleted leaves
                            if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                                leaf_ptr = next_ptr;

                                continue 'leaf_loop;
                            }

                            return None;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
//  Reference-Returning API (Pointer-Backed Storage Only)
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot + RefValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Get a borrowed reference to a value by key.
    ///
    /// This is significantly faster than [`Self::get_with_guard`] for read-heavy workloads
    /// because it avoids atomic reference count operations (Arc clone/drop).
    ///
    /// # Performance
    ///
    /// Under high concurrency, `get_ref` can be **2-5x faster** than `get_with_guard`
    /// because it eliminates cache line bouncing on shared Arc reference counts.
    ///
    /// # Note
    ///
    /// This method is only available for pointer-backed storage modes
    /// (`MassTree24`, `MassTree15`, `MassTree24Inline`, `MassTree15Inline`).
    /// It is not available for true-inline storage (`MassTree24TrueInline`, etc.)
    /// because values are stored as bits, not at stable addresses.
    ///
    /// For true-inline trees, use `get` or `get_copy` instead.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(&V)` - A reference to the value, valid for the guard's lifetime
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get_ref<'g>(&self, key: &[u8], guard: &'g LocalGuard<'_>) -> Option<&'g S::Value> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_impl(&mut search_key, guard, |ptr: *mut u8| {
            // SAFETY: version validated, guard protects from deallocation
            unsafe { &*(ptr.cast::<S::Value>()) }
        })
    }
}
