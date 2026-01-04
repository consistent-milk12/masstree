//! ========================================================================
//!  Generic Optimistic Read Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - Linear search by default (predictable branches, cache-friendly)
//! - Optional SIMD search with `simd` feature flag
//! - Unified implementation via closure for value extraction

use std::ptr as StdPtr;

use super::{
    Key, LayerCapableLeaf, LocalGuard, MassTreeGeneric, NodeAllocatorGeneric, NodeVersion,
    ValueSlot,
};

use crate::leaf_trait::TreePermutation;
use crate::leaf24::KSUF_KEYLENX;
use crate::leaf24::LAYER_KEYLENX;
use crate::link::{is_marked, unmark_ptr};

// ============================================================================
//  LookupResult - Search outcome enum
// ============================================================================

/// Result of searching a leaf node for a key.
///
/// This enum captures the three possible outcomes without interpreting
/// the pointer until after version validation.
enum LookupResult {
    /// Found a value pointer. The `keylenx` confirms it's a value (< `LAYER_KEYLENX`).
    Value(*mut u8),

    /// Found a layer pointer. Need to descend into sublayer.
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
            if let Some(result) = check_slot_match(leaf, s0, search_keylenx, key) {
                return result;
            }
        }

        if ikey1 == target_ikey {
            if let Some(result) = check_slot_match(leaf, s1, search_keylenx, key) {
                return result;
            }
        }

        if ikey2 == target_ikey {
            if let Some(result) = check_slot_match(leaf, s2, search_keylenx, key) {
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
            if let Some(result) = check_slot_match(leaf, slot, search_keylenx, key) {
                return result;
            }
        }

        i += 1;
    }

    LookupResult::NotFound
}

/// Check a slot where ikey already matched. Verifies keylenx and suffix.
///
/// Returns `Some(LookupResult)` if the slot is a value or layer pointer,
/// `None` to continue searching.
#[inline(always)]
fn check_slot_match<S, L>(
    leaf: &L,
    slot: usize,
    search_keylenx: u8,
    key: &Key<'_>,
) -> Option<LookupResult>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let slot_keylenx: u8 = leaf.keylenx(slot);
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

    if slot_ptr.is_null() {
        return None;
    }

    if slot_keylenx == search_keylenx {
        // Potential exact match - verify suffix if present
        let suffix_match: bool = if slot_keylenx == KSUF_KEYLENX {
            leaf.ksuf_equals(slot, key.suffix())
        } else {
            true
        };

        if suffix_match {
            return Some(LookupResult::Value(slot_ptr));
        }
    } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
        // Layer pointer - record for descent after validation
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

    /// Check if key should be in a sibling leaf via B-link.
    ///
    /// Returns `Some(next_leaf_ptr)` if we should follow the B-link,
    /// `None` if key is definitively not found.
    ///
    /// Marked `#[cold]` because B-link traversal is rare in the common case
    /// (no concurrent splits). Keeps the hot path code smaller for better
    /// instruction cache utilization.
    #[cold]
    #[inline(never)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn check_blink_chain(&self, leaf: &L, target_ikey: u64) -> Option<*mut L> {
        let next_raw: *mut L = leaf.next_raw();
        let next_ptr: *mut L = unmark_ptr(next_raw);

        if !next_ptr.is_null() && !is_marked(next_raw) {
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
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
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

    /// Get a value by key using an explicit guard.
    ///
    /// Use this when performing multiple operations to amortize guard overhead.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<S::Output> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_impl(&mut search_key, guard, |ptr| {
            // SAFETY: version validated, ptr points to valid value
            unsafe { S::output_from_raw(ptr) }
        })
    }

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
        self.get_impl(&mut search_key, guard, |ptr| {
            // SAFETY: version validated, guard protects from deallocation
            unsafe { &*(ptr.cast::<S::Value>()) }
        })
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

    /// Single-layer fast path (keys ≤ 8 bytes).
    ///
    /// Completely inline search without `LookupResult` enum overhead.
    /// This is the hot path for most workloads.
    #[inline(always)]
    #[expect(clippy::too_many_lines, reason = "Verbose unrolling of loop")]
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

            // Prefetch ikey cache lines while waiting for stable version.
            // This hides memory latency if the node is locked (version spinning).
            leaf.prefetch_for_search();

            let mut version: u32 = leaf.version().stable();

            'search_loop: loop {
                // Optimized linear search with loop unrolling (3 at a time)
                // Speculative batch load: load slots and ikeys upfront for better ILP
                let perm = leaf.permutation();
                let size = perm.size();
                let mut found_ptr: *mut u8 = std::ptr::null_mut();
                let mut i: usize = 0;

                // Unrolled loop: process 3 slots per iteration
                'unrolled: while i + 3 <= size {
                    // Batch load slots (bit extraction only, no memory access)
                    let s0: usize = perm.get(i);
                    let s1: usize = perm.get(i + 1);
                    let s2: usize = perm.get(i + 2);

                    // Batch load ikeys (memory loads can be issued in parallel)
                    let ikey0: u64 = leaf.ikey(s0);
                    let ikey1: u64 = leaf.ikey(s1);
                    let ikey2: u64 = leaf.ikey(s2);

                    // Check slot 0
                    if ikey0 == target_ikey {
                        let kx0: u8 = leaf.keylenx(s0);

                        if kx0 == search_keylenx {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(s0);

                            if !ptr.is_null() {
                                found_ptr = ptr;
                                break 'unrolled;
                            }
                        }
                    }

                    // Check slot 1
                    if ikey1 == target_ikey {
                        let kx1: u8 = leaf.keylenx(s1);

                        if kx1 == search_keylenx {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(s1);

                            if !ptr.is_null() {
                                found_ptr = ptr;
                                break 'unrolled;
                            }
                        }
                    }

                    // Check slot 2
                    if ikey2 == target_ikey {
                        let kx2: u8 = leaf.keylenx(s2);

                        if kx2 == search_keylenx {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(s2);

                            if !ptr.is_null() {
                                found_ptr = ptr;
                                break 'unrolled;
                            }
                        }
                    }

                    i += 3;
                }

                // Handle remainder (0-2 elements)
                while i < size && found_ptr.is_null() {
                    let slot: usize = perm.get(i);
                    let slot_ikey: u64 = leaf.ikey(slot);

                    if slot_ikey == target_ikey {
                        let slot_keylenx: u8 = leaf.keylenx(slot);

                        if slot_keylenx == search_keylenx {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                            if !ptr.is_null() {
                                found_ptr = ptr;
                            }
                        }
                    }
                    i += 1;
                }

                // Version validation AFTER all reads
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

                // Version validated - interpret result
                if !found_ptr.is_null() {
                    return Some(extract(found_ptr));
                }

                // Not found - check dirty or B-link
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
    #[inline]
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

                // Prefetch ikey cache lines while waiting for stable version.
                leaf.prefetch_for_search();

                let mut version: u32 = leaf.version().stable();

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
                        LookupResult::Value(ptr) => {
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
