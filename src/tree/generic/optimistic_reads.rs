//! ========================================================================
//!  Generic Optimistic Read Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - Binary search for WIDTH > 16 (matching insert path)
//! - Unified implementation via closure for value extraction

use std::cmp::Ordering;

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
//  Binary Search Core (for WIDTH > 16)
// ============================================================================

/// Threshold for switching from linear to binary search.
/// C++ uses 16; we use the same for parity with insert path.
const BINARY_SEARCH_THRESHOLD: usize = 16;

/// Binary search to find the first position with matching ikey.
///
/// Returns the logical position where `target_ikey` starts (or would be),
/// and whether any exact ikey match exists.
#[inline(always)]
fn binary_search_ikey<L, S>(leaf: &L, perm: &L::Perm, target_ikey: u64) -> (usize, bool)
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let size = perm.size();
    let mut l: usize = 0;
    let mut r: usize = size;
    let mut found = false;

    while l < r {
        let m = (l + r) >> 1;
        let slot = perm.get(m);
        let slot_ikey = leaf.ikey(slot);

        match target_ikey.cmp(&slot_ikey) {
            Ordering::Less => r = m,
            Ordering::Equal => {
                found = true;
                r = m; // Continue left to find first match
            }
            Ordering::Greater => l = m + 1,
        }
    }

    (l, found)
}

// ============================================================================
//  Search Helpers (Multi-Layer Path)
// ============================================================================

/// Check a single slot for a value/layer match (multi-layer mode).
#[inline(always)]
fn check_slot_multi_layer<L, S>(
    leaf: &L,
    slot: usize,
    key: &Key<'_>,
    search_keylenx: u8,
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
        return None; // Slot being modified, continue
    }

    if slot_keylenx == search_keylenx {
        // Potential exact match - verify suffix if present
        if slot_keylenx == KSUF_KEYLENX {
            if leaf.ksuf_equals(slot, key.suffix()) {
                return Some(LookupResult::Value(slot_ptr));
            }
        } else {
            return Some(LookupResult::Value(slot_ptr));
        }
    } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
        // Layer pointer - record for descent after validation
        return Some(LookupResult::Layer(slot_ptr));
    }

    None
}

/// Minimum entries before binary search is worthwhile.
/// Below this threshold, linear scan is faster due to lower overhead.
const BINARY_SEARCH_MIN_SIZE: usize = 12;

/// Search a leaf for a key in multi-layer mode (keys > 8 bytes).
///
/// Uses binary search for WIDTH > 16 AND size >= 12, linear search otherwise.
#[inline(always)]
fn search_leaf_multi_layer<S, L>(leaf: &L, key: &Key<'_>) -> LookupResult
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let perm = leaf.permutation();
    let target_ikey: u64 = key.ikey();
    let size = perm.size();

    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
    let search_keylenx: u8 = if key.has_suffix() {
        KSUF_KEYLENX
    } else {
        key.current_len() as u8
    };

    // Use binary search only for leaves with enough entries to benefit
    if L::WIDTH > BINARY_SEARCH_THRESHOLD && size >= BINARY_SEARCH_MIN_SIZE {
        let (start_pos, found) = binary_search_ikey::<L, S>(leaf, &perm, target_ikey);

        if !found {
            return LookupResult::NotFound;
        }

        // Linear scan from first match
        for i in start_pos..size {
            let slot = perm.get(i);
            if leaf.ikey(slot) != target_ikey {
                break;
            }
            if let Some(result) = check_slot_multi_layer::<L, S>(leaf, slot, key, search_keylenx) {
                return result;
            }
        }

        return LookupResult::NotFound;
    }

    // Linear search for small leaves
    for i in 0..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

        if slot_ikey == target_ikey {
            if let Some(result) = check_slot_multi_layer::<L, S>(leaf, slot, key, search_keylenx) {
                return result;
            }
        }
    }

    LookupResult::NotFound
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

        if std::ptr::eq(advanced, leaf) {
            // Same leaf, new version - retry search
            (std::ptr::from_ref(leaf).cast_mut(), new_version, false)
        } else {
            // Different leaf - search there
            (std::ptr::from_ref(advanced).cast_mut(), new_version, true)
        }
    }

    /// Check if key should be in a sibling leaf via B-link.
    ///
    /// Returns `Some(next_leaf_ptr)` if we should follow the B-link,
    /// `None` if key is definitively not found.
    ///
    /// Note: NOT marked #[cold] - B-link traversal is common during concurrent splits.
    #[inline(always)]
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
    /// Note: NOT marked #[cold] - layer descent is hot path for keys > 8 bytes.
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
    /// This is significantly faster than [`get_with_guard`] for read-heavy workloads
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
            let mut version: u32 = leaf.version().stable();

            'search_loop: loop {
                // Inline linear search - no function calls, no enum
                let perm = leaf.permutation();
                let size = perm.size();
                let mut found_ptr: *mut u8 = std::ptr::null_mut();

                for i in 0..size {
                    let slot: usize = perm.get(i);
                    let slot_ikey: u64 = leaf.ikey(slot);

                    if slot_ikey == target_ikey {
                        let slot_keylenx: u8 = leaf.keylenx(slot);
                        if slot_keylenx == search_keylenx {
                            let ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                            if !ptr.is_null() {
                                found_ptr = ptr;
                                break;
                            }
                        }
                        // Layer pointer (keylenx >= 128) with matching ikey means
                        // a longer key exists, but our short key is NOT a match
                    }
                }

                // Version validation AFTER all reads
                if leaf.version().has_changed(version) {
                    let (advanced, new_version) =
                        self.advance_to_key_generic(leaf, key, version, guard);

                    if !std::ptr::eq(advanced, leaf) {
                        leaf_ptr = std::ptr::from_ref(advanced).cast_mut();
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
