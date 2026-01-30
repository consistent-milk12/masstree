use super::{
    InsertSearchResultGeneric, Key, LayerCapableLeaf, MassTreeGeneric, NodeAllocatorGeneric,
    ValueSlot,
};
use crate::leaf_trait::TreePermutation;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};

/// Threshold for switching from linear to binary search.
///
/// For leaves with WIDTH ≤ 16 entries, linear search outperforms binary search due to:
/// - Better branch prediction (sequential access pattern)
/// - Superior cache prefetching (contiguous memory access)
/// - No midpoint calculation overhead
///
/// C++ Masstree uses 16 as the threshold (`ksearch.hh`). Benchmarks confirm this
/// is optimal for modern CPUs with 64-byte cache lines (8 u64 ikeys per line).
const BINARY_SEARCH_THRESHOLD: usize = 16;

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // ========================================================================
    //  Binary Search Core (for WIDTH > 16)
    // ========================================================================

    /// Binary search to find the lower bound position for `target_ikey`.
    ///
    /// Returns the logical position where `target_ikey` starts (or should be inserted).
    /// This finds the LEFTMOST position where `ikey >= target_ikey`, which is necessary
    /// when multiple entries share the same ikey.
    ///
    /// # Algorithm
    ///
    /// Standard lower-bound binary search with 2 comparisons per iteration
    /// (vs 3 for equality-checking variants). The caller determines if a match
    /// exists by comparing the ikey at the returned position.
    ///
    /// # Memory Ordering
    ///
    /// Uses `Relaxed` ordering for ikey reads. The caller's permutation load
    /// with `Acquire` ordering establishes the necessary synchronization.
    ///
    /// # C++ Reference
    ///
    /// Matches `key_lower_bound_by` in `ksearch.hh:64-80`.
    #[inline(always)]
    fn binary_search_lower_bound(leaf: &L, perm: &L::Perm, target_ikey: u64) -> usize {
        let size: usize = perm.size();
        let mut lo: usize = 0;
        let mut hi: usize = size;

        while lo < hi {
            let mid: usize = lo + ((hi - lo) >> 1);
            let slot: usize = perm.get(mid);

            // Relaxed ordering - caller's permutation load provides synchronization
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            // Two-way comparison: fewer branches than three-way
            if slot_ikey < target_ikey {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        lo
    }

    // ========================================================================
    //  Linear Search Core (for WIDTH <= 16)
    // ========================================================================

    /// Linear search for insert position (small leaves, WIDTH ≤ 16).
    ///
    /// For small leaves, linear search outperforms binary search due to
    /// sequential memory access and better branch prediction.
    ///
    /// # Memory Ordering
    ///
    /// Uses `Relaxed` ordering for ikey reads. The caller's permutation load
    /// with `Acquire` ordering establishes the necessary synchronization.
    ///
    /// # C++ Reference
    ///
    /// Matches `key_find_lower_bound_by` in `ksearch.hh:106-121`.
    #[inline]
    fn linear_search_insert(
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
        search_keylenx: u8,
    ) -> InsertSearchResultGeneric {
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();

        for i in 0..size {
            let slot: usize = perm.get(i);

            // Relaxed ordering - caller's permutation load provides synchronization
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            if slot_ikey == target_ikey {
                // Check this slot; continue if it doesn't provide a definitive answer
                if let Some(result) =
                    Self::check_slot_for_insert(leaf, key, i, slot, search_keylenx)
                {
                    return result;
                }
                // Slot didn't match definitively, try next with same ikey
            } else if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    // ========================================================================
    //  Public Search API
    // ========================================================================

    /// Search for insert position in a leaf (generic version).
    ///
    /// Automatically selects the optimal search strategy:
    /// - Linear search for WIDTH ≤ 16 (better cache/branch behavior)
    /// - Binary search for WIDTH > 16 (better asymptotic complexity)
    ///
    /// # Compile-Time Optimization
    ///
    /// The `L::WIDTH <= BINARY_SEARCH_THRESHOLD` check is evaluated at compile time
    /// during monomorphization. The compiler eliminates the dead branch entirely,
    /// so there is no runtime cost for the strategy selection.
    ///
    /// # C++ Reference
    ///
    /// Matches `key_bound<max_size, bound_method_fast>` selection in `ksearch.hh`.
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "API consistency with other search methods"
    )]
    pub(super) fn search_for_insert_generic(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = if key.has_suffix() {
            KSUF_KEYLENX
        } else {
            key.current_len() as u8
        };

        // Compile-time constant: dead branch eliminated during monomorphization
        if L::WIDTH <= BINARY_SEARCH_THRESHOLD {
            return Self::linear_search_insert(leaf, key, perm, search_keylenx);
        }

        // Binary search for larger leaves (WIDTH > 16)
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();
        let start_pos: usize = Self::binary_search_lower_bound(leaf, perm, target_ikey);

        // Linear scan from lower bound to handle entries with matching ikey
        // (different keylenx values, layer pointers, suffix conflicts, etc.)
        for i in start_pos..size {
            let slot: usize = perm.get(i);

            // Relaxed ordering - caller's permutation load provides synchronization
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            // Stop when we pass the target ikey range
            if slot_ikey != target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            // Check this slot for a match
            if let Some(result) = Self::check_slot_for_insert(leaf, key, i, slot, search_keylenx) {
                return result;
            }
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    /// Check a single slot during insert search.
    ///
    /// Returns `Some(result)` if the slot provides a definitive answer,
    /// or `None` if we should continue scanning to the next slot.
    ///
    /// # Slot States
    ///
    /// - **Null pointer**: Slot is being concurrently modified. Return `None` to skip.
    /// - **Layer pointer** (`keylenx >= 128`): Descend if key has suffix, else insert before.
    /// - **Exact match**: Same ikey and keylenx (and suffix if applicable).
    /// - **Conflict**: Same ikey but incompatible keylenx requiring layer creation.
    ///
    /// # Memory Ordering
    ///
    /// Reads `keylenx` and `leaf_value_ptr` with implied ordering from the
    /// caller's permutation snapshot. A null pointer indicates the slot is
    /// mid-modification by another thread.
    #[inline]
    fn check_slot_for_insert(
        leaf: &L,
        key: &Key<'_>,
        logical_pos: usize,
        slot: usize,
        search_keylenx: u8,
    ) -> Option<InsertSearchResultGeneric> {
        let slot_keylenx: u8 = leaf.keylenx(slot);
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        // Null pointer indicates concurrent modification - skip this slot
        if slot_ptr.is_null() {
            return None;
        }

        // Layer pointer (keylenx >= 128) - descend if key has more bytes
        if slot_keylenx >= LAYER_KEYLENX {
            if key.has_suffix() {
                return Some(InsertSearchResultGeneric::Layer { slot });
            }
            // Key terminates here - it sorts before the layer pointer
            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        // Exact match check: same ikey AND same keylenx
        if slot_keylenx == search_keylenx {
            if slot_keylenx == KSUF_KEYLENX {
                // Both have suffixes (keylenx == 64) - compare suffix bytes
                return Some(Self::compare_suffixes(leaf, key, slot));
            }

            // Inline keys (keylenx 0-8) with matching length = same key
            return Some(InsertSearchResultGeneric::Found { slot });
        }

        // Same ikey, different keylenx - check for conflict
        let slot_has_suffix: bool = slot_keylenx == KSUF_KEYLENX;
        let key_has_suffix: bool = key.has_suffix();

        if slot_has_suffix && key_has_suffix {
            // Both have suffixes but different keylenx - need layer
            return Some(Self::make_conflict(slot));
        }

        // Distinct keys with same ikey - determine insertion point
        // Masstree ordering: shorter keys sort before longer keys
        if search_keylenx < slot_keylenx {
            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        // Our key is longer, continue scanning for correct position
        None
    }

    /// Compare suffix bytes for keys with `keylenx == KSUF_KEYLENX`.
    ///
    /// Returns `Found` on exact match, `Conflict` if suffixes differ.
    #[inline]
    fn compare_suffixes(leaf: &L, key: &Key<'_>, slot: usize) -> InsertSearchResultGeneric {
        let key_suffix = key.suffix();

        if let Some(slot_suffix) = leaf.ksuf(slot)
            && key_suffix == slot_suffix
        {
            return InsertSearchResultGeneric::Found { slot };
        }

        // Suffix mismatch or missing - need to create layer
        Self::make_conflict(slot)
    }

    /// Create a conflict result (cold path - suffix conflicts are rare).
    ///
    /// Marked `#[cold]` to hint the compiler to optimize for the non-conflict case.
    /// Not const because we want the cold/inline(never) attributes for code layout.
    #[cold]
    #[inline(never)]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "cold path optimization, const not beneficial"
    )]
    fn make_conflict(slot: usize) -> InsertSearchResultGeneric {
        InsertSearchResultGeneric::Conflict { slot }
    }

    // ========================================================================
    //  Single-Layer Fast Path (keys ≤ 8 bytes)
    // ========================================================================
    //
    // These functions duplicate the generic search logic but are optimized for
    // the common case of short keys that fit in a single layer (≤ 8 bytes).
    //
    // The duplication is intentional: single-layer search is a hot path and
    // benefits from avoiding suffix comparison logic and layer descent checks.
    // Attempting to unify with const generics would add complexity without
    // measurable performance benefit (the compiler already specializes well).

    /// Single-layer fast path for insert search (keys ≤ 8 bytes).
    ///
    /// Optimized version that:
    /// - Skips suffix comparison logic entirely
    /// - Never returns `Layer` or `Conflict` (only `Found` or `NotFound`)
    /// - Reduces code size in the hot path
    ///
    /// # Compile-Time Optimization
    ///
    /// The WIDTH check is evaluated at compile time during monomorphization.
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "API consistency with other search methods"
    )]
    pub(super) fn search_for_insert_single_layer(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // Compile-time constant: dead branch eliminated during monomorphization
        if L::WIDTH <= BINARY_SEARCH_THRESHOLD {
            return Self::linear_search_single_layer(leaf, key, perm, search_keylenx);
        }

        // Binary search for larger leaves (WIDTH > 16)
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();
        let start_pos: usize = Self::binary_search_lower_bound(leaf, perm, target_ikey);

        // Linear scan from lower bound
        for i in start_pos..size {
            let slot: usize = perm.get(i);
            // Relaxed ordering - caller's permutation load provides synchronization
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            // Stop when we pass the target ikey range
            if slot_ikey != target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            // Check this slot
            if let Some(result) = Self::check_slot_single_layer(leaf, i, slot, search_keylenx) {
                return result;
            }
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    /// Check a single slot during single-layer insert search.
    ///
    /// # Implicit Layer/Suffix Handling
    ///
    /// This function does NOT explicitly check for layer pointers (`keylenx >= 128`)
    /// or suffix markers (`keylenx == 64`). These are handled implicitly:
    ///
    /// - Single-layer mode means `search_keylenx` is 0-8
    /// - Layer pointers have `slot_keylenx >= 128`
    /// - Suffix markers have `slot_keylenx == 64`
    /// - The check `search_keylenx < slot_keylenx` catches both cases
    ///
    /// For example: `8 < 128` returns `NotFound` at the correct position.
    #[inline(always)]
    fn check_slot_single_layer(
        leaf: &L,
        logical_pos: usize,
        slot: usize,
        search_keylenx: u8,
    ) -> Option<InsertSearchResultGeneric> {
        let slot_keylenx: u8 = leaf.keylenx(slot);
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        // Null pointer indicates concurrent modification - skip
        if slot_ptr.is_null() {
            return None;
        }

        // Exact match - same ikey and keylenx
        if slot_keylenx == search_keylenx {
            return Some(InsertSearchResultGeneric::Found { slot });
        }

        // Different keylenx - shorter keys sort first
        // Implicitly handles layer pointers (>= 128) and suffix markers (== 64)
        if search_keylenx < slot_keylenx {
            Some(InsertSearchResultGeneric::NotFound { logical_pos })
        } else {
            // Our key is longer, continue scanning
            None
        }
    }

    /// Linear search for single-layer mode (small leaves, WIDTH ≤ 16).
    #[inline]
    fn linear_search_single_layer(
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
        search_keylenx: u8,
    ) -> InsertSearchResultGeneric {
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();

        for i in 0..size {
            let slot: usize = perm.get(i);
            // Relaxed ordering - caller's permutation load provides synchronization
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            if slot_ikey == target_ikey {
                // Check this slot; continue if not definitive
                if let Some(result) = Self::check_slot_single_layer(leaf, i, slot, search_keylenx) {
                    return result;
                }
                // Slot didn't match definitively, try next with same ikey
            } else if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }
}
