use std::cmp::Ordering;

use super::{
    InsertSearchResultGeneric, Key, LayerCapableLeaf, MassTreeGeneric, NodeAllocatorGeneric,
    ValueSlot,
};
use crate::leaf_trait::TreePermutation;
use crate::leaf24::{KSUF_KEYLENX, LAYER_KEYLENX};

/// Threshold for switching from linear to binary search.
/// C++ uses 16; we use the same for parity.
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

    /// Binary search to find the first position with matching ikey.
    ///
    /// Returns the logical position where `target_ikey` starts (or should be inserted),
    /// and whether any exact ikey match exists.
    ///
    /// Unlike simple binary search, this finds the LEFTMOST matching position,
    /// which is necessary when multiple entries share the same ikey.
    ///
    /// This matches C++ `key_lower_bound_by` in `ksearch.hh:64-80`.
    #[inline(always)]
    fn binary_search_ikey_lower_bound(leaf: &L, perm: &L::Perm, target_ikey: u64) -> (usize, bool) {
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
                    // Found a match, but continue searching left to find the first one
                    found = true;
                    r = m;
                }
                Ordering::Greater => l = m + 1,
            }
        }

        (l, found)
    }

    // ========================================================================
    //  Linear Search Core (for WIDTH <= 16)
    // ========================================================================

    /// Linear search for insert position (small leaves).
    ///
    /// For WIDTH <= 16, linear search is faster than binary due to:
    /// - No branch misprediction from binary pattern
    /// - Sequential memory access (better prefetching)
    /// - Simpler loop with no midpoint calculation
    ///
    /// This matches C++ `key_find_lower_bound_by` in `ksearch.hh:106-121`.
    #[inline(always)]
    fn linear_search_insert(
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
        search_keylenx: u8,
    ) -> InsertSearchResultGeneric {
        let target_ikey = key.ikey();
        let size = perm.size();

        for i in 0..size {
            let slot = perm.get(i);
            let slot_ikey = leaf.ikey(slot);

            if slot_ikey == target_ikey {
                // Check this slot; continue if it doesn't provide a definitive answer
                if let Some(result) =
                    Self::check_slot_for_insert(leaf, key, i, slot, search_keylenx)
                {
                    return result;
                }
                // Continue to next slot with same ikey
                continue;
            }

            if slot_ikey > target_ikey {
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
    /// Uses binary search for WIDTH > 16, linear search otherwise.
    /// This matches C++ `key_bound<max_size, bound_method_fast>` selection.
    #[inline] // Not #[inline(always)] - calls other inline fns, avoid cascading bloat
    #[expect(clippy::unused_self, reason = "API Consistency")]
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

        // C++ uses linear for WIDTH <= 16, binary for WIDTH > 16
        if L::WIDTH <= BINARY_SEARCH_THRESHOLD {
            return Self::linear_search_insert(leaf, key, perm, search_keylenx);
        }

        // Binary search for larger leaves (WIDTH > 16)
        let target_ikey = key.ikey();
        let size = perm.size();
        let (start_pos, found) = Self::binary_search_ikey_lower_bound(leaf, perm, target_ikey);

        if !found {
            return InsertSearchResultGeneric::NotFound {
                logical_pos: start_pos,
            };
        }

        // Linear scan from the first matching position to handle multiple entries
        // with the same ikey (different keylenx values, layer pointers, etc.)
        for i in start_pos..size {
            let slot = perm.get(i);
            let slot_ikey = leaf.ikey(slot);

            // Stop when we pass the matching ikeys
            if slot_ikey != target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            // Check this slot for a match
            if let Some(result) = Self::check_slot_for_insert(leaf, key, i, slot, search_keylenx) {
                return result;
            }
            // Slot didn't match, try next
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    /// Check a single slot during insert search.
    ///
    /// Returns `Some(result)` if the slot provides a definitive answer,
    /// or `None` if we should continue scanning.
    #[inline(always)]
    fn check_slot_for_insert(
        leaf: &L,
        key: &Key<'_>,
        logical_pos: usize,
        slot: usize,
        search_keylenx: u8,
    ) -> Option<InsertSearchResultGeneric> {
        let slot_keylenx = leaf.keylenx(slot);
        let slot_ptr = leaf.leaf_value_ptr(slot);

        // Null pointer means slot is being modified - skip and continue
        if slot_ptr.is_null() {
            return None;
        }

        // Layer pointer - descend if key has more bytes
        if slot_keylenx >= LAYER_KEYLENX {
            if key.has_suffix() {
                return Some(InsertSearchResultGeneric::Layer { slot });
            }
            // Key terminates here - it sorts before the layer pointer
            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        // Exact match check
        if slot_keylenx == search_keylenx {
            if slot_keylenx == KSUF_KEYLENX {
                // Both have suffixes - compare them
                let key_suffix = key.suffix();
                if let Some(slot_suffix) = leaf.ksuf(slot) {
                    if key_suffix == slot_suffix {
                        return Some(InsertSearchResultGeneric::Found { slot });
                    }
                    return Some(InsertSearchResultGeneric::Conflict { slot });
                }
                return Some(InsertSearchResultGeneric::Conflict { slot });
            }
            // Inline keys with matching keylenx = same key
            return Some(InsertSearchResultGeneric::Found { slot });
        }

        // Same ikey, different keylenx - check if conflict is needed
        let slot_has_suffix = slot_keylenx == KSUF_KEYLENX;
        let key_has_suffix = key.has_suffix();

        if slot_has_suffix && key_has_suffix {
            return Some(InsertSearchResultGeneric::Conflict { slot });
        }

        // Distinct keys with same ikey - determine if we should insert here
        // Masstree ordering: shorter keys sort before longer keys
        if search_keylenx < slot_keylenx {
            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        // Our key is longer, continue to find the right position
        None
    }

    /// Single-layer fast path for insert search (keys ≤ 8 bytes).
    ///
    /// Optimized version that:
    /// - Skips suffix comparison logic
    /// - Only returns `Found` or `NotFound` (never `Layer` or `Conflict`)
    ///
    /// Uses binary search for WIDTH > 16, linear search otherwise.
    #[inline] // Not #[inline(always)] - calls other inline fns, avoid cascading bloat
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub(super) fn search_for_insert_single_layer(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // C++ uses linear for WIDTH <= 16, binary for WIDTH > 16
        if L::WIDTH <= BINARY_SEARCH_THRESHOLD {
            return Self::linear_search_single_layer(leaf, key, perm, search_keylenx);
        }

        // Binary search for larger leaves (WIDTH > 16)
        let target_ikey = key.ikey();
        let size = perm.size();
        let (start_pos, found) = Self::binary_search_ikey_lower_bound(leaf, perm, target_ikey);

        if !found {
            return InsertSearchResultGeneric::NotFound {
                logical_pos: start_pos,
            };
        }

        // Linear scan from the first matching position
        for i in start_pos..size {
            let slot = perm.get(i);
            let slot_ikey = leaf.ikey(slot);

            // Stop when we pass the matching ikeys
            if slot_ikey != target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            // Check this slot
            if let Some(result) =
                Self::check_slot_for_insert_single_layer(leaf, i, slot, search_keylenx)
            {
                return result;
            }
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    /// Check a single slot during single-layer insert search.
    ///
    /// # Layer Pointer Handling
    ///
    /// This function does NOT explicitly check for layer pointers (`keylenx >= 128`).
    /// Layer pointers are handled implicitly by the ordering comparison:
    /// - Single-layer mode means `search_keylenx` is 0-8
    /// - Layer pointers have `slot_keylenx >= 128`
    /// - The check `search_keylenx < slot_keylenx` (e.g., `8 < 128`) returns `NotFound`
    ///
    /// This also correctly handles suffix markers (`keylenx == 64`).
    #[inline(always)]
    fn check_slot_for_insert_single_layer(
        leaf: &L,
        logical_pos: usize,
        slot: usize,
        search_keylenx: u8,
    ) -> Option<InsertSearchResultGeneric> {
        let slot_keylenx = leaf.keylenx(slot);
        let slot_ptr = leaf.leaf_value_ptr(slot);

        if slot_ptr.is_null() {
            return None;
        }

        // Exact match - same ikey and keylenx
        if slot_keylenx == search_keylenx {
            return Some(InsertSearchResultGeneric::Found { slot });
        }

        // Same ikey, different keylenx - shorter keys sort first
        // This implicitly handles:
        // - Layer pointers (keylenx >= 128): 0-8 < 128, returns NotFound
        // - Suffix markers (keylenx == 64): 0-8 < 64, returns NotFound
        // - Longer inline keys: returns NotFound at correct position
        if search_keylenx < slot_keylenx {
            Some(InsertSearchResultGeneric::NotFound { logical_pos })
        } else {
            // Our key is longer, continue scanning
            None
        }
    }

    /// Linear search for single-layer mode (small leaves).
    #[inline(always)]
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
            let slot_ikey: u64 = leaf.ikey(slot);

            if slot_ikey == target_ikey {
                // Check this slot; continue if it doesn't provide a definitive answer
                if let Some(result) =
                    Self::check_slot_for_insert_single_layer(leaf, i, slot, search_keylenx)
                {
                    return result;
                }
                // Continue to next slot with same ikey
                continue;
            }

            if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }
}
