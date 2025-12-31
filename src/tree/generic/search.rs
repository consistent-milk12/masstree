use super::{
    InsertSearchResultGeneric, Key, LayerCapableLeaf, MassTreeGeneric, NodeAllocatorGeneric,
    ValueSlot,
};

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Search for insert position in a leaf (generic version).
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub(super) fn search_for_insert_generic(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;
        use std::cmp::Ordering;

        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = if key.has_suffix() {
            KSUF_KEYLENX
        } else {
            key.current_len() as u8
        };

        for i in 0..perm.size() {
            let slot: usize = perm.get(i);
            let slot_ikey: u64 = leaf.ikey(slot);

            if slot_ikey == target_ikey {
                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                if slot_ptr.is_null() {
                    continue;
                }

                // Layer pointer - only descend if the new key has more bytes
                if slot_keylenx >= LAYER_KEYLENX {
                    if key.has_suffix() {
                        // Key has more bytes - descend into the layer
                        return InsertSearchResultGeneric::Layer { slot };
                    }
                    // Key terminates here - it must sort before the layer pointer.
                    return InsertSearchResultGeneric::NotFound { logical_pos: i };
                }

                // Exact match check
                if slot_keylenx == search_keylenx {
                    if slot_keylenx == KSUF_KEYLENX {
                        // Both have suffixes - compare them
                        let key_suffix: &[u8] = key.suffix();
                        if let Some(slot_suffix) = leaf.ksuf(slot) {
                            if key_suffix == slot_suffix {
                                // Same suffix = same key
                                return InsertSearchResultGeneric::Found { slot };
                            }
                            // Different suffixes = conflict, need layer
                            return InsertSearchResultGeneric::Conflict { slot };
                        }
                        // No stored suffix (shouldn't happen for KSUF_KEYLENX)
                        // but treat as conflict to be safe
                        return InsertSearchResultGeneric::Conflict { slot };
                    }
                    // Inline keys (no suffix) with matching keylenx = same key
                    return InsertSearchResultGeneric::Found { slot };
                }

                // Same ikey, different keylenx - check if conflict is needed
                let slot_has_suffix: bool = slot_keylenx == KSUF_KEYLENX;
                let key_has_suffix: bool = key.has_suffix();

                if slot_has_suffix && key_has_suffix {
                    // Both have suffixes with same 8-byte prefix - need layer
                    return InsertSearchResultGeneric::Conflict { slot };
                }

                // Distinct keys with the same ikey: insertion order is determined by
                // Masstree `key.compare(ikey, keylenx)` semantics (length vs keylenx).
                if key.compare(slot_ikey, slot_keylenx as usize) == Ordering::Less {
                    return InsertSearchResultGeneric::NotFound { logical_pos: i };
                }
            }

            // Sorted order - found insert position
            if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound {
            logical_pos: perm.size(),
        }
    }

    /// Single-layer fast path for insert search (keys ≤ 8 bytes).
    ///
    /// Optimized version that:
    /// - Skips suffix comparison logic
    /// - Only returns `Found` or `NotFound` (never `Layer` or `Conflict`)
    ///
    /// For layer pointers: an 8-byte key sorts BEFORE a layer pointer with
    /// the same ikey (layer pointers handle keys > 8 bytes).
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub(super) fn search_for_insert_single_layer(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::LAYER_KEYLENX;

        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        for i in 0..perm.size() {
            let slot: usize = perm.get(i);
            let slot_ikey: u64 = leaf.ikey(slot);

            if slot_ikey == target_ikey {
                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                if slot_ptr.is_null() {
                    continue;
                }

                // Layer pointer: short key (≤8 bytes) sorts before layer pointer
                // (layer pointers handle keys > 8 bytes with same prefix)
                if slot_keylenx >= LAYER_KEYLENX {
                    return InsertSearchResultGeneric::NotFound { logical_pos: i };
                }

                // Exact match - same ikey and keylenx
                if slot_keylenx == search_keylenx {
                    return InsertSearchResultGeneric::Found { slot };
                }

                // Same ikey, different keylenx - check insertion order
                // For single-layer, shorter keys sort before longer keys
                if search_keylenx < slot_keylenx {
                    return InsertSearchResultGeneric::NotFound { logical_pos: i };
                }
            }

            // Sorted order - found insert position
            if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound {
            logical_pos: perm.size(),
        }
    }
}
