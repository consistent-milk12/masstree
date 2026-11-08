use super::{InsertSearchResultGeneric, Key, LeafPolicy, MassTreeGeneric, TreeAllocator};
use crate::leaf_trait::TreeLeafNode;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Linear search for insert position (small leaves, WIDTH ≤ 16).
    ///
    /// 3-at-a-time unrolled to reduce loop overhead. The permutation is sorted,
    /// so we can early-exit when any slot's ikey exceeds the target.
    #[inline]
    fn search_for_insert<const SINGLE_LAYER: bool>(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> InsertSearchResultGeneric {
        debug_assert!(!SINGLE_LAYER || !key.has_suffix());
        #[expect(clippy::cast_possible_truncation, reason = "unsuffixed length <= 8")]
        let search_keylenx = if !SINGLE_LAYER && key.has_suffix() {
            KSUF_KEYLENX
        } else {
            key.current_len() as u8
        };
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();
        let mut i: usize = 0;

        // 3-at-a-time unrolled loop
        while i + 3 <= size {
            let s0: usize = perm.get(i);
            let s1: usize = perm.get(i + 1);
            let s2: usize = perm.get(i + 2);

            let ik0: u64 = leaf.ikey_relaxed(s0);
            let ik1: u64 = leaf.ikey_relaxed(s1);
            let ik2: u64 = leaf.ikey_relaxed(s2);

            // Slot 0
            if ik0 == target_ikey {
                if let Some(result) =
                    Self::check_slot_for_insert::<SINGLE_LAYER>(leaf, key, i, s0, search_keylenx)
                {
                    return result;
                }
            } else if ik0 > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            // Slot 1
            if ik1 == target_ikey {
                if let Some(result) = Self::check_slot_for_insert::<SINGLE_LAYER>(
                    leaf,
                    key,
                    i + 1,
                    s1,
                    search_keylenx,
                ) {
                    return result;
                }
            } else if ik1 > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i + 1 };
            }

            // Slot 2
            if ik2 == target_ikey {
                if let Some(result) = Self::check_slot_for_insert::<SINGLE_LAYER>(
                    leaf,
                    key,
                    i + 2,
                    s2,
                    search_keylenx,
                ) {
                    return result;
                }
            } else if ik2 > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i + 2 };
            }

            i += 3;
        }

        // Tail: 0-2 remaining slots
        while i < size {
            let slot: usize = perm.get(i);
            let slot_ikey: u64 = leaf.ikey_relaxed(slot);

            if slot_ikey == target_ikey {
                if let Some(result) =
                    Self::check_slot_for_insert::<SINGLE_LAYER>(leaf, key, i, slot, search_keylenx)
                {
                    return result;
                }
            } else if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }

            i += 1;
        }

        InsertSearchResultGeneric::NotFound { logical_pos: size }
    }

    // ========================================================================
    //  Public Search API
    // ========================================================================

    /// Search for insert position in a leaf (generic version).
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "API consistency with other search methods"
    )]
    pub(super) fn search_for_insert_generic(
        &self,
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> InsertSearchResultGeneric {
        Self::search_for_insert::<false>(leaf, key, perm)
    }

    /// Check a single slot during insert search.
    #[inline]
    fn check_slot_for_insert<const SINGLE_LAYER: bool>(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        logical_pos: usize,
        slot: usize,
        search_keylenx: u8,
    ) -> Option<InsertSearchResultGeneric> {
        let slot_keylenx: u8 = leaf.keylenx_relaxed(slot);

        if leaf.is_value_empty_relaxed(slot) {
            return None;
        }

        if !SINGLE_LAYER && slot_keylenx >= LAYER_KEYLENX {
            if key.has_suffix() {
                return Some(InsertSearchResultGeneric::Layer { slot });
            }

            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        if slot_keylenx == search_keylenx {
            if !SINGLE_LAYER && slot_keylenx == KSUF_KEYLENX {
                return Some(Self::compare_suffixes(leaf, key, slot));
            }

            return Some(InsertSearchResultGeneric::Found { slot });
        }

        if search_keylenx < slot_keylenx {
            return Some(InsertSearchResultGeneric::NotFound { logical_pos });
        }

        None
    }

    /// Compare suffix bytes for keys with `keylenx == KSUF_KEYLENX`.
    #[inline]
    fn compare_suffixes(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        slot: usize,
    ) -> InsertSearchResultGeneric {
        let key_suffix = key.suffix();

        if let Some(slot_suffix) = leaf.ksuf(slot)
            && key_suffix == slot_suffix
        {
            return InsertSearchResultGeneric::Found { slot };
        }

        Self::make_conflict(slot)
    }

    /// Create a conflict result (cold path - suffix conflicts are rare).
    #[cold]
    #[inline(never)]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "cold path optimization, const not beneficial"
    )]
    fn make_conflict(slot: usize) -> InsertSearchResultGeneric {
        InsertSearchResultGeneric::Conflict { slot }
    }

    /// Single-layer fast path for insert search (keys ≤ 8 bytes).
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "API consistency with other search methods"
    )]
    pub(super) fn search_for_insert_single_layer(
        &self,
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> InsertSearchResultGeneric {
        Self::search_for_insert::<true>(leaf, key, perm)
    }
}
