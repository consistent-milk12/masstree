//! Filepath: src/tree/range/helper.rs
//!
//! Lower bound search functions and scan helpers.

use std::cmp::Ordering;

use seize::LocalGuard;

use crate::IKEY_SIZE;
use crate::leaf_trait::TreeLeafNode;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::policy::LeafPolicy;

use super::cursor_key::CursorKey;

// ============================================================================
//  KeyIndexedPosition
// ============================================================================

/// Result of a lower bound search within a leaf.
///
/// Contains both the logical insertion position and optionally the physical
/// slot if an exact ikey match was found.
#[derive(Debug, Clone, Copy)]
pub struct KeyIndexedPosition {
    /// Logical position for insertion (`0..=perm.size()`).
    pub i: usize,

    /// Physical slot if exact ikey match found, else `None`.
    pub p: Option<usize>,
}

impl KeyIndexedPosition {
    /// Create a new position with no exact match.
    #[inline(always)]
    pub const fn not_found(i: usize) -> Self {
        Self { i, p: None }
    }

    /// Create a new position with an exact match.
    #[inline(always)]
    pub const fn found(i: usize, slot: usize) -> Self {
        Self { i, p: Some(slot) }
    }
}

// ============================================================================
//  Forward Scan Helpers
// ============================================================================

/// Check if suffix comparison result allows emit for initial position.
///
/// For forward scans: emit if stored > search, or stored == search and `emit_equal`.
#[inline(always)]
pub const fn initial_ksuf_match(ksuf_compare: Ordering, emit_equal: bool) -> bool {
    match ksuf_compare {
        Ordering::Greater => true,
        Ordering::Equal => emit_equal,
        Ordering::Less => false,
    }
}

// ============================================================================
//  Full-Key Comparison
// ============================================================================

/// Compare the cursor key against a leaf slot using full-key order.
///
/// The ikey comparison alone is insufficient: several distinct keys can share
/// the same first eight bytes (zero-padded short keys, a terminal key and a
/// longer suffix key, or a layer prefix). This routine orders the cursor
/// against the whole slot key:
///
/// - terminal inline keys (`keylenx <= 8`) sort by length (shorter is a prefix
///   and therefore smaller);
/// - suffix keys (`KSUF`) sort after every terminal key with the same ikey,
///   then by suffix bytes;
/// - a layer pointer sorts after terminal keys with the same ikey and compares
///   equal to a cursor with a real suffix (the caller must descend).
#[inline]
pub fn compare_cursor_to_slot<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    slot: usize,
) -> Ordering
where
    P: LeafPolicy,
{
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);

    match cursor_key.current_ikey().cmp(&slot_ikey) {
        Ordering::Less => return Ordering::Less,
        Ordering::Greater => return Ordering::Greater,
        Ordering::Equal => {}
    }

    let keylenx: u8 = leaf.keylenx_relaxed(slot);
    let cursor_len: usize = cursor_key.current_len();

    // Unshift sentinel: the cursor has already consumed this layer pointer.
    // It must compare after any remaining terminal suffix siblings.
    if cursor_len == crate::tree::range::cursor_key::UNSHIFT_SENTINEL_LEN {
        return if keylenx >= LAYER_KEYLENX {
            Ordering::Equal
        } else {
            Ordering::Greater
        };
    }

    if keylenx >= LAYER_KEYLENX {
        // Layer keys are strictly greater than the bare ikey. A cursor with a
        // suffix may be anywhere inside the layer, so report Equal and let the
        // caller descend.
        if cursor_len > IKEY_SIZE {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    } else if keylenx == KSUF_KEYLENX {
        if cursor_len <= IKEY_SIZE {
            // Terminal cursor is a prefix of the longer key.
            Ordering::Less
        } else {
            let stored: &[u8] = leaf.ksuf(slot).unwrap_or(&[]);
            cursor_key.suffix().cmp(stored)
        }
    } else if cursor_len > IKEY_SIZE {
        // Cursor has a suffix; every terminal key with the same ikey is
        // smaller than it.
        Ordering::Greater
    } else {
        cursor_len.cmp(&(keylenx as usize))
    }
}

/// Find the first logical position whose full key is `>=` (or `>`) the cursor.
///
/// `skip_equal == true` means "first key strictly greater than the cursor";
/// this is used once the cursor key has already been emitted. Otherwise the
/// first key equal to the cursor is returned.
#[inline]
pub fn lower_bound_full<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    skip_equal: bool,
) -> usize
where
    P: LeafPolicy,
{
    let size: usize = perm.size();

    for i in 0..size {
        let slot: usize = perm.get(i);

        match compare_cursor_to_slot(cursor_key, leaf, slot) {
            Ordering::Less => return i,
            Ordering::Equal => return if skip_equal { i + 1 } else { i },
            Ordering::Greater => {}
        }
    }

    size
}

/// Find the last logical position whose full key is `<=` (or `<`) the cursor.
///
/// Returns `-1` when no slot is at or below the cursor.
#[inline]
pub fn upper_bound_reverse_full<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    include_equal: bool,
) -> isize
where
    P: LeafPolicy,
{
    let size: usize = perm.size();
    let mut best: isize = -1;

    for i in 0..size {
        let slot: usize = perm.get(i);
        leaf.prefetch_value(slot);

        match compare_cursor_to_slot(cursor_key, leaf, slot) {
            // cursor > slot: the slot is at or below the bound.
            Ordering::Greater => best = i.cast_signed(),
            Ordering::Equal => {
                if include_equal {
                    best = i.cast_signed();
                }
            }
            // cursor < slot: this and every later slot are above the bound.
            Ordering::Less => break,
        }
    }

    best
}

// ============================================================================
//  Lower Bound Search
// ============================================================================

/// Find the lower bound position for a key within a leaf.
#[inline]
pub fn lower_with_position<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
) -> KeyIndexedPosition
where
    P: LeafPolicy,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();

    for i in 0..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        match search_ikey.cmp(&slot_ikey) {
            Ordering::Less => {
                // Search key < slot key: insert before this position
                return KeyIndexedPosition::not_found(i);
            }

            Ordering::Equal => {
                // Exact ikey match found
                return KeyIndexedPosition::found(i, slot);
            }

            Ordering::Greater => {
                // Search key > slot key: continue searching
            }
        }
    }

    // Search key >= all slots: insert at end
    KeyIndexedPosition::not_found(size)
}

/// Find the lower bound position with full key comparison (ikey + suffix).
///
/// Returns the logical position (`0..=perm.size()`) where scanning should
/// resume after accounting for suffix ordering. Unlike [`lower_with_position`],
/// the physical slot is not returned because callers only need the position.
#[inline]
pub fn lower_with_suffix<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
) -> usize
where
    P: LeafPolicy,
{
    let base_pos: KeyIndexedPosition = lower_with_position(cursor_key, leaf, perm);

    // If no ikey match, position is correct
    let Some(slot) = base_pos.p else {
        return base_pos.i;
    };

    // Check if suffix comparison is needed
    let keylenx: u8 = leaf.keylenx_relaxed(slot);

    // Layer pointers: if cursor has suffix (including after unshift()),
    // we've already processed this layer and should skip past it
    if keylenx >= LAYER_KEYLENX {
        if cursor_key.has_suffix() {
            // Cursor has suffix (or is post-unshift sentinel) -> skip layer pointer
            return base_pos.i + 1;
        }
        return base_pos.i;
    }

    // Handle inline keys (no suffix on either side)
    if !cursor_key.has_suffix() {
        // Compare lengths for inline keys
        let cursor_len = cursor_key.current_len();
        let slot_len = keylenx as usize;

        return match cursor_len.cmp(&slot_len) {
            Ordering::Less => base_pos.i, // cursor < slot, position correct
            Ordering::Equal => {
                // Exact match - skip to next position
                base_pos.i + 1
            }
            Ordering::Greater => {
                // cursor > slot, find position after slots with same ikey
                find_position_after_inline(cursor_key, leaf, perm, base_pos.i)
            }
        };
    }

    // If slot has suffix, compare suffix bytes
    if keylenx == KSUF_KEYLENX
        && let Some(stored_suffix) = leaf.ksuf(slot)
    {
        let cursor_suffix: &[u8] = cursor_key.suffix();

        match cursor_suffix.cmp(stored_suffix) {
            Ordering::Less => {
                // Cursor suffix < stored: position is correct (insert before)
                return base_pos.i;
            }

            Ordering::Equal => {
                // Exact match - we already emitted this key, skip to next position
                return base_pos.i + 1;
            }

            Ordering::Greater => {
                // Cursor suffix > stored: insert after this slot
                // But we need to check if there are more slots with same ikey
                return find_position_after_suffix(cursor_key, leaf, perm, base_pos.i);
            }
        }
    }

    // Slot has no suffix but cursor does: cursor is greater
    // Need to find position after slots with same ikey
    find_position_after_suffix(cursor_key, leaf, perm, base_pos.i)
}

/// Find position after inline keys with the same ikey but shorter length.
///
/// When cursor has an inline key that is longer than the slot's inline key
/// `(same ikey, cursor_len > slot_keylenx)`, we need to find the first slot
/// where cursor <= slot.
#[cold]
#[inline(never)]
fn find_position_after_inline<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    start_i: usize,
) -> usize
where
    P: LeafPolicy,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();
    let cursor_len: usize = cursor_key.current_len();

    for i in (start_i + 1)..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey != search_ikey {
            return i;
        }

        // Same ikey: check length/suffix
        let keylenx: u8 = leaf.keylenx_relaxed(slot);

        if keylenx == KSUF_KEYLENX || keylenx >= LAYER_KEYLENX {
            return i;
        }

        // Both inline: compare lengths
        if cursor_len <= keylenx as usize {
            return i;
        }
    }

    size
}

/// When cursor suffix > stored suffix (or cursor has suffix but slot doesn't),
/// we need to skip past any remaining slots with the same ikey.
#[cold]
#[inline(never)]
fn find_position_after_suffix<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    start_i: usize,
) -> usize
where
    P: LeafPolicy,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();

    for i in (start_i + 1)..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey != search_ikey {
            return i;
        }

        // Same ikey: check suffix
        let keylenx: u8 = leaf.keylenx_relaxed(slot);

        if keylenx == KSUF_KEYLENX
            && let Some(stored_suffix) = leaf.ksuf(slot)
            && cursor_key.suffix() <= stored_suffix
        {
            return i;
        }
    }

    size
}

// ============================================================================
//  ReverseScanHelper
// ============================================================================

/// Helper for reverse (descending) range scans.
///
/// Provides direction-specific **stateless** operations for backward iteration.
/// The `upper_bound` flag that was previously stored here now lives in
/// [`ReverseFlags`](super::iterator::iter_flags::ReverseFlags).
#[derive(Clone, Copy, Debug)]
pub struct ReverseScanHelper;

impl ReverseScanHelper {
    /// Retreat to the previous logical position.
    ///
    /// For rev scans: `ki - 1`. Returns -1 when `ki == 0`.
    ///
    /// # Note
    /// We use `isize` to represent -1 as a valid "before first" sentinel.
    #[inline(always)]
    pub const fn prev(ki: isize) -> isize {
        ki - 1
    }

    /// Retreat to the previous leaf in the B-link chain.
    ///
    /// # Safety
    /// The returned pointer is only valid while the guard is held and
    /// version validation has not failed.
    #[inline(always)]
    pub fn retreat<P>(leaf: &LeafNode15<P>) -> *mut LeafNode15<P>
    where
        P: LeafPolicy,
    {
        // SAFETY: Leaf pointer is valid, called during OCC read where
        // version validation catches concurrent modifications.
        unsafe { leaf.prev_unguarded() }
    }

    pub fn stable_reverse<P>(
        leaf: &mut *mut LeafNode15<P>,
        cursor_key: &CursorKey,
        guard: &LocalGuard<'_>,
    ) -> u32
    where
        P: LeafPolicy,
    {
        loop {
            // SAFETY: leaf pointer is valid, protected by guard
            let current: &LeafNode15<P> = unsafe { &**leaf };
            let v: u32 = current.version().stable();

            // Check if we need to advance forward due to concurrent insert
            let next_ptr: *mut LeafNode15<P> = current.safe_next(guard);

            if next_ptr.is_null() {
                return v;
            }

            // SAFETY: next_ptr is non-null, protected by guard
            let next_bound_ikey: u64 = unsafe { &*next_ptr }.ikey_bound();

            // Compare cursor key against next leaf's bound
            match cursor_key.current_ikey().cmp(&next_bound_ikey) {
                // We right in the right leaf
                Ordering::Less => return v,

                Ordering::Equal if cursor_key.current_len() == 0 => {
                    return v;
                }

                _ => {
                    // Key moved to next leaf due to concurrent insert
                    *leaf = next_ptr;
                }
            }
        }
    }

    /// Check if suffix comparison result allows emit for initial position (reverse).
    ///
    /// For reverse: emit if stored < search, or stored == search and emit
    #[inline(always)]
    pub const fn initial_ksuf_match_reverse(ksuf_compare: Ordering, emit_equal: bool) -> bool {
        match ksuf_compare {
            Ordering::Less => true,

            Ordering::Equal => emit_equal,

            Ordering::Greater => false,
        }
    }

    /// Check if a slot is a duplicate of the last emitted key (reverse scan).
    ///
    /// For reverse: duplicate if cursor <= slot (already emitted or after cursor).
    #[inline(always)]
    pub fn is_duplicate_reverse(
        cursor_key: &CursorKey,
        ikey: u64,
        keylenx: u8,
        upper_bound: bool,
    ) -> bool {
        if upper_bound {
            return false;
        }

        // For reverse scan: duplicate if cursor <= slot
        cursor_key.compare(ikey, keylenx as usize) != Ordering::Greater
    }

    /// Find the last position at or before the cursor for a reverse scan.
    ///
    /// When `upper_bound` is true, returns `size - 1` (start from last slot).
    /// Otherwise performs a full-key upper-bound search so equal-ikey groups
    /// (terminal keys, suffix keys, and layer pointers) are ordered correctly.
    ///
    /// # Returns
    ///
    /// Position as `isize` which can be -1 (before first slot).
    #[inline]
    pub fn lower_reverse<P>(
        cursor_key: &CursorKey,
        leaf: &LeafNode15<P>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
        upper_bound: bool,
    ) -> isize
    where
        P: LeafPolicy,
    {
        if upper_bound {
            return perm.size().cast_signed() - 1;
        }

        upper_bound_reverse_full(cursor_key, leaf, perm, true)
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
