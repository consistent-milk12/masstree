//! Filepath: src/tree/range/helper.rs
//!
//! Forward scan helper and lower bound search functions.

use std::cmp::Ordering;

use seize::LocalGuard;

use crate::key::IKEY_SIZE;
use crate::leaf_trait::TreeLeafNode;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::policy::LeafPolicy;

use super::cursor_key::CursorKey;

// ============================================================================
//  KeyIndexedPosition
// ============================================================================

/// Result of a lower bound search within a leaf.
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
//  ForwardScanHelper
// ============================================================================

/// Helper for forward (ascending) range scans.
#[derive(Debug, Clone, Copy)]
pub struct ForwardScanHelper;

impl ForwardScanHelper {
    /// Advance to the next logical position.
    ///
    /// For forward scans: `ki + 1`.
    #[inline(always)]
    #[allow(dead_code, reason = "Forward scan helper API")]
    pub const fn next(ki: usize) -> usize {
        ki + 1
    }

    /// Advance to the next leaf in the B-link chain.
    ///
    /// For forward scans: `leaf.safe_next()`.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while the guard is held.
    #[inline(always)]
    #[allow(dead_code, reason = "Forward scan helper API")]
    pub fn advance<P>(leaf: &LeafNode15<P>) -> *mut LeafNode15<P>
    where
        P: LeafPolicy,
    {
        // SAFETY: Leaf pointer is valid, called during OCC read where
        // version validation catches concurrent modifications.
        unsafe { leaf.safe_next_unguarded() }
    }

    /// Wait for a stable version (spin until not dirty).
    ///
    /// Used when repositioning after a version change.
    #[inline(always)]
    #[allow(dead_code, reason = "Forward scan helper API")]
    pub fn stable<P>(leaf: &LeafNode15<P>) -> u32
    where
        P: LeafPolicy,
    {
        leaf.version().stable()
    }

    /// Check if suffix comparison result allows emit for initial position.
    #[inline(always)]
    pub const fn initial_ksuf_match(ksuf_compare: Ordering, emit_equal: bool) -> bool {
        match ksuf_compare {
            Ordering::Greater => true,

            Ordering::Equal => emit_equal,

            Ordering::Less => false,
        }
    }

    /// Check if a slot is a duplicate of the last emitted key.
    #[inline(always)]
    #[allow(dead_code, reason = "Forward scan helper API")]
    pub fn is_duplicate(cursor_key: &CursorKey, ikey: u64, keylenx: u8) -> bool {
        // For forward scan: duplicate if cursor >= slot
        // cursor.compare returns: Less if cursor < slot, Equal/Greater if cursor >= slot
        cursor_key.compare(ikey, keylenx as usize) != Ordering::Less
    }

    /// Check if a slot with suffix is a duplicate, considering suffix bytes.
    #[inline(always)]
    #[allow(dead_code, reason = "Forward scan helper API")]
    pub fn is_duplicate_with_suffix(cursor_key: &CursorKey, stored_suffix: &[u8]) -> bool {
        // For forward scan: duplicate if cursor suffix >= stored suffix
        cursor_key.compare_suffix(stored_suffix) != Ordering::Less
    }
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

/// Find the lower bound with full key comparison (ikey + suffix).
///
/// This is used when we need to find the exact position considering suffix
/// bytes, not just ikey matching.
#[inline]
pub fn lower_with_suffix<P>(
    cursor_key: &CursorKey,
    leaf: &LeafNode15<P>,
    perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
) -> KeyIndexedPosition
where
    P: LeafPolicy,
{
    let base_pos: KeyIndexedPosition = lower_with_position(cursor_key, leaf, perm);

    // If no ikey match, position is correct
    let Some(slot) = base_pos.p else {
        return base_pos;
    };

    // Check if suffix comparison is needed
    let keylenx: u8 = leaf.keylenx(slot);

    // Layer pointers: if cursor has suffix (including after unshift()),
    // we've already processed this layer and should skip past it
    if keylenx >= LAYER_KEYLENX {
        if cursor_key.has_suffix() {
            // Cursor has suffix (or is post-unshift sentinel) -> skip layer pointer
            return KeyIndexedPosition::not_found(base_pos.i + 1);
        }
        return base_pos;
    }

    // Handle inline keys (no suffix on either side)
    if !cursor_key.has_suffix() {
        // Compare lengths for inline keys
        let cursor_len = cursor_key.current_len();
        let slot_len = keylenx as usize;

        return match cursor_len.cmp(&slot_len) {
            Ordering::Less => base_pos, // cursor < slot, position correct
            Ordering::Equal => {
                // Exact match - skip to next position
                KeyIndexedPosition::not_found(base_pos.i + 1)
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
                return base_pos;
            }

            Ordering::Equal => {
                // Exact match - we already emitted this key, skip to next position
                return KeyIndexedPosition::not_found(base_pos.i + 1);
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
) -> KeyIndexedPosition
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
            // Different ikey: insert before it
            return KeyIndexedPosition::not_found(i);
        }

        // Same ikey: check length/suffix
        let keylenx: u8 = leaf.keylenx(slot);

        if keylenx == KSUF_KEYLENX || keylenx >= LAYER_KEYLENX {
            // Slot has suffix or is layer pointer: cursor (inline) < slot
            return KeyIndexedPosition::not_found(i);
        }

        // Both inline: compare lengths
        if cursor_len <= keylenx as usize {
            return KeyIndexedPosition::not_found(i);
        }

        // cursor_len > slot_len, continue scanning
    }

    // Cursor is greater than all slots
    KeyIndexedPosition::not_found(size)
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
) -> KeyIndexedPosition
where
    P: LeafPolicy,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();

    for i in (start_i + 1)..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey != search_ikey {
            // Found a slot with different ikey: insert before it
            return KeyIndexedPosition::not_found(i);
        }

        // Same ikey: check suffix
        let keylenx: u8 = leaf.keylenx(slot);

        if keylenx == KSUF_KEYLENX
            && let Some(stored_suffix) = leaf.ksuf(slot)
            && cursor_key.suffix() <= stored_suffix
        {
            // Found a slot where cursor suffix <= stored: insert before
            return KeyIndexedPosition::not_found(i);
        }
    }

    // Cursor is greater than all slots
    KeyIndexedPosition::not_found(size)
}

// ============================================================================
//  Keylenx Classification Helpers
// ============================================================================

/// Check if a keylenx indicates a layer pointer.
#[inline(always)]
#[allow(dead_code, reason = "Keylenx classification API")]
pub const fn is_layer_keylenx(keylenx: u8) -> bool {
    keylenx >= LAYER_KEYLENX
}

/// Check if a keylenx indicates a suffix key.
#[inline(always)]
#[allow(dead_code, reason = "Keylenx classification API")]
pub const fn has_suffix_keylenx(keylenx: u8) -> bool {
    keylenx == KSUF_KEYLENX
}

/// Get the inline key length from keylenx.
///
/// For inline keys (keylenx 0-8), returns the key length.
/// For suffix/layer keys, returns 8 (the ikey portion).
#[inline(always)]
#[allow(dead_code, reason = "Keylenx classification API")]
#[expect(clippy::cast_possible_truncation, reason = "Known Value")]
pub const fn inline_key_len(keylenx: u8) -> usize {
    if keylenx <= (IKEY_SIZE as u8) {
        keylenx as usize
    } else {
        IKEY_SIZE
    }
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

    /// Find lower bound position for reverse scan.
    ///
    /// When `upper_bound` is true, returns `size - 1` (start from last slot).
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

        let kx: KeyIndexedPosition = lower_with_position(cursor_key, leaf, perm);

        if kx.p.is_some() {
            kx.i.cast_signed()
        } else {
            kx.i.cast_signed() - 1
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
