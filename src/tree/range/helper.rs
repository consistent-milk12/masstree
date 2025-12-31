//! Filepath: src/tree/range/helper.rs
//!
//! Forward scan helper and lower bound search functions.
//!
//! # Design
//!
//! The scan algorithm is direction-agnostic at the core, with direction-specific
//! logic encapsulated in helper types. `ForwardScanHelper` implements forward
//! iteration; a future `ReverseScanHelper` could implement reverse iteration.
//!
//! # C++ Reference
//!
//! Corresponds to `forward_scan_helper` in `masstree_scan.hh`.

use std::cmp::Ordering;

use crate::key::IKEY_SIZE;
use crate::leaf_trait::{TreeLeafNode, TreePermutation};
use crate::leaf24::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::slot::ValueSlot;

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
    ///
    /// All slots at positions `< i` have ikeys less than the search key.
    /// Slots at positions `>= i` have ikeys greater than or equal.
    pub i: usize,

    /// Physical slot if exact ikey match found, else `None`.
    ///
    /// When `Some(slot)`:
    /// - `leaf.ikey(slot) == search_ikey`
    /// - Still need to compare keylenx/suffix for full key equality
    ///
    /// When `None`:
    /// - No slot has matching ikey
    /// - Position `i` is where a new entry would be inserted
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

    /// Check if an exact ikey match was found.
    #[inline(always)]
    pub const fn has_match(&self) -> bool {
        self.p.is_some()
    }

    /// Get the physical slot (panics if no match).
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Fail fast")]
    pub const fn slot(&self) -> usize {
        self.p
            .expect("slot() called on KeyIndexedPosition without match")
    }
}

// ============================================================================
//  ForwardScanHelper
// ============================================================================

/// Helper for forward (ascending) range scans.
///
/// Provides direction-specific operations that the scan algorithm uses:
/// - Position advancement: `ki + 1`
/// - Leaf advancement: `safe_next()`
/// - Duplicate detection: based on `CursorKey::compare()`
/// - Suffix matching: check if stored suffix >= search suffix
///
/// # C++ Reference
///
/// Corresponds to `forward_scan_helper` in `masstree_scan.hh:37-93`.
#[derive(Debug, Clone, Copy)]
pub struct ForwardScanHelper;

impl ForwardScanHelper {
    /// Advance to the next logical position.
    ///
    /// For forward scans: `ki + 1`.
    #[inline(always)]
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
    pub fn advance<L, S>(leaf: &L) -> *mut L
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
    {
        leaf.safe_next()
    }

    /// Wait for a stable version (spin until not dirty).
    ///
    /// Used when repositioning after a version change.
    #[inline(always)]
    pub fn stable<L, S>(leaf: &L) -> u32
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
    {
        leaf.version().stable()
    }

    /// Check if suffix comparison result allows emit for initial position.
    ///
    /// For forward scans with `find_initial`:
    /// - If stored suffix > search suffix: emit (key is after start bound)
    /// - If stored suffix == search suffix AND `emit_equal`: emit (exact match allowed)
    /// - Otherwise: skip to next
    ///
    /// # Arguments
    ///
    /// - `ksuf_compare`: Result of `stored_suffix.cmp(search_suffix)`
    /// - `emit_equal`: Whether exact matches should be emitted (Included bound)
    ///
    /// # Returns
    ///
    /// `true` if the key at this position should be considered for emission.
    #[inline(always)]
    pub const fn initial_ksuf_match(ksuf_compare: Ordering, emit_equal: bool) -> bool {
        match ksuf_compare {
            Ordering::Greater => true,

            Ordering::Equal => emit_equal,

            Ordering::Less => false,
        }
    }

    /// Check if a slot is a duplicate of the last emitted key.
    ///
    /// For forward scans: a slot is a duplicate if its key <= the cursor key.
    /// This means `cursor.compare(ikey, keylenx) >= 0` (cursor >= slot).
    ///
    /// # Arguments
    ///
    /// - `cursor_key`: The cursor tracking last emitted key
    /// - `ikey`: The slot's ikey
    /// - `keylenx`: The slot's keylenx
    ///
    /// # Returns
    ///
    /// `true` if this slot should be skipped (already emitted or before cursor).
    ///
    /// # C++ Reference
    ///
    /// Matches `forward_scan_helper::is_duplicate()` in `masstree_scan.hh:68-76`.
    #[inline(always)]
    pub fn is_duplicate(cursor_key: &CursorKey, ikey: u64, keylenx: u8) -> bool {
        // For forward scan: duplicate if cursor >= slot
        // cursor.compare returns: Less if cursor < slot, Equal/Greater if cursor >= slot
        cursor_key.compare(ikey, keylenx as usize) != Ordering::Less
    }

    /// Check if a slot with suffix is a duplicate, considering suffix bytes.
    ///
    /// Called when both cursor and slot have suffixes and ikeys are equal.
    ///
    /// # Arguments
    ///
    /// - `cursor_key`: The cursor tracking last emitted key
    /// - `stored_suffix`: The suffix bytes from the leaf slot
    ///
    /// # Returns
    ///
    /// `true` if this slot should be skipped.
    #[inline(always)]
    pub fn is_duplicate_with_suffix(cursor_key: &CursorKey, stored_suffix: &[u8]) -> bool {
        // For forward scan: duplicate if cursor suffix >= stored suffix
        cursor_key.compare_suffix(stored_suffix) != Ordering::Less
    }
}

// ============================================================================
//  Lower Bound Search
// ============================================================================

/// Find the lower bound position for a key within a leaf.
///
/// Searches through the permutation to find:
/// 1. The logical position where the key would be inserted (maintaining order)
/// 2. The physical slot if an exact ikey match exists
///
/// # Algorithm
///
/// Linear search through permutation slots, comparing ikeys. For WIDTH=24,
/// linear search is competitive with binary search due to cache effects.
///
/// ```text
/// For each position i in 0..perm.size():
///     slot = perm.get(i)
///     slot_ikey = leaf.ikey(slot)
///
///     if cursor_key.ikey < slot_ikey:
///         return (i, None)  // Insert before this position
///     if cursor_key.ikey == slot_ikey:
///         return (i, Some(slot))  // Exact ikey match
///
/// return (perm.size(), None)  // Insert at end
///
///
/// # Arguments
///
/// - `cursor_key`: The key to search for
/// - `leaf`: The leaf node to search in
/// - `perm`: The permutation (snapshot) of the leaf
///
/// # Returns
///
/// `KeyIndexedPosition` with insertion point and optional exact match.
///
/// # C++ Reference
///
/// Corresponds to the lower bound logic in `find_initial` and `find_next`
/// from `masstree_scan.hh`.
#[inline]
pub fn lower_with_position<L, S>(
    cursor_key: &CursorKey,
    leaf: &L,
    perm: &L::Perm,
) -> KeyIndexedPosition
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();

    // Linear search through permutation
    for i in 0..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

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
///
/// # Algorithm
///
/// 1. First find by ikey using `lower_with_position`
/// 2. If exact ikey match found and both have suffixes, compare suffixes
/// 3. Return adjusted position based on suffix comparison
///
/// # Arguments
///
/// - `cursor_key`: The key to search for
/// - `leaf`: The leaf node to search in
/// - `perm`: The permutation (snapshot) of the leaf
///
/// # Returns
///
/// `KeyIndexedPosition` with accurate position considering suffixes.
#[inline]
pub fn lower_with_suffix<L, S>(
    cursor_key: &CursorKey,
    leaf: &L,
    perm: &L::Perm,
) -> KeyIndexedPosition
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
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
fn find_position_after_inline<L, S>(
    cursor_key: &CursorKey,
    leaf: &L,
    perm: &L::Perm,
    start_i: usize,
) -> KeyIndexedPosition
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();
    let cursor_len: usize = cursor_key.current_len();

    // Scan forward from start_i to find first slot where cursor <= slot
    for i in (start_i + 1)..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

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
fn find_position_after_suffix<L, S>(
    cursor_key: &CursorKey,
    leaf: &L,
    perm: &L::Perm,
    start_i: usize,
) -> KeyIndexedPosition
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let size: usize = perm.size();
    let search_ikey: u64 = cursor_key.current_ikey();

    // Scan forward from start_i to find first slot with different ikey
    for i in (start_i + 1)..size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

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

        // Continue scanning
    }

    // Cursor is greater than all slots
    KeyIndexedPosition::not_found(size)
}

// ============================================================================
//  Keylenx Classification Helpers
// ============================================================================

/// Check if a keylenx indicates a layer pointer.
#[inline(always)]
pub const fn is_layer_keylenx(keylenx: u8) -> bool {
    keylenx >= LAYER_KEYLENX
}

/// Check if a keylenx indicates a suffix key.
#[inline(always)]
pub const fn has_suffix_keylenx(keylenx: u8) -> bool {
    keylenx == KSUF_KEYLENX
}

/// Get the inline key length from keylenx.
///
/// For inline keys (keylenx 0-8), returns the key length.
/// For suffix/layer keys, returns 8 (the ikey portion).
#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "Known Value")]
pub const fn inline_key_len(keylenx: u8) -> usize {
    if keylenx <= (IKEY_SIZE as u8) {
        keylenx as usize
    } else {
        IKEY_SIZE
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
