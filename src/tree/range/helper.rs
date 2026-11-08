//! Filepath: src/tree/range/helper.rs
//!
//! Lower bound search functions and scan helpers.

use crate::permuter::Permuter15;
use std::cmp::Ordering;

use seize::LocalGuard;

use crate::IKEY_SIZE;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::policy::LeafPolicy;

use super::cursor_key::CursorKey;
use super::scan_state::{ScanSnapshot, ScanSnapshotPtr};

/// Abstracts clone vs zero-copy value emission for per-entry scanning.
pub trait ScanEmitter<P: LeafPolicy> {
    type Snapshot;

    /// Load a value from `leaf[slot]` and wrap it with `key_len`.
    fn emit_value(leaf: &LeafNode15<P>, slot: usize, key_len: usize) -> Option<Self::Snapshot>;
}

/// Clone-based emitter: calls `leaf.load_value(slot)`, returns `ScanSnapshot<P>`.
pub struct CloneEmitter;

impl<P: LeafPolicy> ScanEmitter<P> for CloneEmitter
where
    P::Output: Clone,
{
    type Snapshot = ScanSnapshot<P>;

    #[inline(always)]
    fn emit_value(leaf: &LeafNode15<P>, slot: usize, key_len: usize) -> Option<ScanSnapshot<P>> {
        let output = leaf.load_value(slot)?;
        Some(ScanSnapshot {
            value: output,
            key_len,
        })
    }
}

/// Zero-copy emitter: calls `leaf.load_value_raw(slot)`, returns `ScanSnapshotPtr<P::Value>`.
///
/// For `CAN_WRITE_THROUGH` types, callers must use `ScanSnapshotPtr::value_copy`
/// (atomic read) instead of `value_ref` to avoid aliasing violations.
pub struct PtrEmitter;

impl<P: LeafPolicy> ScanEmitter<P> for PtrEmitter {
    type Snapshot = ScanSnapshotPtr<P::Value>;

    #[inline(always)]
    fn emit_value(
        leaf: &LeafNode15<P>,
        slot: usize,
        key_len: usize,
    ) -> Option<ScanSnapshotPtr<P::Value>> {
        let ptr = leaf.load_value_raw(slot);
        if ptr.is_null() {
            return None;
        }
        Some(ScanSnapshotPtr::from_raw(ptr, key_len))
    }
}

/// One slot's speculative payload, usable only after the enclosing snapshot validates.
pub enum SlotPayload<S> {
    Empty,
    Layer(*mut u8),
    Value(S),
}

/// Complete slot data read under one OCC version. Suffix bytes are immutable
/// while the scan guard protects their storage; no slot reload is needed.
pub struct ValidatedSlot<'a, S> {
    pub ikey: u64,
    pub keylenx: u8,
    pub suffix: Option<&'a [u8]>,
    pub payload: SlotPayload<S>,
}

/// Read the complete key/payload, then validate before any cursor mutation or
/// callback. The caller keeps this leaf and its reachable storage guard-protected.
/// Emitters capture handles only: pointer dereferences happen after validation.
#[inline(always)]
pub fn try_snapshot_slot<P: LeafPolicy, E: ScanEmitter<P>>(
    leaf: &LeafNode15<P>,
    slot: usize,
    version: u32,
) -> Result<ValidatedSlot<'_, E::Snapshot>, ()> {
    let ikey = leaf.ikey_relaxed(slot);
    let keylenx = leaf.keylenx_relaxed(slot);
    let mut suffix = None;
    let payload = if keylenx >= LAYER_KEYLENX {
        let root = leaf.load_layer_raw(slot);
        if root.is_null() {
            SlotPayload::Empty
        } else {
            SlotPayload::Layer(root)
        }
    } else {
        if keylenx == KSUF_KEYLENX {
            suffix = leaf.ksuf(slot);
        }
        let key_len = if keylenx == KSUF_KEYLENX {
            IKEY_SIZE + suffix.map_or(0, <[u8]>::len)
        } else {
            usize::from(keylenx)
        };
        E::emit_value(leaf, slot, key_len).map_or(SlotPayload::Empty, SlotPayload::Value)
    };
    if crate::nodeversion::NodeVersion::is_deleted_version(version)
        || leaf.version().has_changed(version)
    {
        return Err(());
    }
    Ok(ValidatedSlot {
        ikey,
        keylenx,
        suffix,
        payload,
    })
}

/// Build the full key from an already-snapshotted slot.
///
/// The caller must have validated the leaf version after reading the
/// snapshot, so this never re-reads a slot that a writer may have reused.
#[inline(always)]
pub fn build_slot_key_parts(
    cursor_key: &mut CursorKey,
    slot_ikey: u64,
    slot_keylenx: u8,
    suffix: Option<&[u8]>,
) {
    cursor_key.assign_store_ikey(slot_ikey);

    if slot_keylenx == KSUF_KEYLENX {
        if let Some(suffix) = suffix {
            let suffix_len = suffix.len();
            let _ = cursor_key.assign_store_suffix(suffix);
            cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
        } else {
            cursor_key.assign_store_length(IKEY_SIZE);
        }
    } else {
        let len = slot_keylenx as usize;
        cursor_key.assign_store_length(len);
    }
}

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
/// Test-only helper for the legacy forward lower-bound semantics.
#[must_use]
#[cfg(test)]
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
    if cursor_key.is_unshift_sentinel() {
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
    perm: Permuter15,
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
            Ordering::Equal => {
                // An equal layer pointer is not a key: on an excluded bound
                // the scan still has to descend to find keys inside the layer.
                let keylenx: u8 = leaf.keylenx_relaxed(slot);
                if skip_equal && (keylenx < LAYER_KEYLENX || cursor_key.is_unshift_sentinel()) {
                    return i + 1;
                }
                return i;
            }
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
    perm: Permuter15,
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
    perm: Permuter15,
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

    /// Find the last position strictly before the cursor for a reverse scan.
    ///
    /// Used when the cursor already holds an emitted key, so a retry must not
    /// re-emit it. A "fresh sublayer from maximum" scan still starts at the
    /// last slot.
    #[inline]
    pub fn lower_reverse_strict<P>(
        cursor_key: &CursorKey,
        leaf: &LeafNode15<P>,
        perm: Permuter15,
        upper_bound: bool,
    ) -> isize
    where
        P: LeafPolicy,
    {
        if upper_bound {
            return perm.size().cast_signed() - 1;
        }

        upper_bound_reverse_full(cursor_key, leaf, perm, false)
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
        perm: Permuter15,
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
