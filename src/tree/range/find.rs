//! Filepath: src/tree/range/find.rs
//!
//! Core scan algorithm functions implementing the state machine.
//!
//! # Algorithm Overview
//!
//! The scan is implemented as three main functions that work together:
//!
//! 1. [`find_initial`]: Position the scan at the start bound
//! 2. [`find_next`]: Iterate through entries, handling all transitions
//! 3. [`find_retry`]: Reposition after version changes or layer transitions
//!
//! # C++ Reference
//!
//! Corresponds to `scanstackelt::find_initial`, `find_next`, and the retry
//! logic in `masstree_scan.hh`.

use std::cmp::Ordering;

use seize::LocalGuard;

use crate::key::IKEY_SIZE;
use crate::ksearch::upper_bound_internode_generic;
use crate::leaf_trait::{TreeInternode, TreeLeafNode, TreePermutation};
use crate::leaf24::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::nodeversion::NodeVersion;
use crate::prefetch::prefetch_read;
use crate::slot::ValueSlot;

use super::cursor_key::CursorKey;
use super::helper::{
    ForwardScanHelper, KeyIndexedPosition, lower_with_position, lower_with_suffix,
};
use super::scan_state::{
    LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr, ScanStackElement, ScanState,
};

// ============================================================================
//  find_initial
// ============================================================================

/// Find the initial position for a range scan.
///
/// Positions the scan at the correct leaf and slot for the start bound.
/// May descend into sublayers if the start key belongs in a sublayer.
///
/// # Algorithm
///
/// 1. Traverse from root to leaf using `reach_leaf_for_scan`
/// 2. Take stable version and load permutation
/// 3. Find lower bound position for cursor key
/// 4. If exact match found:
///    - For layer pointer: return `Down` to descend
///    - For suffix key: check suffix, return `Emit` or `FindNext`
///    - For inline key: return `Emit` or `FindNext` based on `emit_equal`
/// 5. If no match: return `FindNext` to search forward
///
/// # Arguments
///
/// - `root`: The layer root to search from
/// - `stack`: Stack element to populate with position
/// - `cursor_key`: The cursor positioned at start bound
/// - `layer_stack`: Stack for parent layer contexts
/// - `emit_equal`: Whether exact matches should be emitted (true for Included bound)
/// - `guard`: Memory reclamation guard
///
/// # Returns
///
/// - `(ScanState, Option<ScanSnapshot>)`: Next state and optional value snapshot
///
/// # C++ Reference
///
/// Corresponds to `scanstackelt::find_initial` in `masstree_scan.hh:130-188`.
pub fn find_initial<L, S>(
    root: *const u8,
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    _layer_stack: &mut LayerStack<L>,
    emit_equal: bool,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    S::Output: Clone,
{
    // Initialize stack with root
    stack.set_root(root);

    // Reach the target leaf
    let leaf_ptr: *mut L = reach_leaf_for_scan::<L, S>(root, cursor_key, guard);

    if leaf_ptr.is_null() {
        // Empty tree
        return (ScanState::Up, None);
    }

    stack.set_leaf(leaf_ptr);

    // SAFETY: leaf_ptr is valid (non-null checked above, guard protects it)
    let leaf: &L = unsafe { &*leaf_ptr };

    // Get stable version
    let version: u32 = leaf.version().stable();

    // Check if deleted
    if leaf.version().is_deleted() {
        // Retry from root
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = match leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            // Split in progress, retry
            return (ScanState::Retry, None);
        }
    };

    // Find lower bound position
    let kx: KeyIndexedPosition = lower_with_position(cursor_key, leaf, &perm);

    // Handle position based on whether we found a match
    let (next_state, snapshot) = kx.p.map_or_else(
        || (ScanState::FindNext, None),
        |slot| {
            handle_initial_match::<L, S>(
                leaf, slot, cursor_key, stack, emit_equal, version, &perm, kx.i,
            )
        },
    );

    // Validate version before committing
    if leaf.version().has_changed(version) {
        // Version changed, need to revalidate
        // Check if we need to follow B-links
        if leaf.version().has_split(version) {
            // Key might have moved, retry from root
            return (ScanState::Retry, None);
        }
        // Retry on this leaf
        return (ScanState::Retry, None);
    }

    // Update stack with validated state
    // IMPORTANT: If we found a match (kx.p.is_some()), we need to advance past it.
    // This applies whether we emitted it (emit_equal=true) or skipped it (emit_equal=false).
    // Only when there's no match (kx.p.is_none()) do we start at the insertion point.
    let final_pos = if kx.p.is_some() {
        kx.i + 1 // Advance past matched entry (emitted or skipped)
    } else {
        kx.i // No match, start at insertion point
    };
    stack.update_state(version, perm, final_pos);

    (next_state, snapshot)
}

/// Handle an exact ikey match in `find_initial`.
#[expect(clippy::too_many_arguments, reason = "Internals")]
fn handle_initial_match<L, S>(
    leaf: &L,
    slot: usize,
    cursor_key: &mut CursorKey,
    stack: &mut ScanStackElement<L, S>,
    emit_equal: bool,
    _version: u32,
    _perm: &L::Perm,
    _pos: usize,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    S::Output: Clone,
{
    let keylenx: u8 = leaf.keylenx(slot);

    if keylenx >= LAYER_KEYLENX {
        // Layer pointer - always descend to scan layer contents
        // C++ reference (masstree_scan.hh:218-222):
        //   if (n_->keylenx_is_layer(keylenx)) {
        //       node_stack_.push_back(root_); node_stack_.push_back(n_);
        //       root_ = entry.layer(); return scan_down;
        //   }
        // This is needed even when start key has no suffix (exact 8-byte prefix)
        // because we want to scan all keys under this layer pointer.
        let slot_ikey: u64 = leaf.ikey(slot);
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        cursor_key.assign_store_ikey(slot_ikey);
        stack.set_root(layer_ptr);
        return (ScanState::Down, None);
    }

    if keylenx == KSUF_KEYLENX {
        // Slot has suffix, need to compare
        if let Some(stored_suffix) = leaf.ksuf(slot) {
            let cursor_suffix: &[u8] = cursor_key.suffix();
            let cmp = stored_suffix.cmp(cursor_suffix);

            if ForwardScanHelper::initial_ksuf_match(cmp, emit_equal) {
                // Match - prepare for emit
                let value_ptr = leaf.leaf_value_ptr(slot);
                if !value_ptr.is_null() {
                    // SAFETY: We've validated version, ptr is valid
                    let output: S::Output = unsafe { S::output_from_raw(value_ptr) };
                    let key_len = IKEY_SIZE + stored_suffix.len();

                    // Store key data in cursor for duplicate filtering
                    cursor_key.assign_store_ikey(leaf.ikey(slot));
                    let _ = cursor_key.assign_store_suffix(stored_suffix);
                    cursor_key.assign_store_length(key_len);

                    return (ScanState::Emit, Some(ScanSnapshot::new(output, key_len)));
                }
            }
            // Skip this entry
            return (ScanState::FindNext, None);
        }
    }

    // Inline key (keylenx 0-8)
    if emit_equal {
        // Exact match allowed
        let value_ptr = leaf.leaf_value_ptr(slot);
        if !value_ptr.is_null() {
            // SAFETY: We've validated version, ptr is valid
            let output: S::Output = unsafe { S::output_from_raw(value_ptr) };
            let key_len = keylenx as usize;

            // Store key data in cursor
            cursor_key.assign_store_ikey(leaf.ikey(slot));
            cursor_key.assign_store_length(key_len);

            // Advance past this position for next iteration
            return (ScanState::Emit, Some(ScanSnapshot::new(output, key_len)));
        }
    }

    // Skip to next
    (ScanState::FindNext, None)
}

// ============================================================================
//  find_next
// ============================================================================

/// Find the next entry in the scan sequence.
///
/// This is the main iteration function. It handles:
/// - Iterating through slots in current leaf
/// - Version validation and retry
/// - Duplicate filtering (only when `needs_duplicate_check` is true)
/// - Advancing to next leaf via B-links
/// - Descending into sublayers (layer pointers)
/// - Ascending when layer is exhausted
///
/// # Algorithm
///
/// 1. If leaf is deleted: return `Retry`
/// 2. Get current slot from permutation
/// 3. If slot exists:
///    - Validate version
///    - Check for duplicate (only after Retry)
///    - If layer pointer: return `Down`
///    - Otherwise: prepare snapshot and return `Emit`
/// 4. If no slot (exhausted):
///    - Try advancing to next leaf
///    - If no next leaf: return `Up`
///    - Otherwise: update stack and return `FindNext`
///
/// # Arguments
///
/// - `stack`: Current scan position
/// - `cursor_key`: Cursor tracking last emitted key
/// - `layer_stack`: Parent layer stack (for Down transitions)
/// - `guard`: Memory reclamation guard
///
/// # Returns
///
/// - `(ScanState, Option<ScanSnapshot>)`: Next state and optional value snapshot
///
/// # C++ Reference
///
/// Corresponds to `scanstackelt::find_next` in `masstree_scan.hh:246-317`.
pub fn find_next<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    S::Output: Clone,
{
    // OPTIMIZATION: Skip duplicate check in normal forward iteration.
    // Duplicates can only occur after Retry states (version conflict).
    // The caller (iterator.rs) tracks this via `needs_duplicate_check` flag.
    find_next_inner(stack, cursor_key, layer_stack, guard, false)
}

/// Find the next entry with duplicate checking enabled.
///
/// Called after a Retry state to skip already-emitted entries.
pub fn find_next_with_duplicate_check<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    S::Output: Clone,
{
    find_next_inner(stack, cursor_key, layer_stack, guard, true)
}

/// Inner implementation of `find_next` with configurable duplicate checking.
///
/// # Performance Optimizations
///
/// This function is on the hot path - called for every scanned entry.
/// Key optimizations vs the naive implementation:
///
/// 1. **No search per element**: Uses `stack.kp()` (O(1)) instead of `lower_with_position` (O(n))
/// 2. **Lazy suffix reading**: Only reads suffix when emitting, not for every slot
/// 3. **Minimal cursor updates**: Only updates cursor fields needed for duplicate detection
/// 4. **Batch version validation**: Only check version at leaf boundaries, not per-entry
///
/// # Safety: Why no per-entry version check?
///
/// Following `TreeIndex`'s approach: trust the Guard for memory safety.
/// - Guard prevents use-after-free during iteration
/// - B-link structure means splits only move keys RIGHT (forward direction)
/// - Forward iteration naturally follows this direction
/// - Version is validated when advancing to next leaf (in `advance_leaf`)
/// - If a split moves keys we haven't seen, we'll encounter them in the sibling
#[inline]
fn find_next_inner<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    S::Output: Clone,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        return (ScanState::Up, None);
    }

    let leaf: &L = unsafe { stack.leaf_ref() };

    // Check if leaf is deleted (this is cheap - single atomic load)
    if leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    // Get current slot - O(1) via perm.get(ki)
    let Some(slot) = stack.kp() else {
        // Leaf exhausted, try advancing (this validates version)
        return advance_leaf(stack, cursor_key, guard);
    };

    // Read slot data - Guard ensures memory safety
    // No per-entry version check: validated at leaf boundaries only
    let slot_ikey: u64 = leaf.ikey(slot);
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // Check for duplicate only when needed (after Retry)
    // OPTIMIZATION: In normal forward iteration, stack.next() already advances
    // past the previous entry, so duplicates can't occur
    if needs_duplicate_check {
        // First check ikey + keylenx level
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        let is_dup: bool = match cmp {
            Ordering::Greater => true, // cursor > slot, definitely duplicate

            Ordering::Less => false, // cursor < slot, not duplicate

            Ordering::Equal => {
                // Need suffix comparison if both have suffixes
                if slot_keylenx == KSUF_KEYLENX && cursor_key.has_suffix() {
                    // Read suffix and compare
                    leaf.ksuf(slot).is_none_or(|stored_suffix| {
                        cursor_key.compare_suffix(stored_suffix) != Ordering::Less
                    })
                } else {
                    true // Equal at ikey+keylenx level, is duplicate
                }
            }
        };

        if is_dup {
            stack.next();
            return (ScanState::FindNext, None);
        }
    }

    // Handle layer pointer - descend into sublayer
    if slot_keylenx >= LAYER_KEYLENX {
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
        cursor_key.assign_store_ikey(slot_ikey);
        stack.set_root(slot_ptr);
        return (ScanState::Down, None);
    }

    // Value slot - prepare for emit
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
    if slot_ptr.is_null() {
        stack.next();
        return (ScanState::FindNext, None);
    }

    // Clone output while version is validated
    // SAFETY: Version validated, pointer is valid
    let output: S::Output = unsafe { S::output_from_raw(slot_ptr) };

    // Compute key length - only read suffix NOW if needed
    let key_len: usize = if slot_keylenx == KSUF_KEYLENX {
        // Read suffix only when emitting
        if let Some(suffix) = leaf.ksuf(slot) {
            let suffix_len = suffix.len();
            // Update cursor for duplicate detection on retry
            cursor_key.assign_store_ikey(slot_ikey);
            let _ = cursor_key.assign_store_suffix(suffix);
            cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
            IKEY_SIZE + suffix_len
        } else {
            cursor_key.assign_store_ikey(slot_ikey);
            cursor_key.assign_store_length(IKEY_SIZE);
            IKEY_SIZE
        }
    } else {
        let len = slot_keylenx as usize;
        cursor_key.assign_store_ikey(slot_ikey);
        cursor_key.assign_store_length(len);
        len
    };

    cursor_key.mark_key_complete();

    // Advance position for next call
    stack.next();

    (ScanState::Emit, Some(ScanSnapshot::new(output, key_len)))
}

// ============================================================================
//  Zero-Copy Variants (for scan_ref)
// ============================================================================

/// Find the next entry, returning a raw pointer instead of cloning.
///
/// This is the zero-copy variant of [`find_next`] for use with `scan_ref`.
/// Instead of calling `S::output_from_raw` (which clones Arc values),
/// it returns the raw pointer directly.
///
/// # Safety
///
/// The returned pointer is only valid while:
/// 1. The guard is held
/// 2. The version hasn't changed
///
/// Callers must dereference immediately within the same guard scope.
#[inline]
pub fn find_next_ptr<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    find_next_inner_ptr(stack, cursor_key, layer_stack, guard, false)
}

/// Find the next entry with duplicate checking, returning raw pointer.
///
/// Zero-copy variant of [`find_next_with_duplicate_check`].
#[inline]
pub fn find_next_with_duplicate_check_ptr<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    find_next_inner_ptr(stack, cursor_key, layer_stack, guard, true)
}

/// Inner implementation for zero-copy find_next.
///
/// Nearly identical to [`find_next_inner`] but:
/// - Does NOT call `S::output_from_raw` (no Arc clone)
/// - Returns `ScanSnapshotPtr` with raw pointer instead
///
/// This eliminates 2 atomic operations per entry (increment + decrement).
#[inline]
fn find_next_inner_ptr<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanState, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        return (ScanState::Up, None);
    }

    let leaf: &L = unsafe { stack.leaf_ref() };

    // Check if leaf is deleted (this is cheap - single atomic load)
    if leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    // Get current slot - O(1) via perm.get(ki)
    let Some(slot) = stack.kp() else {
        // Leaf exhausted, try advancing (this validates version)
        return advance_leaf_ptr(stack, cursor_key, guard);
    };

    // Read slot data - Guard ensures memory safety
    let slot_ikey: u64 = leaf.ikey(slot);
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // Check for duplicate only when needed (after Retry)
    if needs_duplicate_check {
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        let is_dup: bool = match cmp {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => {
                if slot_keylenx == KSUF_KEYLENX && cursor_key.has_suffix() {
                    leaf.ksuf(slot).is_none_or(|stored_suffix| {
                        cursor_key.compare_suffix(stored_suffix) != Ordering::Less
                    })
                } else {
                    true
                }
            }
        };

        if is_dup {
            stack.next();
            return (ScanState::FindNext, None);
        }
    }

    // Handle layer pointer - descend into sublayer
    if slot_keylenx >= LAYER_KEYLENX {
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
        cursor_key.assign_store_ikey(slot_ikey);
        stack.set_root(slot_ptr);
        return (ScanState::Down, None);
    }

    // Value slot - prepare for emit (NO CLONING)
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
    if slot_ptr.is_null() {
        stack.next();
        return (ScanState::FindNext, None);
    }

    // KEY DIFFERENCE: We do NOT call S::output_from_raw here!
    // The caller will dereference the pointer directly.

    // Compute key length - only read suffix NOW if needed
    let key_len: usize = if slot_keylenx == KSUF_KEYLENX {
        if let Some(suffix) = leaf.ksuf(slot) {
            let suffix_len = suffix.len();
            cursor_key.assign_store_ikey(slot_ikey);
            let _ = cursor_key.assign_store_suffix(suffix);
            cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
            IKEY_SIZE + suffix_len
        } else {
            cursor_key.assign_store_ikey(slot_ikey);
            cursor_key.assign_store_length(IKEY_SIZE);
            IKEY_SIZE
        }
    } else {
        let len = slot_keylenx as usize;
        cursor_key.assign_store_ikey(slot_ikey);
        cursor_key.assign_store_length(len);
        len
    };

    cursor_key.mark_key_complete();

    // Advance position for next call
    stack.next();

    // Return raw pointer - caller will dereference
    (
        ScanState::Emit,
        Some(ScanSnapshotPtr::from_raw(slot_ptr, key_len)),
    )
}

/// Advance to next leaf, zero-copy variant.
///
/// Same as [`advance_leaf`] but returns `ScanSnapshotPtr`.
fn advance_leaf_ptr<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &CursorKey,
    _guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let leaf: &L = unsafe { stack.leaf_ref() };
    let version: u32 = stack.version();

    // Check if version changed (concurrent modification)
    if leaf.version().has_changed(version) {
        return (ScanState::Retry, None);
    }

    // Get next leaf
    let next: *mut L = ForwardScanHelper::advance(leaf);

    if next.is_null() {
        // No more leaves in this layer
        return (ScanState::Up, None);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &L = unsafe { &*next };

    // Prefetch the next leaf's data arrays
    next_leaf.prefetch();

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted
    if next_leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = match next_leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            return (ScanState::Retry, None);
        }
    };

    // Reposition using full key comparison
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);
    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Advance to the next leaf when current is exhausted.
///
/// Uses `lower_with_suffix` to find the correct starting position in the new
/// leaf, matching the C++ behavior of `helper.lower(ka, this)`.
fn advance_leaf<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &CursorKey,
    _guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let leaf: &L = unsafe { stack.leaf_ref() };
    let version: u32 = stack.version();

    // Check if version changed (concurrent modification)
    if leaf.version().has_changed(version) {
        // Need to reposition
        return (ScanState::Retry, None);
    }

    // Get next leaf
    let next: *mut L = ForwardScanHelper::advance(leaf);

    if next.is_null() {
        // No more leaves in this layer
        return (ScanState::Up, None);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &L = unsafe { &*next };

    // Prefetch the next leaf's data arrays (ikey0, keylenx, leaf_values)
    // This brings multiple cache lines into L1/L2 before we iterate
    next_leaf.prefetch();

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted
    if next_leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = match next_leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            return (ScanState::Retry, None);
        }
    };

    // Reposition using full key comparison (like C++ `helper.lower(ka, this)`).
    // This ensures we skip past any keys <= cursor_key.
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);
    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Refresh stack state after version change and retry.
fn refresh_and_retry<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &CursorKey,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let leaf: &L = unsafe { stack.leaf_ref() };

    // Check for split
    if leaf.version().has_split(stack.version()) {
        // Key might have moved, need to follow B-links or retry from root
        return follow_blinks_or_retry(stack, cursor_key);
    }

    // Just a version bump, refresh state
    let version: u32 = leaf.version().stable();

    // Check if deleted
    if leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    let perm: L::Perm = match leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            return (ScanState::Retry, None);
        }
    };

    // Recompute position using suffix-aware search
    // This is critical: if cursor_key has a suffix, we need to find the position
    // AFTER keys with the same ikey but smaller suffix, not just the first match.
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);

    stack.update_state(version, perm, kx.i);

    // Return FindNext - lower_with_suffix already positioned us correctly
    // past any keys <= cursor_key.
    (ScanState::FindNext, None)
}

/// Follow B-links to find the correct leaf after a split.
#[expect(clippy::similar_names)]
fn follow_blinks_or_retry<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &CursorKey,
) -> (ScanState, Option<ScanSnapshot<S>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let cursor_ikey: u64 = cursor_key.current_ikey();

    // Follow B-links until we find the right leaf
    let mut current: *mut L = stack.leaf_ptr();

    loop {
        // SAFETY: current is protected by guard
        let leaf: &L = unsafe { &*current };

        // Check if cursor key belongs in this leaf
        let bound: u64 = leaf.ikey_bound();

        if cursor_ikey < bound {
            // Key belongs in this leaf or earlier
            // (But we've already passed earlier leaves)
            break;
        }

        // Get next leaf
        let next: *mut L = leaf.safe_next();

        if next.is_null() {
            // End of chain
            break;
        }

        // Check if key is before next leaf's bound
        let next_leaf: &L = unsafe { &*next };
        let next_bound: u64 = next_leaf.ikey_bound();

        if cursor_ikey < next_bound {
            // Key belongs in current leaf
            break;
        }

        // Move to next
        current = next;
    }

    // Update stack with new leaf
    stack.set_leaf(current);

    // Retry will refresh state
    (ScanState::Retry, None)
}

// ============================================================================
//  find_retry
// ============================================================================

/// Reposition after a conflict or layer transition.
///
/// Called after:
/// - Version validation failure
/// - Layer descent (Down → `shift_clear`)
/// - Layer ascent (Up → unshift)
///
/// Re-traverses from the current layer root to find the correct leaf
/// and position for the current cursor key state.
///
/// # Arguments
///
/// - `stack`: Current scan position (root may have changed)
/// - `cursor_key`: Cursor tracking current position
/// - `guard`: Memory reclamation guard
///
/// # Returns
///
/// Next state (usually `FindNext` to continue iteration).
pub fn find_retry<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> ScanState
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // Re-reach leaf from current root
    let leaf_ptr: *mut L = reach_leaf_for_scan::<L, S>(stack.root(), cursor_key, guard);

    if leaf_ptr.is_null() {
        // Layer is empty
        return ScanState::Up;
    }

    stack.set_leaf(leaf_ptr);

    // SAFETY: leaf_ptr is non-null and protected by guard
    let leaf: &L = unsafe { &*leaf_ptr };

    // Get stable version
    let version: u32 = leaf.version().stable();

    // Check if deleted
    if leaf.version().is_deleted() {
        return ScanState::Retry;
    }

    // Load permutation
    let perm: L::Perm = match leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            return ScanState::Retry;
        }
    };

    // Find position using suffix-aware search
    // This ensures we find the correct position when keys share the same ikey
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);

    // Update stack
    stack.update_state(version, perm, kx.i);

    ScanState::FindNext
}

// ============================================================================
//  Traversal Helper
// ============================================================================

/// Traverse from layer root to target leaf.
///
/// Similar to `reach_leaf_concurrent_generic` but uses cursor key's ikey.
fn reach_leaf_for_scan<L, S>(
    start: *const u8,
    cursor_key: &CursorKey,
    _guard: &LocalGuard<'_>,
) -> *mut L
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    if start.is_null() {
        return std::ptr::null_mut();
    }

    let target_ikey: u64 = cursor_key.current_ikey();
    let mut node: *const u8 = start;

    loop {
        // SAFETY: node is valid, both node types have NodeVersion as first field
        #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
        let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

        // Get stable version (spins if dirty)
        let v: u32 = version.stable();

        if version.is_leaf() {
            // Reached a leaf
            return node.cast_mut().cast::<L>();
        }

        // It's an internode - traverse down
        // SAFETY: !is_leaf() confirmed above
        let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };

        // Binary search for child
        let child_idx: usize = upper_bound_internode_generic::<S, L::Internode>(target_ikey, inode);
        let child: *mut u8 = inode.child(child_idx);

        // Prefetch child node
        prefetch_read(child);

        if child.is_null() {
            // Concurrent split in progress - retry from start
            node = start;
            continue;
        }

        // Check if internode changed during our read
        if inode.version().has_changed(v) {
            // Version changed - check for split
            if inode.version().has_split(v) {
                // Key might have escaped to sibling - retry from start
                node = start;
                continue;
            }
            // Just retry this internode
            continue;
        }

        // Descend to child
        node = child;
    }
}

// ============================================================================
//  Handle Layer Transitions (called from iterator)
// ============================================================================

/// Handle descent into a sublayer (Down state).
///
/// Called by the iterator when `find_next` returns `Down`.
///
/// # Actions
///
/// 1. Layer stack already has parent context pushed by `find_next`
/// 2. Stack root already set to layer pointer by `find_next`
/// 3. Clear cursor for sublayer scan
/// 4. Transition to Retry to position in new layer
pub fn handle_down<L, S>(_stack: &mut ScanStackElement<L, S>, cursor_key: &mut CursorKey)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // Clear cursor key for sublayer (scan from minimum)
    cursor_key.shift_clear();
}

/// Handle ascent from exhausted sublayer (Up state).
///
/// Called by the iterator when `find_next` returns `Up`.
///
/// # Arguments
///
/// - `stack`: Current scan position
/// - `cursor_key`: Cursor to unshift
/// - `layer_stack`: Parent layer stack to pop from
/// - `guard`: Memory reclamation guard
///
/// # Returns
///
/// `true` if there's a parent layer to return to, `false` if scan is complete.
pub fn handle_up<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    _guard: &LocalGuard<'_>,
) -> bool
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // Pop parent context
    let Some(parent) = layer_stack.pop() else {
        // No parent, scan is complete
        return false;
    };

    // Restore parent state
    stack.set_root(parent.root);
    stack.set_leaf(parent.leaf_ptr());

    // Unshift cursor (sets len=9 sentinel)
    cursor_key.unshift();

    // Refresh parent leaf state
    // SAFETY: parent.leaf is protected by guard
    let leaf: &L = unsafe { parent.leaf.as_ref() };

    let version: u32 = leaf.version().stable();

    let perm: L::Perm = match leaf.permutation_try() {
        Ok(p) => p,
        Err(_frozen) => {
            // Will retry via find_retry
            stack.update_state(0, L::Perm::empty(), 0);
            return true;
        }
    };

    // Find position (cursor has len=9, will skip past the layer pointer)
    // Use suffix-aware search to handle keys with same ikey correctly
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);

    stack.update_state(version, perm, kx.i);

    true
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Integration tests are in CODE_005 and tests/range_scan_tests.rs
    // These tests would require mock leaf nodes which is complex

    #[test]
    fn test_key_indexed_position() {
        use super::KeyIndexedPosition;

        let not_found = KeyIndexedPosition::not_found(5);
        assert!(!not_found.has_match());
        assert_eq!(not_found.i, 5);

        let found = KeyIndexedPosition::found(3, 10);
        assert!(found.has_match());
        assert_eq!(found.slot(), 10);
    }
}
