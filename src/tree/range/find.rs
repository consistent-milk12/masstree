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
use crate::tree::range::iterator::RangeBound;

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

    // Check if deleted (use version we already loaded to avoid extra atomic)
    if NodeVersion::is_deleted_version(version) {
        // Retry from root
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = leaf.permutation();

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
    // Any version change (insert, split, delete) invalidates our position
    if leaf.version().has_changed(version) {
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
        // Use Relaxed ordering - caller loaded permutation with Acquire, OCC validates at end
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        cursor_key.assign_store_ikey(slot_ikey);
        // Prefetch layer root before descending (hide memory latency)
        prefetch_read(layer_ptr);
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
                    // Use Relaxed ordering - caller loaded permutation with Acquire, OCC validates
                    cursor_key.assign_store_ikey(leaf.ikey_relaxed(slot));
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
            // Use Relaxed ordering - caller loaded permutation with Acquire, OCC validates
            cursor_key.assign_store_ikey(leaf.ikey_relaxed(slot));
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
#[inline]
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
#[inline]
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
///
/// # Code Duplication Note
///
/// This function is intentionally duplicated as [`find_next_inner_ptr`] for zero-copy
/// scans. The key difference: this version calls `S::output_from_raw()` which clones
/// Arc values (2 atomic ops), while the `_ptr` variant returns raw pointers directly.
/// Combining via generics/traits would add overhead on the hot path.
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
    // Use Relaxed ordering - permutation loaded with Acquire, OCC validates at end
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);
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
        // Prefetch layer root before descending (hide memory latency)
        prefetch_read(slot_ptr);
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

/// Inner implementation for zero-copy [`find_next`].
///
/// Nearly identical to [`find_next_inner`] but:
/// - Does NOT call `S::output_from_raw` (no Arc clone)
/// - Returns `ScanSnapshotPtr` with raw pointer instead
///
/// This eliminates 2 atomic operations per entry (increment + decrement).
///
/// See [`find_next_inner`] for the full algorithm documentation.
/// The duplication is intentional for hot-path performance.
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
    // Use Relaxed ordering - permutation loaded with Acquire, OCC validates at end
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);
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
        // Prefetch layer root before descending (hide memory latency)
        prefetch_read(slot_ptr);
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

/// Single-layer fast path for zero-copy [`find_next`].
///
/// Optimized for scans where all keys are ≤ 8 bytes (no layer pointers).
/// This eliminates:
/// - Layer pointer checks (`keylenx >= LAYER_KEYLENX`)
/// - Layer stack operations
/// - Down/Up state handling
///
/// # Fallback
///
/// If we unexpectedly encounter a layer pointer (shouldn't happen for truly
/// single-layer data), we return `(ScanState::Down, None)` to signal the
/// caller to fall back to the standard multi-layer path.
///
/// # Performance
///
/// Uses `#[inline]` to let the compiler decide based on call-site context.
/// The function is medium-sized; forcing inlining could cause I-cache pressure.
#[inline]
pub fn find_next_single_layer_ptr<L, S>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanState, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        // In single-layer mode, null leaf means exhausted (no Up transition)
        return (ScanState::FindNext, None);
    }

    let leaf: &L = unsafe { stack.leaf_ref() };

    // Check if leaf is deleted (cheap - single atomic load)
    if leaf.version().is_deleted() {
        return (ScanState::Retry, None);
    }

    // Get current slot - O(1) via perm.get(ki)
    let Some(slot) = stack.kp() else {
        // Leaf exhausted, try advancing to next leaf
        return advance_leaf_single_layer(stack, cursor_key, guard);
    };

    // Read slot data
    // Use Relaxed ordering - permutation loaded with Acquire, OCC validates at end
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // Check for duplicate only when needed (after Retry)
    if needs_duplicate_check {
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        let is_dup: bool = match cmp {
            Ordering::Less => false,

            Ordering::Greater | Ordering::Equal => {
                // For single-layer, no suffix comparison needed (keylenx <= 8)
                true
            }
        };

        if is_dup {
            stack.next();
            return (ScanState::FindNext, None);
        }
    }

    // DEFENSIVE: If we encounter a layer pointer, signal fallback
    // This shouldn't happen for truly single-layer data, but can occur
    // when single_layer_mode was set based on bounds, not actual keys
    if slot_keylenx >= LAYER_KEYLENX {
        // CRITICAL: Store the slot ikey to cursor before returning Down.
        // This ensures shift_clear() in handle_down preserves the ikey in the buffer.
        // Without this, the full_key() would have null bytes at the parent layer offset.
        cursor_key.assign_store_ikey(slot_ikey);

        // Prefetch the layer pointer for the caller
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        prefetch_read(layer_ptr);

        // NOTE: We do NOT set stack.root here because the caller needs to
        // push the current (parent) context to layer_stack BEFORE setting root.
        // The caller will handle: push layer_stack, then set root, then call handle_down.

        return (ScanState::Down, None);
    }

    // Value slot - prepare for emit
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
    if slot_ptr.is_null() {
        stack.next();
        return (ScanState::FindNext, None);
    }

    // Single-layer keys are always ≤ 8 bytes, no suffix handling
    let key_len: usize = slot_keylenx as usize;
    cursor_key.assign_store_ikey(slot_ikey);
    cursor_key.assign_store_length(key_len);
    cursor_key.mark_key_complete();

    // Advance position for next call
    stack.next();

    // Return raw pointer
    (
        ScanState::Emit,
        Some(ScanSnapshotPtr::from_raw(slot_ptr, key_len)),
    )
}

/// Advance to next leaf in single-layer mode.
///
/// Simplified version of `advance_leaf_ptr` that doesn't handle Up transitions.
///
/// # Note
///
/// The `guard` parameter ensures pointer validity through lifetime binding.
#[inline(always)]
fn advance_leaf_single_layer<L, S>(
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
        // No more leaves - scan exhausted (no Up in single-layer)
        stack.set_leaf(std::ptr::null_mut());
        return (ScanState::FindNext, None);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &L = unsafe { &*next };

    // Prefetch the next leaf's data arrays
    next_leaf.prefetch();

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut L = next_leaf.safe_next();
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &L = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = next_leaf.permutation();

    // Reposition using full key comparison
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);
    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Advance to next leaf, zero-copy variant.
///
/// Same as [`advance_leaf`] but returns `ScanSnapshotPtr`.
///
/// # Note
///
/// The `guard` parameter is unused but required for API consistency
/// and to ensure pointer validity through lifetime binding.
///
/// Uses `#[inline]` - medium-sized function; let compiler decide on inlining.
#[inline]
pub fn advance_leaf_ptr<L, S>(
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

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut L = next_leaf.safe_next();
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &L = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = next_leaf.permutation();

    // Reposition using full key comparison
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);
    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Advance to the next leaf when current is exhausted.
///
/// Uses `lower_with_suffix` to find the correct starting position in the new
/// leaf, matching the C++ behavior of `helper.lower(ka, this)`.
///
/// # Note
///
/// The `guard` parameter ensures pointer validity through lifetime binding.
#[inline]
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

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut L = next_leaf.safe_next();
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &L = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: L::Perm = next_leaf.permutation();

    // Reposition using full key comparison (like C++ `helper.lower(ka, this)`).
    // This ensures we skip past any keys <= cursor_key.
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);
    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
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

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(version) {
        return ScanState::Retry;
    }

    // Load permutation
    let perm: L::Perm = leaf.permutation();

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
///
/// # Note
///
/// The `guard` parameter ensures pointer validity through lifetime binding.
#[inline]
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
        let child_idx: usize = upper_bound_internode_generic::<L::Internode>(target_ikey, inode);
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
/// - `guard`: Memory reclamation guard (ensures pointer validity)
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

    // Unshift cursor (sets sentinel length to skip layer pointer)
    cursor_key.unshift();

    // Refresh parent leaf state
    // SAFETY: parent.leaf is protected by guard
    let leaf: &L = unsafe { parent.leaf.as_ref() };

    let version: u32 = leaf.version().stable();

    let perm: L::Perm = leaf.permutation();

    // Find position (cursor has len=9, will skip past the layer pointer)
    // Use suffix-aware search to handle keys with same ikey correctly
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);

    stack.update_state(version, perm, kx.i);

    true
}

// ============================================================================
//  Intra-Leaf Batch Processing
// ============================================================================

/// Result of processing entries within a single leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LeafBatchResult {
    /// All entries in leaf processed, need to advance to next leaf
    LeafExhausted = 0,

    /// Encountered a layer pointer, need to descend
    LayerEncountered = 1,

    /// Version changed during processing, need retry
    VersionChanged = 2,

    /// Visitor returned false, stop iteration
    Stopped = 3,

    /// End bound exceeded, stop iteration
    EndBoundExceeded = 4,
}

/// Process remaining entries in current leaf in a tight loop.
///
/// This is the core intra-leaf batch optimization. Instead of returning after
/// each entry, we process all remaining entries in the current leaf before
/// returning control to the caller.
///
/// # Algorithm
///
/// For each remaining slot in the permutation:
/// 1. Read slot data `(ikey, keylenx, value_ptr)`
/// 2. If layer pointer → return [`LayerEncountered`] (caller handles descent)
/// 3. If null value → skip
/// 4. Build key and call visitor
/// 5. Check end bound
/// 6. Validate version (OCC) after batch
///
/// # Performance
///
/// This eliminates:
/// - Function call overhead per entry
/// - State machine dispatch per entry
/// - Redundant leaf/version checks
///
/// Expected 2-3x improvement for scans touching many entries per leaf.
#[inline]
pub fn process_leaf_batch_ptr<L, S, F>(
    stack: &mut ScanStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    end_bound: &RangeBound<'_>,
    visitor: &mut F,
    count: &mut usize,
) -> LeafBatchResult
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    F: FnMut(&[u8], &S::Value) -> bool,
{
    // Cache leaf pointer to avoid borrow conflicts
    let leaf_ptr: *const L = stack.leaf_ptr();
    let leaf: &L = unsafe { &*leaf_ptr };
    let perm = stack.perm();
    let perm_size = perm.size();
    let cached_version = stack.version();

    // Check if leaf was deleted since we cached the version
    // This makes the function self-contained (caller also checks, but belt-and-suspenders)
    if leaf.version().is_deleted() {
        return LeafBatchResult::VersionChanged;
    }

    // Process remaining entries in this leaf
    while stack.ki() < perm_size {
        let slot = perm.get(stack.ki());

        // Read slot data with relaxed ordering (permutation provides synchronization)
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = leaf.keylenx(slot);

        // Check for layer pointer - must handle via state machine
        if slot_keylenx >= LAYER_KEYLENX {
            // Set up for layer descent
            let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
            layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
            cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(slot_ptr);
            stack.set_root(slot_ptr);
            return LeafBatchResult::LayerEncountered;
        }

        // Get value pointer
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        if slot_ptr.is_null() {
            stack.next();
            continue;
        }

        // Build key
        let _key_len: usize = if slot_keylenx == KSUF_KEYLENX {
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

        // Check end bound
        let key: &[u8] = cursor_key.full_key();
        if !end_bound.contains(key) {
            return LeafBatchResult::EndBoundExceeded;
        }

        // SAFETY: Guard protects value pointer, slot is valid
        let value_ref: &S::Value = unsafe { &*slot_ptr.cast::<S::Value>() };

        *count += 1;
        stack.next();

        if !visitor(key, value_ref) {
            return LeafBatchResult::Stopped;
        }
    }

    // Validate version after processing batch (OCC)
    if leaf.version().has_changed(cached_version) {
        return LeafBatchResult::VersionChanged;
    }

    LeafBatchResult::LeafExhausted
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Integration tests are in tests/range_scan_tests.rs
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
