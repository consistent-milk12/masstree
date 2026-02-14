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
//! # Performance Characteristics
//!
//! - **O(1)** per-entry iteration within a leaf (permutation index increment)
//! - **O(log n)** cost for layer transitions (tree traversal)
//! - **Prefetching strategy**: 3-way pipelining prefetches current leaf's data arrays,
//!   next leaf (full 6 cache lines), and next-next leaf for memory latency hiding
//!
//! # C++ Reference
//!
//! Corresponds to `scanstackelt::find_initial`, `find_next`, and the retry
//! logic in `masstree_scan.hh`.

use std::cmp::Ordering;
use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::hints::likely;
use crate::key::IKEY_SIZE;
use crate::leaf_trait::TreeLeafNode;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::link::Linker;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;
use crate::tree::range::iterator::RangeBound;

use super::cursor_key::CursorKey;
use super::helper::{
    ForwardScanHelper, KeyIndexedPosition, lower_with_position, lower_with_suffix,
};
use super::scan_state::{
    FindResult, LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr, ScanStackElement,
    ScanState,
};
use super::traversal::reach_leaf_for_scan;

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
/// # Panics
///
/// This function assumes the guard is held and protects all accessed pointers.
/// Undefined behavior may occur if called without proper memory reclamation protection.
///
/// # C++ Reference
///
/// Corresponds to `scanstackelt::find_initial` in `masstree_scan.hh`.
pub fn find_initial<P>(
    root: *const u8,
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    _layer_stack: &mut LayerStack<P>,
    emit_equal: bool,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshot<P>>)
where
    P: LeafPolicy,
    P::Output: Clone,
{
    // Initialize stack with root
    stack.set_root(root);

    // Reach the target leaf
    let leaf_ptr: *mut LeafNode15<P> = reach_leaf_for_scan::<P>(root, cursor_key, guard);

    if leaf_ptr.is_null() {
        // Empty tree
        return (ScanState::Up, None);
    }

    stack.set_leaf(leaf_ptr);

    // SAFETY: leaf_ptr is valid (non-null checked above, guard protects it)
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

    // Get stable version
    let version: u32 = leaf.version().stable();

    // Check if deleted (use version we already loaded to avoid extra atomic)
    if NodeVersion::is_deleted_version(version) {
        // Retry from root
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

    // Find lower bound position
    let kx: KeyIndexedPosition = lower_with_position(cursor_key, leaf, &perm);

    // Handle position based on whether we found a match
    let (next_state, snapshot) = kx.p.map_or_else(
        || (ScanState::FindNext, None),
        |slot| {
            handle_initial_match::<P>(
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
    //
    // Position logic (kx.i is the permutation index, not the physical slot):
    // - Match found (kx.p.is_some()): advance to kx.i + 1 to skip past the matched
    //   entry, regardless of whether we emitted it (emit_equal=true) or skipped it
    //   (emit_equal=false for Excluded bound).
    // - No match (kx.p.is_none()): start at kx.i, which is the insertion point
    //   where the next greater key would be.
    let final_pos = if kx.p.is_some() { kx.i + 1 } else { kx.i };
    stack.update_state(version, perm, final_pos);

    (next_state, snapshot)
}

/// Handle an exact ikey match in `find_initial`.
#[expect(clippy::too_many_arguments, reason = "Internals")]
#[inline(always)]
fn handle_initial_match<P>(
    leaf: &LeafNode15<P>,
    slot: usize,
    cursor_key: &mut CursorKey,
    stack: &mut ScanStackElement<P>,
    emit_equal: bool,
    _version: u32,
    _perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    _pos: usize,
) -> (ScanState, Option<ScanSnapshot<P>>)
where
    P: LeafPolicy,
    P::Output: Clone,
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
        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
        cursor_key.assign_store_ikey(slot_ikey);
        // Prefetch layer root before descending
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
                // Use if-let to eliminate TOCTOU race between is_value_empty and load_value
                if let Some(output) = leaf.load_value(slot) {
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
        // Use if-let to eliminate TOCTOU race between is_value_empty and load_value
        if let Some(output) = leaf.load_value(slot) {
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
/// # Panics
///
/// Assumes the stack contains a valid leaf pointer protected by the guard.
/// Undefined behavior may occur if called with an invalid stack state.
///
/// # C++ Reference
///
/// Corresponds to `scanstackelt::find_next` in `masstree_scan.hh`.
#[inline]
pub fn find_next<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
) -> FindResult<P>
where
    P: LeafPolicy,
    P::Output: Clone,
{
    // OPTIMIZATION: Skip duplicate check in normal forward iteration.
    // Duplicates only occur after Retry states (version conflict), so the
    // iterator state machine calls find_next_with_duplicate_check after retries.
    find_next_inner(stack, cursor_key, layer_stack, guard, false)
}

/// Find the next entry with duplicate checking enabled.
///
/// Called after a Retry state to skip already-emitted entries.
#[inline]
pub fn find_next_with_duplicate_check<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
) -> FindResult<P>
where
    P: LeafPolicy,
    P::Output: Clone,
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
/// scans. The key difference: this version calls `P::output_from_raw()` which clones
/// Arc values (2 atomic ops), while the `_ptr` variant returns raw pointers directly.
/// Combining via generics/traits would add overhead on the hot path.
#[inline]
fn find_next_inner<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> FindResult<P>
where
    P: LeafPolicy,
    P::Output: Clone,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        return FindResult::transition(ScanState::Up);
    }

    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };

    // Check if leaf is deleted (this is cheap - single atomic load)
    if leaf.version().is_deleted() {
        return FindResult::transition(ScanState::Retry);
    }

    // Get current slot - O(1) via perm.get(ki)
    let Some(slot) = stack.kp() else {
        // Leaf exhausted, try advancing (this validates version)
        return advance_leaf(stack, cursor_key, guard);
    };

    // Read slot data - Guard ensures memory safety
    // Use Relaxed ordering - permutation loaded with Acquire, OCC validates at end
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // CRITICAL: Always verify monotonicity to catch concurrent modifications.
    //
    // If the leaf's permutation changed while we were iterating (due to concurrent
    // insert/split), the cached permutation may point to a slot that now contains
    // a different key. Without this check, we could emit keys out of order.
    //
    // This is a cheap check (one u64 comparison) that catches the backward-jump case
    // without requiring per-entry version validation.
    //
    // OPTIMIZATION: Uses stack.last_ikey() instead of cursor_key.current_ikey()
    // to avoid potential cache miss on cursor_key access. last_ikey is stored
    // in the stack element which is already hot in cache.
    //
    // NOTE: This only checks ikey, not full key with suffix. For exact duplicate
    // detection (same ikey, different suffix), we still need `needs_duplicate_check`.
    if slot_ikey < stack.last_ikey() {
        // Slot ikey went backwards - leaf was modified, need to retry
        return FindResult::transition(ScanState::Retry);
    }

    // Check for duplicate only when needed (after Retry)
    // OPTIMIZATION: In normal forward iteration, stack.next() already advances
    // past the previous entry, so duplicates can't occur
    if needs_duplicate_check {
        // First check ikey + keylenx level
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        // After retry repositioning, we typically land past duplicates (Less case).
        // Use likely() hint to help branch prediction.
        let is_dup: bool = if likely(cmp == Ordering::Less) {
            false // cursor < slot, not duplicate (common case)
        } else if cmp == Ordering::Greater {
            true // cursor > slot, definitely duplicate
        } else {
            // Ordering::Equal - need suffix comparison if both have suffixes
            if slot_keylenx == KSUF_KEYLENX && cursor_key.has_suffix() {
                leaf.ksuf(slot).is_none_or(|stored_suffix| {
                    cursor_key.compare_suffix(stored_suffix) != Ordering::Less
                })
            } else {
                true // Equal at ikey+keylenx level, is duplicate
            }
        };

        if is_dup {
            stack.next();
            return FindResult::transition(ScanState::FindNext);
        }
    }

    // Handle layer pointer - descend into sublayer
    if slot_keylenx >= LAYER_KEYLENX {
        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
        layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
        cursor_key.assign_store_ikey(slot_ikey);

        // Prefetch layer root before descending (hide memory latency)
        prefetch_read(layer_ptr);
        stack.set_root(layer_ptr);

        return FindResult::transition(ScanState::Down);
    }

    // Value slot - prepare for emit
    // Use let-else to eliminate TOCTOU race between is_value_empty and load_value
    let Some(output) = leaf.load_value(slot) else {
        // Concurrent modification removed value - skip this slot, OCC will retry
        stack.next();
        return FindResult::transition(ScanState::FindNext);
    };

    // Compute key length - only read suffix NOW if needed
    let key_len: usize = if slot_keylenx == KSUF_KEYLENX {
        // Read suffix only when emitting
        if let Some(suffix) = leaf.ksuf(slot) {
            let suffix_len: usize = suffix.len();

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
        let len: usize = slot_keylenx as usize;

        cursor_key.assign_store_ikey(slot_ikey);
        cursor_key.assign_store_length(len);
        len
    };

    cursor_key.mark_key_complete();

    // Update cached ikey after successful read (for next iteration's monotonicity check)
    stack.set_last_ikey(slot_ikey);

    // Advance position for next call
    stack.next();

    FindResult::emit(ScanSnapshot::new(output, key_len))
}

// ============================================================================
//  Zero-Copy Variants (for scan_ref)
// ============================================================================

/// Find the next entry, returning a raw pointer instead of cloning.
///
/// This is the zero-copy variant of [`find_next`] for use with `scan_ref`.
/// Instead of calling `P::output_from_raw` (which clones Arc values),
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
pub fn find_next_ptr<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    find_next_inner_ptr(stack, cursor_key, layer_stack, guard, false)
}

/// Find the next entry with duplicate checking, returning raw pointer.
///
/// Zero-copy variant of [`find_next_with_duplicate_check`].
#[inline]
pub fn find_next_with_duplicate_check_ptr<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    find_next_inner_ptr(stack, cursor_key, layer_stack, guard, true)
}

/// Inner implementation for zero-copy [`find_next`].
///
/// Nearly identical to [`find_next_inner`] but:
/// - Does NOT call `P::output_from_raw` (no Arc clone)
/// - Returns `ScanSnapshotPtr` with raw pointer instead
///
/// This eliminates 2 atomic operations per entry (increment + decrement).
///
/// See [`find_next_inner`] for the full algorithm documentation.
/// The duplication is intentional for hot-path performance.
///
/// # Safety
///
/// The returned pointer is only valid while:
/// 1. The guard is held (ensures memory isn't reclaimed)
/// 2. The version hasn't changed (OCC validation at leaf boundaries)
///
/// Callers must dereference immediately within the same guard scope.
#[inline]
fn find_next_inner_ptr<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        return (ScanState::Up, None);
    }

    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };

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

    // CRITICAL: Always verify monotonicity (see find_next_inner for rationale)
    // OPTIMIZATION: Uses stack.last_ikey() instead of cursor_key.current_ikey()
    if slot_ikey < stack.last_ikey() {
        return (ScanState::Retry, None);
    }

    // Check for duplicate only when needed (after Retry)
    if needs_duplicate_check {
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        // After retry repositioning, we typically land past duplicates (Less case).
        let is_dup: bool = if likely(cmp == Ordering::Less) {
            false
        } else if cmp == Ordering::Greater {
            true
        } else {
            // Ordering::Equal
            if slot_keylenx == KSUF_KEYLENX && cursor_key.has_suffix() {
                leaf.ksuf(slot).is_none_or(|stored_suffix| {
                    cursor_key.compare_suffix(stored_suffix) != Ordering::Less
                })
            } else {
                true
            }
        };

        if is_dup {
            stack.next();
            return (ScanState::FindNext, None);
        }
    }

    // Handle layer pointer - descend into sublayer
    if slot_keylenx >= LAYER_KEYLENX {
        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
        layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
        cursor_key.assign_store_ikey(slot_ikey);
        // Prefetch layer root before descending
        prefetch_read(layer_ptr);
        stack.set_root(layer_ptr);
        return (ScanState::Down, None);
    }

    // Value slot - prepare for emit (NO CLONING)
    if leaf.is_value_empty(slot) {
        stack.next();
        return (ScanState::FindNext, None);
    }

    // Load raw pointer for zero-copy return — caller will dereference.
    let slot_ptr: *mut u8 = leaf.load_value_raw(slot);

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

    // Update cached ikey after successful read (for next iteration's monotonicity check)
    stack.set_last_ikey(slot_ikey);

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
#[expect(clippy::cast_possible_truncation)]
pub fn find_next_single_layer_ptr<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    // SAFETY: Stack should have valid leaf at this point
    if stack.is_null() {
        // In single-layer mode, null leaf means exhausted (no Up transition)
        return (ScanState::FindNext, None);
    }

    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };

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

    // CRITICAL: Always verify monotonicity (see find_next_inner for rationale)
    // OPTIMIZATION: Uses stack.last_ikey() instead of cursor_key.current_ikey()
    if slot_ikey < stack.last_ikey() {
        return (ScanState::Retry, None);
    }

    // Check for duplicate only when needed (after Retry)
    if needs_duplicate_check {
        let cmp: Ordering = cursor_key.compare(slot_ikey, slot_keylenx as usize);

        // After retry repositioning, we typically land past duplicates (Less case).
        // For single-layer, no suffix comparison needed (keylenx <= 8).
        let is_dup: bool = !likely(cmp == Ordering::Less);

        if is_dup {
            stack.next();
            return (ScanState::FindNext, None);
        }
    }

    // Single-layer fast path only supports inline keys (0..=8).
    // Fallback for any longer encoding and let the multi-layer path re-process.
    if slot_keylenx > IKEY_SIZE as u8 {
        // For real layer pointers, preserve prefetch/cursor behavior to keep
        // fallback cost low and deterministic.
        if slot_keylenx >= LAYER_KEYLENX {
            // CRITICAL: Store the slot ikey to cursor before returning Down.
            // This ensures shift_clear() in handle_down preserves the ikey in the buffer.
            // Without this, the full_key() would have null bytes at the parent layer offset.
            cursor_key.assign_store_ikey(slot_ikey);

            // Prefetch the layer pointer for the caller
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
            prefetch_read(layer_ptr);
        }

        // NOTE: We do NOT set stack.root here because the caller needs to
        // push the current (parent) context to layer_stack BEFORE setting root.
        // The caller will disable single-layer mode and re-process this slot.
        return (ScanState::Down, None);
    }

    // Value slot - prepare for emit
    if leaf.is_value_empty(slot) {
        stack.next();
        return (ScanState::FindNext, None);
    }

    // Load raw pointer for zero-copy return.
    let slot_ptr: *mut u8 = leaf.load_value_raw(slot);

    // Single-layer keys are always ≤ 8 bytes, no suffix handling
    let key_len: usize = slot_keylenx as usize;
    cursor_key.assign_store_ikey(slot_ikey);
    cursor_key.assign_store_length(key_len);
    cursor_key.mark_key_complete();

    // Update cached ikey after successful read (for next iteration's monotonicity check)
    stack.set_last_ikey(slot_ikey);

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
/// The guard parameter (prefixed `_`) ensures pointer validity through lifetime
/// binding even though it's not directly used in this function.
#[inline(always)]
fn advance_leaf_single_layer<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };
    let version: u32 = stack.version();

    // CRITICAL: Split-aware leaf boundary validation (TOCTOU-safe ordering).

    // Step 1: Load raw next pointer FIRST (may be marked)
    let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

    // Step 2: Check if next is marked (split in progress on this boundary)
    if Linker::is_marked(next_raw) {
        leaf.wait_for_split();

        return (ScanState::Retry, None);
    }

    // Step 3: Validate version AFTER loading next_raw (catches concurrent splits)
    if leaf.version().has_changed(version) {
        return (ScanState::Retry, None);
    }

    // Clear mark bit (safe_next equivalent, but we already loaded raw)
    let next: *mut LeafNode15<P> = next_raw.map_addr(|addr| addr & !1);

    if next.is_null() {
        // No more leaves - scan exhausted (no Up in single-layer)
        stack.set_leaf(StdPtr::null_mut());

        return (ScanState::FindNext, None);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &LeafNode15<P> = unsafe { &*next };

    // Prefetch the next leaf's data arrays
    next_leaf.prefetch();

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut LeafNode15<P> = next_leaf.safe_next(guard);
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &LeafNode15<P> = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = next_leaf.permutation();

    // Reposition using full key comparison
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);

    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Advance to next leaf, zero-copy variant.
///
/// Same as [`advance_leaf`] but returns `ScanSnapshotPtr`.
/// The guard parameter (prefixed `_`) ensures pointer validity through lifetime
/// binding even though it's not directly used in this function.
#[inline]
pub fn advance_leaf_ptr<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>)
where
    P: LeafPolicy,
{
    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };
    let version: u32 = stack.version();

    // CRITICAL: Split-aware leaf boundary validation (TOCTOU-safe ordering).
    // See advance_leaf_single_layer for the full rationale.

    // Step 1: Load raw next pointer FIRST
    let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

    // Step 2: Check if next is marked
    if Linker::is_marked(next_raw) {
        leaf.wait_for_split();
        return (ScanState::Retry, None);
    }

    // Step 3: Validate version AFTER loading next_raw
    if leaf.version().has_changed(version) {
        return (ScanState::Retry, None);
    }

    // Clear mark bit
    let next: *mut LeafNode15<P> = next_raw.map_addr(|addr| addr & !1);

    if next.is_null() {
        // No more leaves in this layer
        return (ScanState::Up, None);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &LeafNode15<P> = unsafe { &*next };

    // Prefetch the next leaf's data arrays
    next_leaf.prefetch();

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut LeafNode15<P> = next_leaf.safe_next(guard);
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &LeafNode15<P> = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return (ScanState::Retry, None);
    }

    // Load permutation
    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = next_leaf.permutation();

    // Reposition using full key comparison
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, next_leaf, &perm);

    stack.update_state(next_version, perm, kx.i);

    (ScanState::FindNext, None)
}

/// Advance to the next leaf when current is exhausted.
///
/// Uses `lower_with_suffix` to find the correct starting position in the new
/// leaf, matching the C++ behavior of `helper.lower(ka, this)`.
/// The guard parameter (prefixed `_`) ensures pointer validity through lifetime
/// binding even though it's not directly used in this function.
#[inline]
fn advance_leaf<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> FindResult<P>
where
    P: LeafPolicy,
{
    let leaf: &LeafNode15<P> = unsafe { stack.leaf_ref() };
    let version: u32 = stack.version();

    // CRITICAL: Split-aware leaf boundary validation (TOCTOU-safe ordering).
    // See advance_leaf_single_layer for the full rationale.

    // Step 1: Load raw next pointer FIRST
    let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

    // Step 2: Check if next is marked
    if Linker::is_marked(next_raw) {
        leaf.wait_for_split();
        return FindResult::transition(ScanState::Retry);
    }

    // Step 3: Validate version AFTER loading next_raw
    if leaf.version().has_changed(version) {
        return FindResult::transition(ScanState::Retry);
    }

    // Clear mark bit
    let next: *mut LeafNode15<P> = next_raw.map_addr(|addr| addr & !1);

    // Capture prev leaf info for tracing before mutating stack
    if next.is_null() {
        // No more leaves in this layer
        return FindResult::transition(ScanState::Up);
    }

    // Move to next leaf
    stack.set_leaf(next);

    // SAFETY: next is non-null and protected by guard
    let next_leaf: &LeafNode15<P> = unsafe { &*next };

    // Prefetch the next leaf's data arrays (ikey0, keylenx, leaf_values)
    // This brings multiple cache lines into L1/L2 before we iterate
    next_leaf.prefetch();

    // Prefetch next-next leaf for 3-way pipelining
    // Full prefetch (6 cache lines) instead of just CL0
    let next_next: *mut LeafNode15<P> = next_leaf.safe_next(guard);
    if !next_next.is_null() {
        // SAFETY: next_next is non-null and derived from a valid leaf's B-link
        let next_next_leaf: &LeafNode15<P> = unsafe { &*next_next };
        next_next_leaf.prefetch();
    }

    // Get stable version
    let next_version: u32 = next_leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(next_version) {
        return FindResult::transition(ScanState::Retry);
    }

    // Load permutation
    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = next_leaf.permutation();

    // Reposition using full key comparison (like C++ `helper.lower(ka, this)`).
    // This ensures we skip past any keys <= cursor_key.
    let kx = lower_with_suffix(cursor_key, next_leaf, &perm);

    stack.update_state(next_version, perm, kx.i);

    FindResult::transition(ScanState::FindNext)
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
pub fn find_retry<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> ScanState
where
    P: LeafPolicy,
{
    // Re-reach leaf from current root
    let leaf_ptr: *mut LeafNode15<P> = reach_leaf_for_scan::<P>(stack.root(), cursor_key, guard);

    if leaf_ptr.is_null() {
        // Layer is empty
        return ScanState::Up;
    }

    stack.set_leaf(leaf_ptr);

    // SAFETY: leaf_ptr is non-null and protected by guard
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

    // Get stable version
    let version: u32 = leaf.version().stable();

    // Check if deleted (use version we already loaded)
    if NodeVersion::is_deleted_version(version) {
        return ScanState::Retry;
    }

    // Load permutation
    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

    // Find position using suffix-aware search
    // This ensures we find the correct position when keys share the same ikey
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);

    // Update stack
    stack.update_state(version, perm, kx.i);

    ScanState::FindNext
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
pub fn handle_down<P>(stack: &mut ScanStackElement<P>, cursor_key: &mut CursorKey)
where
    P: LeafPolicy,
{
    // Clear cursor key for sublayer (scan from minimum)
    cursor_key.shift_clear();

    // Reset last_ikey for sublayer - sublayer ikeys are independent of parent
    // Without this, monotonicity check would compare sublayer ikeys against
    // parent's last_ikey, causing false Retry loops
    stack.set_last_ikey(0);
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
/// - `_guard`: Memory reclamation guard (ensures pointer validity through lifetime binding)
///
/// # Returns
///
/// `true` if there's a parent layer to return to, `false` if scan is complete.
pub fn handle_up<P>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    _guard: &LocalGuard<'_>,
) -> bool
where
    P: LeafPolicy,
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
    let leaf: &LeafNode15<P> = unsafe { parent.leaf.as_ref() };

    let version: u32 = leaf.version().stable();

    let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

    // Find position (cursor has len=9, will skip past the layer pointer)
    // Use suffix-aware search to handle keys with same ikey correctly
    let kx: KeyIndexedPosition = lower_with_suffix(cursor_key, leaf, &perm);
    stack.update_state(version, perm, kx.i);

    // Reset last_ikey for parent layer - parent ikeys are independent of sublayer
    // Without this, monotonicity check would compare parent ikeys against
    // sublayer's last_ikey, causing false Retry loops
    stack.set_last_ikey(0);

    true
}

// ============================================================================
//  Intra-Leaf Batch Processing
// ============================================================================

/// Result of processing entries within a single leaf.
///
/// Uses `#[repr(u8)]` for smaller size (1 byte vs default enum size) and
/// faster dispatch via direct byte comparison. This enum is returned from
/// hot-path batch processing functions.
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
#[inline]
pub fn process_leaf_batch_ptr<P, F>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    end_bound: &RangeBound<'_>,
    end_bound_ikey: Option<u64>,
    visitor: &mut F,
    count: &mut usize,
) -> LeafBatchResult
where
    P: LeafPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    // Cache leaf pointer to avoid borrow conflicts
    let leaf_ptr: *const LeafNode15<P> = stack.leaf_ptr();
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
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
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
            layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
            cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            stack.set_root(layer_ptr);
            return LeafBatchResult::LayerEncountered;
        }

        // Fast end bound pre-check using ikey. This avoids full key comparison
        // for the common case where the first 8 bytes already decide the bound.
        let mut needs_full_end_check = true;
        if end_bound_ikey.is_none() {
            needs_full_end_check = false;
        } else if cursor_key.is_at_root_layer()
            && let Some(bound_ikey) = end_bound_ikey
        {
            match slot_ikey.cmp(&bound_ikey) {
                Ordering::Greater => return LeafBatchResult::EndBoundExceeded,
                Ordering::Less => needs_full_end_check = false,
                Ordering::Equal => {}
            }
        }

        // Check value slot
        if leaf.is_value_empty(slot) {
            stack.next();
            continue;
        }

        // Build key in cursor (side effects only, key_len computed via cursor)
        if slot_keylenx == KSUF_KEYLENX {
            if let Some(suffix) = leaf.ksuf(slot) {
                let suffix_len = suffix.len();
                cursor_key.assign_store_ikey(slot_ikey);
                let _ = cursor_key.assign_store_suffix(suffix);
                cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
            } else {
                cursor_key.assign_store_ikey(slot_ikey);
                cursor_key.assign_store_length(IKEY_SIZE);
            }
        } else {
            let len = slot_keylenx as usize;
            cursor_key.assign_store_ikey(slot_ikey);
            cursor_key.assign_store_length(len);
        }

        cursor_key.mark_key_complete();

        // Check end bound
        let key: &[u8] = cursor_key.full_key();
        if needs_full_end_check && !end_bound.contains(key) {
            return LeafBatchResult::EndBoundExceeded;
        }

        // SAFETY: Guard protects value, slot is valid (non-empty, in permutation)
        let value_ref: &P::Value = unsafe { &*leaf.load_value_ptr(slot) };

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

/// Process remaining entries in current leaf, returning values by copy.
///
/// This is the variant of [`process_leaf_batch_ptr`] that works for ALL `LeafPolicy`
/// types, including true-inline storage. Instead of returning `&P::Value` references
/// (which requires pointer-backed storage), it returns `P::Output` by value.
///
/// # Key Differences from `process_leaf_batch_ptr`
///
/// | Aspect | `process_leaf_batch_ptr` | `process_leaf_batch` |
/// |--------|--------------------------|----------------------|
/// | Visitor signature | `FnMut(&[u8], &P::Value)` | `FnMut(&[u8], P::Output)` |
/// | Value access | `&*slot_ptr.cast()` (deref) | `leaf.load_value(slot)` |
/// | Storage support | `RefLeafPolicy` only | All `LeafPolicy` types |
/// | Use case | Zero-copy for Arc/Box | Universal, works with inline |
///
/// # Performance for Inline Storage
///
/// For true-inline storage (`InlinePolicy<V>`), `ValueArray::load()` directly
/// reads the `AtomicU64` bits and reconstructs the value. No pointer
/// indirection or intermediate conversion needed.
///
/// # Algorithm
///
/// Same as `process_leaf_batch_ptr`:
/// 1. Read slot data `(ikey, keylenx, value_ptr)`
/// 2. If layer pointer → return [`LeafBatchResult::LayerEncountered`]
/// 3. If null value → skip
/// 4. Build key and call visitor with loaded value
/// 5. Check end bound
/// 6. Validate version (OCC) after batch
#[inline]
pub fn process_leaf_batch<P, F>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    end_bound: &RangeBound<'_>,
    end_bound_ikey: Option<u64>,
    visitor: &mut F,
    count: &mut usize,
) -> LeafBatchResult
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    // Cache leaf pointer to avoid borrow conflicts
    let leaf_ptr: *const LeafNode15<P> = stack.leaf_ptr();
    // SAFETY: leaf_ptr is valid - protected by guard in caller
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
    let perm = stack.perm();
    let perm_size = perm.size();
    let cached_version = stack.version();

    // Check if leaf was deleted since we cached the version
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
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
            layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
            cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            stack.set_root(layer_ptr);
            return LeafBatchResult::LayerEncountered;
        }

        // Fast end bound pre-check using ikey. This avoids full key comparison
        // for the common case where the first 8 bytes already decide the bound.
        let mut needs_full_end_check = true;
        if end_bound_ikey.is_none() {
            needs_full_end_check = false;
        } else if cursor_key.is_at_root_layer()
            && let Some(bound_ikey) = end_bound_ikey
        {
            match slot_ikey.cmp(&bound_ikey) {
                Ordering::Greater => return LeafBatchResult::EndBoundExceeded,
                Ordering::Less => needs_full_end_check = false,
                Ordering::Equal => {}
            }
        }

        // Check value slot
        if leaf.is_value_empty(slot) {
            stack.next();
            continue;
        }

        // Build key in cursor (side effects only, key_len computed via cursor)
        if slot_keylenx == KSUF_KEYLENX {
            if let Some(suffix) = leaf.ksuf(slot) {
                let suffix_len = suffix.len();
                cursor_key.assign_store_ikey(slot_ikey);
                let _ = cursor_key.assign_store_suffix(suffix);
                cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
            } else {
                cursor_key.assign_store_ikey(slot_ikey);
                cursor_key.assign_store_length(IKEY_SIZE);
            }
        } else {
            let len = slot_keylenx as usize;
            cursor_key.assign_store_ikey(slot_ikey);
            cursor_key.assign_store_length(len);
        }

        cursor_key.mark_key_complete();

        // Check end bound
        let key: &[u8] = cursor_key.full_key();
        if needs_full_end_check && !end_bound.contains(key) {
            return LeafBatchResult::EndBoundExceeded;
        }

        // Load value via typed API — works for all storage types.
        // Use let-else to eliminate TOCTOU race between is_value_empty and load_value
        let Some(output) = leaf.load_value(slot) else {
            // Concurrent modification removed value - skip this slot
            stack.next();
            continue;
        };

        *count += 1;
        stack.next();

        if !visitor(key, output) {
            return LeafBatchResult::Stopped;
        }
    }

    // Validate version after processing batch (OCC)
    if leaf.version().has_changed(cached_version) {
        return LeafBatchResult::VersionChanged;
    }

    LeafBatchResult::LeafExhausted
}

/// Process leaf batch without key materialization (values only).
///
/// This is the performance-critical function for value-only scans. It:
/// - Skips all key building (no `assign_store_ikey`, `assign_store_suffix`)
/// - Uses ikey-only end bound check (approximate for suffix keys)
/// - Directly extracts and passes values to visitor
///
/// # End Bound Approximation
///
/// For bounded scans, we only compare `slot_ikey` against `end_bound_ikey`.
/// This is exact when:
/// - End bound is `Unbounded` (`end_bound_ikey` is `None`)
/// - Keys have no suffix (`keylenx <= 8`)
/// - `slot_ikey != end_bound_ikey`
///
/// It may over-include entries when `slot_ikey == end_bound_ikey` and both
/// have suffixes. This is documented in the public API.
///
/// # Performance
///
/// For 64-byte keys, this saves ~47% of scan time by eliminating:
/// - `assign_store_suffix()` (30.7% of time)
/// - `copy_from_slice()` calls (16.7% of time)
///
/// # Algorithm
///
/// 1. Read slot data `(ikey, keylenx, value_ptr)`
/// 2. If layer pointer → return [`LeafBatchResult::LayerEncountered`]
/// 3. Fast end bound check: `slot_ikey > end_bound_ikey` → stop
/// 4. If null value → skip
/// 5. Call visitor with value only (no key)
/// 6. Validate version (OCC) after batch
#[inline]
pub fn process_leaf_batch_values<P, F>(
    stack: &mut ScanStackElement<P>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<P>,
    end_bound_ikey: Option<u64>,
    visitor: &mut F,
    count: &mut usize,
) -> LeafBatchResult
where
    P: LeafPolicy,
    F: FnMut(P::Output) -> bool,
{
    let leaf_ptr: *const LeafNode15<P> = stack.leaf_ptr();
    // SAFETY: leaf_ptr is valid - protected by guard in caller
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
    let perm = stack.perm();
    let perm_size = perm.size();
    let cached_version = stack.version();

    // Check if leaf was deleted since we cached the version
    if leaf.version().is_deleted() {
        return LeafBatchResult::VersionChanged;
    }

    while stack.ki() < perm_size {
        let slot = perm.get(stack.ki());
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = leaf.keylenx(slot);

        // Handle layer pointer - must use state machine
        if slot_keylenx >= LAYER_KEYLENX {
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
            layer_stack.push(LayerContext::new(stack.root(), stack.leaf_ptr()));
            // Still need to track ikey for layer navigation
            cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            stack.set_root(layer_ptr);
            return LeafBatchResult::LayerEncountered;
        }

        // Fast end bound check (ikey only)
        // This is approximate for suffix keys when slot_ikey == bound_ikey
        if let Some(bound_ikey) = end_bound_ikey
            && slot_ikey > bound_ikey
        {
            return LeafBatchResult::EndBoundExceeded;
        }

        // Check value slot
        if leaf.is_value_empty(slot) {
            stack.next();
            continue;
        }

        // === KEY DIFFERENCE: No key building at all ===
        // Skip: cursor_key.assign_store_ikey(slot_ikey);
        // Skip: cursor_key.assign_store_suffix(suffix);
        // Skip: cursor_key.assign_store_length(...);

        // Load value via typed API.
        // Use let-else to eliminate TOCTOU race between is_value_empty and load_value
        let Some(output) = leaf.load_value(slot) else {
            // Concurrent modification removed value - skip this slot
            stack.next();
            continue;
        };

        *count += 1;
        stack.next();

        // Visitor receives only value, no key
        if !visitor(output) {
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
