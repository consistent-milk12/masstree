//! Filepath: `src/tree/range/reverse_ctx.rs`
//!
//! Reverse scan context, owns all reverse iteration state.

use crate::permuter::Permuter15;
use std::cmp::Ordering;
use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::leaf15::LAYER_KEYLENX;
use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::batch_common::{
    Backward, BatchCtx, CopySlotVisitor, RefSlotVisitor, process_batch_keyed, process_batch_values,
};
use super::cursor_key::CursorKey;
use super::find_rev::LeafBatchResultBack;
use super::helper::ReverseScanHelper;
use super::helper::{
    CloneEmitter, PtrEmitter, ScanEmitter, SlotPayload, build_slot_key_parts,
    compare_cursor_to_parts, try_snapshot_slot,
};
use super::iterator::RangeBound;
use super::iterator::iter_flags::ReverseFlags;
use super::scan_state::{
    BackStackElement, LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr, ScanStateBack,
};
use super::traversal::reach_leaf_for_scan;

#[cfg(test)]
mod unit_tests;

// ============================================================================
//  Internal result types (moved from find_rev.rs)
// ============================================================================

/// Result of attempting to find initial position in a single layer.
enum InitialPositionResult<P: LeafPolicy> {
    /// Found a value to emit.
    Emit(ScanSnapshot<P>),

    /// Need to descend into sublayer at the given root.
    LayerDescent(*const u8),

    /// No match at current position, retreat to find previous.
    FindPrev,

    /// Version conflict, retry from root.
    Retry,
}

// ============================================================================
//  ReverseScanCtx
// ============================================================================

/// Reverse scan context, owns all per-direction state for reverse iteration.
///
/// This struct bundles the scan position, layer stack, cursor key, helper,
/// state machine state, and flags that were previously individual fields on
/// `RangeIter`.
pub struct ReverseScanCtx<P: LeafPolicy> {
    /// Current scan position (leaf, version, permutation, signed ki).
    pub(crate) stack: BackStackElement<P>,

    /// Parent layer stack for sublayer navigation (backward).
    pub(crate) layer_stack: LayerStack<P>,

    /// Cursor tracking current key position (backward).
    pub(crate) cursor_key: CursorKey,

    /// Current state machine state (backward).
    pub(crate) state: ScanStateBack,

    /// Captured snapshot for current entry (backward, if in Emit state).
    pub(crate) snapshot: Option<ScanSnapshot<P>>,

    /// Packed reverse-specific boolean flags.
    pub(crate) flags: ReverseFlags,

    // ========================================================================
    //  Debug-only fields
    // ========================================================================
    /// Last emitted key for backward iteration (debug builds only).
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(crate) debug_last_emitted_key_back: Option<Vec<u8>>,
}

// ============================================================================
//  Construction
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Create a new reverse scan context positioned for a bounded reverse scan.
    ///
    /// `root` is the tree root pointer, `cursor_key` is positioned for reverse
    /// scanning, and `emit_equal` controls whether the end-bound key itself
    /// is emitted.
    #[inline]
    pub fn new_with_bound(root: *const u8, cursor_key: CursorKey, emit_equal: bool) -> Self {
        Self {
            stack: BackStackElement::new(root),
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanStateBack::FindPrev,
            snapshot: None,
            flags: ReverseFlags::with_values(emit_equal),

            #[cfg(debug_assertions)]
            debug_last_emitted_key_back: None,
        }
    }
}

// ============================================================================
//  find_initial_reverse — Initial Positioning
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Find the initial position for a reverse range scan.
    ///
    /// Positions the scan at the correct leaf and slot for the end bound.
    /// Uses an iterative loop instead of recursion for layer descent.
    ///
    /// # Algorithm
    ///
    /// 1. Loop through layers starting from root
    /// 2. For each layer:
    ///    - Reach target leaf via `reach_leaf_for_scan`
    ///    - Handle concurrent inserts via `stable_reverse`
    ///    - Find position via `lower_reverse`
    ///    - If layer pointer: setup descent and continue loop
    ///    - If value: try to emit
    ///    - Otherwise: return `FindPrev` to retreat
    pub fn find_initial_reverse(
        &mut self,
        root: *const u8,
        emit_equal: bool,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        let mut current_root: *const u8 = root;

        // Iterative layer descent loop
        loop {
            self.stack.set_root(current_root);

            // Reach target leaf
            let (mut leaf_ptr, resolved_root) =
                reach_leaf_for_scan::<P>(current_root, &self.cursor_key, guard);
            current_root = resolved_root;
            self.stack.set_root(resolved_root);

            // CRITICAL: Check null BEFORE calling stable_reverse
            if leaf_ptr.is_null() {
                return (ScanStateBack::Up, None);
            }

            // Handle concurrent inserts - may follow forward chain
            let mut version: u32 =
                ReverseScanHelper::stable_reverse(&mut leaf_ptr, &self.cursor_key, guard);
            self.stack.set_leaf(leaf_ptr);

            // Fast path: check deleted version
            if NodeVersion::is_deleted_version(version) {
                return (ScanStateBack::Retry, None);
            }

            // An empty rightmost leaf (e.g. after removing its tail) is still
            // linked until maintenance. Retreat through `prev` instead of
            // terminating the whole scan.
            let mut empty_hops: usize = 0;
            loop {
                // SAFETY: leaf_ptr is valid (null checked above, stable_reverse
                // ensures validity).
                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
                let perm: Permuter15 = leaf.permutation();
                let size: usize = perm.size();

                if size == 0 {
                    let prev: *mut LeafNode15<P> = leaf.prev(guard);

                    if prev.is_null() || empty_hops >= 64 {
                        return (ScanStateBack::Up, None);
                    }

                    empty_hops += 1;
                    leaf_ptr = prev;
                    self.stack.set_leaf(leaf_ptr);

                    // SAFETY: prev is protected by the guard.
                    version = unsafe { &*leaf_ptr }.version().stable();

                    if NodeVersion::is_deleted_version(version) {
                        return (ScanStateBack::Retry, None);
                    }

                    continue;
                }

                // Try to find initial position in this layer
                match self.try_initial_position_reverse(
                    leaf,
                    perm,
                    size,
                    version,
                    current_root,
                    leaf_ptr,
                    emit_equal,
                ) {
                    InitialPositionResult::Emit(snapshot) => {
                        self.flags.mark_cursor_emitted();
                        return (ScanStateBack::Emit, Some(snapshot));
                    }

                    InitialPositionResult::LayerDescent(layer_ptr) => {
                        // Continue outer loop with new layer root
                        current_root = layer_ptr;
                        break;
                    }

                    InitialPositionResult::FindPrev => {
                        // Version check before returning
                        if leaf.version().has_changed(version) {
                            return (ScanStateBack::Retry, None);
                        }
                        return (ScanStateBack::FindPrev, None);
                    }

                    InitialPositionResult::Retry => {
                        return (ScanStateBack::Retry, None);
                    }
                }
            }
        }
    }

    /// Attempt to find initial position within a single layer.
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn try_initial_position_reverse(
        &mut self,
        leaf: &LeafNode15<P>,
        perm: Permuter15,
        size: usize,
        version: u32,
        current_root: *const u8,
        leaf_ptr: *mut LeafNode15<P>,
        emit_equal: bool,
    ) -> InitialPositionResult<P>
    where
        P::Output: Clone,
    {
        // Find position using reverse helper
        let ki: isize = ReverseScanHelper::lower_reverse(
            &self.cursor_key,
            leaf,
            perm,
            self.flags.upper_bound(),
        );

        if !(ki >= 0 && ki.cast_unsigned() < size) {
            if leaf.version().has_changed(version) {
                return InitialPositionResult::Retry;
            }
            self.stack.update_state(version, perm, ki);
            return InitialPositionResult::FindPrev;
        }
        let Ok(slot) =
            try_snapshot_slot::<P, CloneEmitter>(leaf, perm.get(ki.cast_unsigned()), version)
        else {
            return InitialPositionResult::Retry;
        };
        // All cursor/stack changes follow validation, including layer descent.
        self.stack.update_state(version, perm, ki - 1);
        match slot.payload {
            SlotPayload::Layer(root) => {
                let same_prefix = slot.ikey == self.cursor_key.current_ikey();
                let real_suffix =
                    self.cursor_key.has_suffix() && !self.cursor_key.is_unshift_sentinel();
                // A terminal bound excludes only a layer with the SAME prefix.
                // Earlier prefixes are entirely below the end bound.
                if same_prefix && !self.flags.upper_bound() && !real_suffix {
                    return InitialPositionResult::FindPrev;
                }
                self.layer_stack
                    .push(LayerContext::new(current_root, leaf_ptr));
                self.cursor_key.assign_store_ikey(slot.ikey);
                prefetch_read(root);
                if same_prefix && real_suffix {
                    self.cursor_key.shift();
                    self.flags.clear_upper_bound();
                } else {
                    self.cursor_key.shift_clear_reverse();
                    self.flags.set_upper_bound();
                }
                InitialPositionResult::LayerDescent(root.cast_const())
            }
            SlotPayload::Value(snapshot) => {
                let cmp =
                    compare_cursor_to_parts(&self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                if !self.flags.upper_bound()
                    && (cmp == Ordering::Less || (cmp == Ordering::Equal && !emit_equal))
                {
                    return InitialPositionResult::FindPrev;
                }
                build_slot_key_parts(&mut self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                self.cursor_key.mark_key_complete();
                InitialPositionResult::Emit(snapshot)
            }
            SlotPayload::Empty => InitialPositionResult::FindPrev,
        }
    }
}

// ============================================================================
//  find_prev - Main Reverse Scan Algorithm
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Find the previous entry for reverse iteration.
    ///
    /// This is the main workhorse for reverse scanning, called repeatedly
    /// after `find_initial_reverse` positions the scan.
    #[inline]
    pub(crate) fn find_prev(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.find_prev_generic::<CloneEmitter>(guard, false)
    }

    /// Find the previous entry with duplicate checking enabled.
    ///
    /// Called after a Retry state to skip already-emitted entries.
    #[inline]
    pub(crate) fn find_prev_with_dup_check(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.find_prev_generic::<CloneEmitter>(guard, true)
    }

    /// Find the previous entry, returning a raw pointer instead of cloning.
    #[inline]
    #[allow(dead_code, reason = "Zero-copy reverse scan API")]
    pub(crate) fn find_prev_ptr(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshotPtr<P::Value>>)
    where
        P: LeafPolicy,
    {
        self.find_prev_generic::<PtrEmitter>(guard, false)
    }

    /// Find the previous entry with duplicate checking, returning raw pointer.
    #[inline]
    #[allow(dead_code, reason = "Zero-copy reverse scan API")]
    pub(crate) fn find_prev_with_dup_check_ptr(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshotPtr<P::Value>>)
    where
        P: LeafPolicy,
    {
        self.find_prev_generic::<PtrEmitter>(guard, true)
    }

    /// Generic inner implementation of `find_prev` with configurable emission strategy.
    #[inline]
    fn find_prev_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
        needs_duplicate_check: bool,
    ) -> (ScanStateBack, Option<E::Snapshot>) {
        // Fast path: null leaf means we need to go up
        let leaf_ptr: *mut LeafNode15<P> = self.stack.get_leaf_ptr();
        if leaf_ptr.is_null() {
            return (ScanStateBack::Up, None);
        }

        let ki: isize = self.stack.get_ki();

        // Fast path: leaf exhausted (ki went negative)
        // Check BEFORE version to avoid unnecessary atomic load
        if ki < 0 {
            return self.advance_to_prev_leaf_generic::<E>(guard);
        }

        // SAFETY: leaf_ptr is valid (null checked above)
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
        let version: u32 = self.stack.get_version();

        // Version check - if changed, reposition
        if leaf.version().has_changed(version) {
            return self.reposition_back_generic::<E>(guard);
        }

        let perm: Permuter15 = *self.stack.get_perm_ref();
        let size: usize = perm.size();

        // Defensive check: ki might be >= size due to concurrent deletion
        if ki.unsigned_abs() >= size {
            return self.advance_to_prev_leaf_generic::<E>(guard);
        }

        // Process the current slot
        self.process_slot_generic::<E>(leaf, leaf_ptr, perm, version, ki, needs_duplicate_check)
    }

    /// Process a single slot during reverse scan (generic over emission strategy).
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn process_slot_generic<E: ScanEmitter<P>>(
        &mut self,
        leaf: &LeafNode15<P>,
        leaf_ptr: *mut LeafNode15<P>,
        perm: Permuter15,
        version: u32,
        ki: isize,
        needs_duplicate_check: bool,
    ) -> (ScanStateBack, Option<E::Snapshot>) {
        let physical_slot = perm.get(ki.cast_unsigned());
        let Ok(slot) = try_snapshot_slot::<P, E>(leaf, physical_slot, version) else {
            return (ScanStateBack::Retry, None);
        };
        if slot.ikey > self.stack.last_ikey() {
            return (ScanStateBack::Retry, None);
        }
        if ki > 0 {
            leaf.prefetch_value(perm.get((ki - 1).cast_unsigned()));
        }
        if needs_duplicate_check && !self.flags.upper_bound() {
            let sentinel = self.cursor_key.is_unshift_sentinel();
            let cursor_ikey = self.cursor_key.current_ikey();
            let is_dup = if slot.keylenx >= LAYER_KEYLENX {
                if sentinel {
                    slot.ikey == cursor_ikey
                } else {
                    slot.ikey >= cursor_ikey
                }
            } else if sentinel {
                false
            } else {
                compare_cursor_to_parts(&self.cursor_key, slot.ikey, slot.keylenx, slot.suffix)
                    != Ordering::Greater
            };
            if is_dup {
                self.stack.set_ki(ReverseScanHelper::prev(ki));
                return (ScanStateBack::FindPrev, None);
            }
        }
        match slot.payload {
            SlotPayload::Layer(root) => {
                let same_prefix = slot.ikey == self.cursor_key.current_ikey();
                let real_suffix =
                    self.cursor_key.has_suffix() && !self.cursor_key.is_unshift_sentinel();
                self.flags
                    .set_down_mode(same_prefix && real_suffix, !same_prefix);
                self.layer_stack
                    .push(LayerContext::new(self.stack.get_root(), leaf_ptr));
                self.cursor_key.assign_store_ikey(slot.ikey);
                prefetch_read(root);
                self.stack.set_root(root.cast_const());
                self.stack.update_state(version, perm, ki - 1);
                (ScanStateBack::Down, None)
            }
            SlotPayload::Empty => {
                self.stack.set_ki(ReverseScanHelper::prev(ki));
                (ScanStateBack::FindPrev, None)
            }
            SlotPayload::Value(snapshot) => {
                build_slot_key_parts(&mut self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                self.cursor_key.mark_key_complete();
                self.flags.clear_upper_bound();
                self.flags.mark_cursor_emitted();
                self.stack.set_last_ikey(slot.ikey);
                self.stack.set_ki(ReverseScanHelper::prev(ki));
                (ScanStateBack::Emit, Some(snapshot))
            }
        }
    }
}

// ============================================================================
//  advance_to_prev_leaf - O(1) Local Leaf Retreat
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Advance to the previous leaf in the B-link chain.
    ///
    /// Called when the current leaf is exhausted (`ki < 0`). Finds the
    /// rightmost non-empty leaf below the current leaf, following split
    /// siblings that appeared below the current lower bound and skipping empty
    /// leaves left behind by removals.
    #[inline]
    fn advance_to_prev_leaf_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<E::Snapshot>) {
        let current_ptr: *mut LeafNode15<P> = self.stack.get_leaf_ptr();

        // SAFETY: current_ptr was validated before calling this function
        let current_leaf: &LeafNode15<P> = unsafe { &*current_ptr };

        // The cursor is the current leaf's lower bound. Everything below it is
        // in the previous leaf (or a split sibling between the two).
        let current_bound: u64 = current_leaf.ikey_bound();
        self.cursor_key.assign_store_ikey(current_bound);
        self.cursor_key.assign_store_length(0);

        // Right boundary of the region we may still visit. It moves left each
        // time we discard an empty leaf, so the walk cannot bounce between an
        // empty leaf and its predecessor.
        let mut stop_ptr: *mut LeafNode15<P> = current_ptr;
        let mut stop_bound: u64 = current_bound;

        let mut leaf_ptr: *mut LeafNode15<P> = current_leaf.prev(guard);

        if leaf_ptr.is_null() {
            self.stack.set_leaf(StdPtr::null_mut());
            return (ScanStateBack::Up, None);
        }

        let mut hops: usize = 0;

        loop {
            hops += 1;
            if hops > 128 {
                return (ScanStateBack::Retry, None);
            }

            // SAFETY: leaf_ptr is linked and protected by the guard.
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            if leaf.version().is_deleted() {
                leaf_ptr = leaf.prev(guard);

                if leaf_ptr.is_null() {
                    self.stack.set_leaf(StdPtr::null_mut());
                    return (ScanStateBack::Up, None);
                }

                continue;
            }

            // Move right to the newest split sibling that is still below the
            // current leaf. Stop at the remembered right boundary.
            let mut candidate: *mut LeafNode15<P> = leaf_ptr;
            let mut next_hops: usize = 0;

            loop {
                next_hops += 1;
                if next_hops > 64 {
                    break;
                }

                // SAFETY: candidate is protected by the guard.
                let candidate_leaf: &LeafNode15<P> = unsafe { &*candidate };
                let next: *mut LeafNode15<P> = candidate_leaf.safe_next(guard);

                if next.is_null() || next == stop_ptr {
                    break;
                }

                // SAFETY: next is protected by the guard.
                let next_bound: u64 = unsafe { &*next }.ikey_bound();

                if next_bound >= stop_bound {
                    break;
                }

                candidate = next;
            }

            // SAFETY: candidate is protected by the guard.
            let candidate_leaf: &LeafNode15<P> = unsafe { &*candidate };
            let perm: Permuter15 = candidate_leaf.permutation();

            if perm.size() == 0 {
                // Discard this empty leaf and continue left. Moving the stop
                // boundary prevents the next scan from walking back into it.
                stop_ptr = candidate;
                stop_bound = candidate_leaf.ikey_bound();
                leaf_ptr = candidate_leaf.prev(guard);

                if leaf_ptr.is_null() {
                    self.stack.set_leaf(StdPtr::null_mut());
                    return (ScanStateBack::Up, None);
                }

                continue;
            }

            prefetch_read(candidate.cast::<u8>());

            let prev_prev: *mut LeafNode15<P> = candidate_leaf.prev(guard);
            if !prev_prev.is_null() {
                prefetch_read(prev_prev.cast::<u8>());
            }

            let version: u32 = candidate_leaf.version().stable();

            if NodeVersion::is_deleted_version(version) {
                return (ScanStateBack::Retry, None);
            }

            let size: usize = perm.size();
            let ki: isize = (size - 1).cast_signed();

            self.stack.set_leaf(candidate);
            self.stack.set_last_ikey(u64::MAX);
            self.flags.clear_cursor_emitted();
            self.stack.update_state(version, perm, ki);

            return (ScanStateBack::FindPrev, None);
        }
    }
}

// ============================================================================
//  reposition_back - Version Conflict Recovery
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Reposition after version conflict during reverse scan.
    ///
    /// Traverses from the layer root to find the correct leaf for the
    /// current cursor key. O(height) per call.
    pub(crate) fn reposition_back(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.reposition_back_generic::<CloneEmitter>(guard)
    }

    /// Generic reposition after version conflict.
    fn reposition_back_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<E::Snapshot>) {
        const MAX_REPOSITION_RETRIES: u32 = 16;

        let mut root: *const u8 = self.stack.get_root();

        // Handle null root
        if root.is_null() {
            return (ScanStateBack::Up, None);
        }

        // Iterative retry loop (avoids recursion)
        for _retry in 0..MAX_REPOSITION_RETRIES {
            // Traverse from root to leaf
            let (mut leaf_ptr, resolved_root) =
                reach_leaf_for_scan::<P>(root, &self.cursor_key, guard);
            root = resolved_root;
            self.stack.set_root(resolved_root);

            // Check for empty tree / layer
            if leaf_ptr.is_null() {
                self.stack.set_leaf(StdPtr::null_mut());
                return (ScanStateBack::Up, None);
            }

            // Handle concurrent inserts (may follow forward chain)
            let version: u32 =
                ReverseScanHelper::stable_reverse(&mut leaf_ptr, &self.cursor_key, guard);

            // Check for deleted version
            if NodeVersion::is_deleted_version(version) {
                continue;
            }

            // SAFETY: leaf_ptr is valid (null checked, not deleted)
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
            let perm: Permuter15 = leaf.permutation();

            // A retry must not re-emit the key the cursor already holds. The
            // unshift sentinel is a positional marker, not a real key, so it
            // keeps the inclusive search.
            let skip_equal: bool = (self.flags.cursor_emitted()
                || self.flags.needs_duplicate_check())
                && !self.cursor_key.is_unshift_sentinel();

            let ki: isize = if skip_equal {
                ReverseScanHelper::lower_reverse_strict(
                    &self.cursor_key,
                    leaf,
                    perm,
                    self.flags.upper_bound(),
                )
            } else {
                ReverseScanHelper::lower_reverse(
                    &self.cursor_key,
                    leaf,
                    perm,
                    self.flags.upper_bound(),
                )
            };

            // Update stack with new position
            self.stack.set_leaf(leaf_ptr);
            self.stack.set_last_ikey(u64::MAX);
            self.stack.update_state(version, perm, ki);

            return (ScanStateBack::FindPrev, None);
        }

        // Internal retries exhausted under heavy contention.
        // Fall back to outer state machine retry (re-traverse from root).
        (ScanStateBack::Retry, None)
    }
}

// ============================================================================
//  handle_down_back / handle_up_back - Layer Navigation
// ============================================================================

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Handle descent into sublayer for reverse scan.
    ///
    /// Shifts cursor with `shift_clear_reverse()` and sets `upper_bound = true`.
    #[inline(always)]
    pub(crate) fn handle_down_back(&mut self) {
        let follow_suffix: bool = self.flags.take_down_follow_suffix();
        let fresh_subtree: bool = self.flags.take_down_fresh_subtree();

        if follow_suffix {
            // The layer shares the cursor's prefix: carry the real suffix into
            // the sublayer and search downward from the bound.
            self.cursor_key.shift();
            self.flags.clear_upper_bound();
        } else {
            self.cursor_key.shift_clear_reverse();
            self.flags.set_upper_bound();

            if fresh_subtree {
                self.flags.clear_cursor_emitted();
            }
        }

        self.stack.set_last_ikey(u64::MAX);
    }

    /// Handle ascent to parent layer for reverse scan.
    ///
    /// Returns `true` if successfully restored to parent layer,
    /// `false` if layer stack empty (scan complete).
    pub(crate) fn handle_up_back(&mut self, guard: &LocalGuard<'_>) -> bool {
        // Step 1: Pop parent context from layer stack
        let Some(parent_ctx) = self.layer_stack.pop() else {
            return false;
        };

        // Step 2: Restore stack with parent context
        self.stack.set_root(parent_ctx.root);
        self.stack.set_leaf(parent_ctx.leaf.as_ptr());

        // Step 3: Unshift cursor to parent layer
        self.cursor_key.unshift();

        // Step 4: Handle edge case - cursor became empty at root layer
        if self.cursor_key.is_at_empty_root() {
            return self.handle_up_back(guard);
        }

        // Step 5: Get stable version from parent leaf
        let mut leaf_ptr: *mut LeafNode15<P> = parent_ctx.leaf.as_ptr();
        let version: u32 =
            ReverseScanHelper::stable_reverse(&mut leaf_ptr, &self.cursor_key, guard);

        // Update leaf if stable_reverse followed forward chain
        if leaf_ptr != parent_ctx.leaf.as_ptr() {
            self.stack.set_leaf(leaf_ptr);
        }

        // Step 6: Find position in parent leaf
        // SAFETY: leaf_ptr is valid (from NonNull in LayerContext)
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
        let perm: Permuter15 = leaf.permutation();

        // CRITICAL: Clear upper_bound before position search.
        self.flags.clear_upper_bound();

        // First try to locate the layer-pointer slot we just returned from.
        //
        // Matching by the stored pointer alone is wrong after layer-root
        // promotion: the parent slot keeps pointing at the original left leaf,
        // while the sublayer's true root is an internode reached through parent
        // pointers. The parent layer's ikey is unique and stable, so anchor on
        // it instead.
        let layer_ikey: u64 = self.cursor_key.current_ikey();
        let mut anchored_ki: Option<isize> = None;
        for pos in 0..perm.size() {
            let slot: usize = perm.get(pos);
            if leaf.keylenx_relaxed(slot) >= LAYER_KEYLENX && leaf.ikey_relaxed(slot) == layer_ikey
            {
                anchored_ki = Some(pos.cast_signed() - 1);
                break;
            }
        }

        let ki: isize = anchored_ki.unwrap_or_else(|| {
            ReverseScanHelper::lower_reverse(&self.cursor_key, leaf, perm, self.flags.upper_bound())
        });

        // Step 7: Reset monotonicity cache for new leaf context
        self.stack.set_last_ikey(u64::MAX);
        self.flags.mark_cursor_emitted();

        // Step 8: Update stack state
        self.stack.update_state(version, perm, ki);

        true
    }
}

// ============================================================================
//  Intra-Leaf Batch Processing for Reverse Scan
// ============================================================================

/// Build a [`BatchCtx`] from reverse scan fields (split-borrowed).
#[inline(always)]
fn build_reverse_batch_ctx<'a, P: LeafPolicy>(
    stack: &'a BackStackElement<P>,
    cursor_key: &'a mut CursorKey,
    layer_stack: &'a mut LayerStack<P>,
) -> BatchCtx<'a, P> {
    let leaf_ptr: *mut LeafNode15<P> = stack.get_leaf_ptr();
    // SAFETY: leaf_ptr is valid - protected by guard in caller
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
    let perm = *stack.get_perm_ref();
    let perm_size = perm.size();
    BatchCtx {
        leaf,
        perm,
        perm_size,
        cached_version: stack.get_version(),
        ki: stack.get_ki(),
        root: stack.get_root(),
        leaf_ptr,
        cursor_key,
        layer_stack,
        down_follow_suffix: false,
        down_fresh_subtree: false,
    }
}

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Process remaining entries in current leaf in reverse, returning `&P::Value` references.
    #[inline]
    pub(crate) fn process_prev_leaf_batch_ptr<F>(
        &mut self,
        start_bound: &RangeBound<'_>,
        visitor: &mut F,
        count: &mut usize,
    ) -> LeafBatchResultBack
    where
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        let cursor_bound = start_bound.cursor_bound(&self.cursor_key, false);
        let (result, ki, root, follow, fresh) = {
            let mut ctx =
                build_reverse_batch_ctx(&self.stack, &mut self.cursor_key, &mut self.layer_stack);
            let r = process_batch_keyed::<Backward, _, P>(
                &mut ctx,
                start_bound,
                cursor_bound,
                &mut RefSlotVisitor(visitor),
                count,
                &mut self.flags,
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki);
        self.stack.set_root(root);

        if matches!(result, LeafBatchResultBack::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        }

        result
    }

    /// Process remaining entries in current leaf in reverse, returning `P::Output` by value.
    #[inline]
    pub(crate) fn process_prev_leaf_batch<F>(
        &mut self,
        start_bound: &RangeBound<'_>,
        visitor: &mut F,
        count: &mut usize,
    ) -> LeafBatchResultBack
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        let cursor_bound = start_bound.cursor_bound(&self.cursor_key, false);
        let (result, ki, root, follow, fresh) = {
            let mut ctx =
                build_reverse_batch_ctx(&self.stack, &mut self.cursor_key, &mut self.layer_stack);
            let r = process_batch_keyed::<Backward, _, P>(
                &mut ctx,
                start_bound,
                cursor_bound,
                &mut CopySlotVisitor(visitor),
                count,
                &mut self.flags,
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki);
        self.stack.set_root(root);

        if matches!(result, LeafBatchResultBack::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        }

        result
    }

    /// Process previous leaf batch without key materialization (values only).
    #[inline]
    pub(crate) fn process_prev_leaf_batch_values<F>(
        &mut self,
        start_bound: &RangeBound<'_>,
        visitor: &mut F,
        count: &mut usize,
    ) -> LeafBatchResultBack
    where
        F: FnMut(P::Output) -> bool,
    {
        let cursor_bound = start_bound.cursor_bound(&self.cursor_key, false);
        let (result, ki, root, follow, fresh) = {
            let mut ctx =
                build_reverse_batch_ctx(&self.stack, &mut self.cursor_key, &mut self.layer_stack);
            let r = process_batch_values::<Backward, P>(
                &mut ctx,
                start_bound,
                cursor_bound,
                visitor,
                count,
                &mut self.flags,
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki);
        self.stack.set_root(root);

        if matches!(result, LeafBatchResultBack::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        }

        result
    }

    /// Advance to previous leaf in the B-link chain (batch processing variant).
    ///
    /// Returns `true` if successfully advanced, `false` if layer exhausted.
    #[inline]
    pub(crate) fn advance_prev_leaf(&mut self, guard: &LocalGuard<'_>) -> bool {
        // Reuse the guarded retreat loop: it uses the protected `prev` load,
        // follows split siblings that appeared below the current bound, and
        // skips empty/deleted intermediate leaves. The returned snapshot is
        // never populated by a retreat.
        let (state, _) = self.advance_to_prev_leaf_generic::<PtrEmitter>(guard);
        matches!(state, ScanStateBack::FindPrev)
    }
}

// ============================================================================
//  Batch strategy trait + run_batch_reverse unified loop
// ============================================================================

/// Result of a reverse strategy's `emit_initial` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverseBatchAction {
    /// Entry emitted (or skipped), continue scanning.
    Continue,

    /// Visitor returned false, stop scanning.
    Stopped,

    /// Start bound exceeded, mark exhausted and stop.
    Exhausted,
}

/// Strategy trait for reverse batch scanning.
///
/// Three implementations:
///
/// - [`RevIntraLeafRefStrategy`]: `&P::Value` refs via `process_prev_leaf_batch_ptr`
/// - [`RevIntraLeafCopyStrategy`]: `P::Output` by value via `process_prev_leaf_batch`
/// - [`RevValuesOnlyStrategy`]: `P::Output` values only via `process_prev_leaf_batch_values`
///
/// After monomorphization, all trait dispatch is fully inlined, zero overhead.
pub trait ReverseBatchStrategy<P: LeafPolicy> {
    /// Emit the initial entry from `initialize_back()`'s snapshot.
    fn emit_initial(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> ReverseBatchAction;

    /// Process the current leaf's entries in reverse (hot path).
    fn process_leaf(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> LeafBatchResultBack;
}

impl<P: LeafPolicy> ReverseScanCtx<P> {
    /// Unified reverse batch scan loop driven by a strategy.
    ///
    /// Handles the state machine skeleton (Down/Up/Retry/null/deleted checks)
    /// and delegates entry emission and leaf processing to the strategy.
    ///
    /// Callers must handle exhausted/initialized checks before calling this.
    #[inline]
    pub(crate) fn run_batch_reverse<S: ReverseBatchStrategy<P>>(
        &mut self,
        strategy: &mut S,
        start_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let mut count: usize = 0;

        // Handle initial Emit state from initialize_back() if present
        if self.state == ScanStateBack::Emit {
            match strategy.emit_initial(self, start_bound, &mut count) {
                ReverseBatchAction::Continue => {
                    self.state = ScanStateBack::FindPrev;
                }
                ReverseBatchAction::Stopped => return count,
                ReverseBatchAction::Exhausted => {
                    self.flags.mark_exhausted();
                    return count;
                }
            }
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.state {
                ScanStateBack::Down => {
                    self.handle_down_back();
                    self.state = ScanStateBack::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanStateBack::Up => {
                    if !self.handle_up_back(guard) {
                        self.flags.mark_exhausted();
                        return count;
                    }

                    self.state = ScanStateBack::FindPrev;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanStateBack::Retry => {
                    let (new_state, _) = self.reposition_back(guard);
                    self.state = new_state;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanStateBack::Emit | ScanStateBack::FindPrev => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.get_leaf_ptr().is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }

                self.state = ScanStateBack::Up;
                continue;
            }

            // Check leaf deletion
            // SAFETY: null check above ensures leaf_ptr is valid,
            // and the guard protects the node from deallocation.
            let leaf: &LeafNode15<P> = unsafe { &*self.stack.get_leaf_ptr() };
            if leaf.version().is_deleted() {
                self.state = ScanStateBack::Retry;
                continue;
            }

            // Hot path: process leaf batch in reverse
            let result = strategy.process_leaf(self, start_bound, &mut count);

            match result {
                LeafBatchResultBack::LeafExhausted => {
                    if !self.advance_prev_leaf(guard) {
                        if self.layer_stack.is_empty() {
                            self.flags.mark_exhausted();
                            return count;
                        }

                        self.state = ScanStateBack::Up;
                    }
                }

                LeafBatchResultBack::LayerEncountered => {
                    self.state = ScanStateBack::Down;
                }

                LeafBatchResultBack::VersionChanged => {
                    self.state = ScanStateBack::Retry;
                }

                LeafBatchResultBack::Stopped => return count,

                LeafBatchResultBack::StartBoundExceeded => {
                    self.flags.mark_exhausted();
                    return count;
                }
            }
        }
    }
}

// ============================================================================
//  Strategy implementations
// ============================================================================

/// Reverse intra-leaf batch strategy with `&P::Value` references.
///
/// Used by [`super::iterator::RangeIter::rev_for_each_ref`].
pub struct RevIntraLeafRefStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> RevIntraLeafRefStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ReverseBatchStrategy<P> for RevIntraLeafRefStrategy<'_, F>
where
    P: crate::policy::RefPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> ReverseBatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return ReverseBatchAction::Continue;
        };

        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

        if !start_bound.contains_reverse(key) {
            return ReverseBatchAction::Exhausted;
        }

        *count += 1;

        // SAFETY: Guard protects the output. output_as_ref_sound uses
        // atomic read for write-through types, avoiding aliasing violation.
        let mut scratch = std::mem::MaybeUninit::uninit();
        let value_ref: &P::Value = unsafe { P::output_as_ref_sound(&snapshot.value, &mut scratch) };
        let should_continue = (self.visitor)(key, value_ref);

        if should_continue {
            ReverseBatchAction::Continue
        } else {
            ReverseBatchAction::Stopped
        }
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> LeafBatchResultBack {
        ctx.process_prev_leaf_batch_ptr(start_bound, self.visitor, count)
    }
}

/// Reverse intra-leaf batch strategy with `P::Output` by value.
///
/// Used by [`super::iterator::RangeIter::rev_for_each_intra_leaf_batch`].
pub struct RevIntraLeafCopyStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> RevIntraLeafCopyStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ReverseBatchStrategy<P> for RevIntraLeafCopyStrategy<'_, F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> ReverseBatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return ReverseBatchAction::Continue;
        };

        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

        if !start_bound.contains_reverse(key) {
            return ReverseBatchAction::Exhausted;
        }

        *count += 1;

        if (self.visitor)(key, snapshot.value) {
            ReverseBatchAction::Continue
        } else {
            ReverseBatchAction::Stopped
        }
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> LeafBatchResultBack {
        ctx.process_prev_leaf_batch(start_bound, self.visitor, count)
    }
}

/// Reverse values-only batch strategy — no key materialization.
///
/// Used by [`super::iterator::RangeIter::rev_for_each_values_batch`].
pub struct RevValuesOnlyStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> RevValuesOnlyStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ReverseBatchStrategy<P> for RevValuesOnlyStrategy<'_, F>
where
    P: LeafPolicy,
    F: FnMut(P::Output) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> ReverseBatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return ReverseBatchAction::Continue;
        };

        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };
        if !start_bound.contains_reverse(key) {
            return ReverseBatchAction::Exhausted;
        }

        *count += 1;
        if (self.visitor)(snapshot.value) {
            ReverseBatchAction::Continue
        } else {
            ReverseBatchAction::Stopped
        }
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ReverseScanCtx<P>,
        start_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> LeafBatchResultBack {
        ctx.process_prev_leaf_batch_values(start_bound, self.visitor, count)
    }
}
