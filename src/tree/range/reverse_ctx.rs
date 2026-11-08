//! Filepath: `src/tree/range/reverse_ctx.rs`
//!
//! Reverse-specific positioning, guarded leaf retreat and layer navigation.

use crate::permuter::Permuter15;
use std::cmp::Ordering;
use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::leaf15::LAYER_KEYLENX;
use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::cursor_key::CursorKey;
use super::helper::ReverseScanHelper;
use super::helper::{
    CloneEmitter, ScanEmitter, SlotPayload, build_slot_key_parts, compare_cursor_to_parts,
    try_snapshot_slot,
};
use super::scan_state::{LayerContext, ScanSnapshot, ScanState};
use super::traversal::reach_leaf_for_scan;

#[cfg(test)]
mod unit_tests;

// ============================================================================
//  Initial-position result
// ============================================================================

/// Result of attempting to find initial position in a single layer.
enum InitialPositionResult<P: LeafPolicy> {
    /// Found a value to emit.
    Emit(ScanSnapshot<P>),

    /// Need to descend into sublayer at the given root.
    LayerDescent(*const u8),

    /// No match at current position, retreat to find previous.
    Find,

    /// Version conflict, retry from root.
    Retry,
}

pub type ReverseScanCtx<P> = super::scan_ctx::ScanCtx<P, super::direction::Backward>;

impl<P: LeafPolicy> ReverseScanCtx<P> {
    #[inline]
    pub fn new_with_bound(root: *const u8, cursor_key: CursorKey, emit_equal: bool) -> Self {
        Self::new(root, cursor_key, emit_equal)
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
    ///    - Otherwise: return `Find` to retreat
    pub fn find_initial_reverse(
        &mut self,
        root: *const u8,
        emit_equal: bool,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshot<P>>)
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
                return (ScanState::Up, None);
            }

            // Handle concurrent inserts - may follow forward chain
            let mut version: u32 =
                ReverseScanHelper::stable_reverse(&mut leaf_ptr, &self.cursor_key, guard);
            self.stack.set_leaf(leaf_ptr);

            // Fast path: check deleted version
            if NodeVersion::is_deleted_version(version) {
                return (ScanState::Retry, None);
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
                        return (ScanState::Up, None);
                    }

                    empty_hops += 1;
                    leaf_ptr = prev;
                    self.stack.set_leaf(leaf_ptr);

                    // SAFETY: prev is protected by the guard.
                    version = unsafe { &*leaf_ptr }.version().stable();

                    if NodeVersion::is_deleted_version(version) {
                        return (ScanState::Retry, None);
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
                        return (ScanState::Emit, Some(snapshot));
                    }

                    InitialPositionResult::LayerDescent(layer_ptr) => {
                        // Continue outer loop with new layer root
                        current_root = layer_ptr;
                        break;
                    }

                    InitialPositionResult::Find => {
                        // Version check before returning
                        if leaf.version().has_changed(version) {
                            return (ScanState::Retry, None);
                        }
                        return (ScanState::Find, None);
                    }

                    InitialPositionResult::Retry => {
                        return (ScanState::Retry, None);
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
            return InitialPositionResult::Find;
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
                    return InitialPositionResult::Find;
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
                    return InitialPositionResult::Find;
                }
                build_slot_key_parts(&mut self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                self.cursor_key.mark_key_complete();
                InitialPositionResult::Emit(snapshot)
            }
            SlotPayload::Empty => InitialPositionResult::Find,
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
    pub(super) fn advance_to_prev_leaf_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<E::Snapshot>) {
        let current_ptr: *mut LeafNode15<P> = self.stack.leaf_ptr();

        // SAFETY: current_ptr was validated before calling this function
        let current_leaf: &LeafNode15<P> = unsafe { &*current_ptr };

        // The cursor is the current leaf's lower bound. Everything below it is
        // in the previous leaf (or a split sibling between the two).
        let current_bound: u64 = current_leaf.ikey_bound();
        let previous = current_leaf.prev(guard);
        if current_leaf.version().is_deleted()
            || current_leaf.version().has_changed(self.stack.version())
        {
            return (ScanState::Retry, None);
        }
        self.cursor_key.assign_store_ikey(current_bound);
        self.cursor_key.assign_store_length(0);

        // Right boundary of the region we may still visit. It moves left each
        // time we discard an empty leaf, so the walk cannot bounce between an
        // empty leaf and its predecessor.
        let mut stop_ptr: *mut LeafNode15<P> = current_ptr;
        let mut stop_bound: u64 = current_bound;

        let mut leaf_ptr: *mut LeafNode15<P> = previous;

        if leaf_ptr.is_null() {
            self.stack.set_leaf(StdPtr::null_mut());
            return (ScanState::Up, None);
        }

        for _ in 0..128 {
            // SAFETY: leaf_ptr is linked and protected by the guard.
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            if leaf.version().is_deleted() {
                leaf_ptr = leaf.prev(guard);

                if leaf_ptr.is_null() {
                    self.stack.set_leaf(StdPtr::null_mut());
                    return (ScanState::Up, None);
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
            // The version must precede the permutation: pairing an old permutation
            // with a newer stable version could accept physical slots reused by a writer.
            let version = candidate_leaf.version().stable();
            let perm: Permuter15 = candidate_leaf.permutation();
            if NodeVersion::is_deleted_version(version)
                || candidate_leaf.version().has_changed(version)
            {
                return (ScanState::Retry, None);
            }

            if perm.size() == 0 {
                // Discard this empty leaf and continue left. Moving the stop
                // boundary prevents the next scan from walking back into it.
                stop_ptr = candidate;
                stop_bound = candidate_leaf.ikey_bound();
                leaf_ptr = candidate_leaf.prev(guard);

                if leaf_ptr.is_null() {
                    self.stack.set_leaf(StdPtr::null_mut());
                    return (ScanState::Up, None);
                }

                continue;
            }

            prefetch_read(candidate.cast::<u8>());

            let prev_prev: *mut LeafNode15<P> = candidate_leaf.prev(guard);
            if !prev_prev.is_null() {
                prefetch_read(prev_prev.cast::<u8>());
            }

            let size: usize = perm.size();
            let ki: isize = (size - 1).cast_signed();

            self.stack.set_leaf(candidate);
            self.stack.set_last_ikey(u64::MAX);
            self.flags.clear_cursor_emitted();
            self.stack.update_state(version, perm, ki);

            return (ScanState::Find, None);
        }
        (ScanState::Retry, None)
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
    ) -> (ScanState, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.reposition_back_generic::<CloneEmitter>(guard)
    }

    /// Generic reposition after version conflict.
    fn reposition_back_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<E::Snapshot>) {
        const MAX_REPOSITION_RETRIES: u32 = 16;

        let mut root: *const u8 = self.stack.root();

        // Handle null root
        if root.is_null() {
            return (ScanState::Up, None);
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
                return (ScanState::Up, None);
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

            return (ScanState::Find, None);
        }

        // Internal retries exhausted under heavy contention.
        // Fall back to outer state machine retry (re-traverse from root).
        (ScanState::Retry, None)
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
