//! Filepath: `src/tree/range/forward_ctx.rs`
//!
//! Forward-specific positioning, B-link advancement and layer navigation.

use crate::permuter::Permuter15;
use std::cmp::Ordering;

use seize::LocalGuard;

use crate::leaf15::LeafNode15;
use crate::link::Linker;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;
use crate::prefetch::prefetch_read;

use super::helper::{
    CloneEmitter, PtrEmitter, ScanEmitter, SlotPayload, build_slot_key_parts,
    compare_cursor_to_parts, lower_bound_full, try_snapshot_slot,
};
use super::iterator::RangeBound;
use super::scan_state::{LayerContext, ScanSnapshot, ScanSnapshotPtr, ScanState, StepResult};
use super::traversal::reach_leaf_for_scan;

pub type ForwardScanCtx<P> = super::scan_ctx::ScanCtx<P, super::direction::Forward>;

// ============================================================================
//  find_initial — Position at start bound
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Find the initial position for a range scan.
    pub fn find_initial(
        &mut self,
        root: *const u8,
        emit_equal: bool,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.stack.set_root(root);

        let (leaf_ptr, resolved_root) = reach_leaf_for_scan::<P>(root, &self.cursor_key, guard);
        self.stack.set_root(resolved_root);

        if leaf_ptr.is_null() {
            return (ScanState::Up, None);
        }

        self.stack.set_leaf(leaf_ptr);

        // SAFETY: leaf_ptr is valid (non-null checked above, guard protects it)
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        let version: u32 = leaf.version().stable();

        if NodeVersion::is_deleted_version(version) {
            return (ScanState::Retry, None);
        }

        let perm: Permuter15 = leaf.permutation();
        let size: usize = perm.size();

        let pos = lower_bound_full(&self.cursor_key, leaf, perm, false);
        if pos >= size {
            if leaf.version().has_changed(version) {
                return (ScanState::Retry, None);
            }
            self.stack.update_state(version, perm, pos.cast_signed());
            return (ScanState::Find, None);
        }

        let Ok(slot) = try_snapshot_slot::<P, CloneEmitter>(leaf, perm.get(pos), version) else {
            return (ScanState::Retry, None);
        };
        let cmp = compare_cursor_to_parts(&self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
        self.flags.clear_cursor_emitted();
        self.stack.update_state(version, perm, pos.cast_signed());
        if cmp == Ordering::Greater {
            self.stack.next();
            return (ScanState::Find, None);
        }
        match slot.payload {
            SlotPayload::Layer(root) => {
                self.prepare_down(slot.ikey);
                self.layer_stack
                    .push(LayerContext::new(resolved_root, leaf_ptr));
                self.cursor_key.assign_store_ikey(slot.ikey);
                prefetch_read(root);
                self.stack.set_root(root);
                self.stack.next();
                (ScanState::Down, None)
            }
            SlotPayload::Value(snapshot) if cmp == Ordering::Equal => {
                self.stack.next();
                if emit_equal {
                    build_slot_key_parts(
                        &mut self.cursor_key,
                        slot.ikey,
                        slot.keylenx,
                        slot.suffix,
                    );
                    self.flags.mark_cursor_emitted();
                    (ScanState::Emit, Some(snapshot))
                } else {
                    (ScanState::Find, None)
                }
            }
            SlotPayload::Empty => {
                self.stack.next();
                (ScanState::Find, None)
            }
            SlotPayload::Value(_) => (ScanState::Find, None),
        }
    }
}

// ============================================================================
//  Leaf advancement, retry, layer transitions
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Advance to next leaf, zero-copy variant.
    #[inline]
    pub fn advance_leaf(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>) {
        self.advance_leaf_generic::<PtrEmitter>(guard)
    }

    /// Generic advance to next leaf.
    #[inline]
    fn advance_leaf_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<E::Snapshot>) {
        // SAFETY: the scan guard protects the non-null leaf cached by traversal.
        let leaf: &LeafNode15<P> = unsafe { &*self.stack.leaf_ptr() };
        let version: u32 = self.stack.version();

        let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

        if Linker::is_marked(next_raw) {
            leaf.wait_for_split();
            return (ScanState::Retry, None);
        }

        if leaf.version().has_changed(version) {
            return (ScanState::Retry, None);
        }

        let next: *mut LeafNode15<P> = next_raw.map_addr(|addr| addr & !1);

        if next.is_null() {
            return (ScanState::Up, None);
        }

        self.stack.set_leaf(next);

        // SAFETY: next is non-null and protected by guard
        let next_leaf: &LeafNode15<P> = unsafe { &*next };

        next_leaf.prefetch();

        let next_next: *mut LeafNode15<P> = next_leaf.safe_next(guard);
        if !next_next.is_null() {
            // SAFETY: next_next is non-null and derived from a valid leaf's B-link
            let next_next_leaf: &LeafNode15<P> = unsafe { &*next_next };
            next_next_leaf.prefetch();
        }

        let next_version: u32 = next_leaf.version().stable();

        if NodeVersion::is_deleted_version(next_version) {
            return (ScanState::Retry, None);
        }

        let perm: Permuter15 = next_leaf.permutation();

        let skip_equal: bool = self.flags.cursor_emitted() || !self.flags.emit_equal();
        let pos: usize = lower_bound_full(&self.cursor_key, next_leaf, perm, skip_equal);

        self.stack
            .update_state(next_version, perm, pos.cast_signed());

        (ScanState::Find, None)
    }

    /// Reposition after a conflict or layer transition.
    pub fn find_retry(&mut self, guard: &LocalGuard<'_>) -> ScanState {
        let (leaf_ptr, resolved_root) =
            reach_leaf_for_scan::<P>(self.stack.root(), &self.cursor_key, guard);
        self.stack.set_root(resolved_root);

        if leaf_ptr.is_null() {
            return ScanState::Up;
        }

        self.stack.set_leaf(leaf_ptr);

        // SAFETY: leaf_ptr is non-null and protected by guard
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        let version: u32 = leaf.version().stable();

        if NodeVersion::is_deleted_version(version) {
            return ScanState::Retry;
        }

        let perm: Permuter15 = leaf.permutation();

        let skip_equal: bool = self.flags.cursor_emitted() || !self.flags.emit_equal();
        let pos: usize = lower_bound_full(&self.cursor_key, leaf, perm, skip_equal);

        self.stack.update_state(version, perm, pos.cast_signed());

        ScanState::Find
    }

    /// Record how a pending `Down` transition should initialize the cursor.
    ///
    /// Called by every layer-discovery path before it overwrites the cursor's
    /// ikey with the layer's.
    #[inline(always)]
    const fn prepare_down(&mut self, slot_ikey: u64) {
        let same_prefix: bool = slot_ikey == self.cursor_key.current_ikey();
        let real_suffix: bool =
            self.cursor_key.has_suffix() && !self.cursor_key.is_unshift_sentinel();

        self.flags
            .set_down_mode(same_prefix && real_suffix, !same_prefix);
    }

    /// Handle descent into a sublayer (Down state).
    pub fn handle_down(&mut self) {
        let follow_suffix: bool = self.flags.take_down_follow_suffix();
        let fresh_subtree: bool = self.flags.take_down_fresh_subtree();

        if follow_suffix {
            // The layer shares the cursor's prefix: carry the real key suffix
            // into the sublayer so the scan starts at the right continuation.
            self.cursor_key.shift();
        } else {
            self.cursor_key.shift_clear();

            if fresh_subtree {
                // A different prefix: every sublayer key is greater than the
                // last emitted key, so it must not be deduplicated away.
                self.flags.clear_cursor_emitted();
            }
        }

        self.stack.set_last_ikey(0);
    }

    /// Handle ascent from exhausted sublayer (Up state).
    ///
    /// Returns `true` if there's a parent layer, `false` if scan is complete.
    pub fn handle_up(&mut self, _guard: &LocalGuard<'_>) -> bool {
        let Some(parent) = self.layer_stack.pop() else {
            return false;
        };

        self.stack.set_root(parent.root);
        self.stack.set_leaf(parent.leaf_ptr());

        self.cursor_key.unshift();

        // SAFETY: parent.leaf is protected by guard
        let leaf: &LeafNode15<P> = unsafe { parent.leaf.as_ref() };

        let version: u32 = leaf.version().stable();
        let perm: Permuter15 = leaf.permutation();

        // The unshift sentinel sorts after the layer pointer; skip it and any
        // earlier terminal siblings with the same ikey.
        let pos: usize = lower_bound_full(&self.cursor_key, leaf, perm, true);
        self.stack.update_state(version, perm, pos.cast_signed());
        self.flags.mark_cursor_emitted();

        self.stack.set_last_ikey(0);

        true
    }
}

// ============================================================================
//  Per-entry advance methods (fast paths for for_each / Iterator::next)
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Advance without cloning values (zero-copy path).
    ///
    /// Returns `(&[u8], &P::Value)` where both are borrowed references.
    #[inline(always)]
    pub fn advance_no_alloc_ref<'s>(
        &'s mut self,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> Option<(&'s [u8], &'s P::Value)>
    where
        P: RefLeafPolicy,
    {
        // Handle pending emit from initialize() - first entry case
        if self.state == ScanState::Emit && self.snapshot.is_some() {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key = unsafe { self.cursor_key.full_key_unchecked() };

            if !end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            let snapshot: ScanSnapshot<P> = self.snapshot.take()?;

            self.state = ScanState::Find;
            self.storage.last_output = Some(snapshot.value);

            // SAFETY: Guard protects the output from retirement.
            // output_as_ref_sound uses atomic read for write-through types.
            let value_ref: &P::Value = unsafe {
                P::output_as_ref_sound(
                    self.storage.last_output.as_ref().unwrap(),
                    &mut self.storage.scratch_value,
                )
            };

            return Some((key, value_ref));
        }

        loop {
            match self.step_transitions(guard) {
                StepResult::Exhausted => return None,
                StepResult::Continue => continue,
                StepResult::Ready => {}
            }

            let check_duplicate = self.flags.needs_duplicate_check();
            self.flags.clear_duplicate_check();
            let (new_state, snapshot_ptr) = self.find_one::<PtrEmitter>(guard, check_duplicate);

            self.state = new_state;

            if new_state == ScanState::Emit
                && let Some(snap) = snapshot_ptr
            {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key = unsafe { self.cursor_key.full_key_unchecked() };

                if !end_bound.contains(key) {
                    self.flags.mark_exhausted();

                    return None;
                }

                self.state = ScanState::Find;
                // SAFETY: Version validated, guard held, snap pointer valid.
                let value_ref: &P::Value =
                    unsafe { snap.resolve_value_ref::<P>(&mut self.storage.scratch_value) };

                return Some((key, value_ref));
            }
        }
    }
}
