//! Shared scan state, validated per-entry reader and callback batch driver.

#[cfg(debug_assertions)]
use super::cursor_key::CursorDebugState;
use super::{
    batch_common::{BatchCtx, ScanVisitor, process_batch_keyed},
    cursor_key::CursorKey,
    direction::Direction,
    helper::{
        CloneEmitter, ScanEmitter, SlotPayload, build_slot_key_parts, compare_cursor_to_parts,
        try_snapshot_slot,
    },
    iterator::{RangeBound, iter_flags::ScanFlags},
    scan_state::{
        LayerContext, LayerStack, LeafBatchResult, ScanPosition, ScanSnapshot, ScanState,
        StepResult,
    },
};
use crate::{
    leaf15::{LAYER_KEYLENX, LeafNode15},
    policy::LeafPolicy,
    prefetch::prefetch_read,
};
use seize::LocalGuard;
use std::{cmp::Ordering, mem::MaybeUninit};

/// Extra storage needed only by forward borrowed-value iteration and diagnostics.
pub struct ForwardStorage<P: LeafPolicy> {
    pub last_output: Option<P::Output>,
    pub scratch_value: MaybeUninit<P::Value>,
}
impl<P: LeafPolicy> Default for ForwardStorage<P> {
    fn default() -> Self {
        Self {
            last_output: None,
            scratch_value: MaybeUninit::uninit(),
        }
    }
}

pub struct ScanCtx<P: LeafPolicy, D: Direction> {
    pub stack: ScanPosition<P, D>,
    pub layer_stack: LayerStack<P>,
    pub cursor_key: CursorKey,
    pub state: ScanState,
    pub snapshot: Option<ScanSnapshot<P>>,
    pub flags: ScanFlags,
    pub storage: D::Storage<P>,
    #[cfg(debug_assertions)]
    pub debug_last_emitted_key: Option<Vec<u8>>,
    #[cfg(debug_assertions)]
    pub debug_last_cursor_state: Option<CursorDebugState>,
}

impl<P: LeafPolicy, D: Direction> ScanCtx<P, D> {
    pub fn new(root: *const u8, cursor_key: CursorKey, emit_equal: bool) -> Self {
        Self {
            stack: ScanPosition::new(root),
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanState::Find,
            snapshot: None,
            flags: ScanFlags::with_values(emit_equal),
            storage: D::Storage::<P>::default(),
            #[cfg(debug_assertions)]
            debug_last_emitted_key: None,
            #[cfg(debug_assertions)]
            debug_last_cursor_state: None,
        }
    }

    /// Both Iterator directions and copying callbacks advance through this driver.
    #[inline]
    pub fn advance_no_alloc(
        &mut self,
        bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> Option<(&[u8], P::Output)> {
        let snapshot = if self.state == ScanState::Emit && self.snapshot.is_some() {
            self.snapshot.take()?
        } else {
            loop {
                match self.step_transitions(guard) {
                    StepResult::Exhausted => return None,
                    StepResult::Continue => continue,
                    StepResult::Ready => {}
                }
                let check_duplicate = self.flags.needs_duplicate_check();
                self.flags.clear_duplicate_check();
                let (state, snapshot) = self.find_one::<CloneEmitter>(guard, check_duplicate);
                self.state = state;
                if let Some(snapshot) = snapshot {
                    break snapshot;
                }
            }
        };
        let key = self.cursor_key.full_key();
        if !D::key_in_bound(bound, key) {
            self.flags.mark_exhausted();
            return None;
        }
        #[cfg(debug_assertions)]
        {
            if let Some(previous) = &self.debug_last_emitted_key {
                assert_eq!(
                    key.cmp(previous),
                    D::exceeded_ordering(),
                    "scan ordering violation: previous {:?}, current {:?}; cursors {:?} -> {:?}",
                    previous,
                    key,
                    self.debug_last_cursor_state,
                    self.cursor_key.debug_state()
                );
            }
            self.debug_last_emitted_key = Some(key.to_vec());
            self.debug_last_cursor_state = Some(self.cursor_key.debug_state());
        }
        self.state = ScanState::Find;
        Some((key, snapshot.value))
    }

    /// Capture a complete slot, validate, then commit the direction-specific cursor.
    #[inline]
    pub fn find_one<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
        check_duplicate: bool,
    ) -> (ScanState, Option<E::Snapshot>) {
        if self.stack.is_null() {
            return (ScanState::Up, None);
        }
        // SAFETY: The active scan guard protects the non-null leaf cached by traversal.
        let leaf: &LeafNode15<P> = unsafe { &*self.stack.leaf_ptr() };
        if leaf.version().has_changed(self.stack.version()) || leaf.version().is_deleted() {
            return (ScanState::Retry, None);
        }
        let Some(physical_slot) = self.stack.kp() else {
            return (D::advance_leaf(self, guard), None);
        };
        let Ok(slot) = try_snapshot_slot::<P, E>(leaf, physical_slot, self.stack.version()) else {
            return (ScanState::Retry, None);
        };
        if slot.ikey.cmp(&self.stack.last_ikey()) == D::exceeded_ordering().reverse() {
            return (ScanState::Retry, None);
        }
        D::prefetch_neighbor(self.stack.ki(), leaf, self.stack.perm());
        if check_duplicate && !(D::REVERSE && self.flags.upper_bound()) {
            let duplicate = if slot.keylenx >= LAYER_KEYLENX {
                D::REVERSE
                    && if self.cursor_key.is_unshift_sentinel() {
                        slot.ikey == self.cursor_key.current_ikey()
                    } else {
                        slot.ikey >= self.cursor_key.current_ikey()
                    }
            } else if D::REVERSE && self.cursor_key.is_unshift_sentinel() {
                false
            } else {
                let cmp =
                    compare_cursor_to_parts(&self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                cmp != if D::REVERSE {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            };
            if duplicate {
                self.stack.next();
                return (ScanState::Find, None);
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
                    .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));
                self.cursor_key.assign_store_ikey(slot.ikey);
                prefetch_read(root);
                self.stack.set_root(root);
                self.stack.set_ki(D::layer_ki(self.stack.ki()));
                (ScanState::Down, None)
            }
            SlotPayload::Empty => {
                self.stack.next();
                (ScanState::Find, None)
            }
            SlotPayload::Value(snapshot) => {
                build_slot_key_parts(&mut self.cursor_key, slot.ikey, slot.keylenx, slot.suffix);
                self.cursor_key.mark_key_complete();
                self.stack.set_last_ikey(slot.ikey);
                self.stack.next();
                D::on_pre_emit(&mut self.flags);
                (ScanState::Emit, Some(snapshot))
            }
        }
    }

    #[inline]
    pub fn step_transitions(&mut self, guard: &LocalGuard<'_>) -> StepResult {
        match self.state {
            ScanState::Down => {
                D::descend(self);
                self.state = ScanState::Retry;
            }
            ScanState::Up => {
                if !D::ascend(self, guard) {
                    self.flags.mark_exhausted();
                    return StepResult::Exhausted;
                }
                self.state = ScanState::Find;
            }
            ScanState::Retry => {
                self.state = D::retry(self, guard);
            }
            ScanState::Emit | ScanState::Find => return StepResult::Ready,
        }
        self.flags.require_duplicate_check();
        StepResult::Continue
    }

    /// One driver for keyed, reference and values-only callbacks in both directions.
    #[inline]
    pub fn run_batch<V: ScanVisitor<P>>(
        &mut self,
        visitor: &mut V,
        bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let mut count = 0;
        loop {
            match self.step_transitions(guard) {
                StepResult::Exhausted => return count,
                StepResult::Continue => continue,
                StepResult::Ready => {}
            }
            if self.state == ScanState::Emit {
                self.state = ScanState::Find;
                if let Some(snapshot) = self.snapshot.take() {
                    let key = self.cursor_key.full_key();
                    if !D::key_in_bound(bound, key) {
                        self.flags.mark_exhausted();
                        return count;
                    }
                    D::on_pre_emit(&mut self.flags);
                    count += 1;
                    if !visitor.visit_initial(snapshot, key) {
                        return count;
                    }
                }
            }
            if self.stack.is_null() {
                self.state = ScanState::Up;
                continue;
            }
            if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                let (state, snapshot) = self.find_one::<V::Emitter>(guard, true);
                self.state = state;
                if let Some(snapshot) = snapshot {
                    let key = self.cursor_key.full_key();
                    if !D::key_in_bound(bound, key) {
                        self.flags.mark_exhausted();
                        return count;
                    }
                    self.state = ScanState::Find;
                    count += 1;
                    if !visitor.visit(snapshot, key) {
                        return count;
                    }
                }
                continue;
            }
            let result = self.process_leaf(visitor, bound, &mut count);
            self.state = match result {
                LeafBatchResult::LeafExhausted => D::advance_leaf(self, guard),
                LeafBatchResult::LayerEncountered => ScanState::Down,
                LeafBatchResult::VersionChanged => ScanState::Retry,
                LeafBatchResult::Stopped => return count,
                LeafBatchResult::BoundExceeded => {
                    self.flags.mark_exhausted();
                    return count;
                }
            };
        }
    }
    /// Split field borrows once while the direction-generic loop visits this leaf.
    #[inline]
    fn process_leaf<V: ScanVisitor<P>>(
        &mut self,
        visitor: &mut V,
        bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> LeafBatchResult {
        let leaf_ptr = self.stack.leaf_ptr();
        // SAFETY: The non-null leaf and captured suffix/value storage remain
        // protected by the scan guard for every callback in this batch.
        let leaf = unsafe { &*leaf_ptr };
        let cursor_bound = bound.cursor_bound(&self.cursor_key, !D::REVERSE);
        let perm = self.stack.perm();
        let mut batch = BatchCtx {
            leaf,
            perm,
            perm_size: perm.size(),
            cached_version: self.stack.version(),
            ki: self.stack.ki(),
            root: self.stack.root(),
            leaf_ptr,
            cursor_key: &mut self.cursor_key,
            layer_stack: &mut self.layer_stack,
            down_follow_suffix: false,
            down_fresh_subtree: false,
        };
        let result = process_batch_keyed::<D, V, P>(
            &mut batch,
            bound,
            cursor_bound,
            visitor,
            count,
            &mut self.flags,
        );
        self.stack.set_ki(batch.ki);
        self.stack.set_root(batch.root);
        if result == LeafBatchResult::LayerEncountered {
            self.flags
                .set_down_mode(batch.down_follow_suffix, batch.down_fresh_subtree);
        }
        result
    }
}
