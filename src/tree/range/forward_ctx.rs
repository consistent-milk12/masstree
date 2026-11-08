//! Filepath: `src/tree/range/forward_ctx.rs`
//!
//! Forward scan context, owns all forward iteration state.

use crate::permuter::Permuter15;
use std::cmp::Ordering;
use std::ptr as StdPtr;

use seize::LocalGuard;

use std::mem::MaybeUninit;

use crate::key::IKEY_SIZE;
use crate::leaf15::LeafNode15;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX};
use crate::link::Linker;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;
use crate::prefetch::prefetch_read;

use super::batch_common::{
    BatchCtx, CloneEmitter, CopySlotVisitor, Forward, PtrEmitter, RefSlotVisitor, ScanEmitter,
    process_batch_keyed, process_batch_values,
};
#[cfg(debug_assertions)]
use super::cursor_key::CursorDebugState;
use super::cursor_key::CursorKey;
use super::find::LeafBatchResult;
use super::helper::{compare_cursor_to_slot, lower_bound_full};
use super::iterator::RangeBound;
use super::iterator::iter_flags::ForwardFlags;
use super::scan_state::{
    LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr, ScanStackElement, ScanState,
    StepResult,
};
use super::traversal::reach_leaf_for_scan;

// ============================================================================
//  ForwardScanCtx
// ============================================================================

/// Forward scan context - owns all per-direction state for forward iteration.
///
/// This struct bundles the scan position, layer stack, cursor key, state machine
/// state, and flags that were previously individual fields on `RangeIter`.
///
/// # Design
///
/// Methods on this struct replace the free functions from `find.rs`, giving
/// a natural `self.find_next(guard)` interface instead of
/// `find_next(&mut stack, &mut cursor_key, &mut layer_stack, guard)`.
pub struct ForwardScanCtx<P: LeafPolicy> {
    /// Current scan position (leaf, version, permutation, ki).
    pub(crate) stack: ScanStackElement<P>,

    /// Parent layer stack for sublayer navigation.
    pub(crate) layer_stack: LayerStack<P>,

    /// Cursor tracking current key position.
    pub(crate) cursor_key: CursorKey,

    /// Current state machine state.
    pub(crate) state: ScanState,

    /// Captured snapshot for current entry (if in Emit state).
    pub(crate) snapshot: Option<ScanSnapshot<P>>,

    /// Tracks the output from `initialize()`'s snapshot.
    ///
    /// Only used for the first entry case in `advance_no_alloc_ref`,
    /// where we keep the `P::Output` alive so that `output_as_ref` can
    /// return a borrowed `&P::Value` from it.
    pub(crate) last_output: Option<P::Output>,

    /// Scratch storage for atomic value copies (write-through path).
    ///
    /// Used by `advance_no_alloc_ref` to hold an atomically-read value copy
    /// so a `&P::Value` can be returned without aliasing the mutable Box data.
    /// Only written when `P::can_write_through()` is true (V <= 8 bytes).
    pub(crate) scratch_value: MaybeUninit<P::Value>,

    /// Packed forward-specific boolean flags.
    pub(crate) flags: ForwardFlags,

    // ========================================================================
    //  Debug-only fields for ordering violation detection
    // ========================================================================
    /// Last emitted key for forward iteration (debug builds only).
    #[cfg(debug_assertions)]
    pub(crate) debug_last_emitted_key: Option<Vec<u8>>,

    /// Cursor state at last emission (debug builds only).
    #[cfg(debug_assertions)]
    pub(crate) debug_last_cursor_state: Option<CursorDebugState>,

    /// Ring buffer of recent state transitions for debugging.
    #[cfg(debug_assertions)]
    pub(crate) debug_transition_history: [Option<String>; 32],

    /// Write index into `debug_transition_history`.
    #[cfg(debug_assertions)]
    pub(crate) debug_transition_idx: usize,

    /// Number of entries written (saturates at 32).
    #[cfg(debug_assertions)]
    pub(crate) debug_transition_count: usize,
}

// ============================================================================
//  Construction & Debug Helpers
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Create a new forward scan context.
    #[inline]
    pub fn new(
        root: *const u8,
        cursor_key: CursorKey,
        emit_equal: bool,
        single_layer_mode: bool,
    ) -> Self {
        Self {
            stack: ScanStackElement::new(root),
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanState::FindNext,
            snapshot: None,
            last_output: None,
            scratch_value: MaybeUninit::uninit(),
            flags: ForwardFlags::with_values(emit_equal, single_layer_mode),

            #[cfg(debug_assertions)]
            debug_last_emitted_key: None,
            #[cfg(debug_assertions)]
            debug_last_cursor_state: None,
            #[cfg(debug_assertions)]
            debug_transition_history: [const { None }; 32],
            #[cfg(debug_assertions)]
            debug_transition_idx: 0,
            #[cfg(debug_assertions)]
            debug_transition_count: 0,
        }
    }

    /// Assert that keys are emitted in strictly increasing order (debug builds only).
    #[cfg(debug_assertions)]
    #[inline]
    #[allow(
        clippy::panic,
        reason = "Intentional panic for debug-only ordering violation detection"
    )]
    pub(crate) fn assert_ordering(&mut self, key: &[u8]) {
        if let Some(ref last_key) = self.debug_last_emitted_key
            && key <= last_key.as_slice()
        {
            let current_state: CursorDebugState = self.cursor_key.debug_state();
            let last_state: Option<&CursorDebugState> = self.debug_last_cursor_state.as_ref();

            eprintln!("\n=== ORDERING VIOLATION DETECTED (batch path) ===");
            eprintln!("Current key:  {:?}", String::from_utf8_lossy(key));
            eprintln!("Last key:     {:?}", String::from_utf8_lossy(last_key));
            eprintln!("Current key bytes: {key:?}");
            eprintln!("Last key bytes:    {last_key:?}");
            eprintln!("Current cursor: {current_state}");
            if let Some(last) = last_state {
                eprintln!("Last cursor:    {last}");
            }

            eprintln!("\n--- Recent state transitions ---");
            let count = self.debug_transition_count;
            let start = if count < 32 {
                0
            } else {
                self.debug_transition_idx
            };
            for i in 0..count {
                let idx = (start + i) % 32;
                if let Some(ref transition) = self.debug_transition_history[idx] {
                    eprintln!("[{i}] {transition}");
                }
            }
            eprintln!("--- End transitions ---");
            eprintln!("=== END ORDERING VIOLATION ===\n");

            panic!(
                "Scan ordering violation: emitted key {:?} is not > last emitted key {:?}",
                String::from_utf8_lossy(key),
                String::from_utf8_lossy(last_key)
            );
        }

        self.debug_last_emitted_key = Some(key.to_vec());
        self.debug_last_cursor_state = Some(self.cursor_key.debug_state());
        self.record_transition(format!(
            "EMIT: {:?} cursor={}",
            String::from_utf8_lossy(key),
            self.cursor_key.debug_state()
        ));
    }

    /// Record a state transition for debugging (debug builds only).
    #[cfg(debug_assertions)]
    #[inline]
    pub(crate) fn record_transition(&mut self, description: String) {
        self.debug_transition_history[self.debug_transition_idx] = Some(description);
        self.debug_transition_idx = (self.debug_transition_idx + 1) % 32;
        if self.debug_transition_count < 32 {
            self.debug_transition_count += 1;
        }
    }
}

// ============================================================================
//  find_initial — Position at start bound
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Find the initial position for a range scan.
    ///
    /// Corresponds to the free function `find::find_initial`.
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

        // Position calculations and initial emission write into a scratch
        // cursor. Strictly, `self.cursor_key` must still hold the start bound
        // if validation fails and the caller retries.
        let mut probe: CursorKey = self.cursor_key.clone();

        // Full-key lower bound: an equal-ikey group can contain several
        // distinct keys (zero-padded terminals and longer suffix variants).
        let pos: usize = lower_bound_full(&probe, leaf, perm, false);
        let mut final_pos: usize = pos;
        let mut emitted: bool = false;

        let (next_state, snapshot) = if pos >= size {
            (ScanState::FindNext, None)
        } else {
            let slot: usize = perm.get(pos);
            let keylenx: u8 = leaf.keylenx_relaxed(slot);
            let cmp: Ordering = compare_cursor_to_slot(&probe, leaf, slot);

            if cmp == Ordering::Greater {
                // Defensive: the lower-bound search cannot land after the cursor.
                final_pos = pos + 1;
                (ScanState::FindNext, None)
            } else if keylenx >= LAYER_KEYLENX {
                // The first key at or after the cursor lives in this layer.
                // Load the pointer first and validate before publishing it.
                let slot_ikey: u64 = leaf.ikey_relaxed(slot);
                let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);

                if leaf.version().has_changed(version) {
                    return (ScanState::Retry, None);
                }

                self.prepare_down(slot_ikey);
                probe.assign_store_ikey(slot_ikey);
                prefetch_read(layer_ptr);
                self.stack.set_root(layer_ptr);
                final_pos = pos + 1;
                (ScanState::Down, None)
            } else if cmp == Ordering::Equal {
                final_pos = pos + 1;

                if emit_equal {
                    let snapshot = Self::emit_initial_slot(leaf, slot, keylenx, &mut probe);
                    emitted = snapshot.is_some();

                    (
                        if emitted {
                            ScanState::Emit
                        } else {
                            ScanState::FindNext
                        },
                        snapshot,
                    )
                } else {
                    (ScanState::FindNext, None)
                }
            } else {
                // Cursor < slot: the scan is already positioned at this slot.
                (ScanState::FindNext, None)
            }
        };

        if leaf.version().has_changed(version) {
            return (ScanState::Retry, None);
        }

        if emitted {
            self.flags.mark_cursor_emitted();
        } else {
            self.flags.clear_cursor_emitted();
        }

        self.cursor_key = probe;
        self.stack.update_state(version, perm, final_pos);

        (next_state, snapshot)
    }

    /// Build the snapshot for an exact initial match (including suffix keys).
    #[inline]
    fn emit_initial_slot(
        leaf: &LeafNode15<P>,
        slot: usize,
        keylenx: u8,
        cursor_key: &mut CursorKey,
    ) -> Option<ScanSnapshot<P>>
    where
        P::Output: Clone,
    {
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if keylenx == KSUF_KEYLENX {
            let stored_suffix: &[u8] = leaf.ksuf(slot)?;
            let key_len: usize = IKEY_SIZE + stored_suffix.len();
            let output: P::Output = leaf.load_value(slot)?;

            cursor_key.assign_store_ikey(slot_ikey);
            let _ = cursor_key.assign_store_suffix(stored_suffix);
            cursor_key.assign_store_length(key_len);

            Some(ScanSnapshot::new(output, key_len))
        } else {
            let key_len: usize = keylenx as usize;
            let output: P::Output = leaf.load_value(slot)?;

            cursor_key.assign_store_ikey(slot_ikey);
            cursor_key.assign_store_length(key_len);

            Some(ScanSnapshot::new(output, key_len))
        }
    }
}

// ============================================================================
//  find_next variants - Main iteration
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Find the next entry in the scan sequence (clone path).
    ///
    /// Corresponds to `find::find_next`.
    #[inline]
    pub fn find_next(&mut self, guard: &LocalGuard<'_>) -> (ScanState, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.find_next_generic::<CloneEmitter>(guard, false)
    }

    /// Find next with duplicate checking (clone path).
    ///
    /// Corresponds to `find::find_next_with_duplicate_check`.
    #[inline]
    pub fn find_next_with_dup_check(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshot<P>>)
    where
        P::Output: Clone,
    {
        self.find_next_generic::<CloneEmitter>(guard, true)
    }

    /// Find next entry, returning raw pointer (zero-copy path).
    ///
    /// Corresponds to `find::find_next_ptr`.
    #[inline]
    pub fn find_next_ptr(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>) {
        self.find_next_generic::<PtrEmitter>(guard, false)
    }

    /// Find next with duplicate checking, returning raw pointer.
    ///
    /// Corresponds to `find::find_next_with_duplicate_check_ptr`.
    #[inline]
    pub fn find_next_with_dup_check_ptr(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>) {
        self.find_next_generic::<PtrEmitter>(guard, true)
    }

    /// Generic implementation of `find_next` with configurable duplicate checking
    /// and clone vs zero-copy value emission via [`ScanEmitter`].
    #[inline]
    fn find_next_generic<E: ScanEmitter<P>>(
        &mut self,
        guard: &LocalGuard<'_>,
        needs_duplicate_check: bool,
    ) -> (ScanState, Option<E::Snapshot>) {
        if self.stack.is_null() {
            return (ScanState::Up, None);
        }

        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };

        if leaf.version().is_deleted() {
            return (ScanState::Retry, None);
        }

        // A cached permutation can point at a slot that was reused for another
        // key, or at a slot whose value/suffix changed. Revalidate before
        // reading any slot data.
        if leaf.version().has_changed(self.stack.version()) {
            return (ScanState::Retry, None);
        }

        let Some(slot) = self.stack.kp() else {
            return self.advance_leaf_generic::<E>(guard);
        };

        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = leaf.keylenx_relaxed(slot);
        leaf.prefetch_value(slot);

        if slot_ikey < self.stack.last_ikey() {
            return (ScanState::Retry, None);
        }

        if needs_duplicate_check {
            let is_dup: bool = if slot_keylenx >= LAYER_KEYLENX {
                // A layer pointer is a routing entry, not a value; either the
                // scan descends into it or it is already behind the cursor.
                false
            } else {
                compare_cursor_to_slot(&self.cursor_key, leaf, slot) != Ordering::Less
            };

            if is_dup {
                self.stack.next();
                return (ScanState::FindNext, None);
            }
        }

        if slot_keylenx >= LAYER_KEYLENX {
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);

            // Validate before mutating the stack/cursor: the slot may have
            // been converted back to a value by a concurrent update.
            if leaf.version().has_changed(self.stack.version()) {
                return (ScanState::Retry, None);
            }

            self.prepare_down(slot_ikey);
            self.layer_stack
                .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));
            self.cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            self.stack.set_root(layer_ptr);

            return (ScanState::Down, None);
        }

        // Snapshot the suffix once; do not re-read it after validation.
        let suffix: Option<&[u8]> = if slot_keylenx == KSUF_KEYLENX {
            leaf.ksuf(slot)
        } else {
            None
        };

        let key_len: usize = if slot_keylenx == KSUF_KEYLENX {
            suffix.map_or(IKEY_SIZE, |s| IKEY_SIZE + s.len())
        } else {
            slot_keylenx as usize
        };

        let Some(snapshot) = E::emit_value(leaf, slot, key_len) else {
            if leaf.version().has_changed(self.stack.version()) {
                return (ScanState::Retry, None);
            }

            self.stack.next();
            return (ScanState::FindNext, None);
        };

        // Validate after the complete payload read and before publishing the
        // key/cursor. Otherwise a reuse or layer conversion in this gap could
        // emit a stale value or a layer pointer as a value.
        if leaf.version().has_changed(self.stack.version()) {
            return (ScanState::Retry, None);
        }

        self.cursor_key.assign_store_ikey(slot_ikey);

        if let Some(suffix) = suffix {
            let _ = self.cursor_key.assign_store_suffix(suffix);
        }

        self.cursor_key.assign_store_length(key_len);
        self.cursor_key.mark_key_complete();
        self.stack.set_last_ikey(slot_ikey);
        self.stack.next();
        self.flags.mark_cursor_emitted();

        (ScanState::Emit, Some(snapshot))
    }
}

// ============================================================================
//  Single-layer fast path
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Single-layer fast path for zero-copy `find_next`.
    ///
    /// Corresponds to `find::find_next_single_layer_ptr`.
    #[inline]
    #[expect(clippy::cast_possible_truncation)]
    pub fn find_next_single_layer_ptr(
        &mut self,
        guard: &LocalGuard<'_>,
        needs_duplicate_check: bool,
    ) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>) {
        if self.stack.is_null() {
            return (ScanState::FindNext, None);
        }

        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };

        if leaf.version().is_deleted() {
            return (ScanState::Retry, None);
        }

        // Check if leaf was concurrently modified since we cached the version.
        // Symmetric with reverse's find_prev_single_layer (reverse_ctx.rs).
        let version: u32 = self.stack.version();
        if leaf.version().has_changed(version) {
            return (ScanState::Retry, None);
        }

        let Some(slot) = self.stack.kp() else {
            return self.advance_leaf_single_layer(guard);
        };

        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = leaf.keylenx_relaxed(slot);
        leaf.prefetch_value(slot);

        if slot_ikey < self.stack.last_ikey() {
            return (ScanState::Retry, None);
        }

        if needs_duplicate_check {
            let is_dup: bool = if slot_keylenx >= LAYER_KEYLENX {
                false
            } else {
                compare_cursor_to_slot(&self.cursor_key, leaf, slot) != Ordering::Less
            };

            if is_dup {
                self.stack.next();
                return (ScanState::FindNext, None);
            }
        }

        // True layer pointer — prepare for descent
        if slot_keylenx >= LAYER_KEYLENX {
            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);

            if leaf.version().has_changed(version) {
                return (ScanState::Retry, None);
            }

            self.prepare_down(slot_ikey);
            self.cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            return (ScanState::Down, None);
        }

        // Suffix key (KSUF) or other non-inline — bail to multi-layer path.
        // Returning FindNext after disabling single-layer mode causes the caller
        // to re-enter the loop and fall through to the multi-layer path which
        // handles KSUF correctly via find_next_ptr.
        if slot_keylenx > IKEY_SIZE as u8 {
            self.flags.disable_single_layer_mode();
            return (ScanState::FindNext, None);
        }

        let slot_ptr: *mut u8 = leaf.load_value_raw(slot);
        if slot_ptr.is_null() {
            self.stack.next();
            return (ScanState::FindNext, None);
        }

        // Validate after the pointer/value read and before publishing it.
        if leaf.version().has_changed(version) {
            return (ScanState::Retry, None);
        }

        let key_len: usize = slot_keylenx as usize;
        self.cursor_key.assign_store_ikey(slot_ikey);
        self.cursor_key.assign_store_length(key_len);
        self.cursor_key.mark_key_complete();

        self.stack.set_last_ikey(slot_ikey);
        self.stack.next();
        self.flags.mark_cursor_emitted();

        (
            ScanState::Emit,
            Some(ScanSnapshotPtr::from_raw(slot_ptr, key_len)),
        )
    }

    /// Advance to next leaf in single-layer mode.
    #[inline(always)]
    fn advance_leaf_single_layer(
        &mut self,
        guard: &LocalGuard<'_>,
    ) -> (ScanState, Option<ScanSnapshotPtr<P::Value>>) {
        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
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
            self.stack.set_leaf(StdPtr::null_mut());
            return (ScanState::FindNext, None);
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

        // Before the first emission the cursor is still the *start* key, so an
        // exact match must be included when the bound is inclusive. After an
        // emission the cursor is the last emitted key and must be skipped.
        let skip_equal: bool = self.flags.cursor_emitted() || !self.flags.emit_equal();
        let pos: usize = lower_bound_full(&self.cursor_key, next_leaf, perm, skip_equal);

        self.stack.update_state(next_version, perm, pos);

        (ScanState::FindNext, None)
    }
}

// ============================================================================
//  Leaf advancement, retry, layer transitions
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Advance to next leaf, zero-copy variant.
    ///
    /// Corresponds to `find::advance_leaf_ptr`.
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
        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
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

        self.stack.update_state(next_version, perm, pos);

        (ScanState::FindNext, None)
    }

    /// Reposition after a conflict or layer transition.
    ///
    /// Corresponds to `find::find_retry`.
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

        self.stack.update_state(version, perm, pos);

        ScanState::FindNext
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
    ///
    /// Corresponds to `find::handle_down`.
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
    /// Corresponds to `find::handle_up`.
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
        self.stack.update_state(version, perm, pos);
        self.flags.mark_cursor_emitted();

        self.stack.set_last_ikey(0);

        true
    }

    /// Handle Down/Up/Retry transitions in a single method.
    ///
    /// Centralizes the state-machine transition logic shared across all five
    /// forward scan entry points (`run_batch`, `advance_no_alloc`,
    /// `advance_no_alloc_ref`, `RangeIter::advance`, `for_each_batch_ref`).
    #[inline(always)]
    pub(crate) fn step_transitions(&mut self, guard: &LocalGuard<'_>) -> StepResult {
        match self.state {
            ScanState::Down => {
                self.flags.disable_single_layer_mode();
                self.handle_down();
                self.state = ScanState::Retry;
                self.flags.require_duplicate_check();
                StepResult::Continue
            }
            ScanState::Up => {
                if !self.handle_up(guard) {
                    self.flags.mark_exhausted();
                    return StepResult::Exhausted;
                }
                self.state = ScanState::FindNext;
                self.flags.require_duplicate_check();
                StepResult::Continue
            }
            ScanState::Retry => {
                self.state = self.find_retry(guard);
                self.flags.require_duplicate_check();
                StepResult::Continue
            }
            ScanState::Emit | ScanState::FindNext => StepResult::Ready,
        }
    }
}

// ============================================================================
//  Intra-leaf batch processing
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Process remaining entries in current leaf, returning `&P::Value` references.
    #[inline]
    pub fn process_leaf_batch_ptr<F>(
        &mut self,
        end_bound: &RangeBound<'_>,
        visitor: &mut F,
        count: &mut usize,
    ) -> LeafBatchResult
    where
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        let cursor_bound = end_bound.cursor_bound(&self.cursor_key, true);
        let before: usize = *count;
        let (result, ki, root, follow, fresh) = {
            let mut ctx = self.build_batch_ctx();
            let r = process_batch_keyed::<Forward, _, P>(
                &mut ctx,
                end_bound,
                cursor_bound,
                &mut RefSlotVisitor(visitor),
                count,
                &mut (),
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki.cast_unsigned());
        self.stack.set_root(root);

        if matches!(result, LeafBatchResult::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        } else if *count > before {
            self.flags.mark_cursor_emitted();
        }

        result
    }

    /// Process remaining entries in current leaf, returning `P::Output` by value.
    #[inline]
    pub fn process_leaf_batch<F>(
        &mut self,
        end_bound: &RangeBound<'_>,
        visitor: &mut F,
        count: &mut usize,
    ) -> LeafBatchResult
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        let cursor_bound = end_bound.cursor_bound(&self.cursor_key, true);
        let before: usize = *count;
        let (result, ki, root, follow, fresh) = {
            let mut ctx = self.build_batch_ctx();
            let r = process_batch_keyed::<Forward, _, P>(
                &mut ctx,
                end_bound,
                cursor_bound,
                &mut CopySlotVisitor(visitor),
                count,
                &mut (),
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki.cast_unsigned());
        self.stack.set_root(root);

        if matches!(result, LeafBatchResult::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        } else if *count > before {
            self.flags.mark_cursor_emitted();
        }

        result
    }

    /// Process leaf batch without key materialization (values only).
    #[inline]
    pub fn process_leaf_batch_values(
        &mut self,
        end_bound: &RangeBound<'_>,
        visitor: &mut impl FnMut(P::Output) -> bool,
        count: &mut usize,
    ) -> LeafBatchResult {
        let cursor_bound = end_bound.cursor_bound(&self.cursor_key, true);
        let before: usize = *count;
        let (result, ki, root, follow, fresh) = {
            let mut ctx = self.build_batch_ctx();
            let r = process_batch_values::<Forward, P>(
                &mut ctx,
                end_bound,
                cursor_bound,
                visitor,
                count,
                &mut (),
            );
            (
                r,
                ctx.ki,
                ctx.root,
                ctx.down_follow_suffix,
                ctx.down_fresh_subtree,
            )
        };
        self.stack.set_ki(ki.cast_unsigned());
        self.stack.set_root(root);

        if matches!(result, LeafBatchResult::LayerEncountered) {
            self.flags.set_down_mode(follow, fresh);
        } else if *count > before {
            self.flags.mark_cursor_emitted();
        }

        result
    }

    /// Build a [`BatchCtx`] from the forward scan stack.
    #[inline(always)]
    fn build_batch_ctx(&mut self) -> BatchCtx<'_, P> {
        let leaf_ptr: *mut LeafNode15<P> = self.stack.leaf_ptr();
        // SAFETY: leaf_ptr is valid - protected by guard in caller
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
        let perm = self.stack.perm();
        let perm_size = perm.size();
        BatchCtx {
            leaf,
            perm,
            perm_size,
            cached_version: self.stack.version(),
            ki: self.stack.ki().cast_signed(),
            root: self.stack.root(),
            leaf_ptr,
            cursor_key: &mut self.cursor_key,
            layer_stack: &mut self.layer_stack,
            down_follow_suffix: false,
            down_fresh_subtree: false,
        }
    }
}

// ============================================================================
//  Batch strategy trait + run_batch unified loop
// ============================================================================

/// Result of a strategy's `emit_initial` or `dup_check_entry` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAction {
    /// Entry emitted (or skipped), continue scanning.
    Continue,
    /// Visitor returned false — stop scanning.
    Stopped,
    /// End bound exceeded — mark exhausted and stop.
    Exhausted,
}

/// Strategy trait for forward batch scanning.
///
/// Defines how to emit the initial entry, handle the dup-check slow path,
/// and process leaf batches. Three implementations:
///
/// - [`IntraLeafRefStrategy`]: `&P::Value` refs via `process_leaf_batch_ptr`
/// - [`IntraLeafCopyStrategy`]: `P::Output` by value via `process_leaf_batch`
/// - [`ValuesOnlyStrategy`]: `P::Output` values only via `process_leaf_batch_values`
///
/// After monomorphization, all trait dispatch is fully inlined — zero overhead.
pub trait ForwardBatchStrategy<P: LeafPolicy> {
    /// Emit the initial entry from `initialize()`'s snapshot.
    fn emit_initial(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> BatchAction;

    /// Handle one entry via the dup-check slow path after retry/layer transitions.
    fn dup_check_entry(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> BatchAction;

    /// Process the current leaf's entries (hot path).
    fn process_leaf(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> LeafBatchResult;
}

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Unified batch scan loop driven by a strategy.
    ///
    /// Handles the state machine skeleton (Down/Up/Retry/null/deleted checks)
    /// and delegates entry emission, duplicate checking, and leaf processing
    /// to the strategy.
    ///
    /// Callers must handle exhausted/initialized checks before calling this.
    #[inline]
    pub(crate) fn run_batch<S: ForwardBatchStrategy<P>>(
        &mut self,
        strategy: &mut S,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let mut count: usize = 0;

        // Handle initial Emit state from initialize() if present
        if self.state == ScanState::Emit {
            match strategy.emit_initial(self, end_bound, &mut count) {
                BatchAction::Continue => {
                    if count > 0 {
                        self.flags.mark_cursor_emitted();
                    }

                    self.state = ScanState::FindNext;
                }

                BatchAction::Stopped => return count,

                BatchAction::Exhausted => {
                    self.flags.mark_exhausted();
                    return count;
                }
            }
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.step_transitions(guard) {
                StepResult::Exhausted => return count,
                StepResult::Continue => continue,
                StepResult::Ready => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }
                self.state = ScanState::Up;
                continue;
            }

            // Check leaf deletion
            // SAFETY: stack.is_null() check above ensures leaf_ptr is valid,
            // and the guard protects the node from deallocation.
            let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // Dup-check slow path (after retry/layer transitions)
            if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();

                match strategy.dup_check_entry(self, end_bound, guard, &mut count) {
                    BatchAction::Continue => continue,

                    BatchAction::Stopped => return count,

                    BatchAction::Exhausted => {
                        self.flags.mark_exhausted();
                        return count;
                    }
                }
            }

            // Hot path: process leaf batch
            let result = strategy.process_leaf(self, end_bound, guard, &mut count);

            match result {
                LeafBatchResult::LeafExhausted => {
                    let (state, _) = self.advance_leaf(guard);
                    self.state = state;
                }

                LeafBatchResult::LayerEncountered => {
                    self.state = ScanState::Down;
                }

                LeafBatchResult::VersionChanged => {
                    self.state = ScanState::Retry;
                }

                LeafBatchResult::Stopped => return count,

                LeafBatchResult::EndBoundExceeded => {
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

/// Intra-leaf batch strategy with `&P::Value` references.
///
/// Used by [`RangeIter::for_each_intra_leaf_batch_ref`].
pub struct IntraLeafRefStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> IntraLeafRefStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ForwardBatchStrategy<P> for IntraLeafRefStrategy<'_, F>
where
    P: RefLeafPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return BatchAction::Continue;
        };
        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

        if !end_bound.contains(key) {
            return BatchAction::Exhausted;
        }

        let value_ref: &P::Value =
            unsafe { P::output_as_ref_sound(&snapshot.value, &mut ctx.scratch_value) };
        *count += 1;

        if (self.visitor)(key, value_ref) {
            BatchAction::Continue
        } else {
            BatchAction::Stopped
        }
    }

    #[inline(always)]
    fn dup_check_entry(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let (new_state, snapshot_ptr) = ctx.find_next_with_dup_check_ptr(guard);
        ctx.state = new_state;

        if new_state == ScanState::Emit
            && let Some(snap) = snapshot_ptr
        {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

            if !end_bound.contains(key) {
                return BatchAction::Exhausted;
            }

            let value_ref: &P::Value =
                unsafe { snap.resolve_value_ref::<P>(&mut ctx.scratch_value) };
            *count += 1;
            ctx.state = ScanState::FindNext;

            if !(self.visitor)(key, value_ref) {
                return BatchAction::Stopped;
            }
        }

        BatchAction::Continue
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        _guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> LeafBatchResult {
        ctx.process_leaf_batch_ptr(end_bound, self.visitor, count)
    }
}

/// Intra-leaf batch strategy with `P::Output` by value.
///
/// Used by [`RangeIter::for_each_intra_leaf_batch`].
pub struct IntraLeafCopyStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> IntraLeafCopyStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ForwardBatchStrategy<P> for IntraLeafCopyStrategy<'_, F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return BatchAction::Continue;
        };

        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

        if !end_bound.contains(key) {
            return BatchAction::Exhausted;
        }

        *count += 1;

        if (self.visitor)(key, snapshot.value) {
            BatchAction::Continue
        } else {
            BatchAction::Stopped
        }
    }

    #[inline(always)]
    fn dup_check_entry(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let (new_state, snapshot) = ctx.find_next_with_dup_check(guard);
        ctx.state = new_state;

        if new_state == ScanState::Emit
            && let Some(snap) = snapshot
        {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };

            if !end_bound.contains(key) {
                return BatchAction::Exhausted;
            }

            *count += 1;
            ctx.state = ScanState::FindNext;

            if !(self.visitor)(key, snap.value) {
                return BatchAction::Stopped;
            }
        }

        BatchAction::Continue
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        _guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> LeafBatchResult {
        ctx.process_leaf_batch(end_bound, self.visitor, count)
    }
}

/// Values-only batch strategy — no key materialization.
///
/// Used by [`RangeIter::for_each_values_batch`].
pub struct ValuesOnlyStrategy<'a, F> {
    visitor: &'a mut F,
}

impl<'a, F> ValuesOnlyStrategy<'a, F> {
    #[inline]
    pub const fn new(visitor: &'a mut F) -> Self {
        Self { visitor }
    }
}

impl<P, F> ForwardBatchStrategy<P> for ValuesOnlyStrategy<'_, F>
where
    P: LeafPolicy,
    F: FnMut(P::Output) -> bool,
{
    #[inline(always)]
    fn emit_initial(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let Some(snapshot) = ctx.snapshot.take() else {
            return BatchAction::Continue;
        };

        let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };
        if !end_bound.contains(key) {
            return BatchAction::Exhausted;
        }

        *count += 1;

        if (self.visitor)(snapshot.value) {
            BatchAction::Continue
        } else {
            BatchAction::Stopped
        }
    }

    #[inline(always)]
    fn dup_check_entry(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> BatchAction {
        let (new_state, snapshot) = ctx.find_next_with_dup_check(guard);
        ctx.state = new_state;

        if new_state == ScanState::Emit
            && let Some(snap) = snapshot
        {
            let key: &[u8] = unsafe { ctx.cursor_key.full_key_unchecked() };
            if !end_bound.contains(key) {
                return BatchAction::Exhausted;
            }

            *count += 1;

            ctx.state = ScanState::FindNext;
            if !(self.visitor)(snap.value) {
                return BatchAction::Stopped;
            }
        }

        BatchAction::Continue
    }

    #[inline(always)]
    fn process_leaf(
        &mut self,
        ctx: &mut ForwardScanCtx<P>,
        end_bound: &RangeBound<'_>,
        _guard: &LocalGuard<'_>,
        count: &mut usize,
    ) -> LeafBatchResult {
        ctx.process_leaf_batch_values(end_bound, self.visitor, count)
    }
}

// ============================================================================
//  Per-entry advance methods (fast paths for for_each / Iterator::next)
// ============================================================================

impl<P: LeafPolicy> ForwardScanCtx<P> {
    /// Advance without allocating key Vec (clone path).
    ///
    /// Returns `(&[u8], P::Output)` where the key slice is borrowed from
    /// the internal `cursor_key` buffer.
    ///
    /// This function inlines the common case `(FindNext → Emit)` to avoid
    /// state machine dispatch overhead.
    #[inline(always)]
    pub fn advance_no_alloc(
        &mut self,
        end_bound: &RangeBound<'_>,
        guard: &LocalGuard<'_>,
    ) -> Option<(&[u8], P::Output)>
    where
        P::Output: Clone,
    {
        // Fast path: if we have a pending emit, process it first
        if self.state == ScanState::Emit
            && let Some(snapshot) = self.snapshot.take()
        {
            #[cfg(debug_assertions)]
            {
                let key_copy = self.cursor_key.full_key().to_vec();
                self.assert_ordering(&key_copy);
            }

            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

            if !end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            self.state = ScanState::FindNext;

            return Some((key, snapshot.value));
        }

        loop {
            #[cfg(debug_assertions)]
            let (pre_state, pre_cursor) = (self.state, self.cursor_key.debug_state());

            match self.step_transitions(guard) {
                StepResult::Exhausted => {
                    #[cfg(debug_assertions)]
                    self.record_transition(format!("Up -> Exhausted: pre={pre_cursor}"));
                    return None;
                }
                StepResult::Continue => {
                    #[cfg(debug_assertions)]
                    self.record_transition(format!(
                        "{pre_state:?} -> {:?}: pre={pre_cursor}, post={}",
                        self.state,
                        self.cursor_key.debug_state()
                    ));
                    continue;
                }
                StepResult::Ready => {}
            }

            // Main hot path: FindNext
            let (new_state, snapshot) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                self.find_next_with_dup_check(guard)
            } else {
                self.find_next(guard)
            };

            self.state = new_state;

            // Fast path: if Emit, return immediately
            if new_state == ScanState::Emit
                && let Some(snap) = snapshot
            {
                #[cfg(debug_assertions)]
                {
                    let key_copy = self.cursor_key.full_key().to_vec();
                    self.assert_ordering(&key_copy);
                }

                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key = unsafe { self.cursor_key.full_key_unchecked() };

                if !end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return None;
                }

                self.state = ScanState::FindNext;
                return Some((key, snap.value));
            }

            self.snapshot = snapshot;
        }
    }

    /// Advance without cloning values (zero-copy path).
    ///
    /// Returns `(&[u8], &P::Value)` where both are borrowed references.
    #[inline(always)]
    #[expect(clippy::too_many_lines, reason = "Complex allocation logic")]
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

            self.state = ScanState::FindNext;
            self.last_output = Some(snapshot.value);

            // SAFETY: Guard protects the output from retirement.
            // output_as_ref_sound uses atomic read for write-through types.
            let value_ref: &P::Value = unsafe {
                P::output_as_ref_sound(self.last_output.as_ref().unwrap(), &mut self.scratch_value)
            };

            return Some((key, value_ref));
        }

        loop {
            // ================================================================
            // Single-layer fast path (keys ≤ 8 bytes)
            // ================================================================
            if self.flags.single_layer_mode() {
                if self.state == ScanState::Retry {
                    self.state = self.find_retry(guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                let (new_state, snapshot_ptr) =
                    self.find_next_single_layer_ptr(guard, self.flags.needs_duplicate_check());

                if self.flags.needs_duplicate_check() {
                    self.flags.clear_duplicate_check();
                }

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot_ptr {
                            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                            let key = unsafe { self.cursor_key.full_key_unchecked() };

                            if !end_bound.contains(key) {
                                self.flags.mark_exhausted();
                                return None;
                            }

                            self.state = ScanState::FindNext;
                            // SAFETY: Version validated, guard held, snap pointer valid.
                            let value_ref: &P::Value =
                                unsafe { snap.resolve_value_ref::<P>(&mut self.scratch_value) };

                            return Some((key, value_ref));
                        }
                    }

                    ScanState::FindNext => {
                        if self.stack.is_null() {
                            self.flags.mark_exhausted();
                            return None;
                        }

                        continue;
                    }

                    ScanState::Retry => continue,

                    ScanState::Down => {
                        // Encountered layer pointer - fall back to multi-layer
                        self.flags.disable_single_layer_mode();

                        self.layer_stack
                            .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));

                        let Some(slot) = self.stack.kp() else {
                            debug_assert!(
                                false,
                                "Down state entered without valid slot - state machine bug"
                            );

                            self.state = ScanState::Retry;
                            continue;
                        };

                        // SAFETY: find_next_single_layer_ptr validated the leaf version,
                        // and the guard protects the node from deallocation.
                        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
                        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
                        self.stack.set_root(layer_ptr);

                        // Fall through to handle Down below
                    }

                    ScanState::Up => {
                        self.flags.mark_exhausted();

                        return None;
                    }
                }
            }

            // ================================================================
            // Multi-layer path (handles Down/Up transitions)
            // ================================================================

            match self.step_transitions(guard) {
                StepResult::Exhausted => return None,
                StepResult::Continue => continue,
                StepResult::Ready => {}
            }

            let (new_state, snapshot_ptr) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                self.find_next_with_dup_check_ptr(guard)
            } else {
                self.find_next_ptr(guard)
            };

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

                self.state = ScanState::FindNext;
                // SAFETY: Version validated, guard held, snap pointer valid.
                let value_ref: &P::Value =
                    unsafe { snap.resolve_value_ref::<P>(&mut self.scratch_value) };

                return Some((key, value_ref));
            }
        }
    }
}
