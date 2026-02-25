//! Shared helpers for forward and reverse batch processing.
//!
//! Extracted to eliminate key-building duplication across `find.rs` and `find_rev.rs`.
//!
//! # `ScanEmitter` Trait
//!
//! Abstracts clone vs zero-copy value emission for per-entry scanning.
//! Used by both forward (`find_next_generic`) and reverse (`find_prev_generic`).
//! Two implementations:
//! - [`CloneEmitter`]: calls `load_value()`, returns `ScanSnapshot<P>`. Requires `P::Output: Clone`.
//! - [`PtrEmitter`]: calls `load_value_raw()` + null check, returns `ScanSnapshotPtr<P::Value>`.
//!
//! # `SlotVisitor` Trait
//!
//! Abstracts the ref vs copy value-loading difference for batch processing. Two implementations:
//! - [`RefSlotVisitor`]: loads `&P::Value` via pointer dereference (zero-copy)
//! - [`CopySlotVisitor`]: loads `P::Output` via `load_value()` (universal)
//!
//! # `BatchDirection` Trait
//!
//! Abstracts iteration direction for batch processing. Zero-sized marker types
//! [`Forward`] and [`Backward`] implement all direction-specific operations.
//! After monomorphization, all trait dispatch is fully inlined — zero overhead.

use std::cmp::Ordering;

use crate::hints::unlikely;
use crate::key::IKEY_SIZE;
use crate::leaf_trait::TreeLeafNode;
use crate::leaf15::KSUF_KEYLENX;
use crate::leaf15::LAYER_KEYLENX;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::cursor_key::CursorKey;
use super::find::LeafBatchResult;
use super::find_rev::LeafBatchResultBack;
use super::iterator::RangeBound;
use super::iterator::iter_flags::ReverseFlags;
use super::scan_state::{LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr};

/// Build the full key in `cursor_key` from slot data.
///
/// Stores the ikey unconditionally, then handles suffix or inline length.
/// Callers should call `cursor_key.mark_key_complete()` after this returns.
#[inline(always)]
pub fn build_slot_key<P: LeafPolicy>(
    cursor_key: &mut CursorKey,
    leaf: &LeafNode15<P>,
    slot: usize,
    slot_ikey: u64,
    slot_keylenx: u8,
) {
    cursor_key.assign_store_ikey(slot_ikey);

    if slot_keylenx == KSUF_KEYLENX {
        if let Some(suffix) = leaf.ksuf(slot) {
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

/// Visitor for batch slot processing — abstracts ref vs copy value loading.
///
/// Each implementation defines how to load and deliver a value from a leaf slot
/// to the user's callback. The batch loop calls [`visit`](SlotVisitor::visit)
/// after building the key, letting the visitor handle value access.
pub trait SlotVisitor<P: LeafPolicy> {
    /// Visit a slot after key has been built in `cursor_key`.
    ///
    /// # Returns
    ///
    /// - `None`: Value was concurrently removed (TOCTOU race) — skip this slot
    /// - `Some(true)`: Continue scanning
    /// - `Some(false)`: Visitor requested stop
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool>;
}

/// Ref visitor: loads `&P::Value` via pointer dereference (zero-copy).
///
/// For use with `RefLeafPolicy` types where values are pointer-backed (Arc, Box).
pub struct RefSlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for RefSlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    #[inline(always)]
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool> {
        // SAFETY: Guard protects value, slot is valid (in permutation).
        // Null-check handles TOCTOU race — value may have been concurrently
        // removed between caller's is_value_empty check and this dereference.
        let ptr = unsafe { leaf.load_value_ptr(slot) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: Non-null pointer to a valid P::Value, protected by OCC guard.
        let value_ref: &P::Value = unsafe { &*ptr };
        Some((self.0)(key, value_ref))
    }
}

/// Copy visitor: loads `P::Output` via `load_value()` (universal).
///
/// Works for all `LeafPolicy` types including true-inline storage.
pub struct CopySlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for CopySlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    #[inline(always)]
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool> {
        // Use load_value to handle TOCTOU race — value may have been
        // concurrently removed between is_value_empty check and here.
        let output = leaf.load_value(slot)?;
        Some((self.0)(key, output))
    }
}

/// Abstracts clone vs zero-copy value emission for per-entry scanning.
///
/// Used by both forward (`find_next_generic`) and reverse (`find_prev_generic`).
/// Two implementations:
/// - [`CloneEmitter`]: calls `load_value()`, returns `ScanSnapshot<P>`. Requires `P::Output: Clone`.
/// - [`PtrEmitter`]: calls `load_value_raw()` + null check, returns `ScanSnapshotPtr<P::Value>`.
///
/// After monomorphization, trait dispatch is fully inlined — zero overhead.
pub trait ScanEmitter<P: LeafPolicy> {
    type Snapshot;

    /// Load a value from `leaf[slot]` and wrap it with `key_len`.
    ///
    /// Returns `None` if the value was concurrently removed (TOCTOU race).
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

/// Direction-specific operations for batch leaf processing.
///
/// Implemented by zero-sized marker types [`Forward`] and [`Backward`].
/// All methods are trivially inlineable, so monomorphization produces
/// identical code to hand-written direction-specific loops.
pub trait BatchDirection: Sized {
    /// Result type for batch processing.
    type Result: Copy;

    /// Direction-specific flags type. `()` for forward, [`ReverseFlags`] for backward.
    type Flags: ?Sized;

    /// Advance position by one step in this direction.
    fn step(ki: isize) -> isize;

    /// Position to store when a layer pointer is encountered.
    fn layer_ki(ki: isize) -> isize;

    /// Prefetch neighbor slot's value to hide memory latency.
    /// Forward: no-op. Backward: prefetches previous slot.
    fn prefetch_neighbor<P: LeafPolicy>(
        ki: isize,
        leaf: &LeafNode15<P>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    );

    /// The [`Ordering`] that indicates the bound has been exceeded.
    /// Forward: `Greater` (`slot_ikey` > `end_bound`). Backward: `Less` (`slot_ikey` < `start_bound`).
    fn exceeded_ordering() -> Ordering;

    /// Full key bound check.
    /// Forward: `bound.contains(key)`. Backward: `bound.contains_reverse(key)`.
    fn key_in_bound(bound: &RangeBound<'_>, key: &[u8]) -> bool;

    /// Called before emitting a value. Forward: no-op. Backward: clears `upper_bound` flag.
    fn on_pre_emit(flags: &mut Self::Flags);

    // Result constructors
    fn exhausted() -> Self::Result;
    fn layer_encountered() -> Self::Result;
    fn version_changed() -> Self::Result;
    fn stopped() -> Self::Result;
    fn bound_exceeded() -> Self::Result;
}

/// Forward direction marker for batch processing.
pub struct Forward;

impl BatchDirection for Forward {
    type Result = LeafBatchResult;
    type Flags = ();

    #[inline(always)]
    fn step(ki: isize) -> isize {
        ki + 1
    }

    #[inline(always)]
    fn layer_ki(ki: isize) -> isize {
        ki
    }

    #[inline(always)]
    fn prefetch_neighbor<P: LeafPolicy>(
        _ki: isize,
        _leaf: &LeafNode15<P>,
        _perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) {
        // Forward scan does not prefetch neighbor slot values.
    }

    #[inline(always)]
    fn exceeded_ordering() -> Ordering {
        Ordering::Greater
    }

    #[inline(always)]
    fn key_in_bound(bound: &RangeBound<'_>, key: &[u8]) -> bool {
        bound.contains(key)
    }

    #[inline(always)]
    fn on_pre_emit(_flags: &mut ()) {}

    #[inline(always)]
    fn exhausted() -> LeafBatchResult {
        LeafBatchResult::LeafExhausted
    }

    #[inline(always)]
    fn layer_encountered() -> LeafBatchResult {
        LeafBatchResult::LayerEncountered
    }

    #[inline(always)]
    fn version_changed() -> LeafBatchResult {
        LeafBatchResult::VersionChanged
    }

    #[inline(always)]
    fn stopped() -> LeafBatchResult {
        LeafBatchResult::Stopped
    }

    #[inline(always)]
    fn bound_exceeded() -> LeafBatchResult {
        LeafBatchResult::EndBoundExceeded
    }
}

/// Backward direction marker for batch processing.
pub struct Backward;

impl BatchDirection for Backward {
    type Result = LeafBatchResultBack;
    type Flags = ReverseFlags;

    #[inline(always)]
    fn step(ki: isize) -> isize {
        ki - 1
    }

    #[inline(always)]
    fn layer_ki(ki: isize) -> isize {
        ki - 1
    }

    #[inline(always)]
    fn prefetch_neighbor<P: LeafPolicy>(
        ki: isize,
        leaf: &LeafNode15<P>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) {
        if ki > 0 {
            let prev_slot: usize = perm.get((ki - 1).cast_unsigned());
            leaf.prefetch_value(prev_slot);
        }
    }

    #[inline(always)]
    fn exceeded_ordering() -> Ordering {
        Ordering::Less
    }

    #[inline(always)]
    fn key_in_bound(bound: &RangeBound<'_>, key: &[u8]) -> bool {
        bound.contains_reverse(key)
    }

    #[inline(always)]
    fn on_pre_emit(flags: &mut ReverseFlags) {
        flags.clear_upper_bound();
    }

    #[inline(always)]
    fn exhausted() -> LeafBatchResultBack {
        LeafBatchResultBack::LeafExhausted
    }

    #[inline(always)]
    fn layer_encountered() -> LeafBatchResultBack {
        LeafBatchResultBack::LayerEncountered
    }

    #[inline(always)]
    fn version_changed() -> LeafBatchResultBack {
        LeafBatchResultBack::VersionChanged
    }

    #[inline(always)]
    fn stopped() -> LeafBatchResultBack {
        LeafBatchResultBack::Stopped
    }

    #[inline(always)]
    fn bound_exceeded() -> LeafBatchResultBack {
        LeafBatchResultBack::StartBoundExceeded
    }
}

/// Extracted batch processing state, direction-agnostic.
///
/// Callers build this from their direction-specific stack type, call the
/// unified batch function, then write back `ki` and `root`.
pub struct BatchCtx<'a, P: LeafPolicy> {
    pub leaf: &'a LeafNode15<P>,
    pub perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm,
    pub perm_size: usize,
    pub cached_version: u32,
    pub ki: isize,
    pub root: *const u8,
    pub leaf_ptr: *mut LeafNode15<P>,
    pub cursor_key: &'a mut CursorKey,
    pub layer_stack: &'a mut LayerStack<P>,
}

/// Direction-parameterized keyed batch processor.
#[inline]
pub fn process_batch_keyed<D, V, P>(
    ctx: &mut BatchCtx<'_, P>,
    bound: &RangeBound<'_>,
    bound_ikey: Option<u64>,
    slot_visitor: &mut V,
    count: &mut usize,
    flags: &mut D::Flags,
) -> D::Result
where
    D: BatchDirection,
    V: SlotVisitor<P>,
    P: LeafPolicy,
{
    if ctx.leaf.version().is_deleted() {
        return D::version_changed();
    }

    while ctx.ki >= 0 && ctx.ki.unsigned_abs() < ctx.perm_size {
        let slot: usize = ctx.perm.get(ctx.ki.unsigned_abs());
        let slot_ikey: u64 = ctx.leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = ctx.leaf.keylenx(slot);

        D::prefetch_neighbor::<P>(ctx.ki, ctx.leaf, &ctx.perm);

        // Layer pointer check
        if unlikely(slot_keylenx >= LAYER_KEYLENX) {
            let layer_ptr: *mut u8 = ctx.leaf.load_layer_raw(slot);
            ctx.layer_stack
                .push(LayerContext::new(ctx.root, ctx.leaf_ptr));
            ctx.cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            ctx.root = layer_ptr.cast_const();
            ctx.ki = D::layer_ki(ctx.ki);
            return D::layer_encountered();
        }

        // Empty value check
        if unlikely(ctx.leaf.is_value_empty(slot)) {
            ctx.ki = D::step(ctx.ki);
            continue;
        }

        // Fast bound pre-check using ikey
        let mut needs_full_bound_check = true;
        if bound_ikey.is_none() {
            needs_full_bound_check = false;
        } else if ctx.cursor_key.is_at_root_layer()
            && let Some(b_ikey) = bound_ikey
        {
            match slot_ikey.cmp(&b_ikey) {
                ord if ord == D::exceeded_ordering() => return D::bound_exceeded(),
                Ordering::Equal => {}
                _ => needs_full_bound_check = false,
            }
        }

        // Build key
        build_slot_key(ctx.cursor_key, ctx.leaf, slot, slot_ikey, slot_keylenx);
        ctx.cursor_key.mark_key_complete();

        D::on_pre_emit(flags);

        // Full key bound check
        let key: &[u8] = ctx.cursor_key.full_key();
        if needs_full_bound_check && !D::key_in_bound(bound, key) {
            return D::bound_exceeded();
        }

        // Visit slot
        match slot_visitor.visit(ctx.leaf, slot, key) {
            None => {
                ctx.ki = D::step(ctx.ki);
            }
            Some(should_continue) => {
                *count += 1;
                ctx.ki = D::step(ctx.ki);
                if !should_continue {
                    return D::stopped();
                }
            }
        }
    }

    if ctx.leaf.version().has_changed(ctx.cached_version) {
        return D::version_changed();
    }

    D::exhausted()
}

// ============================================================================
//  Unified Values-Only Batch Processor
// ============================================================================

/// Direction-parameterized values-only batch processor (no key materialization).
#[inline]
pub fn process_batch_values<D, P>(
    ctx: &mut BatchCtx<'_, P>,
    bound_ikey: Option<u64>,
    visitor: &mut impl FnMut(P::Output) -> bool,
    count: &mut usize,
    flags: &mut D::Flags,
) -> D::Result
where
    D: BatchDirection,
    P: LeafPolicy,
{
    if ctx.leaf.version().is_deleted() {
        return D::version_changed();
    }

    while ctx.ki >= 0 && ctx.ki.unsigned_abs() < ctx.perm_size {
        let slot: usize = ctx.perm.get(ctx.ki.unsigned_abs());
        let slot_ikey: u64 = ctx.leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = ctx.leaf.keylenx(slot);

        D::prefetch_neighbor::<P>(ctx.ki, ctx.leaf, &ctx.perm);

        // Layer pointer check
        if unlikely(slot_keylenx >= LAYER_KEYLENX) {
            let layer_ptr: *mut u8 = ctx.leaf.load_layer_raw(slot);
            ctx.layer_stack
                .push(LayerContext::new(ctx.root, ctx.leaf_ptr));
            ctx.cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(layer_ptr);
            ctx.root = layer_ptr.cast_const();
            ctx.ki = D::layer_ki(ctx.ki);
            return D::layer_encountered();
        }

        // Fast bound check (ikey only)
        if let Some(b_ikey) = bound_ikey
            && slot_ikey.cmp(&b_ikey) == D::exceeded_ordering()
        {
            return D::bound_exceeded();
        }

        // Empty value check
        if unlikely(ctx.leaf.is_value_empty(slot)) {
            ctx.ki = D::step(ctx.ki);
            continue;
        }

        D::on_pre_emit(flags);

        let Some(output) = ctx.leaf.load_value(slot) else {
            ctx.ki = D::step(ctx.ki);
            continue;
        };

        *count += 1;
        ctx.ki = D::step(ctx.ki);

        if !visitor(output) {
            return D::stopped();
        }
    }

    if ctx.leaf.version().has_changed(ctx.cached_version) {
        return D::version_changed();
    }

    D::exhausted()
}
