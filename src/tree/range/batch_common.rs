//! Shared helpers for forward and reverse batch processing.

use crate::permuter::Permuter15;
use std::cmp::Ordering;

use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::cursor_key::CursorKey;
use super::find::LeafBatchResult;
use super::find_rev::LeafBatchResultBack;
use super::helper::{
    CloneEmitter, PtrEmitter, ScanEmitter, SlotPayload, build_slot_key_parts, try_snapshot_slot,
};
use super::iterator::CursorBound;
use super::iterator::RangeBound;
use super::iterator::iter_flags::ReverseFlags;
use super::scan_state::{LayerContext, LayerStack, ScanSnapshot, ScanSnapshotPtr};

#[cfg(test)]
mod unit_tests;

/// Delivers a value captured before OCC validation. Visitors never reload slots.
pub trait SlotVisitor<P: LeafPolicy> {
    type Emitter: ScanEmitter<P>;

    fn visit(&mut self, snapshot: <Self::Emitter as ScanEmitter<P>>::Snapshot, key: &[u8]) -> bool;
}

/// Reference visitor over a validated, guard-protected pointer snapshot.
pub struct RefSlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for RefSlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    type Emitter = PtrEmitter;

    #[inline(always)]
    fn visit(&mut self, snapshot: ScanSnapshotPtr<P::Value>, key: &[u8]) -> bool {
        if P::can_write_through() {
            // SAFETY: The pointer was loaded before OCC validation and remains
            // protected by the scan guard. This sealed policy permits atomic
            // copies; removal/reuse cannot change the captured allocation's key.
            let value = unsafe { snapshot.value_copy() };
            (self.0)(key, &value)
        } else {
            // SAFETY: Validation identified a terminal pointer; the guard keeps
            // it alive. Non-write-through values are replaced, never mutated.
            (self.0)(key, unsafe { snapshot.value_ref() })
        }
    }
}

/// Copy visitor over a validated output snapshot, including inline values.
pub struct CopySlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for CopySlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    type Emitter = CloneEmitter;

    #[inline(always)]
    fn visit(&mut self, snapshot: ScanSnapshot<P>, key: &[u8]) -> bool {
        (self.0)(key, snapshot.value)
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
    fn prefetch_neighbor<P: LeafPolicy>(ki: isize, leaf: &LeafNode15<P>, perm: Permuter15);

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
    fn prefetch_neighbor<P: LeafPolicy>(ki: isize, leaf: &LeafNode15<P>, perm: Permuter15) {
        let next_ki: usize = (ki + 1).cast_unsigned();
        if next_ki < perm.size() {
            let next_slot: usize = perm.get(next_ki);
            leaf.prefetch_value(next_slot);
        }
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
    fn prefetch_neighbor<P: LeafPolicy>(ki: isize, leaf: &LeafNode15<P>, perm: Permuter15) {
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
        flags.mark_cursor_emitted();
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
    pub perm: Permuter15,
    pub perm_size: usize,
    pub cached_version: u32,
    pub ki: isize,
    pub root: *const u8,
    pub leaf_ptr: *mut LeafNode15<P>,
    pub cursor_key: &'a mut CursorKey,
    pub layer_stack: &'a mut LayerStack<P>,
    /// Whether the pending layer descent shares the cursor's prefix and must
    /// carry its real suffix into the sublayer.
    pub down_follow_suffix: bool,
    /// Whether the pending layer descent starts a different prefix subtree.
    pub down_fresh_subtree: bool,
}

/// Direction-parameterized keyed batch processor.
#[inline]
pub fn process_batch_keyed<D, V, P>(
    ctx: &mut BatchCtx<'_, P>,
    bound: &RangeBound<'_>,
    cursor_bound: CursorBound,
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
        D::prefetch_neighbor::<P>(ctx.ki, ctx.leaf, ctx.perm);

        let Ok(captured) = try_snapshot_slot::<P, V::Emitter>(ctx.leaf, slot, ctx.cached_version)
        else {
            return D::version_changed();
        };
        let slot_ikey = captured.ikey;
        let output = match captured.payload {
            SlotPayload::Layer(layer_ptr) => {
                match cursor_bound {
                    CursorBound::Outside => return D::bound_exceeded(),
                    CursorBound::At(bound_ikey)
                        if slot_ikey.cmp(&bound_ikey) == D::exceeded_ordering() =>
                    {
                        return D::bound_exceeded();
                    }
                    CursorBound::At(_) | CursorBound::Inside => {}
                }
                let same_prefix = slot_ikey == ctx.cursor_key.current_ikey();
                let real_suffix =
                    ctx.cursor_key.has_suffix() && !ctx.cursor_key.is_unshift_sentinel();
                ctx.down_follow_suffix = same_prefix && real_suffix;
                ctx.down_fresh_subtree = !same_prefix;
                ctx.layer_stack
                    .push(LayerContext::new(ctx.root, ctx.leaf_ptr));
                ctx.cursor_key.assign_store_ikey(slot_ikey);
                prefetch_read(layer_ptr);
                ctx.root = layer_ptr.cast_const();
                ctx.ki = D::layer_ki(ctx.ki);
                return D::layer_encountered();
            }
            SlotPayload::Empty => {
                ctx.ki = D::step(ctx.ki);
                continue;
            }
            SlotPayload::Value(value) => value,
        };

        // Fast ikey bound check. A full-key comparison is only needed when the
        // slot shares the bound's 8-byte ikey; every interior slot is ordered
        // by its ikey alone. `CursorBound::Inside` means the whole sublayer is
        // in range, so no check is needed at all.
        let mut needs_full_bound: bool = false;

        match cursor_bound {
            CursorBound::Outside => return D::bound_exceeded(),

            CursorBound::At(b_ikey) => match slot_ikey.cmp(&b_ikey) {
                ord if ord == D::exceeded_ordering() => return D::bound_exceeded(),
                Ordering::Equal => needs_full_bound = true,
                _ => {}
            },

            CursorBound::Inside => {}
        }

        // Phase 3: commit the validated snapshot and check the full-key bound.
        build_slot_key_parts(ctx.cursor_key, slot_ikey, captured.keylenx, captured.suffix);
        ctx.cursor_key.mark_key_complete();

        let key: &[u8] = ctx.cursor_key.full_key();
        if needs_full_bound && !D::key_in_bound(bound, key) {
            return D::bound_exceeded();
        }

        D::on_pre_emit(flags);

        *count += 1;
        ctx.ki = D::step(ctx.ki);
        if !slot_visitor.visit(output, key) {
            return D::stopped();
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

/// Values-only adapter over the same validated batch loop. The cursor still
/// records suffix bytes so a retry can resume after the last emitted full key.
#[inline]
pub fn process_batch_values<D, P>(
    ctx: &mut BatchCtx<'_, P>,
    bound: &RangeBound<'_>,
    cursor_bound: CursorBound,
    visitor: &mut impl FnMut(P::Output) -> bool,
    count: &mut usize,
    flags: &mut D::Flags,
) -> D::Result
where
    D: BatchDirection,
    P: LeafPolicy,
{
    process_batch_keyed::<D, _, P>(
        ctx,
        bound,
        cursor_bound,
        &mut CopySlotVisitor(|_key: &[u8], value| visitor(value)),
        count,
        flags,
    )
}
