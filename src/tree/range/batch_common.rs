//! Shared helpers for forward and reverse batch processing.

use crate::permuter::Permuter15;
use std::cmp::Ordering;

use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::cursor_key::CursorKey;
use super::helper::{
    CloneEmitter, PtrEmitter, ScanEmitter, SlotPayload, build_slot_key_parts, try_snapshot_slot,
};
use super::iterator::CursorBound;
use super::iterator::RangeBound;
use super::iterator::iter_flags::ScanFlags;
use super::scan_state::LeafBatchResult;
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

use super::direction::Direction;
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
    flags: &mut ScanFlags,
) -> LeafBatchResult
where
    D: Direction,
    V: SlotVisitor<P>,
    P: LeafPolicy,
{
    if ctx.leaf.version().is_deleted() {
        return LeafBatchResult::VersionChanged;
    }

    while ctx.ki >= 0 && ctx.ki.unsigned_abs() < ctx.perm_size {
        let slot: usize = ctx.perm.get(ctx.ki.unsigned_abs());
        D::prefetch_neighbor::<P>(ctx.ki, ctx.leaf, ctx.perm);

        let Ok(captured) = try_snapshot_slot::<P, V::Emitter>(ctx.leaf, slot, ctx.cached_version)
        else {
            return LeafBatchResult::VersionChanged;
        };
        let slot_ikey = captured.ikey;
        let output = match captured.payload {
            SlotPayload::Layer(layer_ptr) => {
                match cursor_bound {
                    CursorBound::Outside => return LeafBatchResult::BoundExceeded,
                    CursorBound::At(bound_ikey)
                        if slot_ikey.cmp(&bound_ikey) == D::exceeded_ordering() =>
                    {
                        return LeafBatchResult::BoundExceeded;
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
                return LeafBatchResult::LayerEncountered;
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
            CursorBound::Outside => return LeafBatchResult::BoundExceeded,

            CursorBound::At(b_ikey) => match slot_ikey.cmp(&b_ikey) {
                ord if ord == D::exceeded_ordering() => return LeafBatchResult::BoundExceeded,
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
            return LeafBatchResult::BoundExceeded;
        }

        D::on_pre_emit(flags);

        *count += 1;
        ctx.ki = D::step(ctx.ki);
        if !slot_visitor.visit(output, key) {
            return LeafBatchResult::Stopped;
        }
    }

    if ctx.leaf.version().has_changed(ctx.cached_version) {
        return LeafBatchResult::VersionChanged;
    }

    LeafBatchResult::LeafExhausted
}

// ============================================================================
//  Unified Values-Only Batch Processor
// ============================================================================

/// A scan visitor also accepts the copy snapshot produced by initial positioning.
pub trait ScanVisitor<P: LeafPolicy>: SlotVisitor<P> {
    fn visit_initial(&mut self, snapshot: ScanSnapshot<P>, key: &[u8]) -> bool;
}

impl<P: LeafPolicy, F: FnMut(&[u8], P::Output) -> bool> ScanVisitor<P> for CopySlotVisitor<F> {
    #[inline(always)]
    fn visit_initial(&mut self, snapshot: ScanSnapshot<P>, key: &[u8]) -> bool {
        (self.0)(key, snapshot.value)
    }
}
impl<P: crate::policy::RefPolicy, F: FnMut(&[u8], &P::Value) -> bool> ScanVisitor<P>
    for RefSlotVisitor<F>
{
    #[inline(always)]
    fn visit_initial(&mut self, snapshot: ScanSnapshot<P>, key: &[u8]) -> bool {
        let mut scratch = std::mem::MaybeUninit::uninit();
        // SAFETY: Initial positioning validated this terminal handle; the scan
        // guard owns its lifetime through the callback. Write-through values use
        // an atomic copy into scratch rather than aliasing mutable box data.
        let value = unsafe { P::output_as_ref_sound(&snapshot.value, &mut scratch) };
        (self.0)(key, value)
    }
}

/// Values-only delivery retains the same full cursor key for exact bounds/retry.
pub struct ValuesSlotVisitor<F>(pub F);
impl<P: LeafPolicy, F: FnMut(P::Output) -> bool> SlotVisitor<P> for ValuesSlotVisitor<F> {
    type Emitter = CloneEmitter;
    #[inline(always)]
    fn visit(&mut self, snapshot: ScanSnapshot<P>, _key: &[u8]) -> bool {
        (self.0)(snapshot.value)
    }
}
impl<P: LeafPolicy, F: FnMut(P::Output) -> bool> ScanVisitor<P> for ValuesSlotVisitor<F> {
    #[inline(always)]
    fn visit_initial(&mut self, snapshot: ScanSnapshot<P>, _key: &[u8]) -> bool {
        (self.0)(snapshot.value)
    }
}
