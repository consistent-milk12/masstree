//! Compile-time scan direction and the leaf/layer navigation that differs by direction.

use seize::LocalGuard;
use std::cmp::Ordering;

use super::{
    helper::PtrEmitter,
    iterator::{RangeBound, iter_flags::ScanFlags},
    scan_ctx::{ForwardStorage, ScanCtx},
    scan_state::ScanState,
};
use crate::{leaf15::LeafNode15, permuter::Permuter15, policy::LeafPolicy};

pub trait Direction: Sized {
    const REVERSE: bool;
    const INITIAL_IKEY: u64;
    type Storage<P: LeafPolicy>: Default;

    #[inline(always)]
    fn step(ki: isize) -> isize {
        if Self::REVERSE { ki - 1 } else { ki + 1 }
    }
    #[inline(always)]
    fn layer_ki(ki: isize) -> isize {
        if Self::REVERSE { ki - 1 } else { ki }
    }
    #[inline(always)]
    fn prefetch_neighbor<P: LeafPolicy>(ki: isize, leaf: &LeafNode15<P>, perm: Permuter15) {
        let next = Self::step(ki);
        if next >= 0 && next.cast_unsigned() < perm.size() {
            leaf.prefetch_value(perm.get(next.cast_unsigned()));
        }
    }
    #[inline(always)]
    fn exceeded_ordering() -> Ordering {
        if Self::REVERSE {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
    #[inline(always)]
    fn key_in_bound(bound: &RangeBound<'_>, key: &[u8]) -> bool {
        if Self::REVERSE {
            bound.contains_reverse(key)
        } else {
            bound.contains(key)
        }
    }
    #[inline(always)]
    fn on_pre_emit(flags: &mut ScanFlags) {
        flags.clear_upper_bound();
        flags.mark_cursor_emitted();
    }

    fn advance_leaf<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>)
    -> ScanState;
    fn retry<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> ScanState;
    fn descend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>);
    fn ascend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> bool;
}

pub struct Forward;
pub struct Backward;

impl Direction for Forward {
    const REVERSE: bool = false;
    const INITIAL_IKEY: u64 = 0;
    type Storage<P: LeafPolicy> = ForwardStorage<P>;

    #[inline]
    fn advance_leaf<P: LeafPolicy>(
        ctx: &mut ScanCtx<P, Self>,
        guard: &LocalGuard<'_>,
    ) -> ScanState {
        ctx.advance_leaf(guard).0
    }
    #[inline]
    fn retry<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> ScanState {
        ctx.find_retry(guard)
    }
    #[inline]
    fn descend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>) {
        ctx.handle_down();
    }
    #[inline]
    fn ascend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> bool {
        ctx.handle_up(guard)
    }
}

impl Direction for Backward {
    const REVERSE: bool = true;
    const INITIAL_IKEY: u64 = u64::MAX;
    type Storage<P: LeafPolicy> = ();

    #[inline]
    fn advance_leaf<P: LeafPolicy>(
        ctx: &mut ScanCtx<P, Self>,
        guard: &LocalGuard<'_>,
    ) -> ScanState {
        ctx.advance_to_prev_leaf_generic::<PtrEmitter>(guard).0
    }
    #[inline]
    fn retry<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> ScanState {
        ctx.reposition_back(guard).0
    }
    #[inline]
    fn descend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>) {
        ctx.handle_down_back();
    }
    #[inline]
    fn ascend<P: LeafPolicy>(ctx: &mut ScanCtx<P, Self>, guard: &LocalGuard<'_>) -> bool {
        ctx.handle_up_back(guard)
    }
}
