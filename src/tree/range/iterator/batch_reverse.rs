//! Reverse batch iteration methods. Each method wraps
//! the shared direction-parametric `ScanCtx::run_batch` with a value visitor.

use crate::alloc_trait::TreeAllocator;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;

use crate::tree::range::batch_common::{CopySlotVisitor, RefSlotVisitor, ValuesSlotVisitor};

use super::RangeIter;

impl<P, A> RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Reverse intra-leaf batch iteration with zero-copy references.
    ///
    /// Processes leaves in tight loops with OCC validation before each
    /// callback. The shared driver handles layer transitions.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn rev_for_each_ref<F>(mut self, mut visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        if self.back_exhausted() {
            return 0;
        }

        if !self.back_initialized() {
            self.initialize_back();
            if self.back_exhausted() {
                return 0;
            }
        }

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = RefSlotVisitor(&mut visitor);
        rev.run_batch(&mut strategy, &self.start_bound, self.guard)
    }

    /// Reverse intra-leaf batch iteration returning values by copy.
    ///
    /// Works for ALL storage types including true-inline. Same leaf-level
    /// batching as `rev_for_each_ref` but returns `P::Output` by value.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn rev_for_each_intra_leaf_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        if self.back_exhausted() {
            return 0;
        }

        if !self.back_initialized() {
            self.initialize_back();
            if self.back_exhausted() {
                return 0;
            }
        }

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = CopySlotVisitor(&mut visitor);
        rev.run_batch(&mut strategy, &self.start_bound, self.guard)
    }

    /// Reverse value-only batch iteration with exact full-key bounds.
    ///
    /// Cursor key bytes are retained for bounds and retry progress; callbacks
    /// receive only the value.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn rev_for_each_values_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(P::Output) -> bool,
    {
        if self.back_exhausted() {
            return 0;
        }

        if !self.back_initialized() {
            self.initialize_back();
            if self.back_exhausted() {
                return 0;
            }
        }

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = ValuesSlotVisitor(&mut visitor);
        rev.run_batch(&mut strategy, &self.start_bound, self.guard)
    }
}
