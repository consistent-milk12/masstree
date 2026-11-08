//! Forward batch iteration methods for maximum performance.

use crate::alloc_trait::TreeAllocator;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;

use super::RangeIter;

use crate::tree::range::batch_common::{CopySlotVisitor, RefSlotVisitor, ValuesSlotVisitor};

impl<P, A> RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Zero-allocation iteration with a visitor closure.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }

        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            if let Some(entry) = self.fwd.advance_no_alloc(&self.end_bound, self.guard) {
                count += 1;

                if !visitor(entry.0, entry.1) {
                    break 'l;
                }
            } else {
                break 'l;
            }
        }

        count
    }

    /// Zero-copy iteration with borrowed `&P::Value` references.
    ///
    /// Eliminates Arc increment per entry vs [`for_each`](Self::for_each).
    /// References are valid only within the callback scope.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each_ref<F>(mut self, mut visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }

        if !self.fwd.flags.initialized() {
            self.initialize();
            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            if let Some((key, value_ref)) =
                self.fwd.advance_no_alloc_ref(&self.end_bound, self.guard)
            {
                count += 1;
                if !visitor(key, value_ref) {
                    break 'l;
                }
            } else {
                break 'l;
            }
        }

        count
    }

    /// Batch scan with zero-copy references and reduced dispatch overhead.
    ///
    /// Uses the shared leaf-batch loop, validating each captured slot before
    /// its callback and retaining full-key progress across retries.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each_batch_ref<F>(self, visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        self.for_each_intra_leaf_batch_ref(visitor)
    }

    /// Intra-leaf batch iteration with zero-copy references.
    ///
    /// Processes leaves in tight loops with OCC validation before each
    /// callback. The shared driver handles sublayer transitions.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each_intra_leaf_batch_ref<F>(mut self, mut visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }
        if !self.fwd.flags.initialized() {
            self.initialize();
            if self.fwd.flags.exhausted() {
                return 0;
            }
        }
        self.fwd.run_batch(
            &mut RefSlotVisitor(&mut visitor),
            &self.end_bound,
            self.guard,
        )
    }

    /// Intra-leaf batch iteration returning values by copy.
    ///
    /// Works for ALL `LeafPolicy` types including true-inline storage.
    /// Same leaf-level batching as `for_each_intra_leaf_batch_ref` but returns
    /// `P::Output` by value. For Arc storage where zero-copy matters, prefer
    /// the `_ref` variant.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each_intra_leaf_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }
        if !self.fwd.flags.initialized() {
            self.initialize();
            if self.fwd.flags.exhausted() {
                return 0;
            }
        }
        self.fwd.run_batch(
            &mut CopySlotVisitor(&mut visitor),
            &self.end_bound,
            self.guard,
        )
    }

    /// Value-only batch iteration.
    ///
    /// ```no_run
    /// use masstree::MassTree;
    /// let tree: MassTree<u64> = MassTree::new();
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    /// tree.iter(&guard).for_each_values_batch(|value| {
    ///     sum += value;
    ///     true
    /// });
    /// ```
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each_values_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(P::Output) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }
        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        self.fwd.run_batch(
            &mut ValuesSlotVisitor(&mut visitor),
            &self.end_bound,
            self.guard,
        )
    }

    /// Fallible iteration with zero-copy references.
    ///
    /// ```ignore
    /// let result = tree.iter(&guard).try_for_each_ref(|key, value| {
    ///     writer.write_entry(key, value)?;
    ///     Ok(true)
    /// });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the visitor's error.
    #[inline]
    #[must_use = "returns the count or error - check the result"]
    pub fn try_for_each_ref<F, E>(mut self, mut visitor: F) -> Result<usize, E>
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> Result<bool, E>,
    {
        if self.fwd.flags.exhausted() {
            return Ok(0);
        }

        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                return Ok(0);
            }
        }

        let mut count: usize = 0;

        loop {
            if let Some((key, value_ref)) =
                self.fwd.advance_no_alloc_ref(&self.end_bound, self.guard)
            {
                count += 1;

                match visitor(key, value_ref) {
                    Ok(true) => {}

                    Ok(false) => return Ok(count),

                    Err(e) => return Err(e),
                }
            } else {
                return Ok(count);
            }
        }
    }
}
