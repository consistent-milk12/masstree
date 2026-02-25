//! Filepath: `src/tree/range/iterator/batch_reverse.rs`
//!
//! Reverse batch iteration methods for maximum performance.
//!
//! Each method is a thin wrapper around [`ReverseScanCtx::run_batch_reverse`]
//! with a different strategy type. After monomorphization, the strategy trait
//! dispatch is fully inlined — zero overhead.

use crate::alloc_trait::TreeAllocator;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;

use crate::tree::range::reverse_ctx::{
    RevIntraLeafCopyStrategy, RevIntraLeafRefStrategy, RevValuesOnlyStrategy,
};

use super::RangeIter;

impl<P, A> RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// High-performance reverse iteration with zero-copy references.
    ///
    /// This is the fastest reverse iteration method. It processes entire
    /// leaves in tight loops, minimizing per-entry overhead.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to previous leaf
    /// - Single OCC validation per leaf
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &P::Value)`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
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

        // Lazy initialization of back cursor
        if !self.back_initialized() {
            self.initialize_back();

            if self.back_exhausted() {
                return 0;
            }
        }

        // Pre-extract start bound ikey for fast comparison in batch processor
        let start_bound_ikey: Option<u64> = self.start_bound.extract_ikey();

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = RevIntraLeafRefStrategy::new(&mut visitor);
        rev.run_batch_reverse(
            &mut strategy,
            &self.start_bound,
            start_bound_ikey,
            self.guard,
        )
    }

    /// Highest-performance batch-optimized reverse iteration (value by copy).
    ///
    /// This is the non-reference variant that works with ALL storage types including
    /// true-inline (`MassTree15Inline`). Unlike `rev_for_each_ref` which returns
    /// `&P::Value` references, this returns `P::Output` by value.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to previous leaf
    /// - Single OCC validation per leaf
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions
    ///
    /// # Availability
    ///
    /// Available for ALL storage types:
    /// - `MassTree15<V>` (Box-based)
    /// - `MassTree15Inline<V>` (true-inline)
    ///
    /// # Arguments
    ///
    /// - `visitor`: Callback function `fn(&[u8], P::Output) -> bool`
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn rev_for_each_intra_leaf_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        if self.back_exhausted() {
            return 0;
        }

        // Lazy initialization of back cursor
        if !self.back_initialized() {
            self.initialize_back();
            if self.back_exhausted() {
                return 0;
            }
        }

        // Pre-extract start bound ikey for fast comparison in batch processor
        let start_bound_ikey: Option<u64> = self.start_bound.extract_ikey();

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = RevIntraLeafCopyStrategy::new(&mut visitor);
        rev.run_batch_reverse(
            &mut strategy,
            &self.start_bound,
            start_bound_ikey,
            self.guard,
        )
    }

    /// Highest-performance reverse value-only batch iteration (no key materialization).
    ///
    /// This is the fastest reverse scan method when you only need values. Keys are not
    /// built or copied, saving up to 56 bytes of copying per entry for long keys.
    ///
    /// # Performance
    ///
    /// For 64-byte keys: ~1.5-2x faster than `rev_for_each_intra_leaf_batch` when
    /// the visitor would ignore the key parameter anyway.
    ///
    /// # Start Bound Behavior (Reverse Iteration)
    ///
    /// - `Unbounded`: Exact (scans all entries)
    /// - `Included`/`Excluded`: **Approximate** for keys with suffix
    ///
    /// For bounded scans, the start check uses ikey comparison only. This means:
    /// - Keys where `ikey > bound_ikey`: correctly included
    /// - Keys where `ikey < bound_ikey`: correctly excluded
    /// - Keys where `ikey == bound_ikey`: **may over-include** entries
    ///
    /// If you need exact start bounds with long keys, use `rev_for_each_intra_leaf_batch`.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `P::Output`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn rev_for_each_values_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(P::Output) -> bool,
    {
        if self.back_exhausted() {
            return 0;
        }

        // Lazy initialization of back cursor
        if !self.back_initialized() {
            self.initialize_back();

            if self.back_exhausted() {
                return 0;
            }
        }

        // Pre-extract start bound ikey for fast comparison (reverse uses start as lower bound)
        let start_bound_ikey: Option<u64> = self.start_bound.extract_ikey();

        // SAFETY: rev is Some after initialize_back
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let mut strategy = RevValuesOnlyStrategy::new(&mut visitor);
        rev.run_batch_reverse(
            &mut strategy,
            &self.start_bound,
            start_bound_ikey,
            self.guard,
        )
    }
}
