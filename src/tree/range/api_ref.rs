//! Filepath: `src/tree/range/api_ref.rs`
//!
//! Reference-based range scan methods for [`crate::MassTreeGeneric`].
//!
//! These methods require [`RefValueSlot`] because they return `&V` references.
//! True-inline storage (`TrueInlineSlot<V>`) does NOT implement [`RefValueSlot`],
//! so these methods are compile-time unavailable for `MassTree15Inline`/`MassTree24Inline`.
//!
//! # Why This Separation?
//!
//! True-inline storage encodes values as bits in pointer addresses (XOR with magic constant).
//! The "pointer" returned by `output_to_raw()` is NOT a valid memory address.
//! Dereferencing it causes SIGSEGV. By gating `_ref` methods behind `RefValueSlot`,
//! we get compile-time prevention of this unsoundness.
//!
//! # Safe Alternatives for Inline Storage
//!
//! Use [`MassTreeGeneric::scan`] instead, which returns owned values by copy.

use seize::LocalGuard;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::LayerCapableLeaf;
use crate::ref_value_slot::RefValueSlot;
use crate::slot::ValueSlot;
use crate::tree::MassTreeGeneric;

use super::iterator::RangeBound;

// ============================================================================
//  Reference-Based Scan API (requires RefValueSlot)
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot + RefValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Scan a range with zero-copy value references.
    ///
    /// Unlike [`Self::scan`] which clones values (Arc increment for `MassTree`),
    /// this returns borrowed `&V` references. This eliminates 2 atomic
    /// operations per entry, significantly improving scan throughput.
    ///
    /// # Availability
    ///
    /// This method is only available for pointer-backed storage modes:
    /// - `MassTree15<V>` (Arc-based)
    /// - `MassTree24<V>` (Arc-based)
    /// - `MassTree24Inline<V>` (Box/index-based)
    ///
    /// It is **NOT** available for true-inline storage (`MassTree15Inline`).
    /// Use [`Self::scan`] instead for inline storage.
    ///
    /// # Performance
    ///
    /// For `MassTree<V>` (Arc-based): 2-3x faster than `scan()` for scan-heavy workloads.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], &V) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited (including the last one if stopped early).
    ///
    /// # Safety
    ///
    /// The value references passed to the visitor are only valid for the
    /// duration of the callback. Do not store them. The guard ensures
    /// the underlying data isn't deallocated during iteration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    ///
    /// tree.scan_ref(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += *value;  // Direct access, no Arc overhead
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_ref<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        // Use zero-copy for_each_ref internally
        self.range(start, end, guard).for_each_ref(visitor)
    }

    /// Batch scan with zero-copy value references and reduced dispatch overhead.
    ///
    /// This is a performance-optimized variant of [`Self::scan_ref`] that reduces
    /// state machine dispatch overhead while maintaining identical correctness
    /// guarantees.
    ///
    /// # Availability
    ///
    /// Only available for pointer-backed storage. See [`Self::scan_ref`] for details.
    ///
    /// # Correctness
    ///
    /// This method delegates to [`super::iterator::RangeIter::for_each_batch_ref`], which:
    /// - Uses per-entry OCC validation (same as `scan_ref`)
    /// - Properly updates cursor key for duplicate filtering
    /// - Handles layer transitions correctly (including start-bound descent)
    ///
    /// # Performance
    ///
    /// Expected 1.3-1.5x improvement over `scan_ref` for large scans, from:
    /// - Eliminating `match state {}` dispatch overhead
    /// - Inlining the common `FindNext` → `Emit` path
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], &V) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited (including the last one if stopped early).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    ///
    /// tree.scan_batch_ref(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += *value;
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_batch_ref<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        // Delegate to RangeIter::for_each_batch_ref which has correct initialization
        self.range(start, end, guard).for_each_batch_ref(visitor)
    }

    /// Intra-leaf batch scan with maximum performance.
    ///
    /// This is the highest-performance scan method. It processes entire
    /// leaves in tight loops, minimizing per-entry overhead.
    ///
    /// # Availability
    ///
    /// Only available for pointer-backed storage. See [`Self::scan_ref`] for details.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to next leaf
    /// - Single OCC validation per leaf (vs per-entry in `scan_batch_ref`)
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions (sublayers)
    ///
    /// Expected 2-3x improvement over `scan_batch_ref` for large scans.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], &V) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    ///
    /// // Fastest scan for large ranges
    /// tree.scan_intra_leaf_batch_ref(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += *value;
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_intra_leaf_batch_ref<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        self.range(start, end, guard)
            .for_each_intra_leaf_batch_ref(visitor)
    }

    /// Highest-performance reverse iteration with zero-copy value references.
    ///
    /// This is the batch-optimized reverse scan method, equivalent to
    /// [`Self::scan_intra_leaf_batch_ref`] but iterating in descending key order.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to previous leaf
    /// - Single OCC validation per leaf
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions
    ///
    /// Expected 2-3x improvement over standard `iter().rev()` iteration.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound (lower bound for reverse = stopping point)
    /// - `end`: End bound (upper bound for reverse = starting point)
    /// - `visitor`: Callback function `fn(&[u8], &V) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    ///
    /// // Fastest reverse scan for large ranges
    /// tree.scan_rev_batch_ref(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += *value;
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_rev_batch_ref<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        self.range(start, end, guard).rev_for_each_ref(visitor)
    }
}
