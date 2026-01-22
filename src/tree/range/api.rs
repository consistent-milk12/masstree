//! Filepath: src/tree/range/api.rs
//!
//! Public API methods for range scans on [`crate::MassTreeGeneric`].

use seize::LocalGuard;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::LayerCapableLeaf;
use crate::slot::ValueSlot;
use crate::tree::MassTreeGeneric;

use super::iterator::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};

// ============================================================================
//  Range Scan API for MassTreeGeneric
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // ========================================================================
    //  Iterator API
    // ========================================================================

    /// Create an iterator over a key range.
    ///
    /// Returns an iterator that yields [`ScanEntry`] items containing
    /// owned keys and cloned values in lexicographic order.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A [`RangeIter`] that yields entries in the specified range.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    ///
    /// for entry in tree.range(
    ///     RangeBound::Included(b"aaa"),
    ///     RangeBound::Excluded(b"zzz"),
    ///     &guard
    /// ) {
    ///     println!("{:?} -> {:?}", entry.key, entry.value);
    /// }
    ///
    pub fn range<'a, 'g>(
        &'a self,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> RangeIter<'a, 'g, S, L, A> {
        self.verify_guard(guard);
        RangeIter::new(self, start, end, guard)
    }

    /// Create an iterator over all entries.
    ///
    /// Equivalent to `range(RangeBound::Unbounded, RangeBound::Unbounded, guard)`.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A [`RangeIter`] that yields all entries in the tree.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let count = tree.iter(&guard).count();
    /// println!("Tree has {} entries", count);
    ///
    pub fn iter<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> RangeIter<'a, 'g, S, L, A> {
        self.range(RangeBound::Unbounded, RangeBound::Unbounded, guard)
    }

    /// Create an iterator over all keys.
    ///
    /// Returns an iterator that yields owned key `Vec<u8>` values.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A [`KeysIter`] that yields all keys in the tree.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let keys: Vec<Vec<u8>> = tree.keys(&guard).collect();
    ///
    pub fn keys<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> KeysIter<'a, 'g, S, L, A> {
        self.iter(guard).keys()
    }

    /// Create an iterator over all values.
    ///
    /// Returns an iterator that yields cloned values.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A [`ValuesIter`] that yields all values in the tree.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// let values: Vec<Arc<String>> = tree.values(&guard).collect();
    ///
    pub fn values<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> ValuesIter<'a, 'g, S, L, A> {
        self.iter(guard).values()
    }

    // ========================================================================
    //  First / Last Access
    // ========================================================================

    /// Get the first (smallest) key-value pair in the tree.
    ///
    /// Creates a guard internally. For repeated access, prefer
    /// [`first_with_guard`](Self::first_with_guard).
    ///
    /// # Returns
    ///
    /// * `Some(ScanEntry)` - The entry with the lexicographically smallest key
    /// * `None` - If the tree is empty
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree::<u64>::new();
    /// tree.insert(b"banana", 2).unwrap();
    /// tree.insert(b"apple", 1).unwrap();
    /// tree.insert(b"cherry", 3).unwrap();
    ///
    /// let first = tree.first().unwrap();
    /// assert_eq!(first.key(), b"apple");
    /// ```
    #[must_use]
    #[inline]
    pub fn first(&self) -> Option<ScanEntry<S::Output>> {
        let guard = self.guard();
        self.first_with_guard(&guard)
    }

    /// Get the first (smallest) key-value pair using an existing guard.
    #[must_use]
    #[inline]
    pub fn first_with_guard<'a>(&'a self, guard: &LocalGuard<'a>) -> Option<ScanEntry<S::Output>> {
        self.iter(guard).next()
    }

    /// Get the last (largest) key-value pair in the tree.
    ///
    /// Creates a guard internally. For repeated access, prefer
    /// [`last_with_guard`](Self::last_with_guard).
    ///
    /// # Returns
    ///
    /// * `Some(ScanEntry)` - The entry with the lexicographically largest key
    /// * `None` - If the tree is empty
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree::<u64>::new();
    /// tree.insert(b"banana", 2).unwrap();
    /// tree.insert(b"apple", 1).unwrap();
    /// tree.insert(b"cherry", 3).unwrap();
    ///
    /// let last = tree.last().unwrap();
    /// assert_eq!(last.key(), b"cherry");
    /// ```
    #[must_use]
    #[inline]
    pub fn last(&self) -> Option<ScanEntry<S::Output>> {
        let guard = self.guard();
        self.last_with_guard(&guard)
    }

    /// Get the last (largest) key-value pair using an existing guard.
    #[must_use]
    #[inline]
    pub fn last_with_guard<'a>(&'a self, guard: &LocalGuard<'a>) -> Option<ScanEntry<S::Output>> {
        self.iter(guard).next_back()
    }

    // ========================================================================
    //  Visitor API
    // ========================================================================

    /// Scan a range with a visitor callback.
    ///
    /// The visitor receives borrowed key bytes and cloned value output.
    /// Return `false` from the visitor to stop scanning early.
    ///
    /// This is more efficient than the iterator API when you don't need
    /// to own the keys, as it avoids allocating `Vec<u8>` for each key.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], S::Output) -> bool`
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
    /// let mut count = 0;
    ///
    /// tree.scan(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |key, value| {
    ///         count += 1;
    ///         println!("Key {:?} -> {:?}", key, value);
    ///         count < 100 // Stop after 100 entries
    ///     },
    ///     &guard
    /// );
    ///
    pub fn scan<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        // Use zero-allocation for_each internally
        self.range(start, end, guard).for_each(visitor)
    }

    /// Highest-performance batch-optimized range scan.
    ///
    /// This is the fastest scan method for all storage types including inline.
    /// Unlike [`scan`](Self::scan), this method processes entries in batches
    /// within each leaf node, reducing per-entry overhead.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to next leaf
    /// - Single OCC validation per leaf (vs per-entry in `scan`)
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions (sublayers)
    ///
    /// # Availability
    ///
    /// Available for ALL storage types including:
    /// - `MassTree15<V>` (Arc-based)
    /// - `MassTree24<V>` (Arc-based)
    /// - `MassTree15Inline<V>` (true-inline)
    /// - `MassTree24Inline<V>` (Box/index-based)
    ///
    /// For pointer-backed storage that can return references, consider
    /// [`scan_intra_leaf_batch_ref`](Self::scan_intra_leaf_batch_ref) in `api_ref.rs`
    /// which avoids cloning values.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], S::Output) -> bool`
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
    /// tree.scan_intra_leaf_batch(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += value;
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_intra_leaf_batch<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        self.range(start, end, guard)
            .for_each_intra_leaf_batch(visitor)
    }

    /// Highest-performance batch-optimized reverse range scan.
    ///
    /// This is the fastest reverse scan method for all storage types including inline.
    /// Unlike [`scan_intra_leaf_batch`](Self::scan_intra_leaf_batch), this iterates
    /// in descending key order.
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
    /// Available for ALL storage types including:
    /// - `MassTree15<V>` (Arc-based)
    /// - `MassTree24<V>` (Arc-based)
    /// - `MassTree15Inline<V>` (true-inline)
    /// - `MassTree24Inline<V>` (Box/index-based)
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound (lower bound - stopping point for reverse)
    /// - `end`: End bound (upper bound - starting point for reverse)
    /// - `visitor`: Callback function `fn(&[u8], S::Output) -> bool`
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
    /// tree.scan_rev_batch(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |_key, value| {
    ///         sum += value;
    ///         true
    ///     },
    ///     &guard
    /// );
    /// ```
    pub fn scan_rev_batch<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        self.range(start, end, guard)
            .rev_for_each_intra_leaf_batch(visitor)
    }

    /// Highest-performance value-only scan (no key materialization).
    ///
    /// This is the fastest scan method when you only need values. Keys are not
    /// built or copied, saving up to 56 bytes of copying per entry for long keys.
    ///
    /// # Performance
    ///
    /// For 64-byte keys: ~1.5-2x faster than `scan_intra_leaf_batch` when
    /// the visitor would ignore the key parameter anyway.
    ///
    /// # End Bound Behavior
    ///
    /// - `Unbounded`: Exact (scans all entries)
    /// - `Included`/`Excluded`: **Approximate** for keys with suffix
    ///
    /// For bounded scans, the end check uses ikey comparison only. This means:
    /// - Keys where `ikey < bound_ikey`: correctly included
    /// - Keys where `ikey > bound_ikey`: correctly excluded
    /// - Keys where `ikey == bound_ikey`: **may over-include** entries
    ///
    /// If you need exact end bounds with long keys, use `scan_intra_leaf_batch`.
    ///
    /// # When to Use
    ///
    /// - Aggregations (sum, count, min, max)
    /// - Existence checks
    /// - Any scan where you process values but don't need keys
    ///
    /// # When NOT to Use
    ///
    /// - When you need the key for each entry
    /// - When exact end bound semantics matter for long keys
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(S::Output) -> bool`
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
    /// // Fastest: unbounded value scan
    /// tree.scan_values(
    ///     RangeBound::Unbounded,
    ///     RangeBound::Unbounded,
    ///     |value| { sum += value; true },
    ///     &guard
    /// );
    /// ```
    pub fn scan_values<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(S::Output) -> bool,
    {
        self.range(start, end, guard).for_each_values_batch(visitor)
    }

    /// Highest-performance reverse value-only scan (no key materialization).
    ///
    /// This is the fastest reverse scan method when you only need values.
    /// Same as [`scan_values`](Self::scan_values) but iterates in descending order.
    ///
    /// # Performance
    ///
    /// For 64-byte keys: ~1.5-2x faster than `scan_rev_batch` when
    /// the visitor would ignore the key parameter anyway.
    ///
    /// # Start Bound Behavior (Reverse Iteration)
    ///
    /// - `Unbounded`: Exact (scans all entries)
    /// - `Included`/`Excluded`: **Approximate** for keys with suffix
    ///
    /// If you need exact start bounds with long keys, use `scan_rev_batch`.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound (lower bound - stopping point for reverse)
    /// - `end`: End bound (upper bound - starting point for reverse)
    /// - `visitor`: Callback function `fn(S::Output) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    pub fn scan_values_rev<F>(
        &self,
        start: RangeBound<'_>,
        end: RangeBound<'_>,
        visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(S::Output) -> bool,
    {
        self.range(start, end, guard)
            .rev_for_each_values_batch(visitor)
    }

    /// Scan all entries with a prefix.
    ///
    /// Convenience method for scanning all keys that start with a given prefix.
    ///
    /// # Arguments
    ///
    /// - `prefix`: The key prefix to match
    /// - `visitor`: Callback function `fn(&[u8], S::Output) -> bool`
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
    ///
    /// tree.scan_prefix(b"user:", |key, value| {
    ///     println!("User key: {:?}", key);
    ///     true // Continue
    /// }, &guard);
    ///
    pub fn scan_prefix<F>(&self, prefix: &[u8], mut visitor: F, guard: &LocalGuard<'_>) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        // Compute exclusive upper bound
        // This is the prefix with its last byte incremented
        // e.g., "abc" -> "abd", "ab\xff" -> "ac", etc.
        let upper_bound: Option<Vec<u8>> = compute_prefix_upper_bound(prefix);

        let end: RangeBound<'_> = upper_bound
            .as_ref()
            .map_or(RangeBound::Unbounded, |bound| RangeBound::Excluded(bound));

        // Use zero-allocation for_each with prefix check
        self.range(RangeBound::Included(prefix), end, guard)
            .for_each(|key, value| {
                // Double-check prefix match (handles edge cases)
                if !key.starts_with(prefix) {
                    return false;
                }
                visitor(key, value)
            })
    }

    // ========================================================================
    //  Convenience Collectors
    // ========================================================================

    /// Collect all entries into a Vec.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A vector of all entries in the tree.
    pub fn collect_entries(&self, guard: &LocalGuard<'_>) -> Vec<ScanEntry<S::Output>> {
        self.iter(guard).collect()
    }

    /// Collect all keys into a Vec.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A vector of all keys in the tree.
    pub fn collect_keys(&self, guard: &LocalGuard<'_>) -> Vec<Vec<u8>> {
        self.keys(guard).collect()
    }

    /// Collect all values into a Vec.
    ///
    /// # Arguments
    ///
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A vector of all values in the tree.
    pub fn collect_values(&self, guard: &LocalGuard<'_>) -> Vec<S::Output> {
        self.values(guard).collect()
    }
}

// ============================================================================
//  Helper Functions
// ============================================================================

/// Compute the exclusive upper bound for a prefix scan.
///
/// Returns `None` if the prefix cannot be incremented (all 0xFF bytes).
#[expect(clippy::indexing_slicing, reason = "Checked")]
fn compute_prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None; // Unbounded
    }

    let mut upper = prefix.to_vec();

    // Find the rightmost byte that can be incremented
    for i in (0..upper.len()).rev() {
        if upper[i] < 0xFF {
            upper[i] += 1;
            upper.truncate(i + 1);

            return Some(upper);
        }
    }

    // All bytes are 0xFF, no upper bound possible
    None
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
