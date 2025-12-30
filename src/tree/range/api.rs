//! Filepath: src/tree/range/api.rs
//!
//! Public API methods for range scans on [`MassTreeGeneric`].

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

    /// Scan a range with zero-copy value references.
    ///
    /// Unlike [`scan`] which clones values (Arc increment for `MassTree`),
    /// this returns borrowed `&V` references. This eliminates 2 atomic
    /// operations per entry, significantly improving scan throughput.
    ///
    /// # Performance
    ///
    /// For `MassTree<V>` (Arc-based): 2-3x faster than `scan()` for scan-heavy workloads.
    /// For `MassTree24Inline<V>` (Copy-based): Similar performance to `scan()`.
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
