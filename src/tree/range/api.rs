//! Filepath: src/tree/range/api.rs
//!
//! Public API methods for range scans on [`crate::MassTreeGeneric`].

use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;
use crate::key::{IKEY_SIZE, MAX_KEY_LENGTH};
use crate::leaf15::{LeafNode15, LAYER_KEYLENX};
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::tree::MassTreeGeneric;
use crate::Permuter;

use super::cursor_key::CursorKey;
use super::helper::{lower_with_position, KeyIndexedPosition};
use super::iterator::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
use super::traversal::reach_leaf_for_scan;

// ============================================================================
//  Range Scan API for MassTreeGeneric
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
    ) -> RangeIter<'a, 'g, P, A> {
        self.verify_guard(guard);
        RangeIter::new(self, start, end, guard)
    }

    /// Create a forward-only range iterator (skips backward state initialization).
    ///
    /// Used internally by forward-only scan methods (`scan`, `scan_prefix`,
    /// `scan_intra_leaf_batch`, `scan_values`) to avoid initializing ~300 bytes
    /// of backward iteration state that will never be accessed.
    pub(crate) fn range_forward<'a, 'g>(
        &'a self,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> RangeIter<'a, 'g, P, A> {
        self.verify_guard(guard);
        RangeIter::new_forward_only(self, start, end, guard)
    }

    /// Create a forward-only iterator rooted at a specific sublayer.
    pub(crate) fn range_forward_from_root<'a, 'g>(
        &'a self,
        layer_root: *const u8,
        cursor_key: CursorKey,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> RangeIter<'a, 'g, P, A> {
        self.verify_guard(guard);
        RangeIter::new_forward_only_from_root(layer_root, cursor_key, start, end, guard)
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
    pub fn iter<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> RangeIter<'a, 'g, P, A> {
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
    pub fn keys<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> KeysIter<'a, 'g, P, A> {
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
    /// let values: Vec<ValuePtr<String>> = tree.values(&guard).collect();
    ///
    pub fn values<'a, 'g>(&'a self, guard: &'g LocalGuard<'a>) -> ValuesIter<'a, 'g, P, A> {
        self.iter(guard).values()
    }

    // ========================================================================
    //  First / Last Access
    // ========================================================================

    /// Get the first (smallest) key-value pair in the tree.
    ///
    /// Creates a guard internally. Returns an owned clone of the value.
    /// For repeated access, prefer [`first_with_guard`](Self::first_with_guard).
    ///
    /// # Returns
    ///
    /// * `Some(ScanEntry)` - The entry with the lexicographically smallest key
    /// * `None` - If the tree is empty
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree15::<u64>::new();
    /// tree.insert(b"banana", 2);
    /// tree.insert(b"apple", 1);
    /// tree.insert(b"cherry", 3);
    ///
    /// let first = tree.first().unwrap();
    /// assert_eq!(first.key(), b"apple");
    /// ```
    #[must_use]
    #[inline]
    pub fn first(&self) -> Option<ScanEntry<P::Value>>
    where
        P::Value: Clone,
    {
        let guard = self.guard();
        self.first_with_guard(&guard)
            .map(|entry| ScanEntry::new(entry.key, P::clone_value_from_output(&entry.value)))
    }

    /// Get the first (smallest) key-value pair using an existing guard.
    #[must_use]
    #[inline]
    pub fn first_with_guard<'a>(&'a self, guard: &LocalGuard<'a>) -> Option<ScanEntry<P::Output>> {
        self.iter(guard).next()
    }

    /// Get the last (largest) key-value pair in the tree.
    ///
    /// Creates a guard internally. Returns an owned clone of the value.
    /// For repeated access, prefer [`last_with_guard`](Self::last_with_guard).
    ///
    /// # Returns
    ///
    /// * `Some(ScanEntry)` - The entry with the lexicographically largest key
    /// * `None` - If the tree is empty
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree15::<u64>::new();
    /// tree.insert(b"banana", 2);
    /// tree.insert(b"apple", 1);
    /// tree.insert(b"cherry", 3);
    ///
    /// let last = tree.last().unwrap();
    /// assert_eq!(last.key(), b"cherry");
    /// ```
    #[must_use]
    #[inline]
    pub fn last(&self) -> Option<ScanEntry<P::Value>>
    where
        P::Value: Clone,
    {
        let guard = self.guard();
        self.last_with_guard(&guard)
            .map(|entry| ScanEntry::new(entry.key, P::clone_value_from_output(&entry.value)))
    }

    /// Get the last (largest) key-value pair using an existing guard.
    #[must_use]
    #[inline]
    pub fn last_with_guard<'a>(&'a self, guard: &LocalGuard<'a>) -> Option<ScanEntry<P::Output>> {
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
    /// - `visitor`: Callback function `fn(&[u8], P::Output) -> bool`
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
        F: FnMut(&[u8], P::Output) -> bool,
    {
        self.range_forward(start, end, guard).for_each(visitor)
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
    /// - `MassTree15Inline<V>` (true-inline)
    ///
    /// For pointer-backed storage that can return references, consider
    /// [`scan_intra_leaf_batch_ref`](Self::scan_intra_leaf_batch_ref) in `api_ref.rs`
    /// which avoids cloning values.
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `visitor`: Callback function `fn(&[u8], P::Output) -> bool`
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
        F: FnMut(&[u8], P::Output) -> bool,
    {
        self.range_forward(start, end, guard)
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
    /// - `MassTree15Inline<V>` (true-inline)
    ///
    /// # Arguments
    ///
    /// - `start`: Start bound (lower bound - stopping point for reverse)
    /// - `end`: End bound (upper bound - starting point for reverse)
    /// - `visitor`: Callback function `fn(&[u8], P::Output) -> bool`
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
        F: FnMut(&[u8], P::Output) -> bool,
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
    /// - `visitor`: Callback function `fn(P::Output) -> bool`
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
        F: FnMut(P::Output) -> bool,
    {
        self.range_forward(start, end, guard)
            .for_each_values_batch(visitor)
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
    /// - `visitor`: Callback function `fn(P::Output) -> bool`
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
        F: FnMut(P::Output) -> bool,
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
    /// - `visitor`: Callback function `fn(&[u8], P::Output) -> bool`
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
    ///```
    ///
    /// # Panics
    /// Invariant check.
    pub fn scan_prefix<F>(&self, prefix: &[u8], mut visitor: F, guard: &LocalGuard<'_>) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        self.scan_prefix_inner(prefix, guard, |exact_value, iter| {
            let mut count = 0;
            if let Some(value) = exact_value {
                count += 1;
                if !visitor(prefix, value) {
                    return count;
                }
            }
            count + iter.for_each_intra_leaf_batch(visitor)
        })
    }

    /// Value-only prefix scan (no key materialization).
    ///
    /// Like [`scan_prefix`](Self::scan_prefix) but skips building key bytes,
    /// saving up to 56 bytes of copying per entry for long keys.
    ///
    /// # End Bound Accuracy
    ///
    /// - For ikey-aligned prefixes (multiples of 8 bytes): **exact**
    /// - For non-aligned prefixes: **approximate** (may over-include entries
    ///   sharing the same ikey as the boundary)
    ///
    /// # Arguments
    ///
    /// - `prefix`: The key prefix to match
    /// - `visitor`: Callback function `fn(P::Output) -> bool`
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Panics
    ///
    /// Panics if `prefix.len()` exceeds `MAX_KEY_LENGTH` (256 bytes).
    pub fn scan_prefix_values<F>(
        &self,
        prefix: &[u8],
        mut visitor: F,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        F: FnMut(P::Output) -> bool,
    {
        self.scan_prefix_inner(prefix, guard, |exact_value, iter| {
            let mut count = 0;
            if let Some(value) = exact_value {
                count += 1;
                if !visitor(value) {
                    return count;
                }
            }
            count + iter.for_each_values_batch(visitor)
        })
    }

    // ========================================================================
    //  Shared Prefix Scan Logic
    // ========================================================================

    /// Shared implementation for `scan_prefix` and `scan_prefix_values`.
    ///
    /// Performs prefix validation, upper-bound computation, and trie-aware
    /// fast-path descent. Delegates visitor-specific logic to `scan_fn`:
    /// - First argument: `Some(value)` if the exact prefix key exists at a
    ///   chunk boundary, `None` otherwise.
    /// - Second argument: a forward-only `RangeIter` positioned for the scan.
    #[inline]
    fn scan_prefix_inner(
        &self,
        prefix: &[u8],
        guard: &LocalGuard<'_>,
        scan_fn: impl FnOnce(Option<P::Output>, RangeIter<'_, '_, P, A>) -> usize,
    ) -> usize {
        assert!(
            prefix.len() <= MAX_KEY_LENGTH,
            "key length {} exceeds maximum {}",
            prefix.len(),
            MAX_KEY_LENGTH
        );

        // Compute exclusive upper bound on the stack (no heap allocation).
        // Increments the rightmost non-0xFF byte: "abc" -> "abd", "ab\xff" -> "ac".
        let mut upper_buf = [0u8; MAX_KEY_LENGTH];
        let upper_len = compute_prefix_upper_bound_into(prefix, &mut upper_buf);

        let end: RangeBound<'_> = upper_len.map_or(RangeBound::Unbounded, |len| {
            RangeBound::Excluded(&upper_buf[..len])
        });

        // Trie-aware fast path: descend through exact 8-byte chunks when
        // matching layer pointers exist, then scan from that sublayer root.
        if let Some((layer_root, descended_chunks)) = self.descend_prefix_layers(prefix, guard)
            && descended_chunks > 0
        {
            let mut cursor = CursorKey::from_slice(prefix);
            for _ in 0..descended_chunks {
                if cursor.has_suffix() {
                    cursor.shift();
                } else {
                    cursor.shift_clear();
                }
            }

            let prefix_at_chunk_boundary = prefix.len() == descended_chunks * IKEY_SIZE;

            if prefix_at_chunk_boundary {
                let exact_value = self.get_with_guard(prefix, guard);
                let iter = self.range_forward_from_root(
                    layer_root,
                    cursor,
                    RangeBound::Unbounded,
                    RangeBound::Unbounded,
                    guard,
                );
                return scan_fn(exact_value, iter);
            }

            let iter = self.range_forward_from_root(
                layer_root,
                cursor,
                RangeBound::Included(prefix),
                end,
                guard,
            );
            return scan_fn(None, iter);
        }

        let iter = self.range_forward(RangeBound::Included(prefix), end, guard);
        scan_fn(None, iter)
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
    pub fn collect_entries(&self, guard: &LocalGuard<'_>) -> Vec<ScanEntry<P::Output>> {
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
    pub fn collect_values(&self, guard: &LocalGuard<'_>) -> Vec<P::Output> {
        self.values(guard).collect()
    }
}

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Descend through as many full 8-byte prefix chunks as possible.
    ///
    /// Returns the sublayer root and number of consumed 8-byte chunks.
    fn descend_prefix_layers(
        &self,
        prefix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> Option<(*const u8, usize)> {
        let full_chunks: usize = prefix.len() / IKEY_SIZE;

        if full_chunks == 0 {
            return None;
        }

        let mut current_root = self.load_root_ptr_generic(guard);
        let mut descended = 0usize;

        while descended < full_chunks {
            let chunk_ikey = read_full_chunk_ikey(prefix, descended);

            let Some(next_root) = find_layer_child_root::<P>(current_root, chunk_ikey, guard)
            else {
                break;
            };

            current_root = next_root;
            descended += 1;
        }

        Some((current_root, descended))
    }
}

// ============================================================================
//  Helper Functions
// ============================================================================

/// Find child sublayer root for an exact ikey layer-pointer entry in `root`.
fn find_layer_child_root<P>(
    root: *const u8,
    chunk_ikey: u64,
    guard: &LocalGuard<'_>,
) -> Option<*const u8>
where
    P: LeafPolicy,
{
    if root.is_null() {
        return None;
    }

    let cursor = CursorKey::from_slice(&chunk_ikey.to_be_bytes());
    let leaf_ptr: *mut LeafNode15<P> = reach_leaf_for_scan::<P>(root, &cursor, guard);

    if leaf_ptr.is_null() {
        return None;
    }

    // SAFETY: leaf_ptr is protected by guard and null-checked above.
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };
    let version: u32 = leaf.version().stable();

    if NodeVersion::is_deleted_version(version) {
        return None;
    }

    let perm: Permuter = leaf.permutation();
    let kx: KeyIndexedPosition = lower_with_position(&cursor, leaf, &perm);
    let _ = kx.p?;

    let mut i: usize = kx.i;

    while i < perm.size() {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey != chunk_ikey {
            break;
        }

        if leaf.keylenx(slot) >= LAYER_KEYLENX && !leaf.is_value_empty(slot) {
            let layer_ptr: *const u8 = leaf.load_layer_raw(slot).cast_const();

            if leaf.version().has_changed(version) {
                return None;
            }

            if !layer_ptr.is_null() {
                return Some(layer_ptr);
            }

            return None;
        }

        i += 1;
    }

    if leaf.version().has_changed(version) {
        return None;
    }

    None
}

#[expect(clippy::indexing_slicing, reason = "chunk bounds are caller-checked")]
fn read_full_chunk_ikey(prefix: &[u8], chunk_idx: usize) -> u64 {
    let start: usize = chunk_idx * IKEY_SIZE;
    let end: usize = start + IKEY_SIZE;

    #[expect(clippy::expect_used, reason = "slice length is guaranteed to be 8")]
    let bytes: [u8; IKEY_SIZE] = prefix[start..end].try_into().expect("slice is 8 bytes");

    u64::from_be_bytes(bytes)
}

/// Compute the exclusive upper bound for a prefix scan into a caller-provided buffer.
///
/// Copies the prefix into `buf`, then increments the rightmost non-0xFF byte.
/// Returns `Some(len)` with the length of the upper bound, or `None` if the prefix
/// is empty or all 0xFF bytes (unbounded).
///
/// This avoids heap allocation by writing directly into a stack buffer.
#[expect(clippy::indexing_slicing, reason = "Checked")]
fn compute_prefix_upper_bound_into(prefix: &[u8], buf: &mut [u8; MAX_KEY_LENGTH]) -> Option<usize> {
    assert!(
        prefix.len() <= MAX_KEY_LENGTH,
        "key length {} exceeds maximum {}",
        prefix.len(),
        MAX_KEY_LENGTH
    );

    if prefix.is_empty() {
        return None; // Unbounded
    }

    buf[..prefix.len()].copy_from_slice(prefix);

    // Find the rightmost byte that can be incremented
    for i in (0..prefix.len()).rev() {
        if buf[i] < 0xFF {
            buf[i] += 1;
            return Some(i + 1);
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
