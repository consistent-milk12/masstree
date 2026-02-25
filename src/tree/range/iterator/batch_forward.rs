//! Filepath: `src/tree/range/iterator/batch_forward.rs`
//!
//! Forward batch iteration methods for maximum performance.

use crate::alloc_trait::TreeAllocator;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;

use super::RangeIter;

use crate::tree::range::forward_ctx::{
    IntraLeafCopyStrategy, IntraLeafRefStrategy, ValuesOnlyStrategy,
};
use crate::tree::range::scan_state::ScanState;

impl<P, A> RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Zero-allocation iteration with a visitor closure.
    ///
    /// This is significantly faster than the `Iterator` trait because it:
    /// - Avoids allocating `Vec<u8>` for each key
    /// - Uses references directly from internal buffers
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], P::Output)`. Return `true` to continue,
    ///   `false` to stop early.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    #[inline]
    #[must_use = "returns the number of entries visited"]
    pub fn for_each<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Fast path: process current entry without allocation
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

    /// Zero-copy iteration with borrowed value references.
    ///
    /// Unlike [`Self::for_each`] which clones values (Arc increment for `LeafValue`),
    /// this returns `&P::Value` references tied to the guard lifetime.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &P::Value)`. Return `true` to continue,
    ///   `false` to stop early.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Lifetime Guarantees
    ///
    /// References are borrowed from internal buffers and protected by the guard.
    /// The closure signature prevents storing them beyond the callback scope.
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

        // Lazy initialization
        if !self.fwd.flags.initialized() {
            self.initialize();
            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Use the zero-copy advance method
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

    /// Batch iteration with zero-copy value references and reduced dispatch overhead.
    ///
    /// This is the highest-performance iteration method. It eliminates state machine
    /// dispatch overhead while maintaining identical correctness to [`Self::for_each_ref`].
    ///
    /// # Correctness
    ///
    /// Unlike approaches that validate only once per leaf, this method:
    /// - Uses per-entry OCC validation (same as `for_each_ref`)
    /// - Properly updates cursor key for duplicate filtering
    /// - Handles layer transitions correctly (dynamically switches from single-layer
    ///   to multi-layer mode when `Down` is encountered)
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
    #[expect(clippy::too_many_lines, reason = "Complex state management logic")]
    pub fn for_each_batch_ref<F>(mut self, mut visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        if self.fwd.flags.exhausted() {
            return 0;
        }

        // Lazy initialization - reuses existing RangeIter::initialize()
        // which correctly handles start-bound descent (shift vs shift_clear)
        if !self.fwd.flags.initialized() {
            self.initialize();
            if self.fwd.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // NOTE: We don't use advance_no_alloc_ref here because it has issues
        // with multi-layer keys. Instead, we use the batch loop for all entries
        // which correctly handles cursor_key updates via find_next_ptr.

        // If state is Emit with a snapshot from initialize(), handle it specially
        // by extracting the snapshot and emitting directly
        if self.fwd.state == ScanState::Emit {
            if let Some(snapshot) = self.fwd.snapshot.take() {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key: &[u8] = unsafe { self.fwd.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.fwd.flags.mark_exhausted();
                    return 0;
                }

                // Borrow value directly from snapshot (no raw pointer conversion)
                let value_ref: &P::Value = P::output_as_ref(&snapshot.value);
                let should_continue = {
                    count += 1;
                    visitor(key, value_ref)
                };

                if !should_continue {
                    return count;
                }
            }
            self.fwd.state = ScanState::FindNext;
        }

        // Main batch loop - uses find_next_ptr which correctly updates cursor_key

        loop {
            // ================================================================
            // Handle rare states (layer transitions, retries, exhaustion)
            // ================================================================

            // Handle pending state transitions first (like advance_no_alloc_ref)
            match self.fwd.state {
                ScanState::Down => {
                    self.fwd.flags.disable_single_layer_mode();
                    self.fwd.handle_down();
                    self.fwd.state = ScanState::Retry;
                    self.fwd.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Up => {
                    if !self.fwd.handle_up(self.guard) {
                        self.fwd.flags.mark_exhausted();

                        return count;
                    }

                    self.fwd.state = ScanState::FindNext;
                    self.fwd.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Retry => {
                    self.fwd.state = self.fwd.find_retry(self.guard);
                    self.fwd.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.fwd.stack.is_null() {
                if self.fwd.layer_stack.is_empty() {
                    self.fwd.flags.mark_exhausted();
                    return count;
                }

                self.fwd.state = ScanState::Up;

                continue;
            }

            // Check leaf deletion
            let leaf: &LeafNode15<P> = unsafe { self.fwd.stack.leaf_ref() };

            if leaf.version().is_deleted() {
                self.fwd.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // Main hot path: FindNext → Emit (inlined)
            // ================================================================

            let (new_state, snapshot_ptr) = if self.fwd.flags.needs_duplicate_check() {
                self.fwd.flags.clear_duplicate_check();
                self.fwd.find_next_with_dup_check_ptr(self.guard)
            } else {
                self.fwd.find_next_ptr(self.guard)
            };

            self.fwd.state = new_state;

            match new_state {
                ScanState::Emit => {
                    if let Some(snap) = snapshot_ptr {
                        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                        let key: &[u8] = unsafe { self.fwd.cursor_key.full_key_unchecked() };

                        // Check end bound
                        if !self.end_bound.contains(key) {
                            self.fwd.flags.mark_exhausted();
                            return count;
                        }

                        // SAFETY: find_next_ptr validated version, guard protects pointer
                        let value_ref: &P::Value = unsafe { &*snap.value_ptr };

                        count += 1;
                        self.fwd.state = ScanState::FindNext;

                        if !visitor(key, value_ref) {
                            return count;
                        }
                    }
                    // Continue to next entry
                }

                // Other states are handled at the top of the loop
                ScanState::FindNext | ScanState::Down | ScanState::Up | ScanState::Retry => {}
            }
        }
    }

    /// Intra-leaf batch iteration with maximum performance.
    ///
    /// This is the highest-performance iteration method. It processes entire
    /// leaves in tight loops, minimizing per-entry overhead.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to next leaf
    /// - Amortized OCC validation overhead (validates once, processes batch)
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
        let end_bound_ikey: Option<u64> = self.end_bound.extract_ikey();
        self.fwd.run_batch(
            &mut IntraLeafRefStrategy::new(&mut visitor, end_bound_ikey),
            &self.end_bound,
            self.guard,
        )
    }

    /// Intra-leaf batch iteration returning values by copy.
    ///
    /// This is the variant of [`Self::for_each_intra_leaf_batch_ref`] that works for ALL
    /// `LeafPolicy` types, including true-inline storage. Instead of returning `&P::Value`
    /// references, it returns `P::Output` by value.
    ///
    /// # Performance Characteristics
    ///
    /// Same optimizations as `for_each_intra_leaf_batch_ref`:
    /// - Processes all entries in a leaf before moving to next leaf
    /// - Amortized OCC validation overhead (validates once, processes batch)
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions
    ///
    /// # Use Cases
    ///
    /// - **True-inline storage**: Required since inline values cannot be returned by reference
    /// - **Copy types**: When cloning is cheap (integers, small structs)
    /// - **Arc storage**: Works but incurs refcount operations per entry
    ///
    /// For Arc-based storage where zero-copy matters, prefer `for_each_intra_leaf_batch_ref`.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], P::Output)`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
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
        let end_bound_ikey: Option<u64> = self.end_bound.extract_ikey();
        self.fwd.run_batch(
            &mut IntraLeafCopyStrategy::new(&mut visitor, end_bound_ikey),
            &self.end_bound,
            self.guard,
        )
    }

    /// Highest-performance value-only batch iteration (no key materialization).
    ///
    /// This is the fastest scan method when you only need values. Keys are not
    /// built or copied, saving up to 56 bytes of copying per entry for long keys.
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
    /// If you need exact end bounds with long keys, use `for_each_intra_leaf_batch`.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `P::Output`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use masstree::MassTree;
    /// let tree: MassTree<u64> = MassTree::new();
    /// let guard = tree.guard();
    /// let mut sum = 0u64;
    /// tree.iter(&guard).for_each_values_batch(|value| {
    ///     sum += value; // value is u64 directly (MassTree uses inline storage)
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
            &mut ValuesOnlyStrategy::new(&mut visitor),
            &self.end_bound,
            self.guard,
        )
    }

    /// Fallible iteration with zero-copy value references.
    ///
    /// Like [`Self::for_each_ref`], but the visitor can return an error to stop
    /// iteration early. This is useful when processing entries might fail (e.g.,
    /// serialization, validation, I/O).
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &P::Value)`. Return `Ok(true)` to
    ///   continue, `Ok(false)` to stop early, or `Err(E)` to stop with an error.
    ///
    /// # Returns
    ///
    /// - `Ok(count)`: Number of entries successfully visited
    /// - `Err(e)`: The error returned by the visitor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = tree.iter(&guard).try_for_each_ref(|key, value| {
    ///     if key.len() > MAX_KEY_LEN {
    ///         return Err(ValidationError::KeyTooLong);
    ///     }
    ///     writer.write_entry(key, value)?;
    ///     Ok(true)
    /// });
    ///
    /// match result {
    ///     Ok(count) => println!("Wrote {} entries", count),
    ///     Err(e) => eprintln!("Failed: {}", e),
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the visitor returns an error.
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

        // Lazy initialization
        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                return Ok(0);
            }
        }

        let mut count: usize = 0;

        loop {
            // Use the zero-copy advance method
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
