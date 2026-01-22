//! Filepath: `src/tree/range/iterator/batch_reverse.rs`
//!
//! Reverse batch iteration methods for maximum performance.

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::LayerCapableLeaf;
use crate::ref_value_slot::RefValueSlot;
use crate::slot::ValueSlot;

use super::RangeIter;
use super::cleanup_guard::CleanupGuard;

use crate::tree::range::find_rev::ReverseScan;
use crate::tree::range::scan_state::ScanStateBack;

impl<S, L, A> RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
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
    /// - `visitor`: Closure receiving `(&[u8], &S::Value)`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    #[expect(clippy::too_many_lines)]
    pub fn rev_for_each_ref<F>(mut self, mut visitor: F) -> usize
    where
        S: RefValueSlot,
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        use crate::tree::range::find_rev::{
            LeafBatchResultBack, advance_prev_leaf_ptr, process_prev_leaf_batch_ptr,
        };

        if self.flags.back_exhausted() {
            return 0;
        }

        // Lazy initialization of back cursor
        if !self.flags.back_initialized() {
            self.initialize_back();

            if self.flags.back_exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // Handle initial Emit state from initialize_back() if present
        if self.back_state == ScanStateBack::Emit {
            if let Some(snapshot) = self.back_snapshot.take() {
                let key: &[u8] = self.back_cursor_key.full_key();

                if !self.start_bound.contains_reverse(key) {
                    self.flags.mark_back_exhausted();
                    return 0;
                }

                let should_continue =
                    CleanupGuard::<S>::with_output_ref(&snapshot.value, |value_ref| {
                        count += 1;
                        visitor(key, value_ref)
                    });

                if !should_continue {
                    return count;
                }
            }

            self.back_state = ScanStateBack::FindPrev;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.back_state {
                ScanStateBack::Down => {
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);
                    self.back_state = ScanStateBack::Retry;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Up => {
                    if !ReverseScan::handle_up_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_layer_stack,
                        &mut self.back_helper,
                        self.guard,
                    ) {
                        self.flags.mark_back_exhausted();
                        return count;
                    }

                    self.back_state = ScanStateBack::FindPrev;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Retry => {
                    let (new_state, _) = ReverseScan::reposition_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_helper,
                        self.guard,
                    );
                    self.back_state = new_state;
                    self.flags.require_back_duplicate_check();
                    continue;
                }

                ScanStateBack::Emit | ScanStateBack::FindPrev => {}
            }

            // Check for null stack (layer exhausted)
            if self.back_stack.get_leaf_ptr().is_null() {
                if self.back_layer_stack.is_empty() {
                    self.flags.mark_back_exhausted();
                    return count;
                }

                self.back_state = ScanStateBack::Up;
                continue;
            }

            // Check leaf deletion
            let leaf: &L = unsafe { &*self.back_stack.get_leaf_ptr() };

            if leaf.version().is_deleted() {
                self.back_state = ScanStateBack::Retry;
                continue;
            }

            // ================================================================
            // INTRA-LEAF BATCH: Process all remaining entries in this leaf
            // ================================================================

            let result = process_prev_leaf_batch_ptr(
                &mut self.back_stack,
                &mut self.back_cursor_key,
                &mut self.back_layer_stack,
                &self.start_bound,
                &mut self.back_helper,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResultBack::LeafExhausted => {
                    // Advance to previous leaf
                    if !advance_prev_leaf_ptr(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        self.guard,
                    ) {
                        // No previous leaf - check if we need to go up
                        if self.back_layer_stack.is_empty() {
                            self.flags.mark_back_exhausted();

                            return count;
                        }

                        self.back_state = ScanStateBack::Up;
                    }
                }

                LeafBatchResultBack::LayerEncountered => {
                    self.back_state = ScanStateBack::Down;
                }

                LeafBatchResultBack::VersionChanged => {
                    self.back_state = ScanStateBack::Retry;
                }

                LeafBatchResultBack::Stopped => {
                    return count;
                }

                LeafBatchResultBack::StartBoundExceeded => {
                    self.flags.mark_back_exhausted();
                    return count;
                }
            }
        }
    }

    /// Highest-performance batch-optimized reverse iteration (value by copy).
    ///
    /// This is the non-reference variant that works with ALL storage types including
    /// true-inline (`MassTree15Inline`). Unlike `rev_for_each_ref` which returns
    /// `&S::Value` references, this returns `S::Output` by value.
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
    /// - `MassTree15<V>` (Arc-based)
    /// - `MassTree24<V>` (Arc-based)
    /// - `MassTree15Inline<V>` (true-inline)
    /// - `MassTree24Inline<V>` (Box/index-based)
    ///
    /// # Arguments
    ///
    /// - `visitor`: Callback function `fn(&[u8], S::Output) -> bool`
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    #[expect(clippy::too_many_lines)]
    pub fn rev_for_each_intra_leaf_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        use crate::tree::range::find_rev::{
            LeafBatchResultBack, advance_prev_leaf_ptr, process_prev_leaf_batch,
        };

        if self.flags.back_exhausted() {
            return 0;
        }

        // Lazy initialization of back cursor
        if !self.flags.back_initialized() {
            self.initialize_back();
            if self.flags.back_exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // Handle initial Emit state from initialize_back() if present
        if self.back_state == ScanStateBack::Emit {
            if let Some(snapshot) = self.back_snapshot.take() {
                let key: &[u8] = self.back_cursor_key.full_key();

                if !self.start_bound.contains_reverse(key) {
                    self.flags.mark_back_exhausted();
                    return 0;
                }

                // Use output_from_raw to get value by copy
                let ptr: *mut u8 = S::output_to_raw(&snapshot.value);
                let output: S::Output = unsafe { S::output_from_raw(ptr) };
                // Note: ptr was created from snapshot.value which is still alive,
                // and output_from_raw clones/copies, so we need to clean up ptr
                unsafe { S::cleanup_output_raw(ptr) };

                count += 1;
                let should_continue = visitor(key, output);

                if !should_continue {
                    return count;
                }
            }

            self.back_state = ScanStateBack::FindPrev;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.back_state {
                ScanStateBack::Down => {
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);
                    self.back_state = ScanStateBack::Retry;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Up => {
                    if !ReverseScan::handle_up_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_layer_stack,
                        &mut self.back_helper,
                        self.guard,
                    ) {
                        self.flags.mark_back_exhausted();
                        return count;
                    }

                    self.back_state = ScanStateBack::FindPrev;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Retry => {
                    let (new_state, _) = ReverseScan::reposition_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_helper,
                        self.guard,
                    );
                    self.back_state = new_state;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Emit | ScanStateBack::FindPrev => {}
            }

            // Check for null stack (layer exhausted)
            if self.back_stack.get_leaf_ptr().is_null() {
                if self.back_layer_stack.is_empty() {
                    self.flags.mark_back_exhausted();
                    return count;
                }

                self.back_state = ScanStateBack::Up;

                continue;
            }

            // Check leaf deletion
            let leaf: &L = unsafe { &*self.back_stack.get_leaf_ptr() };

            if leaf.version().is_deleted() {
                self.back_state = ScanStateBack::Retry;

                continue;
            }

            // ================================================================
            // INTRA-LEAF BATCH: Process all remaining entries in this leaf
            // ================================================================

            let result = process_prev_leaf_batch(
                &mut self.back_stack,
                &mut self.back_cursor_key,
                &mut self.back_layer_stack,
                &self.start_bound,
                &mut self.back_helper,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResultBack::LeafExhausted => {
                    // Advance to previous leaf
                    if !advance_prev_leaf_ptr(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        self.guard,
                    ) {
                        // No previous leaf - check if we need to go up
                        if self.back_layer_stack.is_empty() {
                            self.flags.mark_back_exhausted();
                            return count;
                        }

                        self.back_state = ScanStateBack::Up;
                    }
                }

                LeafBatchResultBack::LayerEncountered => {
                    self.back_state = ScanStateBack::Down;
                }

                LeafBatchResultBack::VersionChanged => {
                    self.back_state = ScanStateBack::Retry;
                }

                LeafBatchResultBack::Stopped => {
                    return count;
                }

                LeafBatchResultBack::StartBoundExceeded => {
                    self.flags.mark_back_exhausted();

                    return count;
                }
            }
        }
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
    /// - `visitor`: Closure receiving `S::Output`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[must_use = "returns the number of entries visited"]
    #[expect(clippy::too_many_lines)]
    pub fn rev_for_each_values_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(S::Output) -> bool,
    {
        use crate::tree::range::find_rev::{
            LeafBatchResultBack, advance_prev_leaf_ptr, process_prev_leaf_batch_values,
        };

        if self.flags.back_exhausted() {
            return 0;
        }

        // Lazy initialization of back cursor
        if !self.flags.back_initialized() {
            self.initialize_back();

            if self.flags.back_exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // Pre-extract start bound ikey for fast comparison (reverse uses start as lower bound)
        let start_bound_ikey: Option<u64> = self.start_bound.extract_ikey();

        // Handle initial Emit state from initialize_back() if present
        if self.back_state == ScanStateBack::Emit {
            if let Some(snapshot) = self.back_snapshot.take() {
                // Skip start bound check for values-only (approximate)
                // The bound was already checked during initialization

                // Use output_from_raw to get value by copy
                let ptr: *mut u8 = S::output_to_raw(&snapshot.value);
                let output: S::Output = unsafe { S::output_from_raw(ptr) };
                unsafe { S::cleanup_output_raw(ptr) };

                count += 1;
                let should_continue = visitor(output);

                if !should_continue {
                    return count;
                }
            }
            self.back_state = ScanStateBack::FindPrev;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.back_state {
                ScanStateBack::Down => {
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);
                    self.back_state = ScanStateBack::Retry;
                    self.flags.require_back_duplicate_check();
                    continue;
                }

                ScanStateBack::Up => {
                    if !ReverseScan::handle_up_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_layer_stack,
                        &mut self.back_helper,
                        self.guard,
                    ) {
                        self.flags.mark_back_exhausted();
                        return count;
                    }

                    self.back_state = ScanStateBack::FindPrev;
                    self.flags.require_back_duplicate_check();

                    continue;
                }

                ScanStateBack::Retry => {
                    let (new_state, _) = ReverseScan::reposition_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_helper,
                        self.guard,
                    );

                    self.back_state = new_state;
                    self.flags.require_back_duplicate_check();
                    continue;
                }

                ScanStateBack::Emit | ScanStateBack::FindPrev => {}
            }

            // Check for null stack (layer exhausted)
            if self.back_stack.get_leaf_ptr().is_null() {
                if self.back_layer_stack.is_empty() {
                    self.flags.mark_back_exhausted();

                    return count;
                }

                self.back_state = ScanStateBack::Up;

                continue;
            }

            // Check leaf deletion
            let leaf: &L = unsafe { &*self.back_stack.get_leaf_ptr() };
            if leaf.version().is_deleted() {
                self.back_state = ScanStateBack::Retry;
                continue;
            }

            // ================================================================
            // VALUE-ONLY BATCH: Process all remaining entries without key building
            // ================================================================

            let result = process_prev_leaf_batch_values(
                &mut self.back_stack,
                &mut self.back_cursor_key,
                &mut self.back_layer_stack,
                start_bound_ikey,
                &mut self.back_helper,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResultBack::LeafExhausted => {
                    // Advance to previous leaf
                    if !advance_prev_leaf_ptr(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        self.guard,
                    ) {
                        // No previous leaf - check if we need to go up
                        if self.back_layer_stack.is_empty() {
                            self.flags.mark_back_exhausted();

                            return count;
                        }

                        self.back_state = ScanStateBack::Up;
                    }
                }

                LeafBatchResultBack::LayerEncountered => {
                    self.back_state = ScanStateBack::Down;
                }

                LeafBatchResultBack::VersionChanged => {
                    self.back_state = ScanStateBack::Retry;
                }

                LeafBatchResultBack::Stopped => {
                    return count;
                }

                LeafBatchResultBack::StartBoundExceeded => {
                    self.flags.mark_back_exhausted();
                    return count;
                }
            }
        }
    }
}
