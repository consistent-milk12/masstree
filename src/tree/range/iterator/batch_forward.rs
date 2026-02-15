//! Filepath: `src/tree/range/iterator/batch_forward.rs`
//!
//! Forward batch iteration methods for maximum performance.

use crate::alloc_trait::TreeAllocator;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::policy::RefPolicy as RefLeafPolicy;

use super::RangeIter;

use crate::tree::range::find::{
    find_next, find_next_ptr, find_next_single_layer_ptr, find_next_with_duplicate_check,
    find_next_with_duplicate_check_ptr, find_retry, handle_down, handle_up,
};
use crate::tree::range::scan_state::{LayerContext, ScanState};

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
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();

            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Fast path: process current entry without allocation
            if let Some(entry) = self.advance_no_alloc() {
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

    /// Advance without allocating key Vec.
    ///
    /// Returns `(&[u8], P::Output)` where the key slice is borrowed from
    /// the internal `cursor_key` buffer.
    ///
    /// This function inlines the common case `(FindNext → Emit)` to avoid:
    /// - State machine dispatch overhead
    /// - Function call overhead to `find_next()`
    ///
    /// Only rare cases (Down, Up, Retry) use function calls.
    #[inline(always)]
    #[expect(
        clippy::too_many_lines,
        reason = "State machine with debug instrumentation"
    )]
    pub(super) fn advance_no_alloc(&mut self) -> Option<(&[u8], P::Output)> {
        // Fast path: if we have a pending emit, process it first
        if self.state == ScanState::Emit
            && let Some(snapshot) = self.snapshot.take()
        {
            // DEBUG: Assert strict ordering (must copy key to avoid borrow conflict)
            #[cfg(debug_assertions)]
            {
                let key_copy = self.cursor_key.full_key().to_vec();
                self.assert_ordering(&key_copy);
            }

            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

            if !self.end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            self.state = ScanState::FindNext;

            return Some((key, snapshot.value));
        }

        loop {
            // Handle rare states first (will break out of loop on Emit)
            match self.state {
                ScanState::Down => {
                    #[cfg(debug_assertions)]
                    let pre_cursor = self.cursor_key.debug_state();

                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();

                    #[cfg(debug_assertions)]
                    self.record_transition(format!(
                        "Down -> Retry: pre={}, post={}",
                        pre_cursor,
                        self.cursor_key.debug_state()
                    ));

                    continue;
                }

                ScanState::Up => {
                    #[cfg(debug_assertions)]
                    let pre_cursor = self.cursor_key.debug_state();

                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();

                        #[cfg(debug_assertions)]
                        self.record_transition(format!("Up -> Exhausted: pre={pre_cursor}"));

                        return None;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();

                    #[cfg(debug_assertions)]
                    self.record_transition(format!(
                        "Up -> FindNext: pre={}, post={}",
                        pre_cursor,
                        self.cursor_key.debug_state()
                    ));

                    continue;
                }

                ScanState::Retry => {
                    #[cfg(debug_assertions)]
                    let pre_cursor = self.cursor_key.debug_state();

                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();

                    #[cfg(debug_assertions)]
                    self.record_transition(format!(
                        "Retry -> {:?}: pre={}, post={}",
                        self.state,
                        pre_cursor,
                        self.cursor_key.debug_state()
                    ));

                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Main hot path: FindNext (inlined from find_next)
            let (new_state, snapshot) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                find_next_with_duplicate_check(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
                .into_parts()
            } else {
                find_next(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
                .into_parts()
            };

            self.state = new_state;

            // Fast path: if Emit, return immediately without another loop iteration
            if new_state == ScanState::Emit
                && let Some(snap) = snapshot
            {
                // DEBUG: Assert strict ordering (must copy key to avoid borrow conflict)
                #[cfg(debug_assertions)]
                {
                    let key_copy = self.cursor_key.full_key().to_vec();
                    self.assert_ordering(&key_copy);
                }

                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key = unsafe { self.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return None;
                }

                self.state = ScanState::FindNext;
                return Some((key, snap.value));
            }

            self.snapshot = snapshot;
        }
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
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Use the zero-copy advance method
            if let Some((key, value_ref)) = self.advance_no_alloc_ref() {
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
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization - reuses existing RangeIter::initialize()
        // which correctly handles start-bound descent (shift vs shift_clear)
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // NOTE: We don't use advance_no_alloc_ref here because it has issues
        // with multi-layer keys. Instead, we use the batch loop for all entries
        // which correctly handles cursor_key updates via find_next_ptr.

        // If state is Emit with a snapshot from initialize(), handle it specially
        // by extracting the snapshot and emitting directly
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
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
            self.state = ScanState::FindNext;
        }

        // Main batch loop - uses find_next_ptr which correctly updates cursor_key

        loop {
            // ================================================================
            // Handle rare states (layer transitions, retries, exhaustion)
            // ================================================================

            // Handle pending state transitions first (like advance_no_alloc_ref)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();

                        return count;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }

                self.state = ScanState::Up;

                continue;
            }

            // Check leaf deletion
            let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };

            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // Main hot path: FindNext → Emit (inlined)
            // ================================================================

            let (new_state, snapshot_ptr) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                find_next_with_duplicate_check_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            } else {
                find_next_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            };

            self.state = new_state;

            match new_state {
                ScanState::Emit => {
                    if let Some(snap) = snapshot_ptr {
                        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                        let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                        // Check end bound
                        if !self.end_bound.contains(key) {
                            self.flags.mark_exhausted();
                            return count;
                        }

                        // SAFETY: find_next_ptr validated version, guard protects pointer
                        let value_ref: &P::Value = unsafe { &*snap.value_ptr };

                        count += 1;
                        self.state = ScanState::FindNext;

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
    #[expect(clippy::too_many_lines)]
    pub fn for_each_intra_leaf_batch_ref<F>(mut self, mut visitor: F) -> usize
    where
        P: RefLeafPolicy,
        F: FnMut(&[u8], &P::Value) -> bool,
    {
        use crate::tree::range::find::{LeafBatchResult, advance_leaf_ptr, process_leaf_batch_ptr};

        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();

            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;
        let end_bound_ikey: Option<u64> = self.end_bound.extract_ikey();

        // Handle initial Emit state from initialize() if present
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return 0;
                }

                let value_ref: &P::Value = P::output_as_ref(&snapshot.value);
                let should_continue: bool = {
                    count += 1;
                    visitor(key, value_ref)
                };

                if !should_continue {
                    return count;
                }
            }
            self.state = ScanState::FindNext;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();

                        return count;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }
                self.state = ScanState::Up;

                continue;
            }

            // Check leaf deletion
            let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };

            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;

                continue;
            }

            // ================================================================
            // DUPLICATE CHECK SLOW PATH: After Retry/Down/Up, use per-entry
            // path to filter already-emitted keys before resuming batch mode.
            //
            // This is critical for correctness: after a VersionChanged retry,
            // we may reposition to a leaf containing keys we already emitted.
            // The batch functions don't check CursorKey, so we must use the
            // non-batch path (which has duplicate filtering) for at least one
            // entry before resuming batching.
            // ================================================================
            if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();

                let (new_state, snapshot_ptr) = find_next_with_duplicate_check_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                );

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot_ptr {
                            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                            let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                            if !self.end_bound.contains(key) {
                                self.flags.mark_exhausted();
                                return count;
                            }

                            // SAFETY: find_next_with_duplicate_check_ptr validated version,
                            // guard protects pointer
                            let value_ref: &P::Value = unsafe { &*snap.value_ptr };

                            count += 1;
                            self.state = ScanState::FindNext;

                            if !visitor(key, value_ref) {
                                return count;
                            }
                        }
                    }

                    // Other states continue the loop
                    ScanState::FindNext | ScanState::Down | ScanState::Up | ScanState::Retry => {}
                }

                continue;
            }

            // ================================================================
            // INTRA-LEAF BATCH: Process all remaining entries in this leaf
            // (Fast path - no duplicate checking needed)
            // ================================================================

            let result = process_leaf_batch_ptr(
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                &self.end_bound,
                end_bound_ikey,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResult::LeafExhausted => {
                    // Advance to next leaf
                    let (state, _) =
                        advance_leaf_ptr(&mut self.stack, &self.cursor_key, self.guard);
                    self.state = state;
                }

                LeafBatchResult::LayerEncountered => {
                    self.state = ScanState::Down;
                }

                LeafBatchResult::VersionChanged => {
                    self.state = ScanState::Retry;
                }

                LeafBatchResult::Stopped => {
                    return count;
                }

                LeafBatchResult::EndBoundExceeded => {
                    self.flags.mark_exhausted();

                    return count;
                }
            }
        }
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
    #[expect(clippy::too_many_lines)]
    pub fn for_each_intra_leaf_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], P::Output) -> bool,
    {
        use crate::tree::range::find::{LeafBatchResult, advance_leaf_ptr, process_leaf_batch};

        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;
        let end_bound_ikey: Option<u64> = self.end_bound.extract_ikey();

        // Handle initial Emit state from initialize() if present
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return 0;
                }

                count += 1;
                let should_continue = visitor(key, snapshot.value);

                if !should_continue {
                    return count;
                }
            }
            self.state = ScanState::FindNext;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();
                        return count;
                    }
                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }
                self.state = ScanState::Up;
                continue;
            }

            // Check leaf deletion
            // SAFETY: stack.is_null() check above ensures leaf_ptr is valid,
            // and the guard protects the node from deallocation.
            let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // DUPLICATE CHECK SLOW PATH: After Retry/Down/Up, use per-entry
            // path to filter already-emitted keys before resuming batch mode.
            // See for_each_intra_leaf_batch_ref for detailed rationale.
            // ================================================================
            if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();

                let (new_state, snapshot) = find_next_with_duplicate_check(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
                .into_parts();

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot {
                            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                            let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                            if !self.end_bound.contains(key) {
                                self.flags.mark_exhausted();
                                return count;
                            }

                            count += 1;
                            self.state = ScanState::FindNext;

                            if !visitor(key, snap.value) {
                                return count;
                            }
                        }
                    }

                    // Other states continue the loop
                    ScanState::FindNext | ScanState::Down | ScanState::Up | ScanState::Retry => {}
                }

                continue;
            }

            // ================================================================
            // INTRA-LEAF BATCH: Process all remaining entries in this leaf
            // (Fast path - no duplicate checking needed)
            // ================================================================

            let result = process_leaf_batch(
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                &self.end_bound,
                end_bound_ikey,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResult::LeafExhausted => {
                    // Advance to next leaf
                    let (state, _) =
                        advance_leaf_ptr(&mut self.stack, &self.cursor_key, self.guard);
                    self.state = state;
                }

                LeafBatchResult::LayerEncountered => {
                    self.state = ScanState::Down;
                }

                LeafBatchResult::VersionChanged => {
                    self.state = ScanState::Retry;
                }

                LeafBatchResult::Stopped => {
                    return count;
                }

                LeafBatchResult::EndBoundExceeded => {
                    self.flags.mark_exhausted();

                    return count;
                }
            }
        }
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
    #[expect(clippy::too_many_lines)]
    pub fn for_each_values_batch<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(P::Output) -> bool,
    {
        use crate::tree::range::find::{
            LeafBatchResult, advance_leaf_ptr, process_leaf_batch_values,
        };

        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();

            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // Handle initial Emit state from initialize() if present
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                // Values-only mode uses approximate ikey-based end bound checking
                // (see doc "End Bound Behavior" section for trade-offs)
                count += 1;
                let should_continue = visitor(snapshot.value);

                if !should_continue {
                    return count;
                }
            }

            self.state = ScanState::FindNext;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();

                        return count;
                    }
                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();

                    return count;
                }

                self.state = ScanState::Up;

                continue;
            }

            // Check leaf deletion
            // SAFETY: stack.is_null() check above ensures leaf_ptr is valid,
            // and the guard protects the node from deallocation.
            let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // DUPLICATE CHECK SLOW PATH: After Retry/Down/Up, use per-entry
            // path to filter already-emitted keys before resuming batch mode.
            // See for_each_intra_leaf_batch_ref for detailed rationale.
            // ================================================================
            if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();

                let (new_state, snapshot) = find_next_with_duplicate_check(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
                .into_parts();

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot {
                            // Values-only: skip end bound check (approximate ikey-based)
                            // This matches the batch function's behavior
                            count += 1;
                            self.state = ScanState::FindNext;

                            if !visitor(snap.value) {
                                return count;
                            }
                        }
                    }

                    // Other states continue the loop
                    ScanState::FindNext | ScanState::Down | ScanState::Up | ScanState::Retry => {}
                }

                continue;
            }

            // ================================================================
            // VALUE-ONLY BATCH: Process all remaining entries without key building
            // (Fast path - no duplicate checking needed)
            // ================================================================

            // Layer-aware ikey extraction: align end-bound ikey with the current
            // trie depth so descended scans don't compare sublayer ikeys against
            // root-layer bound bytes.
            let end_bound_ikey: Option<u64> =
                self.end_bound.extract_ikey_at(self.cursor_key.offset());

            let result = process_leaf_batch_values(
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                end_bound_ikey,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResult::LeafExhausted => {
                    // Advance to next leaf
                    let (state, _) =
                        advance_leaf_ptr(&mut self.stack, &self.cursor_key, self.guard);
                    self.state = state;
                }

                LeafBatchResult::LayerEncountered => {
                    self.state = ScanState::Down;
                }

                LeafBatchResult::VersionChanged => {
                    self.state = ScanState::Retry;
                }

                LeafBatchResult::Stopped => {
                    return count;
                }

                LeafBatchResult::EndBoundExceeded => {
                    self.flags.mark_exhausted();
                    return count;
                }
            }
        }
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
        if self.flags.exhausted() {
            return Ok(0);
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();

            if self.flags.exhausted() {
                return Ok(0);
            }
        }

        let mut count: usize = 0;

        loop {
            // Use the zero-copy advance method
            if let Some((key, value_ref)) = self.advance_no_alloc_ref() {
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

    /// Advance without cloning values.
    ///
    /// Returns `(&[u8], &P::Value)` where both are borrowed references.
    /// The value is obtained by dereferencing the raw pointer directly,
    /// avoiding Arc clone overhead.
    ///
    /// # Note on Initial Entry
    ///
    /// After `initialize()`, there may be a pending emit in `self.snapshot`.
    /// For the first entry, we convert the Output to a raw pointer and dereference.
    /// This requires that `P::Output` is dereferenceable to `P::Value`.
    ///
    /// # Safety
    ///
    /// The returned references are valid because:
    /// 1. The guard prevents deallocation during iteration
    /// 2. Version validation ensures the slot hasn't been modified
    #[inline(always)]
    #[expect(clippy::too_many_lines, reason = "Complex allocation logic")]
    pub(super) fn advance_no_alloc_ref(&mut self) -> Option<(&[u8], &P::Value)>
    where
        P: RefLeafPolicy,
    {
        // Handle pending emit from initialize() - first entry case
        if self.state == ScanState::Emit && self.snapshot.is_some() {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let key = unsafe { self.cursor_key.full_key_unchecked() };

            if !self.end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            // Take the snapshot to get the value
            let snapshot = self.snapshot.take()?;

            // Transition to FindNext for next call
            self.state = ScanState::FindNext;

            // Store the output so it stays alive for the returned reference.
            // Replaces the previous output (dropping it automatically).
            self.last_output = Some(snapshot.value);
            let value_ref: &P::Value = P::output_as_ref(self.last_output.as_ref().unwrap());

            return Some((key, value_ref));
        }

        loop {
            // ================================================================
            // Single-layer fast path (keys ≤ 8 bytes)
            // ================================================================
            if self.flags.single_layer_mode() {
                // Retry handling in single-layer mode
                if self.state == ScanState::Retry {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                let (new_state, snapshot_ptr) = find_next_single_layer_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    self.guard,
                    self.flags.needs_duplicate_check(),
                );

                if self.flags.needs_duplicate_check() {
                    self.flags.clear_duplicate_check();
                }

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot_ptr {
                            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                            let key = unsafe { self.cursor_key.full_key_unchecked() };

                            if !self.end_bound.contains(key) {
                                self.flags.mark_exhausted();
                                return None;
                            }

                            self.state = ScanState::FindNext;
                            let value_ref: &P::Value = unsafe { &*snap.value_ptr };

                            return Some((key, value_ref));
                        }
                    }

                    ScanState::FindNext => {
                        if self.stack.is_null() {
                            self.flags.mark_exhausted();
                            return None;
                        }

                        continue;
                    }

                    ScanState::Retry => continue,

                    ScanState::Down => {
                        // Encountered layer pointer - fall back to multi-layer
                        self.flags.disable_single_layer_mode();

                        // Push PARENT context to layer_stack before setting new root.
                        // find_next_single_layer_ptr already stored the ikey to cursor.
                        self.layer_stack
                            .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));

                        // Read the layer pointer from current slot and set as new root.
                        // Stack position is still at the layer pointer slot.
                        let Some(slot) = self.stack.kp() else {
                            // Defensive: Down state guarantees valid slot, but if
                            // invariant is violated, recover via retry rather than panic.
                            debug_assert!(
                                false,
                                "Down state entered without valid slot - state machine bug"
                            );

                            self.state = ScanState::Retry;
                            continue;
                        };

                        // SAFETY: find_next_single_layer_ptr validated the leaf version,
                        // and the guard protects the node from deallocation.
                        let leaf: &LeafNode15<P> = unsafe { self.stack.leaf_ref() };
                        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
                        self.stack.set_root(layer_ptr);

                        // Don't continue; fall through to handle Down below
                    }

                    ScanState::Up => {
                        self.flags.mark_exhausted();

                        return None;
                    }
                }
            }

            // ================================================================
            // Multi-layer path (handles Down/Up transitions)
            // ================================================================

            // Handle rare states first
            match self.state {
                ScanState::Down => {
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();

                        return None;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();

                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Use zero-copy find_next variants
            let (new_state, snapshot_ptr) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                find_next_with_duplicate_check_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            } else {
                find_next_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            };

            self.state = new_state;

            // If Emit, return the reference directly
            if new_state == ScanState::Emit
                && let Some(snap) = snapshot_ptr
            {
                // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                let key = unsafe { self.cursor_key.full_key_unchecked() };

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();

                    return None;
                }

                self.state = ScanState::FindNext;

                // SAFETY:
                // - Guard prevents deallocation during iteration
                // - Version was validated in find_next_inner_ptr before returning
                // - Pointer is properly aligned (stored in leaf slot with correct layout)
                // - We dereference directly (not via snap.value_ref()) because the
                //   reference must outlive the local `snap` variable
                let value_ref: &P::Value = unsafe { &*snap.value_ptr };

                return Some((key, value_ref));
            }

            // All non-Emit states (Up, Down, Retry, FindNext) continue the loop.
            // Exhaustion is detected by stack.is_null() or handle_up() returning false.
        }
    }
}
