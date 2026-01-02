//! ========================================================================
//!  Batch Insert Implementation
//! ========================================================================
//!
//! Provides high-throughput batch insertion by:
//! - Sorting entries by ikey for cache locality
//! - Grouping entries by target leaf
//! - Acquiring a single lock per leaf group
//! - Amortizing traversal overhead across multiple inserts
//!
//! # Performance Characteristics
//!
//! - **Best case:** 3-10x faster than individual inserts for bulk loads
//! - **Worst case:** Same as individual inserts (random keys, every key in different leaf)
//!
//! # Limitations
//!
//! - Entries are processed in ikey order, not insertion order
//! - Old values are returned in batch, not per-insert
//! - Layer descent (keys > 8 bytes pointing to sublayers) falls back to single insert

use std::sync::atomic::Ordering as AtomicOrdering;

use seize::{Guard, LocalGuard};

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric,
    key::Key,
    leaf_trait::LayerCapableLeaf,
    nodeversion::LockGuard,
    slot::ValueSlot,
    tree::InsertError,
    is_marked, unmark_ptr,
};

use super::{InsertSearchResultGeneric, TreePermutation};

// ============================================================================
//  Batch Entry Types
// ============================================================================

/// A single entry in a batch insert operation.
///
/// Contains the key bytes and the pre-converted output value.
/// The output is created via `S::into_output()` before sorting to ensure
/// allocation happens exactly once per entry.
#[must_use]
#[expect(missing_debug_implementations, reason = "Debug on S::Output may not be available")]
pub struct BatchEntry<S: ValueSlot> {
    /// The key bytes (owned for sorting).
    pub key: Vec<u8>,

    /// The pre-converted output value.
    ///
    /// Created via `S::into_output(value)` to ensure the Arc (if any)
    /// is allocated once and reused across retries.
    pub output: S::Output,

    /// Cached ikey for the first 8 bytes (used for sorting).
    ikey: u64,
}

impl<S: ValueSlot> BatchEntry<S> {
    /// Create a new batch entry from key and value.
    ///
    /// Converts the value to output immediately to ensure single allocation.
    #[inline]
    pub fn new(key: Vec<u8>, value: S::Value) -> Self {
        let ikey = Self::compute_ikey(&key);
        let output = S::into_output(value);
        Self { key, output, ikey }
    }

    /// Create a batch entry from key and pre-converted output.
    ///
    /// Use this when you already have an `S::Output` (e.g., from a previous
    /// failed batch that needs retry).
    #[inline]
    pub fn from_output(key: Vec<u8>, output: S::Output) -> Self {
        let ikey = Self::compute_ikey(&key);
        Self { key, output, ikey }
    }

    /// Compute the ikey (first 8 bytes as big-endian u64).
    #[inline]
    fn compute_ikey(key: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        let len = key.len().min(8);
        buf[..len].copy_from_slice(&key[..len]);
        u64::from_be_bytes(buf)
    }

    /// Get the cached ikey.
    #[inline]
    pub fn ikey(&self) -> u64 {
        self.ikey
    }

    /// Check if this key has a suffix (> 8 bytes).
    #[inline]
    pub fn has_suffix(&self) -> bool {
        self.key.len() > 8
    }
}

// ============================================================================
//  Batch Insert Result
// ============================================================================

/// Result of a batch insert operation.
///
/// # Type Parameter
///
/// * `O` - The output type (`Arc<V>` for Arc mode, `V` for Inline mode)
#[derive(Debug, Clone)]
#[must_use]
pub struct BatchInsertResult<O> {
    /// Number of new keys inserted.
    pub inserted: usize,

    /// Number of existing keys updated.
    pub updated: usize,

    /// Old values from updated keys (in no particular order).
    ///
    /// For `MassTree24<V>`, this is `Vec<Arc<V>>`.
    /// For `MassTree24Inline<V>`, this is `Vec<V>`.
    pub old_values: Vec<O>,

    /// Number of entries that failed and need individual retry.
    ///
    /// Entries fail when they require layer descent (sublayer operations).
    /// The caller should retry these entries individually via `insert()`.
    pub failed: usize,
}

impl<O> Default for BatchInsertResult<O> {
    fn default() -> Self {
        Self {
            inserted: 0,
            updated: 0,
            old_values: Vec::new(),
            failed: 0,
        }
    }
}

impl<O> BatchInsertResult<O> {
    /// Create a new empty result.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a result with pre-allocated capacity for old values.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inserted: 0,
            updated: 0,
            old_values: Vec::with_capacity(capacity),
            failed: 0,
        }
    }

    /// Record a successful new key insertion.
    #[inline]
    pub fn record_insert(&mut self) {
        self.inserted += 1;
    }

    /// Record a successful update with old value.
    #[inline]
    pub fn record_update(&mut self, old_value: O) {
        self.updated += 1;
        self.old_values.push(old_value);
    }

    /// Record a failed entry.
    #[inline]
    pub fn record_failure(&mut self) {
        self.failed += 1;
    }

    /// Total entries processed (inserted + updated + failed).
    #[inline]
    pub fn total(&self) -> usize {
        self.inserted + self.updated + self.failed
    }

    /// Check if all entries succeeded.
    #[inline]
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0
    }
}

// ============================================================================
//  Helper Types
// ============================================================================

/// Result of trying to insert a single entry in a batch.
enum BatchEntryResult<O> {
    /// Entry was inserted as a new key.
    Inserted,

    /// Entry updated an existing key, returning the old value.
    Updated(O),

    /// Leaf is full, need to stop batch and retry after split.
    NeedsSplit,

    /// Layer descent needed - mark for individual retry.
    NeedsLayerDescent,

    /// Entry doesn't belong to this leaf (ikey >= next leaf's bound).
    #[expect(dead_code)]
    BelongsToSibling,

    /// Slot is being modified by another thread, retry.
    Retry,
}

/// Result of finding a usable slot for insertion.
enum FindSlotResult {
    /// Found a usable slot.
    Found { slot: usize, back_offset: usize },

    /// No usable slot, split needed.
    NeedsSplit,
}

/// Errors from membership validation.
enum MembershipError {
    /// A split is in progress - wait and retry.
    SplitInProgress,
    /// Key has moved to a sibling leaf - retry traversal.
    KeyMovedToSibling,
}

// ============================================================================
//  Batch Insert Implementation
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
    //  Public Batch Insert API
    // ========================================================================

    /// Insert multiple key-value pairs in a single batch operation.
    ///
    /// This is the main public API for batch inserts. It:
    /// 1. Converts all values to outputs (single allocation per entry)
    /// 2. Sorts entries by ikey for cache locality
    /// 3. Groups entries by target leaf
    /// 4. Inserts with minimal lock acquisitions
    ///
    /// # Arguments
    ///
    /// * `entries` - Iterator of (key, value) pairs to insert
    ///
    /// # Returns
    ///
    /// A `BatchInsertResult` containing:
    /// - Number of new keys inserted
    /// - Number of existing keys updated
    /// - Old values from updated keys
    /// - Number of failed entries (require individual retry)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use masstree::MassTree24;
    ///
    /// let tree: MassTree24<u64> = MassTree24::new();
    ///
    /// let entries = vec![
    ///     (b"key1".to_vec(), 1u64),
    ///     (b"key2".to_vec(), 2u64),
    ///     (b"key3".to_vec(), 3u64),
    /// ];
    ///
    /// let result = tree.insert_batch(entries).unwrap();
    /// assert_eq!(result.inserted, 3);
    /// assert_eq!(result.updated, 0);
    /// ```
    ///
    /// # Performance
    ///
    /// For N entries going to M leaves:
    /// - Lock acquisitions: O(M) instead of O(N)
    /// - Tree traversals: O(M) instead of O(N)
    /// - Best when entries cluster by key prefix
    ///
    /// # Note on Clone Bound
    ///
    /// This method requires `S::Output: Clone` because entries may need to be
    /// retried after a split. For `Arc<V>`, cloning is cheap (refcount bump).
    /// For `Copy` types in `Inline` mode, cloning is also cheap.
    ///
    /// # Errors
    ///
    /// Returns `Err` only for unrecoverable errors. Individual entry failures
    /// are tracked in the result's `failed` count.
    pub fn insert_batch<I>(
        &self,
        entries: I,
    ) -> Result<BatchInsertResult<S::Output>, InsertError>
    where
        I: IntoIterator<Item = (Vec<u8>, S::Value)>,
    {
        let guard = self.guard();
        self.insert_batch_with_guard(entries, &guard)
    }

    /// Insert multiple key-value pairs using an existing guard.
    ///
    /// Use this when performing multiple batch operations under the same
    /// guard to amortize guard creation overhead.
    ///
    /// # Arguments
    ///
    /// * `entries` - Iterator of (key, value) pairs to insert
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// A `BatchInsertResult` with insertion statistics.
    ///
    /// # Errors
    ///
    /// Returns `Err` only for unrecoverable errors.
    pub fn insert_batch_with_guard<I>(
        &self,
        entries: I,
        guard: &LocalGuard<'_>,
    ) -> Result<BatchInsertResult<S::Output>, InsertError>
    where
        I: IntoIterator<Item = (Vec<u8>, S::Value)>,
    {
        // Convert to BatchEntry (allocates outputs once)
        let mut batch: Vec<BatchEntry<S>> = entries
            .into_iter()
            .map(|(key, value)| BatchEntry::new(key, value))
            .collect();

        if batch.is_empty() {
            return Ok(BatchInsertResult::new());
        }

        // Sort by ikey for cache locality and leaf clustering
        batch.sort_unstable_by_key(|entry| entry.ikey());

        // Process the sorted batch
        self.process_sorted_batch(&mut batch, guard)
    }

    /// Insert pre-constructed batch entries.
    ///
    /// Use this when you need finer control over the batch entries,
    /// or when retrying failed entries from a previous batch.
    ///
    /// # Arguments
    ///
    /// * `entries` - Mutable slice of batch entries (will be sorted in place)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// A `BatchInsertResult` with insertion statistics.
    ///
    /// # Note
    ///
    /// The entries slice will be sorted by ikey in place.
    pub fn insert_batch_entries(
        &self,
        entries: &mut [BatchEntry<S>],
        guard: &LocalGuard<'_>,
    ) -> Result<BatchInsertResult<S::Output>, InsertError> {
        if entries.is_empty() {
            return Ok(BatchInsertResult::new());
        }

        // Sort by ikey for cache locality
        entries.sort_unstable_by_key(|entry| entry.ikey());

        // Convert slice to Vec for processing
        let mut batch: Vec<BatchEntry<S>> = entries
            .iter()
            .map(|e| BatchEntry::from_output(e.key.clone(), e.output.clone()))
            .collect();

        self.process_sorted_batch(&mut batch, guard)
    }

    // ========================================================================
    //  Internal Batch Processing
    // ========================================================================

    /// Process a sorted batch of entries.
    ///
    /// Entries must be sorted by ikey before calling this method.
    fn process_sorted_batch(
        &self,
        batch: &mut Vec<BatchEntry<S>>,
        guard: &LocalGuard<'_>,
    ) -> Result<BatchInsertResult<S::Output>, InsertError> {
        let mut result = BatchInsertResult::with_capacity(batch.len() / 4);
        let mut index = 0;

        while index < batch.len() {
            // Get the current entry to find its leaf
            let entry = &batch[index];
            let key = Key::new(&entry.key);

            // Load root pointer
            let mut layer_root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

            // Retry loop for finding and locking the correct leaf
            let entries_processed = 'retry: loop {
                // Handle layer root promotion
                layer_root = self.maybe_parent_generic(layer_root);

                // Traverse to leaf
                let leaf_ptr: *mut L =
                    self.reach_leaf_concurrent_generic(layer_root, &key, false, guard);
                let leaf: &L = unsafe { &*leaf_ptr };

                // B-link advance if needed
                let (leaf, exceeded_hop_limit) =
                    self.advance_to_key_by_bound_generic(leaf, &key, guard);

                if exceeded_hop_limit {
                    layer_root = self.root_ptr.load(AtomicOrdering::Acquire);
                    continue 'retry;
                }

                // Capture pre-lock state for OCC validation
                let pre_lock_version = leaf.version().stable();
                let pre_lock_perm_raw = leaf.permutation_raw();

                // Lock the leaf
                let mut lock = leaf.version().lock();

                // Validate post-lock state
                if !self.validate_post_lock_batch(leaf, pre_lock_version, pre_lock_perm_raw) {
                    drop(lock);
                    continue 'retry;
                }

                // Check for deleted layer (gc'd sublayer)
                if leaf.deleted_layer() {
                    drop(lock);
                    layer_root = self.root_ptr.load(AtomicOrdering::Acquire);
                    continue 'retry;
                }

                // Validate membership
                if self.validate_membership_batch(leaf, &key).is_err() {
                    drop(lock);
                    continue 'retry;
                }

                // Now we hold the lock - process as many entries as fit in this leaf
                let processed = self.insert_batch_into_locked_leaf(
                    leaf,
                    &mut lock,
                    batch,
                    index,
                    &mut result,
                    guard,
                );

                drop(lock);

                // If we processed nothing and the leaf is full, we need to fall back
                // to regular insert to trigger a split
                if processed == 0 {
                    // Leaf is full - use regular insert which handles splits
                    break 'retry 0;
                }

                break 'retry processed;
            };

            // If we made no progress, fall back to individual insert for this entry
            // This triggers a split if needed
            if entries_processed == 0 {
                let entry = &batch[index];
                let mut key = Key::new(&entry.key);
                match self.insert_concurrent_generic(&mut key, entry.output.clone(), guard) {
                    Ok(old) => {
                        if let Some(old_value) = old {
                            result.record_update(old_value);
                        } else {
                            result.record_insert();
                        }
                    }
                    Err(_) => {
                        result.record_failure();
                    }
                }
                index += 1;
            } else {
                index += entries_processed;
            }
        }

        Ok(result)
    }

    /// Insert as many entries as possible into a locked leaf.
    ///
    /// Returns the number of entries processed (inserted, updated, or marked failed).
    fn insert_batch_into_locked_leaf(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        batch: &[BatchEntry<S>],
        start_index: usize,
        result: &mut BatchInsertResult<S::Output>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let mut processed = 0;
        let mut perm = leaf.permutation();

        // Determine the ikey upper bound for this leaf
        let next_raw: *mut L = leaf.next_raw();
        let next_ptr: *mut L = unmark_ptr(next_raw);
        let upper_bound: Option<u64> = if next_ptr.is_null() {
            None
        } else {
            // SAFETY: next_ptr is valid, protected by guard
            Some(unsafe { (*next_ptr).ikey_bound() })
        };

        // Handle empty leaf reuse
        if leaf.is_empty() && start_index < batch.len() {
            let entry = &batch[start_index];
            let key = Key::new(&entry.key);
            if self.can_reuse_empty_leaf_batch(leaf, &key) {
                self.insert_into_empty_leaf_batch(leaf, lock, &key, &entry.output, guard);
                result.record_insert();
                processed = 1;
                perm = leaf.permutation();
            }
        }

        // Process entries that belong to this leaf
        while start_index + processed < batch.len() {
            let entry = &batch[start_index + processed];

            // Check if this entry belongs to a sibling leaf
            if let Some(bound) = upper_bound {
                if entry.ikey() >= bound {
                    break;
                }
            }

            // Check if leaf has space - if not, stop and let caller retry
            // The next iteration of process_sorted_batch will handle the split
            if perm.size() >= L::WIDTH {
                break;
            }

            // Create key for this entry
            let key = Key::new(&entry.key);
            let is_single_layer = !key.has_suffix();

            // Try to insert this entry
            let insert_result = if is_single_layer {
                self.try_insert_entry_single_layer(
                    leaf,
                    lock,
                    &key,
                    &entry.output,
                    &mut perm,
                    guard,
                )
            } else {
                self.try_insert_entry_multi_layer(
                    leaf,
                    lock,
                    &key,
                    &entry.output,
                    &mut perm,
                    guard,
                )
            };

            match insert_result {
                BatchEntryResult::Inserted => {
                    result.record_insert();
                    processed += 1;
                }
                BatchEntryResult::Updated(old_value) => {
                    result.record_update(old_value);
                    processed += 1;
                }
                BatchEntryResult::NeedsSplit => {
                    // Leaf is full - stop processing
                    // Caller will retry and trigger split via normal insert
                    break;
                }
                BatchEntryResult::NeedsLayerDescent => {
                    // Layer descent needed - mark as failed for individual retry
                    result.record_failure();
                    processed += 1;
                }
                BatchEntryResult::BelongsToSibling => {
                    // Entry belongs to a sibling - stop
                    break;
                }
                BatchEntryResult::Retry => {
                    // Slot being modified - stop and retry
                    break;
                }
            }
        }

        processed
    }

    // ========================================================================
    //  Single Entry Insertion Helpers
    // ========================================================================

    /// Try to insert a single entry in single-layer mode (keys ≤ 8 bytes).
    fn try_insert_entry_single_layer(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &S::Output,
        perm: &mut L::Perm,
        guard: &LocalGuard<'_>,
    ) -> BatchEntryResult<S::Output> {
        let search_result = self.search_for_insert_single_layer(leaf, key, perm);

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                let old_ptr = leaf.leaf_value_ptr(slot);
                if old_ptr.is_null() {
                    return BatchEntryResult::Retry;
                }

                // Update existing value and return old value
                let old_value = self.update_value_in_slot_batch(leaf, lock, slot, value.clone(), guard);
                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey = key.ikey();

                match self.find_usable_slot_batch(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        self.insert_into_slot_batch(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            key,
                            value,
                            guard,
                        );
                        self.count.fetch_add(1, AtomicOrdering::Relaxed);

                        // Update perm for next iteration
                        *perm = leaf.permutation();
                        BatchEntryResult::Inserted
                    }

                    FindSlotResult::NeedsSplit => BatchEntryResult::NeedsSplit,
                }
            }

            InsertSearchResultGeneric::Layer { .. }
            | InsertSearchResultGeneric::Conflict { .. } => {
                // These shouldn't happen in single-layer mode
                BatchEntryResult::Retry
            }
        }
    }

    /// Try to insert a single entry in multi-layer mode (keys > 8 bytes).
    fn try_insert_entry_multi_layer(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &S::Output,
        perm: &mut L::Perm,
        guard: &LocalGuard<'_>,
    ) -> BatchEntryResult<S::Output> {
        let search_result = self.search_for_insert_generic(leaf, key, perm);

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                let old_ptr = leaf.leaf_value_ptr(slot);
                if old_ptr.is_null() {
                    return BatchEntryResult::Retry;
                }

                // Update existing value and return old value
                let old_value = self.update_value_in_slot_batch(leaf, lock, slot, value.clone(), guard);
                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey = key.ikey();

                match self.find_usable_slot_batch(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        self.insert_into_slot_batch(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            key,
                            value,
                            guard,
                        );
                        self.count.fetch_add(1, AtomicOrdering::Relaxed);

                        // Update perm for next iteration
                        *perm = leaf.permutation();
                        BatchEntryResult::Inserted
                    }

                    FindSlotResult::NeedsSplit => BatchEntryResult::NeedsSplit,
                }
            }

            InsertSearchResultGeneric::Layer { .. } => {
                // Layer descent needed - can't handle in batch mode
                BatchEntryResult::NeedsLayerDescent
            }

            InsertSearchResultGeneric::Conflict { slot } => {
                // Suffix conflict - create layer
                // This is complex, mark for individual retry
                let _ = slot; // suppress unused warning
                BatchEntryResult::NeedsLayerDescent
            }
        }
    }

    // ========================================================================
    //  Helper Methods (with _batch suffix to avoid visibility changes)
    // ========================================================================

    /// Find a usable slot for insertion.
    #[inline(always)]
    fn find_usable_slot_batch(&self, leaf: &L, perm: &L::Perm, ikey: u64) -> FindSlotResult {
        if perm.size() >= L::WIDTH {
            return FindSlotResult::NeedsSplit;
        }

        let slot = perm.back();

        if slot == 0 && !leaf.can_reuse_slot0(ikey) {
            let free_count = L::WIDTH - perm.size();

            for offset in 1..free_count {
                let candidate = perm.back_at_offset(offset);
                if candidate != 0 {
                    return FindSlotResult::Found {
                        slot: candidate,
                        back_offset: offset,
                    };
                }
            }

            return FindSlotResult::NeedsSplit;
        }

        FindSlotResult::Found {
            slot,
            back_offset: 0,
        }
    }

    /// Validate post-lock state.
    #[inline(always)]
    fn validate_post_lock_batch(
        &self,
        leaf: &L,
        pre_lock_version: u32,
        pre_lock_perm_raw: <L::Perm as TreePermutation>::Raw,
    ) -> bool {
        !leaf.version().has_changed(pre_lock_version) && leaf.permutation_raw() == pre_lock_perm_raw
    }

    /// Validate membership.
    #[inline(always)]
    fn validate_membership_batch(&self, leaf: &L, key: &Key<'_>) -> Result<(), MembershipError> {
        let next_raw: *mut L = leaf.next_raw();

        if is_marked(next_raw) {
            leaf.wait_for_split();
            return Err(MembershipError::SplitInProgress);
        }

        let next_ptr: *mut L = unmark_ptr(next_raw);

        if !next_ptr.is_null() {
            // SAFETY: next_ptr is valid, protected by guard
            let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };

            if key.ikey() >= next_bound {
                return Err(MembershipError::KeyMovedToSibling);
            }
        }

        Ok(())
    }

    /// Check if an empty leaf can be reused.
    #[inline(always)]
    fn can_reuse_empty_leaf_batch(&self, leaf: &L, key: &Key<'_>) -> bool {
        if leaf.prev().is_null() {
            return true;
        }
        leaf.ikey_bound() == key.ikey()
    }

    /// Insert into an empty leaf.
    fn insert_into_empty_leaf_batch(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) {
        leaf.clear_empty_state();
        let slot = 0;
        self.assign_slot_generic(leaf, lock, slot, key, value, guard);
        let new_perm = L::Perm::make_sorted(1);
        leaf.set_permutation(new_perm);
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Update a value in an existing slot, returning the old value.
    fn update_value_in_slot_batch(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        slot: usize,
        new_value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> S::Output {
        let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        // Clone old value for return BEFORE we store new pointer
        // SAFETY: old_ptr is non-null and came from output_to_raw
        let old_output: S::Output = unsafe { S::output_from_raw(old_ptr) };
        let new_ptr: *mut u8 = S::output_consume_to_raw(new_value);

        lock.mark_insert();
        leaf.set_leaf_value_ptr(slot, new_ptr);

        // Defer retirement of the old value
        if !old_ptr.is_null() {
            // SAFETY: old_ptr came from output_to_raw
            unsafe {
                guard.defer_retire(old_ptr, |ptr, _| {
                    S::cleanup_value_ptr(ptr);
                });
            }
        }

        old_output
    }

    /// Insert a new value into a slot.
    #[expect(clippy::too_many_arguments)]
    fn insert_into_slot_batch(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        slot: usize,
        back_offset: usize,
        logical_pos: usize,
        perm: L::Perm,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) {
        // Assign the slot
        self.assign_slot_generic(leaf, lock, slot, key, value, guard);

        // Update permutation
        let mut new_perm = perm;

        if back_offset > 0 {
            let back_pos = L::WIDTH - 1;
            let chosen_pos = back_pos - back_offset;
            new_perm.swap_free_slots(back_pos, chosen_pos);
        }

        let allocated = new_perm.insert_from_back(logical_pos);
        debug_assert_eq!(allocated, slot, "allocated unexpected slot");

        leaf.set_permutation(new_perm);
    }
}
