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

use seize::LocalGuard;

use crate::leaf15::LeafNode15;
use crate::{
    Linker, MassTreeGeneric, TreeAllocator,
    key::Key,
    leaf_trait::TreeLeafNode,
    nodeversion::LockGuard,
    policy::{LeafPolicy, RetireHandle},
};

use super::{InsertSearchResultGeneric, TreePermutation};

// ============================================================================
//  Batch Entry Types
// ============================================================================

/// A single entry in a batch insert operation.
///
/// Contains the key bytes and the pre-converted output value.
/// The output is created via `P::into_output()` before sorting to ensure
/// allocation happens exactly once per entry.
#[must_use]
#[expect(
    missing_debug_implementations,
    reason = "Debug on P::Output may not be available"
)]
pub struct BatchEntry<P: LeafPolicy> {
    /// The key bytes (owned for sorting).
    pub key: Vec<u8>,

    /// The pre-converted output value.
    ///
    /// Created via `P::into_output(value)` to ensure the Arc (if any)
    /// is allocated once and reused across retries.
    pub output: P::Output,

    /// Cached ikey for the first 8 bytes (used for sorting).
    ikey: u64,
}

impl<P: LeafPolicy> BatchEntry<P> {
    /// Create a new batch entry from key and value.
    ///
    /// Converts the value to output immediately to ensure single allocation.
    #[inline]
    pub fn new(key: Vec<u8>, value: P::Value) -> Self {
        let ikey = Self::compute_ikey(&key);
        let output = P::into_output(value);
        Self { key, output, ikey }
    }

    /// Create a batch entry from key and pre-converted output.
    ///
    /// Use this when you already have an `P::Output` (e.g., from a previous
    /// failed batch that needs retry).
    #[inline(always)]
    pub fn from_output(key: Vec<u8>, output: P::Output) -> Self {
        let ikey = Self::compute_ikey(&key);
        Self { key, output, ikey }
    }

    /// Compute the ikey (first 8 bytes as big-endian u64).
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "len is bounded by min(key.len(), 8), so slicing is safe"
    )]
    fn compute_ikey(key: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        let len = key.len().min(8);
        buf[..len].copy_from_slice(&key[..len]);
        u64::from_be_bytes(buf)
    }

    /// Get the cached ikey.
    #[inline(always)]
    pub const fn ikey(&self) -> u64 {
        self.ikey
    }

    /// Check if this key has a suffix (> 8 bytes).
    #[inline(always)]
    pub const fn has_suffix(&self) -> bool {
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
/// * `O` - The output type (`ValuePtr<V>` for Box mode, `V` for Inline mode)
#[derive(Debug, Clone)]
#[must_use]
pub struct BatchInsertResult<O> {
    /// Number of new keys inserted.
    pub inserted: usize,

    /// Number of existing keys updated.
    pub updated: usize,

    /// Old values from updated keys (in no particular order).
    ///
    /// For `MassTree24<V>`, this is `Vec<ValuePtr<V>>`.
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
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inserted: 0,
            updated: 0,
            old_values: Vec::with_capacity(capacity),
            failed: 0,
        }
    }

    /// Record a successful new key insertion.
    #[inline(always)]
    pub const fn record_insert(&mut self) {
        self.inserted += 1;
    }

    /// Record a successful update with old value.
    #[inline]
    pub fn record_update(&mut self, old_value: O) {
        self.updated += 1;
        self.old_values.push(old_value);
    }

    /// Record a failed entry.
    #[inline(always)]
    pub const fn record_failure(&mut self) {
        self.failed += 1;
    }

    /// Total entries processed (inserted + updated + failed).
    #[must_use]
    #[inline(always)]
    pub const fn total(&self) -> usize {
        self.inserted + self.updated + self.failed
    }

    /// Check if all entries succeeded.
    #[must_use]
    #[inline(always)]
    pub const fn all_succeeded(&self) -> bool {
        self.failed == 0
    }
}

// ============================================================================
//  Helper Types
// ============================================================================

/// Result of trying to insert a single entry in a batch.
enum BatchEntryResult<O> {
    /// Entry was inserted as a new key. Contains a pointer to a suffix bag
    /// that must be retired after the lock is dropped (null if none).
    Inserted(*mut u8),

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
    /// Key has moved to a sibling leaf - retry traversal (walk right).
    KeyMovedToSibling,
    /// Key is below this leaf's lower bound - must restart from root.
    /// This cannot be recovered by walking right; requires full re-traversal.
    KeyBelowLowerBound,
}

// ============================================================================
//  Batch Insert Implementation
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
    /// let result = tree.insert_batch(entries);
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
    /// This method requires `P::Output: Clone` because entries may need to be
    /// retried after a split. For `ValuePtr<V>`, cloning is trivial (Copy).
    /// For `Copy` types in `Inline` mode, cloning is also cheap.
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption (should not happen in normal operation).
    pub fn insert_batch<I>(&self, entries: I) -> BatchInsertResult<P::Value>
    where
        I: IntoIterator<Item = (Vec<u8>, P::Value)>,
        P::Value: Clone,
    {
        let guard = self.guard();
        let result = self.insert_batch_with_guard(entries, &guard);
        BatchInsertResult {
            inserted: result.inserted,
            updated: result.updated,
            old_values: result
                .old_values
                .iter()
                .map(|o| P::clone_value_from_output(o))
                .collect(),
            failed: result.failed,
        }
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
    /// # Panics
    ///
    /// Panics on internal tree corruption.
    pub fn insert_batch_with_guard<I>(
        &self,
        entries: I,
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output>
    where
        I: IntoIterator<Item = (Vec<u8>, P::Value)>,
    {
        // Convert to BatchEntry (allocates outputs once)
        let mut batch: Vec<BatchEntry<P>> = entries
            .into_iter()
            .map(|(key, value)| BatchEntry::new(key, value))
            .collect();

        if batch.is_empty() {
            return BatchInsertResult::new();
        }

        // Sort by ikey for cache locality and leaf clustering
        batch.sort_unstable_by_key(BatchEntry::ikey);

        // Process the sorted batch
        self.process_sorted_batch(&batch, guard)
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
        entries: &mut [BatchEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output> {
        if entries.is_empty() {
            return BatchInsertResult::new();
        }

        // Sort by ikey for cache locality
        entries.sort_unstable_by_key(BatchEntry::ikey);

        // Convert slice to Vec for processing
        let batch: Vec<BatchEntry<P>> = entries
            .iter()
            .map(|e| BatchEntry::from_output(e.key.clone(), e.output.clone()))
            .collect();

        self.process_sorted_batch(&batch, guard)
    }

    // ========================================================================
    //  Internal Batch Processing
    // ========================================================================

    /// Process a sorted batch of entries.
    ///
    /// Entries must be sorted by ikey before calling this method.
    #[expect(
        clippy::indexing_slicing,
        reason = "Index bounds are checked in the while condition"
    )]
    fn process_sorted_batch(
        &self,
        batch: &[BatchEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output> {
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
                // NOTE: We track leaf_ptr (*mut LeafNode15<P>) to preserve mutable provenance
                let mut leaf_ptr: *mut LeafNode15<P> =
                    self.reach_leaf_concurrent_generic(layer_root, &key, false, guard);

                // B-link advance if needed (returns *mut LeafNode15<P> to preserve provenance)
                let (advanced_ptr, exceeded_hop_limit) =
                    self.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                if exceeded_hop_limit {
                    layer_root = self.root_ptr.load(AtomicOrdering::Acquire);
                    continue 'retry;
                }

                leaf_ptr = advanced_ptr;
                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

                // Capture pre-lock state for OCC validation
                let pre_lock_version = leaf.version().stable();
                let pre_lock_perm_raw = leaf.permutation_raw();

                // Lock the leaf (pure spin - yields cause syscall overhead under contention)
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
                let mut deferred_retires: Vec<*mut u8> = Vec::new();
                let processed = self.insert_batch_into_locked_leaf(
                    leaf,
                    &mut lock,
                    batch,
                    index,
                    &mut result,
                    &mut deferred_retires,
                    guard,
                );

                drop(lock);

                // Retire old suffix bags OUTSIDE the lock
                for ptr in deferred_retires {
                    // SAFETY: ptr is a valid suffix bag pointer from a completed operation.
                    unsafe {
                        LeafNode15::<P>::retire_suffix_bag_ptr(ptr, guard);
                    }
                }

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
                    Err(e) => {
                        // Internal errors indicate bugs - should not happen in normal operation
                        panic!("Batch insert failed unexpectedly: {e:?}. This indicates a bug.");
                    }
                }
                index += 1;
            } else {
                index += entries_processed;
            }
        }

        result
    }

    /// Insert as many entries as possible into a locked leaf.
    ///
    /// Returns the number of entries processed (inserted, updated, or marked failed).
    /// Deferred suffix bag pointers are collected in `deferred_retires` and must be
    /// retired via `retire_suffix_bag_ptr` after the lock is dropped.
    #[expect(
        clippy::too_many_arguments,
        reason = "Batch insertion requires context"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "Index bounds checked in while condition"
    )]
    fn insert_batch_into_locked_leaf(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        batch: &[BatchEntry<P>],
        start_index: usize,
        result: &mut BatchInsertResult<P::Output>,
        deferred_retires: &mut Vec<*mut u8>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let mut processed = 0;
        let mut perm = leaf.permutation();

        // Determine the ikey upper bound for this leaf
        // SAFETY: Called under lock - no concurrent retirement.
        let next_raw: *mut LeafNode15<P> = unsafe { leaf.next_raw_unguarded() };
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);
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
                let deferred =
                    self.insert_into_empty_leaf_batch(leaf, lock, &key, &entry.output, guard);
                if !deferred.is_null() {
                    deferred_retires.push(deferred);
                }
                result.record_insert();
                processed = 1;
                perm = leaf.permutation();
            }
        }

        // Process entries that belong to this leaf
        while start_index + processed < batch.len() {
            let entry = &batch[start_index + processed];

            // Check if this entry belongs to a sibling leaf
            if let Some(bound) = upper_bound
                && entry.ikey() >= bound
            {
                break;
            }

            // Check if leaf has space - if not, stop and let caller retry
            // The next iteration of process_sorted_batch will handle the split
            if perm.size() >= LeafNode15::<P>::WIDTH {
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
                self.try_insert_entry_multi_layer(leaf, lock, &key, &entry.output, &mut perm, guard)
            };

            match insert_result {
                BatchEntryResult::Inserted(deferred) => {
                    if !deferred.is_null() {
                        deferred_retires.push(deferred);
                    }
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
    #[expect(clippy::too_many_arguments, reason = "Insertion requires full context")]
    fn try_insert_entry_single_layer(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &P::Output,
        perm: &mut <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        guard: &LocalGuard<'_>,
    ) -> BatchEntryResult<P::Output> {
        let search_result = self.search_for_insert_single_layer(leaf, key, perm);

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                // Update existing value and return old value
                let old_value = self.update_value_in_slot_batch(leaf, lock, slot, value, guard);
                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey = key.ikey();

                match self.find_usable_slot_batch(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let deferred_retire = self.insert_into_slot_batch(
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
                        self.count.increment();

                        // Update perm for next iteration
                        *perm = leaf.permutation();
                        BatchEntryResult::Inserted(deferred_retire)
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
    #[expect(clippy::too_many_arguments, reason = "Insertion requires full context")]
    fn try_insert_entry_multi_layer(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &P::Output,
        perm: &mut <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        guard: &LocalGuard<'_>,
    ) -> BatchEntryResult<P::Output> {
        let search_result = self.search_for_insert_generic(leaf, key, perm);

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                // Update existing value and return old value
                let old_value = self.update_value_in_slot_batch(leaf, lock, slot, value, guard);
                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey = key.ikey();

                match self.find_usable_slot_batch(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let deferred_retire = self.insert_into_slot_batch(
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
                        self.count.increment();

                        // Update perm for next iteration
                        *perm = leaf.permutation();
                        BatchEntryResult::Inserted(deferred_retire)
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
    #[expect(
        clippy::unused_self,
        reason = "Method for consistency with other helpers"
    )]
    fn find_usable_slot_batch(
        &self,
        leaf: &LeafNode15<P>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
        ikey: u64,
    ) -> FindSlotResult {
        if perm.size() >= LeafNode15::<P>::WIDTH {
            return FindSlotResult::NeedsSplit;
        }

        let slot = perm.back();

        if slot == 0 && !leaf.can_reuse_slot0(ikey) {
            let free_count = LeafNode15::<P>::WIDTH - perm.size();

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
    #[expect(
        clippy::unused_self,
        reason = "Method for consistency with other helpers"
    )]
    fn validate_post_lock_batch(
        &self,
        leaf: &LeafNode15<P>,
        pre_lock_version: u32,
        pre_lock_perm_raw: <<LeafNode15<P> as TreeLeafNode<P>>::Perm as TreePermutation>::Raw,
    ) -> bool {
        !leaf.version().has_changed(pre_lock_version) && leaf.permutation_raw() == pre_lock_perm_raw
    }

    /// Validate membership.
    #[inline(always)]
    #[expect(
        clippy::unused_self,
        reason = "Method for consistency with other helpers"
    )]
    fn validate_membership_batch(
        &self,
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
    ) -> Result<(), MembershipError> {
        // SAFETY: Called under lock - no concurrent retirement.
        let next_raw: *mut LeafNode15<P> = unsafe { leaf.next_raw_unguarded() };

        if Linker::is_marked(next_raw) {
            leaf.wait_for_split();
            return Err(MembershipError::SplitInProgress);
        }

        // Check lower bound for non-leftmost leaves.
        // If key < ikey_bound and prev exists, we're "too far right" due to
        // concurrent splits. Recovery requires restart from root (can't walk left).
        // SAFETY: Called under lock - no concurrent retirement.
        if !unsafe { leaf.prev_unguarded() }.is_null() {
            let lower_bound: u64 = leaf.ikey_bound();
            if key.ikey() < lower_bound {
                return Err(MembershipError::KeyBelowLowerBound);
            }
        }

        // Check upper bound (key hasn't moved to right sibling)
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

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
    #[expect(
        clippy::unused_self,
        reason = "Method for consistency with other helpers"
    )]
    fn can_reuse_empty_leaf_batch(&self, leaf: &LeafNode15<P>, key: &Key<'_>) -> bool {
        // SAFETY: Called under lock - no concurrent retirement.
        if unsafe { leaf.prev_unguarded() }.is_null() {
            return true;
        }
        leaf.ikey_bound() == key.ikey()
    }

    /// Insert into an empty leaf. Returns a pointer to a suffix bag that must be
    /// retired after the lock is dropped (null if none).
    fn insert_into_empty_leaf_batch(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &P::Output,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        leaf.clear_empty_state();
        let slot = 0;
        let deferred_retire = self.assign_slot_generic(leaf, lock, slot, key, value, guard, None);
        let new_perm = <LeafNode15<P> as TreeLeafNode<P>>::Perm::make_sorted(1);
        leaf.set_permutation(new_perm);
        self.count.increment();
        deferred_retire
    }

    /// Update a value in an existing slot, returning the old value.
    #[expect(
        clippy::unused_self,
        reason = "Method for consistency with other helpers"
    )]
    fn update_value_in_slot_batch(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        new_value: &P::Output,
        guard: &LocalGuard<'_>,
    ) -> P::Output {
        // Load old value before overwrite.
        // load_value returns Option<P::Output>; slot is guaranteed occupied (Found path).
        let old_output: P::Output = leaf
            .load_value(slot)
            .expect("slot should have value in Found path");

        lock.mark_insert();

        // Store new value in place; get retirement handle for old data.
        // We hold the lock. Slot contains a terminal value (Found path).
        let retire: RetireHandle = leaf.update_in_place(slot, new_value);

        // Defer retirement of old value data if needed.
        // SAFETY: handle was produced by update_in_place() on this leaf.
        unsafe { P::retire_handle(retire, guard) };

        old_output
    }

    /// Insert a new value into a slot. Returns a pointer to a suffix bag that must be
    /// retired after the lock is dropped (null if none).
    #[expect(clippy::too_many_arguments)]
    fn insert_into_slot_batch(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        back_offset: usize,
        logical_pos: usize,
        perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        key: &Key<'_>,
        value: &P::Output,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        // Assign the slot
        let deferred_retire = self.assign_slot_generic(leaf, lock, slot, key, value, guard, None);

        // Update permutation
        let mut new_perm = perm;

        if back_offset > 0 {
            let back_pos = LeafNode15::<P>::WIDTH - 1;
            let chosen_pos = back_pos - back_offset;
            new_perm.swap_free_slots(back_pos, chosen_pos);
        }

        let allocated = new_perm.insert_from_back(logical_pos);
        debug_assert_eq!(allocated, slot, "allocated unexpected slot");

        leaf.set_permutation(new_perm);
        deferred_retire
    }
}
