use std::sync::atomic::Ordering as AtomicOrdering;

use seize::LocalGuard;

use crate::Permuter;
use crate::leaf15::LeafNode15;
use crate::{
    Linker, MassTreeGeneric, TreeAllocator, key::Key, leaf_trait::TreeLeafNode,
    nodeversion::LockGuard, policy::LeafPolicy,
};

use super::{FindSlotResult, InsertSearchResultGeneric};

// ============================================================================
//  Batch Entry Types
// ============================================================================

/// A single entry in a batch insert operation.
#[must_use]
#[expect(
    missing_debug_implementations,
    reason = "Debug on P::Output may not be available"
)]
pub struct BatchEntry<P: LeafPolicy> {
    /// The key bytes (owned for sorting).
    pub key: Vec<u8>,

    /// The pre-converted output value.
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
    pub old_values: Vec<O>,

    /// Number of entries that failed and need individual retry.
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

    /// Slot is being modified by another thread, retry.
    Retry,
}

// FindSlotResult and MembershipError are shared with insert.rs, defined in super (generic.rs).

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
    /// # Example
    ///
    /// ```rust
    /// use masstree::MassTree15;
    ///
    /// let tree: MassTree15<u64> = MassTree15::new();
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
    /// Takes ownership of the entries to avoid cloning keys and outputs.
    /// Use this when you need finer control over the batch entries,
    /// or when retrying failed entries from a previous batch.
    pub fn insert_batch_entries(
        &self,
        mut entries: Vec<BatchEntry<P>>,
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output> {
        if entries.is_empty() {
            return BatchInsertResult::new();
        }

        // Sort by ikey for cache locality
        entries.sort_unstable_by_key(BatchEntry::ikey);

        self.process_sorted_batch(&entries, guard)
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

                let pre_lock_version = leaf.version().stable();
                let pre_lock_perm_raw = leaf.permutation_raw();
                let mut lock = leaf.version().lock();

                // Validate post-lock state
                if !self.validate_post_lock(leaf, pre_lock_version, pre_lock_perm_raw) {
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
                if self.validate_membership(leaf, &key).is_err() {
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
        let mut processed: usize = 0;
        let mut perm: Permuter = leaf.permutation();

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

        if leaf.is_empty() && start_index < batch.len() {
            let entry: &BatchEntry<P> = &batch[start_index];
            let key: Key<'_> = Key::new(&entry.key);

            if self.can_reuse_empty_leaf(leaf, &key) {
                let deferred: *mut u8 =
                    self.insert_into_empty_leaf_batch(leaf, lock, &key, &entry.output, guard);

                if !deferred.is_null() {
                    deferred_retires.push(deferred);
                }

                result.record_insert();
                processed = 1;
                perm = leaf.permutation();
            }
        }

        while start_index + processed < batch.len() {
            let entry = &batch[start_index + processed];

            if let Some(bound) = upper_bound
                && entry.ikey() >= bound
            {
                break;
            }

            if perm.size() >= LeafNode15::<P>::WIDTH {
                break;
            }

            let key: Key<'_> = Key::new(&entry.key);
            let is_single_layer: bool = !key.has_suffix();

            let insert_result = self.try_insert_entry(
                leaf,
                lock,
                &key,
                &entry.output,
                &mut perm,
                is_single_layer,
                guard,
            );

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

                BatchEntryResult::NeedsSplit
                | BatchEntryResult::NeedsLayerDescent
                | BatchEntryResult::Retry => {
                    break;
                }
            }
        }

        processed
    }

    // ========================================================================
    //  Single Entry Insertion Helpers
    // ========================================================================

    /// Try to insert a single entry into a locked leaf.
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Insertion requires full context")]
    fn try_insert_entry(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &P::Output,
        perm: &mut <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        single_layer_mode: bool,
        guard: &LocalGuard<'_>,
    ) -> BatchEntryResult<P::Output> {
        let search_result = if single_layer_mode {
            self.search_for_insert_single_layer(leaf, key, perm)
        } else {
            self.search_for_insert_generic(leaf, key, perm)
        };

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                let old_value = self.update_existing_value(leaf, lock, slot, value, guard);
                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey = key.ikey();

                match self.find_usable_slot(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let deferred_retire = self.insert_new_value(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            key,
                            value,
                            guard,
                            None,
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
                if single_layer_mode {
                    // Layer/Conflict shouldn't occur in single-layer mode
                    BatchEntryResult::Retry
                } else {
                    // Layer descent or suffix conflict — too complex for batch,
                    // fall back to individual insert
                    BatchEntryResult::NeedsLayerDescent
                }
            }
        }
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
        let slot: usize = 0;
        let deferred_retire: *mut u8 =
            self.assign_slot_generic(leaf, lock, slot, key, value, guard, None);

        // Use Relaxed ordering since we hold the lock - the lock's Release
        // fence on drop provides the necessary synchronization.
        let new_perm: Permuter = <LeafNode15<P> as TreeLeafNode<P>>::Perm::make_sorted(1);
        leaf.set_permutation_relaxed(new_perm);
        self.count.increment();

        deferred_retire
    }
}
