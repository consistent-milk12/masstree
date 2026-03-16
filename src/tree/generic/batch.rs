use std::sync::atomic::Ordering as AtomicOrdering;

use seize::LocalGuard;

use std::ptr as StdPtr;

use crate::Permuter;
use crate::leaf15::LeafNode15;
use crate::{
    Linker, MassTreeGeneric, TreeAllocator, key::Key, leaf_trait::TreeLeafNode,
    nodeversion::LockGuard, policy::LeafPolicy,
};

use super::{FindSlotResult, InsertSearchResultGeneric};

/// Max suffix bag retires per locked batch insert.
/// One per slot (WIDTH=15) plus one for empty-leaf reuse = 16.
const MAX_BATCH_RETIRES: usize = crate::leaf15::WIDTH_15 + 1;

/// Buffer for deferred suffix bag retirements during batch insertion.
///
/// Collects pointers to old suffix bags that must be retired outside the
/// leaf lock. Bounded by `MAX_BATCH_RETIRES`.
struct RetireBuf {
    ptrs: [*mut u8; MAX_BATCH_RETIRES],
    count: usize,
}

impl RetireBuf {
    #[inline]
    const fn new() -> Self {
        Self {
            ptrs: [StdPtr::null_mut(); MAX_BATCH_RETIRES],
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, ptr: *mut u8) {
        debug_assert!(self.count < self.ptrs.len());
        self.ptrs[self.count] = ptr;
        self.count += 1;
    }

    #[inline]
    fn as_slice(&self) -> &[*mut u8] {
        &self.ptrs[..self.count]
    }
}

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
        let ikey: u64 = Self::compute_ikey(&key);
        let output: P::Output = P::into_output(value);

        Self { key, output, ikey }
    }

    /// Create a batch entry from key and pre-converted output.
    #[inline(always)]
    pub fn from_output(key: Vec<u8>, output: P::Output) -> Self {
        let ikey: u64 = Self::compute_ikey(&key);

        Self { key, output, ikey }
    }

    /// Compute the ikey (first 8 bytes as big-endian u64).
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "len is bounded by min(key.len(), 8), so slicing is safe"
    )]
    fn compute_ikey(key: &[u8]) -> u64 {
        let mut buf: [u8; 8] = [0u8; 8];
        let len: usize = key.len().min(8);
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

/// A single entry in a batch insert operation that defers output allocation.
#[must_use]
struct BatchValueEntry<P: LeafPolicy> {
    /// The key bytes (owned for sorting).
    key: Vec<u8>,

    /// Deferred value payload. Taken only when a new slot is inserted or a
    /// slow-path retry must fall back to the generic insert path.
    value: Option<P::Value>,

    /// Cached ikey for the first 8 bytes (used for sorting).
    ikey: u64,
}

impl<P: LeafPolicy> BatchValueEntry<P> {
    #[inline]
    fn new(key: Vec<u8>, value: P::Value) -> Self {
        let ikey: u64 = BatchEntry::<P>::compute_ikey(&key);

        Self {
            key,
            value: Some(value),
            ikey,
        }
    }

    #[inline(always)]
    const fn ikey(&self) -> u64 {
        self.ikey
    }

    #[inline(always)]
    fn value_ref(&self) -> &P::Value {
        debug_assert!(self.value.is_some(), "batch value already consumed");
        // SAFETY: value is always Some until take_value is called exactly once.
        // The batch processing loop guarantees each entry hits either the update
        // path (value_ref) or the insert path (take_value), never both.
        unsafe { self.value.as_ref().unwrap_unchecked() }
    }

    #[inline(always)]
    fn take_value(&mut self) -> P::Value {
        debug_assert!(self.value.is_some(), "batch value already consumed");
        // SAFETY: see value_ref. Each entry is consumed exactly once.
        unsafe { self.value.take().unwrap_unchecked() }
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
        let guard: LocalGuard<'_> = self.guard();
        if P::CAN_WRITE_THROUGH {
            self.insert_batch_values_with_guard(entries, &guard)
        } else {
            let result: BatchInsertResult<P::Output> =
                self.insert_batch_with_guard(entries, &guard);

            BatchInsertResult {
                inserted: result.inserted,
                updated: result.updated,
                old_values: result
                    .old_values
                    .iter()
                    .map(|o: &P::Output| P::clone_value_from_output(o))
                    .collect(),
                failed: result.failed,
            }
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
        if P::CAN_WRITE_THROUGH {
            let value_result: BatchInsertResult<P::Value> =
                self.insert_batch_values_with_guard(entries, guard);

            return BatchInsertResult {
                inserted: value_result.inserted,
                updated: value_result.updated,
                old_values: value_result
                    .old_values
                    .into_iter()
                    .map(P::into_output)
                    .collect(),
                failed: value_result.failed,
            };
        }

        // Convert to BatchEntry (allocates outputs once)
        let mut batch: Vec<BatchEntry<P>> = entries
            .into_iter()
            .map(|(key, value): (Vec<u8>, P::Value)| BatchEntry::new(key, value))
            .collect();

        if batch.is_empty() {
            return BatchInsertResult::new();
        }

        // Sort by ikey for cache locality and leaf clustering
        batch.sort_unstable_by_key(BatchEntry::ikey);

        // Process the sorted batch
        self.process_sorted_batch(&batch, guard)
    }

    fn insert_batch_values_with_guard<I>(
        &self,
        entries: I,
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Value>
    where
        I: IntoIterator<Item = (Vec<u8>, P::Value)>,
    {
        let mut batch: Vec<BatchValueEntry<P>> = entries
            .into_iter()
            .map(|(key, value): (Vec<u8>, P::Value)| BatchValueEntry::new(key, value))
            .collect();

        if batch.is_empty() {
            return BatchInsertResult::new();
        }

        batch.sort_unstable_by_key(BatchValueEntry::ikey);
        self.process_sorted_value_batch(&mut batch, guard)
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
    #[expect(
        clippy::too_many_lines,
        reason = "Batch processing loop with pre-allocation and retry logic"
    )]
    fn process_sorted_batch(
        &self,
        batch: &[BatchEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output> {
        let mut result: BatchInsertResult<P::Output> =
            BatchInsertResult::with_capacity(batch.len() / 4);
        let mut index: usize = 0;

        while index < batch.len() {
            // Get the current entry to find its leaf
            let entry: &BatchEntry<P> = &batch[index];
            let key: Key<'_> = Key::new(&entry.key);

            // Load root pointer
            let mut layer_root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

            // Retry loop for finding and locking the correct leaf
            let entries_processed: usize = 'retry: loop {
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

                let pre_lock_version: u32 = leaf.version().stable();
                let pre_lock_perm_raw: u64 = leaf.permutation_raw();
                let pre_lock_perm: Permuter = leaf.permutation();

                // Pre-allocate suffix buffer outside the lock to reduce
                // critical section time, matching the pattern in insert.rs.
                let pre_allocated_vec: Option<Vec<u8>> = if key.has_suffix() {
                    Self::maybe_pre_allocate_suffix(&key, pre_lock_perm.size())
                } else {
                    None
                };

                let mut lock = leaf.version().lock_bounded();

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

                // Now we hold the lock - process as many entries as fit in this leaf.
                let mut retire_buf = RetireBuf::new();
                let processed: usize = self.insert_batch_into_locked_leaf(
                    leaf,
                    &mut lock,
                    batch,
                    index,
                    &mut result,
                    &mut retire_buf,
                    guard,
                    pre_allocated_vec,
                );

                drop(lock);

                // Retire old suffix bags OUTSIDE the lock
                for &ptr in retire_buf.as_slice() {
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
                let entry: &BatchEntry<P> = &batch[index];
                let mut key: Key<'_> = Key::new(&entry.key);

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

    #[expect(
        clippy::indexing_slicing,
        reason = "Index bounds are checked in the while condition"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "Batch processing loop with deferred allocation and retry logic"
    )]
    fn process_sorted_value_batch(
        &self,
        batch: &mut [BatchValueEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Value> {
        let mut result: BatchInsertResult<P::Value> =
            BatchInsertResult::with_capacity(batch.len() / 4);
        let mut index: usize = 0;

        while index < batch.len() {
            let entry: &BatchValueEntry<P> = &batch[index];
            let key: Key<'_> = Key::new(&entry.key);

            let mut layer_root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

            let entries_processed: usize = 'retry: loop {
                layer_root = self.maybe_parent_generic(layer_root);

                let mut leaf_ptr: *mut LeafNode15<P> =
                    self.reach_leaf_concurrent_generic(layer_root, &key, false, guard);

                let (advanced_ptr, exceeded_hop_limit) =
                    self.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                if exceeded_hop_limit {
                    layer_root = self.root_ptr.load(AtomicOrdering::Acquire);
                    continue 'retry;
                }

                leaf_ptr = advanced_ptr;
                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

                let pre_lock_version: u32 = leaf.version().stable();
                let pre_lock_perm_raw: u64 = leaf.permutation_raw();
                let pre_lock_perm: Permuter = leaf.permutation();
                let pre_allocated_vec: Option<Vec<u8>> = if key.has_suffix() {
                    Self::maybe_pre_allocate_suffix(&key, pre_lock_perm.size())
                } else {
                    None
                };

                let mut lock = leaf.version().lock_bounded();

                if !self.validate_post_lock(leaf, pre_lock_version, pre_lock_perm_raw) {
                    drop(lock);
                    continue 'retry;
                }

                if leaf.deleted_layer() {
                    drop(lock);
                    layer_root = self.root_ptr.load(AtomicOrdering::Acquire);
                    continue 'retry;
                }

                if self.validate_membership(leaf, &key).is_err() {
                    drop(lock);
                    continue 'retry;
                }

                let mut retire_buf = RetireBuf::new();
                let processed: usize = self.insert_value_batch_into_locked_leaf(
                    leaf,
                    &mut lock,
                    batch,
                    index,
                    &mut result,
                    &mut retire_buf,
                    guard,
                    pre_allocated_vec,
                );

                drop(lock);

                for &ptr in retire_buf.as_slice() {
                    unsafe {
                        LeafNode15::<P>::retire_suffix_bag_ptr(ptr, guard);
                    }
                }

                if processed == 0 {
                    break 'retry 0;
                }

                break 'retry processed;
            };

            if entries_processed == 0 {
                let entry: &mut BatchValueEntry<P> = &mut batch[index];
                let value: P::Value = entry.take_value();
                let mut key: Key<'_> = Key::new(&entry.key);

                match self.insert_concurrent_value(&mut key, value, guard) {
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
        retire_buf: &mut RetireBuf,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> usize {
        let mut processed: usize = 0;
        let mut perm: Permuter = leaf.permutation();
        let mut pre_allocated_vec: Option<Vec<u8>> = pre_allocated;

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
                    retire_buf.push(deferred);
                }

                result.record_insert();
                processed = 1;
                perm = leaf.permutation();
            }
        }

        while start_index + processed < batch.len() {
            let entry: &BatchEntry<P> = &batch[start_index + processed];

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
                pre_allocated_vec.take(),
            );

            match insert_result {
                BatchEntryResult::Inserted(deferred) => {
                    if !deferred.is_null() {
                        retire_buf.push(deferred);
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

    #[expect(
        clippy::too_many_arguments,
        reason = "Batch insertion requires context"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "Index bounds checked in while condition"
    )]
    fn insert_value_batch_into_locked_leaf(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        batch: &mut [BatchValueEntry<P>],
        start_index: usize,
        result: &mut BatchInsertResult<P::Value>,
        retire_buf: &mut RetireBuf,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> usize {
        let mut processed: usize = 0;
        let mut perm: Permuter = leaf.permutation();
        let mut pre_allocated_vec: Option<Vec<u8>> = pre_allocated;

        let next_raw: *mut LeafNode15<P> = unsafe { leaf.next_raw_unguarded() };
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);
        let upper_bound: Option<u64> = if next_ptr.is_null() {
            None
        } else {
            Some(unsafe { (*next_ptr).ikey_bound() })
        };

        if leaf.is_empty() && start_index < batch.len() {
            let can_reuse: bool = {
                let entry: &BatchValueEntry<P> = &batch[start_index];
                let key: Key<'_> = Key::new(&entry.key);
                self.can_reuse_empty_leaf(leaf, &key)
            };

            if can_reuse {
                let value: P::Value = batch[start_index].take_value();
                let key: Key<'_> = Key::new(&batch[start_index].key);
                let deferred: *mut u8 =
                    self.insert_into_empty_leaf_batch_value(leaf, lock, &key, value, guard);

                if !deferred.is_null() {
                    retire_buf.push(deferred);
                }

                result.record_insert();
                processed = 1;
                perm = leaf.permutation();
            }
        }

        while start_index + processed < batch.len() {
            let entry_index: usize = start_index + processed;

            if let Some(bound) = upper_bound
                && batch[entry_index].ikey() >= bound
            {
                break;
            }

            if perm.size() >= LeafNode15::<P>::WIDTH {
                break;
            }

            let insert_result = self.try_insert_value_entry(
                leaf,
                lock,
                &mut batch[entry_index],
                &mut perm,
                guard,
                pre_allocated_vec.take(),
            );

            match insert_result {
                BatchEntryResult::Inserted(deferred) => {
                    if !deferred.is_null() {
                        retire_buf.push(deferred);
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
        pre_allocated: Option<Vec<u8>>,
    ) -> BatchEntryResult<P::Output> {
        let search_result: InsertSearchResultGeneric = if single_layer_mode {
            self.search_for_insert_single_layer(leaf, key, perm)
        } else {
            self.search_for_insert_generic(leaf, key, perm)
        };

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                let old_value: P::Output =
                    self.update_existing_value(leaf, lock, slot, value, guard);

                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey: u64 = key.ikey();

                match self.find_usable_slot(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let deferred_retire: *mut u8 = self.insert_new_value(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            key,
                            value,
                            guard,
                            pre_allocated,
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
                    // Layer descent or suffix conflict
                    BatchEntryResult::NeedsLayerDescent
                }
            }
        }
    }

    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Insertion requires full context")]
    fn try_insert_value_entry(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        entry: &mut BatchValueEntry<P>,
        perm: &mut <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> BatchEntryResult<P::Value> {
        let key: Key<'_> = Key::new(&entry.key);
        let single_layer_mode: bool = !key.has_suffix();

        let search_result: InsertSearchResultGeneric = if single_layer_mode {
            self.search_for_insert_single_layer(leaf, &key, perm)
        } else {
            self.search_for_insert_generic(leaf, &key, perm)
        };

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                let old_value: P::Value =
                    self.update_existing_value_write_through(leaf, lock, slot, entry.value_ref());

                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey: u64 = key.ikey();

                match self.find_usable_slot(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let output: P::Output = P::into_output(entry.take_value());
                        let insert_key: Key<'_> = Key::new(&entry.key);
                        let deferred_retire: *mut u8 = self.insert_new_value(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            &insert_key,
                            &output,
                            guard,
                            pre_allocated,
                        );
                        self.count.increment();

                        *perm = leaf.permutation();

                        BatchEntryResult::Inserted(deferred_retire)
                    }

                    FindSlotResult::NeedsSplit => BatchEntryResult::NeedsSplit,
                }
            }

            InsertSearchResultGeneric::Layer { .. }
            | InsertSearchResultGeneric::Conflict { .. } => {
                if single_layer_mode {
                    BatchEntryResult::Retry
                } else {
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

        let new_perm: Permuter = <LeafNode15<P> as TreeLeafNode<P>>::Perm::make_sorted(1);
        leaf.set_permutation_relaxed(new_perm);
        self.count.increment();

        deferred_retire
    }

    fn insert_into_empty_leaf_batch_value(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: P::Value,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        let output: P::Output = P::into_output(value);
        self.insert_into_empty_leaf_batch(leaf, lock, key, &output, guard)
    }
}
