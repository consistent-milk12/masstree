use std::fmt::{self as StdFmt, Debug, Formatter};
use std::marker::PhantomData;

use seize::LocalGuard;

use std::ptr as StdPtr;

use crate::Permuter;
use crate::leaf15::LeafNode15;
use crate::{
    MassTreeGeneric, TreeAllocator, ValueRef, key::Key, nodeversion::LockGuard, policy::LeafPolicy,
};

use super::{FindSlotResult, InsertSearchResultGeneric};

/// Max suffix bag retires per locked batch insert.
/// One per slot (WIDTH=15) plus one for empty-leaf reuse = 16.
const MAX_BATCH_RETIRES: usize = crate::leaf15::WIDTH_15 + 1;

/// Buffer for deferred suffix bag retirements during batch insertion.
///
/// Collects pointers to old suffix bags that must be retired outside the
/// leaf lock. Bounded by `MAX_BATCH_RETIRES`.
struct RetireBuf<'g, 'c, P: LeafPolicy> {
    guard: &'g LocalGuard<'c>,
    _policy: PhantomData<P>,
    ptrs: [*mut u8; MAX_BATCH_RETIRES],
    count: usize,
}

impl<'g, 'c, P: LeafPolicy> RetireBuf<'g, 'c, P> {
    #[inline]
    const fn new(guard: &'g LocalGuard<'c>) -> Self {
        Self {
            guard,
            _policy: PhantomData,
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
}

impl<P: LeafPolicy> Drop for RetireBuf<'_, '_, P> {
    fn drop(&mut self) {
        for &ptr in &self.ptrs[..self.count] {
            // SAFETY: Each pointer was displaced exactly once by this batch,
            // is no longer reachable, and remains protected by the same guard.
            unsafe { LeafNode15::<P>::retire_suffix_bag_ptr(ptr, self.guard) };
        }
    }
}

// ============================================================================
//  Batch Entry Types
// ============================================================================

/// A single entry in a batch insert operation.
///
/// The entry owns a real user value; it never exposes a borrowed tree value
/// handle that could alias a value still stored in the tree. Dropping an
/// entry drops the value normally.
#[must_use]
pub struct BatchEntry<P: LeafPolicy> {
    /// The key bytes (owned for sorting).
    key: Vec<u8>,

    /// The owned user value, converted to a storage output only when the
    /// entry is inserted.
    value: Option<P::Value>,

    /// Cached ikey for the first 8 bytes (used for sorting).
    ikey: u64,

    /// Input-order tiebreaker, assigned when the batch is submitted.
    seq: usize,
}

impl<P: LeafPolicy> BatchEntry<P> {
    /// Create a new batch entry from key and value.
    ///
    /// # Panics
    ///
    /// Panics if `key` is longer than [`MAX_KEY_LENGTH`](crate::key::MAX_KEY_LENGTH).
    /// Validation happens here, before any tree mutation.
    #[inline]
    pub fn new(key: Vec<u8>, value: P::Value) -> Self {
        assert!(
            key.len() <= crate::key::MAX_KEY_LENGTH,
            "batch key length {} exceeds maximum {}",
            key.len(),
            crate::key::MAX_KEY_LENGTH
        );

        let ikey: u64 = Self::compute_ikey(&key);

        Self {
            key,
            value: Some(value),
            ikey,
            seq: 0,
        }
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

    /// Get the key bytes.
    #[inline(always)]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get a reference to the owned value, if it has not been consumed.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Option::as_ref is not const on this toolchain"
    )]
    #[inline(always)]
    pub fn value(&self) -> Option<&P::Value> {
        self.value.as_ref()
    }

    /// Take the owned value. Returns `None` if it was already consumed.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "BatchEntry has a destructor, so this cannot be const"
    )]
    #[inline(always)]
    pub fn into_value(mut self) -> Option<P::Value> {
        self.value.take()
    }

    /// Check if this key has a suffix (> 8 bytes).
    #[inline(always)]
    pub const fn has_suffix(&self) -> bool {
        self.key.len() > 8
    }
}

impl<P: LeafPolicy> Debug for BatchEntry<P>
where
    P::Value: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("BatchEntry")
            .field("key", &self.key)
            .field("value", &self.value)
            .field("ikey", &self.ikey)
            .finish_non_exhaustive()
    }
}

// ============================================================================
//  Batch Insert Result
// ============================================================================

/// Result of a batch insert operation.
///
/// # Type Parameter
///
/// * `O` - The value representation returned for updated keys. `insert_batch`
///   returns owned `P::Value`s; the guard-scoped variants return
///   [`ValueRef`](crate::ValueRef)s tied to the caller's guard.
#[derive(Debug, Clone)]
#[must_use]
pub struct BatchInsertResult<O> {
    /// Number of new keys inserted.
    pub inserted: usize,

    /// Number of existing keys updated.
    pub updated: usize,

    /// Old values from updated keys (in no particular order).
    pub old_values: Vec<O>,
}

impl<O> Default for BatchInsertResult<O> {
    fn default() -> Self {
        Self {
            inserted: 0,
            updated: 0,
            old_values: Vec::new(),
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

    /// Total entries processed (inserted + updated).
    ///
    /// A batch either applies an entry or panics on an internal invariant
    /// violation; there is no per-entry failure counter.
    #[must_use]
    #[inline(always)]
    pub const fn total(&self) -> usize {
        self.inserted + self.updated
    }

    /// Transform every old value with `f`, preserving the counters.
    #[inline]
    pub fn map_old_values<U>(self, f: impl FnMut(O) -> U) -> BatchInsertResult<U> {
        BatchInsertResult {
            inserted: self.inserted,
            updated: self.updated,
            old_values: self.old_values.into_iter().map(f).collect(),
        }
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
    /// Returns owned old values, so the result is independent of any guard.
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
    /// Panics if a key exceeds the maximum key length (before mutating the
    /// tree) or on internal tree corruption.
    pub fn insert_batch<I>(&self, entries: I) -> BatchInsertResult<P::Value>
    where
        I: IntoIterator<Item = (Vec<u8>, P::Value)>,
        P::Value: Clone,
    {
        let guard: LocalGuard<'_> = self.guard();
        let result: BatchInsertResult<ValueRef<'_, P::Value>> =
            self.insert_batch_with_guard(entries, &guard);

        result.map_old_values(ValueRef::into_owned)
    }

    /// Insert multiple key-value pairs using an existing guard.
    ///
    /// Old values are guard-scoped [`ValueRef`](crate::ValueRef)s: write-through
    /// values are atomic copies, boxed values stay alive until the guard is
    /// dropped even after retirement.
    ///
    /// Use this when performing multiple batch operations under the same
    /// guard to amortize guard creation overhead.
    ///
    /// # Panics
    ///
    /// Panics if a key exceeds the maximum key length (before mutating the
    /// tree) or on internal tree corruption.
    pub fn insert_batch_with_guard<'g, I>(
        &self,
        entries: I,
        guard: &'g LocalGuard<'_>,
    ) -> BatchInsertResult<ValueRef<'g, P::Value>>
    where
        I: IntoIterator<Item = (Vec<u8>, P::Value)>,
    {
        self.verify_guard(guard);
        // Collect owning entries and validate every key before any conversion
        // to non-owning storage handles or mutation of the tree.
        let batch = entries
            .into_iter()
            .map(|(key, value)| BatchEntry::new(key, value))
            .collect();
        self.insert_batch_entries(batch, guard)
    }

    /// Insert pre-constructed, value-owning batch entries.
    ///
    /// Equal keys are processed in input order (last value wins). Old values
    /// are guarded borrows for boxed objects or owned write-through snapshots.
    pub fn insert_batch_entries<'g>(
        &self,
        mut entries: Vec<BatchEntry<P>>,
        guard: &'g LocalGuard<'_>,
    ) -> BatchInsertResult<ValueRef<'g, P::Value>> {
        self.verify_guard(guard);
        for (seq, entry) in entries.iter_mut().enumerate() {
            entry.seq = seq;
        }
        entries.sort_unstable_by(|a, b| (a.ikey, &a.key, a.seq).cmp(&(b.ikey, &b.key, b.seq)));
        self.process_sorted_batch(&mut entries, guard)
    }

    /// Try one entry while the leaf is locked. Values remain owned until the
    /// destination is known, including on split/layer fallbacks and unwinding.
    fn try_insert_batch_entry<'g>(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        perm: &mut Permuter,
        entry: &mut BatchEntry<P>,
        guard: &'g LocalGuard<'_>,
    ) -> BatchEntryResult<ValueRef<'g, P::Value>> {
        let key = Key::new(&entry.key);
        let search = if key.has_suffix() {
            self.search_for_insert_generic(leaf, &key, perm)
        } else {
            self.search_for_insert_single_layer(leaf, &key, perm)
        };
        match search {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }
                let value = entry.value.take().expect("batch entry already consumed");
                let old = if P::can_write_through() {
                    ValueRef::Owned(
                        self.update_existing_value_write_through(leaf, lock, slot, &value),
                    )
                } else {
                    let output = P::into_output(value);
                    let old = self.update_existing_value(leaf, lock, slot, &output, guard);
                    // SAFETY: The displaced value was retired under this guard.
                    // It remains alive for the returned borrow's lifetime.
                    unsafe { P::output_to_value_ref(&old) }
                };
                BatchEntryResult::Updated(old)
            }
            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let FindSlotResult::Found { slot, back_offset } =
                    self.find_usable_slot(leaf, perm, key.ikey())
                else {
                    return BatchEntryResult::NeedsSplit;
                };
                let output =
                    P::into_output(entry.value.take().expect("batch entry already consumed"));
                let deferred = self.insert_new_value(
                    leaf,
                    lock,
                    slot,
                    back_offset,
                    logical_pos,
                    *perm,
                    &key,
                    &output,
                    guard,
                    None,
                );
                self.count.increment();
                *perm = leaf.permutation();
                BatchEntryResult::Inserted(deferred)
            }
            InsertSearchResultGeneric::Layer { .. }
            | InsertSearchResultGeneric::Conflict { .. } => BatchEntryResult::NeedsLayerDescent,
        }
    }

    fn fallback_insert_batch_entry<'g>(
        &self,
        entry: &mut BatchEntry<P>,
        guard: &'g LocalGuard<'_>,
    ) -> Option<ValueRef<'g, P::Value>> {
        let value = entry.value.take().expect("batch entry already consumed");
        let mut key = Key::new(&entry.key);
        if P::can_write_through() {
            self.insert_concurrent_value(&mut key, value, guard)
                .expect("batch fallback insert failed")
                .map(ValueRef::Owned)
        } else {
            self.insert_concurrent_generic(&mut key, P::into_output(value), guard)
                .expect("batch fallback insert failed")
                .map(|output| {
                    // SAFETY: The old value was retired under this guard and
                    // remains alive for the returned borrow's lifetime.
                    unsafe { P::output_to_value_ref(&output) }
                })
        }
    }

    /// Process a sorted slice of value-owning entries for either policy.
    ///
    /// Handles traversal, OCC validation, locking, locked-leaf insertion,
    /// retirement, and fallback to individual insert on split.
    fn process_sorted_batch<'g>(
        &self,
        batch: &mut [BatchEntry<P>],
        guard: &'g LocalGuard<'_>,
    ) -> BatchInsertResult<ValueRef<'g, P::Value>> {
        let mut result = BatchInsertResult::new();
        let mut index: usize = 0;

        while index < batch.len() {
            let key: Key<'_> = Key::new(&batch[index].key);

            let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);

            let entries_processed: usize = 'retry: loop {
                layer_root = self.maybe_parent_generic(layer_root, guard);

                let mut leaf_ptr: *mut LeafNode15<P> =
                    self.reach_leaf_concurrent_generic(layer_root, &key, false, guard);

                let (advanced_ptr, exceeded_hop_limit) =
                    self.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                if exceeded_hop_limit {
                    layer_root = self.load_root_ptr_generic(guard);
                    continue 'retry;
                }

                leaf_ptr = advanced_ptr;
                // SAFETY: Traversal loaded this live leaf through the guard;
                // retirement cannot reclaim it until the guard is released.
                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

                let pre_lock_version: u32 = leaf.version().stable();
                let pre_lock_perm_raw: u64 = leaf.permutation_raw();
                // Declared before the lock so unwinding unlocks first, then
                // retires any suffix bags displaced by earlier entries.
                let mut retire_buf: RetireBuf<'_, '_, P> = RetireBuf::new(guard);

                let mut lock = leaf.version().lock_bounded();

                if !self.validate_post_lock(leaf, pre_lock_version, pre_lock_perm_raw) {
                    drop(lock);
                    continue 'retry;
                }

                if leaf.deleted_layer() {
                    drop(lock);
                    layer_root = self.load_root_ptr_generic(guard);
                    continue 'retry;
                }

                if self.validate_membership(leaf, &key).is_err() {
                    drop(lock);
                    continue 'retry;
                }

                let processed: usize = self.insert_batch_into_locked_leaf_inner(
                    leaf,
                    &mut lock,
                    batch,
                    index,
                    &mut result,
                    &mut retire_buf,
                    guard,
                );

                drop(lock);

                drop(retire_buf);

                if processed == 0 {
                    break 'retry 0;
                }

                break 'retry processed;
            };

            if entries_processed == 0 {
                if let Some(old) = self.fallback_insert_batch_entry(&mut batch[index], guard) {
                    result.record_update(old);
                } else {
                    result.record_insert();
                }
                index += 1;
            } else {
                index += entries_processed;
            }
        }

        result
    }

    /// Insert a run of entries while holding one leaf lock.
    ///
    /// Inserts as many entries as possible into a locked leaf. Handles
    /// empty leaf reuse, upper bound clustering, and per-entry dispatch.
    #[expect(
        clippy::too_many_arguments,
        reason = "Batch insertion requires context"
    )]
    fn insert_batch_into_locked_leaf_inner<'g>(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        batch: &mut [BatchEntry<P>],
        start_index: usize,
        result: &mut BatchInsertResult<ValueRef<'g, P::Value>>,
        retire_buf: &mut RetireBuf<'_, '_, P>,
        guard: &'g LocalGuard<'_>,
    ) -> usize {
        let mut processed: usize = 0;
        let mut perm: Permuter = leaf.permutation();

        // A sibling may retire independently; use the guard's protected load.
        let next_ptr: *mut LeafNode15<P> = leaf.safe_next(guard);
        let upper_bound: Option<u64> = if next_ptr.is_null() {
            None
        } else {
            // SAFETY: next_ptr is valid, protected by guard
            Some(unsafe { (*next_ptr).ikey_bound() })
        };

        if leaf.is_empty()
            && start_index < batch.len()
            && self.can_reuse_empty_leaf(leaf, &Key::new(&batch[start_index].key))
        {
            let entry = &mut batch[start_index];
            let output = P::into_output(entry.value.take().expect("batch entry already consumed"));
            let deferred = self.insert_into_empty_leaf(
                leaf,
                lock,
                &Key::new(&entry.key),
                &output,
                guard,
                None,
            );
            self.count.increment();

            if !deferred.is_null() {
                retire_buf.push(deferred);
            }

            result.record_insert();
            processed = 1;
            perm = leaf.permutation();
        }

        while start_index + processed < batch.len() {
            let entry_index: usize = start_index + processed;

            if let Some(bound) = upper_bound
                && batch[entry_index].ikey >= bound
            {
                break;
            }

            let insert_result =
                self.try_insert_batch_entry(leaf, lock, &mut perm, &mut batch[entry_index], guard);

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
}
