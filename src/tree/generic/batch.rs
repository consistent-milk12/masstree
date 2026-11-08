use std::fmt::{self as StdFmt, Debug, Formatter};
use std::sync::atomic::Ordering as AtomicOrdering;

use seize::LocalGuard;

use std::ptr as StdPtr;

use crate::Permuter;
use crate::leaf15::LeafNode15;
use crate::{
    Linker, MassTreeGeneric, TreeAllocator, ValueRef,
    key::Key,
    leaf_trait::TreeLeafNode,
    nodeversion::LockGuard,
    policy::{LeafPolicy, PreparedValue},
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
            .finish()
    }
}

/// Internal entry with a pre-converted output allocation.
///
/// Conversion happens once, outside the tree locks. Every entry is consumed by
/// the batch loop (the fallback path inserts the remainder individually), so
/// the output is never dropped without being published.
struct OutputEntry<P: LeafPolicy> {
    key: Vec<u8>,
    output: P::Output,
    ikey: u64,
    /// Original input position, used as the final sort tiebreaker so equal
    /// keys are processed in input order (last value wins deterministically).
    seq: usize,
}

impl<P: LeafPolicy> OutputEntry<P> {
    #[inline]
    fn from_value(key: Vec<u8>, value: P::Value, seq: usize) -> Self {
        assert!(
            key.len() <= crate::key::MAX_KEY_LENGTH,
            "batch key length {} exceeds maximum {}",
            key.len(),
            crate::key::MAX_KEY_LENGTH
        );

        let ikey: u64 = BatchEntry::<P>::compute_ikey(&key);
        let output: P::Output = P::into_output(value);

        Self {
            key,
            output,
            ikey,
            seq,
        }
    }

    #[inline(always)]
    const fn ikey(&self) -> u64 {
        self.ikey
    }
}

/// Sort key for a batch entry: ikey cluster, then full key bytes, then input
/// order. Sorting by only the cached ikey lets an unstable sort reorder equal
/// keys arbitrarily, changing which duplicate value wins.
#[inline(always)]
const fn batch_sort_key(ikey: u64, key: &[u8], seq: usize) -> (u64, &[u8], usize) {
    (ikey, key, seq)
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

    /// Original input position (stable duplicate ordering).
    seq: usize,
}

impl<P: LeafPolicy> BatchValueEntry<P> {
    #[inline]
    fn new(key: Vec<u8>, value: P::Value, seq: usize) -> Self {
        assert!(
            key.len() <= crate::key::MAX_KEY_LENGTH,
            "batch key length {} exceeds maximum {}",
            key.len(),
            crate::key::MAX_KEY_LENGTH
        );

        let ikey: u64 = BatchEntry::<P>::compute_ikey(&key);

        Self {
            key,
            value: Some(value),
            ikey,
            seq,
        }
    }

    #[inline(always)]
    const fn ikey(&self) -> u64 {
        self.ikey
    }

    #[inline(always)]
    const fn seq(&self) -> usize {
        self.seq
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
//  Batch Operations Trait (unifies generic and write-through batch paths)
// ============================================================================

/// Abstraction over batch entry access patterns.
///
/// `GenericBatch` wraps `&[BatchEntry<P>]` (pre-allocated outputs).
/// `ValueBatch` wraps `&mut [BatchValueEntry<P>]` (deferred allocation).
/// The unified `process_sorted_batch_inner` and `insert_batch_into_locked_leaf_inner`
/// are parameterized over `B: BatchOps` to eliminate duplicated traversal logic.
trait BatchOps<P: LeafPolicy, A: TreeAllocator<P>> {
    type OldValue;

    fn len(&self) -> usize;
    fn entry_ikey(&self, index: usize) -> u64;
    fn entry_key_bytes(&self, index: usize) -> &[u8];

    fn can_reuse_empty_at(
        &self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
    ) -> bool;

    fn insert_empty_leaf_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        guard: &LocalGuard<'_>,
    ) -> *mut u8;

    #[expect(
        clippy::too_many_arguments,
        reason = "Batch entry insertion requires full context"
    )]
    fn try_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        perm: &mut Permuter,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> BatchEntryResult<Self::OldValue>;

    fn fallback_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        result: &mut BatchInsertResult<Self::OldValue>,
        guard: &LocalGuard<'_>,
    );
}

// ============================================================================
//  GenericBatch: wraps &[OutputEntry<P>] (pre-allocated outputs)
// ============================================================================

struct GenericBatch<'a, P: LeafPolicy>(&'a [OutputEntry<P>]);

impl<P: LeafPolicy, A: TreeAllocator<P>> BatchOps<P, A> for GenericBatch<'_, P> {
    type OldValue = P::Output;

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn entry_ikey(&self, index: usize) -> u64 {
        self.0[index].ikey()
    }

    #[inline(always)]
    fn entry_key_bytes(&self, index: usize) -> &[u8] {
        &self.0[index].key
    }

    #[inline]
    fn can_reuse_empty_at(
        &self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
    ) -> bool {
        let key: Key<'_> = Key::new(&self.0[index].key);
        tree.can_reuse_empty_leaf(leaf, &key)
    }

    #[inline]
    fn insert_empty_leaf_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        let entry: &OutputEntry<P> = &self.0[index];
        let key: Key<'_> = Key::new(&entry.key);
        tree.insert_into_empty_leaf_batch(leaf, lock, &key, &entry.output, guard)
    }

    #[inline]
    fn try_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        perm: &mut Permuter,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> BatchEntryResult<P::Output> {
        let entry: &OutputEntry<P> = &self.0[index];
        let key: Key<'_> = Key::new(&entry.key);
        let single_layer_mode: bool = !key.has_suffix();

        let search_result: InsertSearchResultGeneric = if single_layer_mode {
            tree.search_for_insert_single_layer(leaf, &key, perm)
        } else {
            tree.search_for_insert_generic(leaf, &key, perm)
        };

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                let old_value: P::Output =
                    tree.update_existing_value(leaf, lock, slot, &entry.output, guard);

                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey: u64 = key.ikey();

                match tree.find_usable_slot(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let deferred_retire: *mut u8 = tree.insert_new_value(
                            leaf,
                            lock,
                            slot,
                            back_offset,
                            logical_pos,
                            *perm,
                            &key,
                            &entry.output,
                            guard,
                            pre_allocated,
                        );
                        tree.count.increment();
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

    fn fallback_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        result: &mut BatchInsertResult<P::Output>,
        guard: &LocalGuard<'_>,
    ) {
        let entry: &OutputEntry<P> = &self.0[index];
        let mut key: Key<'_> = Key::new(&entry.key);

        match tree.insert_concurrent_generic(&mut key, entry.output.clone(), guard) {
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
    }
}

// ============================================================================
//  ValueBatch: wraps &mut [BatchValueEntry<P>] (deferred allocation)
// ============================================================================

struct ValueBatch<'a, P: LeafPolicy>(&'a mut [BatchValueEntry<P>]);

impl<P: LeafPolicy, A: TreeAllocator<P>> BatchOps<P, A> for ValueBatch<'_, P> {
    type OldValue = P::Value;

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn entry_ikey(&self, index: usize) -> u64 {
        self.0[index].ikey()
    }

    #[inline(always)]
    fn entry_key_bytes(&self, index: usize) -> &[u8] {
        &self.0[index].key
    }

    #[inline]
    fn can_reuse_empty_at(
        &self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
    ) -> bool {
        let key: Key<'_> = Key::new(&self.0[index].key);
        tree.can_reuse_empty_leaf(leaf, &key)
    }

    #[inline]
    fn insert_empty_leaf_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        let value: P::Value = self.0[index].take_value();
        let key: Key<'_> = Key::new(&self.0[index].key);
        let output: P::Output = P::into_output(value);
        tree.insert_into_empty_leaf_batch(leaf, lock, &key, &output, guard)
    }

    #[inline]
    fn try_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        perm: &mut Permuter,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> BatchEntryResult<P::Value> {
        let key: Key<'_> = Key::new(&self.0[index].key);
        let single_layer_mode: bool = !key.has_suffix();

        let search_result: InsertSearchResultGeneric = if single_layer_mode {
            tree.search_for_insert_single_layer(leaf, &key, perm)
        } else {
            tree.search_for_insert_generic(leaf, &key, perm)
        };

        match search_result {
            InsertSearchResultGeneric::Found { slot } => {
                if leaf.is_value_empty(slot) {
                    return BatchEntryResult::Retry;
                }

                let old_value: P::Value = tree.update_existing_value_write_through(
                    leaf,
                    lock,
                    slot,
                    self.0[index].value_ref(),
                );

                BatchEntryResult::Updated(old_value)
            }

            InsertSearchResultGeneric::NotFound { logical_pos } => {
                let ikey: u64 = key.ikey();

                match tree.find_usable_slot(leaf, perm, ikey) {
                    FindSlotResult::Found { slot, back_offset } => {
                        let output: P::Output = P::into_output(self.0[index].take_value());
                        let insert_key: Key<'_> = Key::new(&self.0[index].key);
                        let deferred_retire: *mut u8 = tree.insert_new_value(
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
                        tree.count.increment();
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

    fn fallback_insert_at(
        &mut self,
        index: usize,
        tree: &MassTreeGeneric<P, A>,
        result: &mut BatchInsertResult<P::Value>,
        guard: &LocalGuard<'_>,
    ) {
        let entry: &mut BatchValueEntry<P> = &mut self.0[index];
        let value: P::Value = entry.take_value();
        let mut key: Key<'_> = Key::new(&entry.key);

        match tree.insert_concurrent_value(&mut key, value, guard) {
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
    }
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
        if P::can_write_through() {
            let value_result: BatchInsertResult<P::Value> =
                self.insert_batch_values_with_guard(entries, guard);

            return value_result.map_old_values(ValueRef::Owned);
        }

        // Convert to OutputEntry (allocates outputs once)
        let mut batch: Vec<OutputEntry<P>> = entries
            .into_iter()
            .enumerate()
            .map(|(seq, (key, value)): (usize, (Vec<u8>, P::Value))| {
                OutputEntry::from_value(key, value, seq)
            })
            .collect();

        if batch.is_empty() {
            return BatchInsertResult::new();
        }

        // Sort by ikey for cache locality, full key, and original position so
        // duplicate keys keep input order (deterministic last-wins).
        batch.sort_unstable_by(|a: &OutputEntry<P>, b: &OutputEntry<P>| {
            batch_sort_key(a.ikey, &a.key, a.seq).cmp(&batch_sort_key(b.ikey, &b.key, b.seq))
        });

        let output_result: BatchInsertResult<P::Output> = self.process_sorted_batch(&batch, guard);

        output_result.map_old_values(|output: P::Output| {
            // SAFETY: Each old value was retired under `guard`; the allocation
            // cannot be reclaimed until the guard is dropped.
            unsafe { P::output_to_value_ref(&output) }
        })
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
            .enumerate()
            .map(|(seq, (key, value)): (usize, (Vec<u8>, P::Value))| {
                BatchValueEntry::new(key, value, seq)
            })
            .collect();

        if batch.is_empty() {
            return BatchInsertResult::new();
        }

        batch.sort_unstable_by(|a: &BatchValueEntry<P>, b: &BatchValueEntry<P>| {
            batch_sort_key(a.ikey, &a.key, a.seq()).cmp(&batch_sort_key(b.ikey, &b.key, b.seq()))
        });
        self.process_sorted_value_batch(&mut batch, guard)
    }

    /// Insert pre-constructed batch entries.
    ///
    /// Takes ownership of the entries. Each entry owns a real user value; the
    /// value is converted to its storage representation only here, once, and
    /// the entry no longer aliases any value still stored in the tree.
    ///
    /// Old values are guard-scoped [`ValueRef`](crate::ValueRef)s.
    ///
    /// # Panics
    ///
    /// Panics if a `BatchEntry` value was already consumed, or on an internal
    /// tree invariant violation.
    pub fn insert_batch_entries<'g>(
        &self,
        entries: Vec<BatchEntry<P>>,
        guard: &'g LocalGuard<'_>,
    ) -> BatchInsertResult<ValueRef<'g, P::Value>> {
        self.verify_guard(guard);
        if entries.is_empty() {
            return BatchInsertResult::new();
        }

        if P::can_write_through() {
            let mut value_entries: Vec<BatchValueEntry<P>> = entries
                .into_iter()
                .enumerate()
                .map(|(seq, entry): (usize, BatchEntry<P>)| {
                    let key: Vec<u8> = entry.key;
                    let value: P::Value = entry
                        .value
                        .expect("BatchEntry value already consumed before insertion");
                    BatchValueEntry::new(key, value, seq)
                })
                .collect();

            value_entries.sort_unstable_by(|a: &BatchValueEntry<P>, b: &BatchValueEntry<P>| {
                batch_sort_key(a.ikey, &a.key, a.seq).cmp(&batch_sort_key(b.ikey, &b.key, b.seq))
            });

            let value_result: BatchInsertResult<P::Value> =
                self.process_sorted_value_batch(&mut value_entries, guard);

            return value_result.map_old_values(ValueRef::Owned);
        }

        // Convert to internal outputs; the input order supplies the duplicate
        // tiebreaker.
        let mut outputs: Vec<OutputEntry<P>> = entries
            .into_iter()
            .enumerate()
            .map(|(seq, entry): (usize, BatchEntry<P>)| {
                let key: Vec<u8> = entry.key;
                let value: P::Value = entry
                    .value
                    .expect("BatchEntry value already consumed before insertion");
                OutputEntry::from_value(key, value, seq)
            })
            .collect();

        outputs.sort_unstable_by(|a: &OutputEntry<P>, b: &OutputEntry<P>| {
            batch_sort_key(a.ikey, &a.key, a.seq).cmp(&batch_sort_key(b.ikey, &b.key, b.seq))
        });

        let output_result: BatchInsertResult<P::Output> =
            self.process_sorted_batch(&outputs, guard);

        output_result.map_old_values(|output: P::Output| {
            // SAFETY: Each old value was retired under `guard`; it remains
            // allocated until the guard is dropped.
            unsafe { P::output_to_value_ref(&output) }
        })
    }

    // ========================================================================
    //  Internal Batch Processing
    // ========================================================================

    /// Process a sorted batch of entries (generic path).
    ///
    /// Delegates to `process_sorted_batch_inner` via `GenericBatch`.
    fn process_sorted_batch(
        &self,
        batch: &[OutputEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Output> {
        let mut ops: GenericBatch<'_, P> = GenericBatch(batch);
        self.process_sorted_batch_inner(&mut ops, guard)
    }

    /// Process a sorted batch of value entries (write-through path).
    ///
    /// Delegates to `process_sorted_batch_inner` via `ValueBatch`.
    fn process_sorted_value_batch(
        &self,
        batch: &mut [BatchValueEntry<P>],
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<P::Value> {
        let mut ops: ValueBatch<'_, P> = ValueBatch(batch);
        self.process_sorted_batch_inner(&mut ops, guard)
    }

    /// Unified sorted batch processing, parameterized by `BatchOps`.
    ///
    /// Handles traversal, OCC validation, locking, locked-leaf insertion,
    /// retirement, and fallback to individual insert on split.
    fn process_sorted_batch_inner<B: BatchOps<P, A>>(
        &self,
        batch: &mut B,
        guard: &LocalGuard<'_>,
    ) -> BatchInsertResult<B::OldValue> {
        let mut result: BatchInsertResult<B::OldValue> = BatchInsertResult::new();
        let mut index: usize = 0;

        while index < batch.len() {
            let key: Key<'_> = Key::new(batch.entry_key_bytes(index));

            let mut layer_root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

            let entries_processed: usize = 'retry: loop {
                layer_root = self.maybe_parent_generic(layer_root, guard);

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
                let processed: usize = self.insert_batch_into_locked_leaf_inner(
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

                if processed == 0 {
                    break 'retry 0;
                }

                break 'retry processed;
            };

            if entries_processed == 0 {
                batch.fallback_insert_at(index, self, &mut result, guard);
                index += 1;
            } else {
                index += entries_processed;
            }
        }

        result
    }

    /// Unified locked-leaf batch insertion, parameterized by `BatchOps`.
    ///
    /// Inserts as many entries as possible into a locked leaf. Handles
    /// empty leaf reuse, upper bound clustering, and per-entry dispatch.
    #[expect(
        clippy::too_many_arguments,
        reason = "Batch insertion requires context"
    )]
    fn insert_batch_into_locked_leaf_inner<B: BatchOps<P, A>>(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        batch: &mut B,
        start_index: usize,
        result: &mut BatchInsertResult<B::OldValue>,
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

        if leaf.is_empty()
            && start_index < batch.len()
            && batch.can_reuse_empty_at(start_index, self, leaf)
        {
            let deferred: *mut u8 =
                batch.insert_empty_leaf_at(start_index, self, leaf, lock, guard);

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
                && batch.entry_ikey(entry_index) >= bound
            {
                break;
            }

            let insert_result = batch.try_insert_at(
                entry_index,
                self,
                leaf,
                lock,
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
        // INS-04: encode before clearing the empty state or touching the slot.
        let prepared: PreparedValue<P> = LeafNode15::<P>::prepare_value(value);

        leaf.clear_empty_state();
        let slot: usize = 0;
        let deferred_retire: *mut u8 =
            self.assign_slot_generic(leaf, lock, slot, key, &prepared, guard, None);

        let new_perm: Permuter = <LeafNode15<P> as TreeLeafNode<P>>::Perm::make_sorted(1);
        leaf.set_permutation(new_perm);
        self.count.increment();

        deferred_retire
    }
}
