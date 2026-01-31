//! Entry API for conditional key access
//!
//! Provides [`Entry`], [`OccupiedEntry`], and [`VacantEntry`] types for
//! ergonomic conditional insertion and modification patterns.
//!
//! # Performance
//!
//! Entry operations are optimized for single tree traversals:
//! - `entry_with_guard()`: 1 traversal (lookup)
//! - `or_insert()` on Vacant: 1 traversal (insert with pre-cloned output)
//! - `and_modify()` on Occupied: 1 traversal (insert with pre-cloned output)
//!
//! # Concurrency
//!
//! Entry operations are **NOT atomic**. Between lookup and modification,
//! other threads may modify the same key. This follows "last-writer-wins"
//! semantics.
//! - `or_insert` may overwrite a value inserted concurrently another thread
//! - `or_insert` on [`Occupied`] may return a stale cached value without reinsertion
//!   (if the key was deleted between `entry_with_guard` and `or_insert`)
//! - `and_modify` may lose updates from concurrent mods
//! - `or_insert_key` closure runs eagerly for vacant entries, even if the key
//!   becomes occupied before the actual insert (wasted work + overwrite)
//! - [`Occupied`]/[`Vacant`] classification may change before operations complete
//!
//! For atomic operations, use external synchronization.

use std::fmt::{self as StdFmt, Debug, Formatter};

use seize::LocalGuard;

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric,
    leaf_trait::LayerCapableLeaf,
    slot::ValueSlot,
    tree::RemoveError,
    value::traits::LeafValueLoad,
};

/// Result type for [`OccupiedEntry::try_remove_entry`].
///
/// - `Ok(Some((key, value)))` - Entry was removed successfully
/// - `Ok(None)` - Key was already deleted by another thread
/// - `Err(RemoveError)` - Removal failed (retry limit exceeded)
pub type RemoveEntryResult<O> = Result<Option<(Vec<u8>, O)>, RemoveError>;

/// A view into a single entry in a tree, which may either be vacant or occupied.
///
/// This enum is constructed using [`entry_with_guard()`](MassTreeGeneric::entry_with_guard).
///
/// # Differences from [`HashMap::Entry`](std::collections::HashMap)
///
/// - Returns values by-value (`S::Output`), not by mutable reference
/// - Requires a guard for concurrent access
/// - `and_modify` takes a transform function `FnOnce(&S::Output) -> S::Value`
/// - Key is borrowed, not owned (zero allocation for entry creation)
/// - Provides fallible `try_*` variants that return `Result`
///
/// # Concurrency Warning
///
/// Entry operations are NOT atomic. The occupied/vacant classification is
/// best-effort and may change due to concurrent modifications. Notably:
///
/// - `or_insert` may overwrite a concurrently inserted value
/// - `or_insert` on Occupied returns cached value without reinserting (stale if deleted)
/// - `or_insert_with` closure runs eagerly, may waste work under races
/// - Methods handle races gracefully without panicking
///
/// See module documentation for details.
///
/// # Example
///
/// ```rust,ignore
/// use masstree::MassTree;
///
/// let tree: MassTree<u64> = MassTree::new();
/// let guard = tree.guard();
///
/// // Insert if absent, get value either way
/// let value = tree.entry_with_guard(b"counter", &guard)
///     .or_insert(0);
///
/// // Modify existing or insert default
/// let value = tree.entry_with_guard(b"counter", &guard)
///     .and_modify(|v| *v + 1)
///     .or_insert(0);
/// ```
pub enum Entry<'t, 'e, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// An occupied entry (key exists in tree at classification time).
    Occupied(OccupiedEntry<'t, 'e, S, L, A>),

    /// A vacant entry (key does not exist classification time).
    Vacant(VacantEntry<'t, 'e, S, L, A>),
}

/// A view into an occupied entry in a tree.
///
/// Created by [`Entry::Occupied`] variant from
/// [`entry_with_guard`](MassTreeGeneric::entry_with_guard).
///
/// NOTE: The "occupied" status was determined at entry creation time.
/// Under concurrency, the key may have been deleted since then. Methods
/// handle this gracefully by returning [`Option`] or [`Result`].
pub struct OccupiedEntry<'t, 'e, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// The key (borrowed from caller, zero allocation).
    key: &'e [u8],

    /// The current value (snapshot at entry creation time).
    value: S::Output,

    /// Reference to the tree for insertion.
    tree: &'t MassTreeGeneric<S, L, A>,

    /// Guard for concurrent access.
    guard: &'e LocalGuard<'t>,
}

/// A view into a vacant entry in a tree.
///
/// Created by [`Entry::Vacant`] variant from
/// [`entry_with_guard`](MassTreeGeneric::entry_with_guard).
///
/// NOTE: The "vacant" status was determined at entry creation time.
/// Under concurrency, another thread may have inserted a value since then.
/// `insert` will overwrite any such value (last-writer-wins).
pub struct VacantEntry<'t, 'e, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// The key (borrowed from caller, zero alloc)
    key: &'e [u8],

    /// Reference to the tree for insertion.
    tree: &'t MassTreeGeneric<S, L, A>,

    /// Guard for concurrent access.
    guard: &'e LocalGuard<'t>,
}

// ============================================================================
//  Entry Implementation
// ============================================================================

impl<'t, 'e, S, L, A> Entry<'t, 'e, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Returns the key for this entry.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> &[u8] {
        match self {
            Entry::Occupied(o) => o.key(),

            Entry::Vacant(v) => v.key(),
        }
    }

    /// Fallible version [`or_insert`](Self::or_insert).
    ///
    /// # Errors
    ///
    /// Panics if insertion fails (allocation error). For fallible insertion,
    /// use [`or_try_insert`](Self::or_try_insert).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let value = tree.entry_with_guard(b"key", &guard).or_insert(42);
    /// ```
    /// Returns the entry's value if occupied, or inserts the default and returns it.
    #[inline(always)]
    pub fn or_insert(self, default: S::Value) -> S::Output {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// Returns the entry's value if occupied, or computes and inserts a default.
    #[inline(always)]
    pub fn or_insert_with<F>(self, default: F) -> S::Output
    where
        F: FnOnce() -> S::Value,
    {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => v.insert(default()),
        }
    }

    /// Returns the entry's value if occupied, or computes a default from the key.
    #[inline(always)]
    pub fn or_insert_with_key<F>(self, default: F) -> S::Output
    where
        F: FnOnce(&[u8]) -> S::Value,
    {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => {
                let value: S::Value = default(v.key());
                v.insert(value)
            }
        }
    }

    pub fn or_default(self) -> S::Output
    where
        S::Value: Default,
    {
        self.or_insert(Default::default())
    }

    /// Modifies the value if occupied using the provided function.
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&S::Output) -> S::Value,
    {
        match self {
            Entry::Occupied(mut o) => {
                let new_value: S::Value = f(&o.value);
                let new_output: S::Output = S::into_output(new_value);
                let return_output: S::Output = new_output.clone();

                let _old = o.tree
                    .insert_output_with_guard(o.key, new_output, o.guard);
                o.value = return_output;

                Entry::Occupied(o)
            }

            Entry::Vacant(v) => Entry::Vacant(v),
        }
    }

    /// Inserts a value and returns an `OccupiedEntry`.
    #[inline]
    pub fn insert_entry(self, value: S::Value) -> OccupiedEntry<'t, 'e, S, L, A> {
        match self {
            Entry::Occupied(mut o) => {
                o.insert(value);
                o
            }

            Entry::Vacant(v) => {
                let key: &[u8] = v.key;
                let tree: &MassTreeGeneric<S, L, A> = v.tree;
                let guard: &LocalGuard<'_> = v.guard;

                // Convert to output before insert
                let output: S::Output = S::into_output(value);
                let return_output: S::Output = output.clone();
                let _old = tree.insert_output_with_guard(key, output, guard);

                OccupiedEntry {
                    key,
                    value: return_output,
                    tree,
                    guard,
                }
            }
        }
    }

    #[must_use]
    #[inline(always)]
    pub const fn get(&self) -> Option<&S::Output> {
        match self {
            Entry::Occupied(o) => Some(o.get()),

            Entry::Vacant(_) => None,
        }
    }
}

// ============================================================================
//  OccupiedEntry Implementation
// ============================================================================

impl<S, L, A> OccupiedEntry<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Gets a reference to the key in the entry.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> &[u8] {
        self.key
    }

    /// Gets a reference to the value in the entry.
    ///
    /// NOTE: This is the value at entry creation time (or last modification
    /// via this entry). If another thread has modified the key since then,
    /// this may be stale.
    #[inline(always)]
    pub const fn get(&self) -> &S::Output {
        &self.value
    }

    /// Converts the [`OccupiedEntry`] into the value
    #[inline(always)]
    pub fn into_value(self) -> S::Output {
        self.value
    }

    /// Sets the value of the entry, and returns the old value.
    ///
    /// # Returns
    ///
    /// - `Some(old)` - Key existed, old value returned
    /// - `None` - Key was deleted concurrently, new value inserted anyway
    #[inline(always)]
    pub fn insert(&mut self, value: S::Value) -> Option<S::Output> {
        // Convert to output before insert
        let output = S::into_output(value);
        let return_output = output.clone();
        let old = self
            .tree
            .insert_output_with_guard(self.key, output, self.guard);

        // Update cached value
        self.value = return_output;

        old
    }

    /// Removes the entry from the tree and returns the actually removed value.
    ///
    /// Unlike some Entry implementations, this returns the value that was
    /// actually in the tree at removal time, not a stale snapshot.
    ///
    /// # Panics
    ///
    /// Panics if removal fails (extremely rare - only on retry limit exceeded).
    /// Use [`try_remove`](Self::try_remove) for fallible operation.
    #[inline(always)]
    #[expect(
        clippy::panic,
        reason = "Convenience wrapper; use try_remove for fallible version"
    )]
    pub fn remove(self) -> Option<S::Output> {
        match self.try_remove() {
            Ok(value) => value,

            Err(e) => panic!("OccupiedEntry::remove failed: {e:?}"),
        }
    }

    /// Fallible version of [`remove`](Self::remove)
    ///
    /// # Errors
    ///
    /// Returns error if removal failed
    #[inline(always)]
    pub fn try_remove(self) -> Result<Option<S::Output>, RemoveError> {
        self.tree.remove_with_guard(self.key, self.guard)
    }

    /// Removes the entry from the tree and returns the key and removed value.
    ///
    /// NOTE: Returns the key as a [`Vec<u8>`] copy since the [`Entry`] borrows the key.
    ///
    /// # Panics
    ///
    /// Panics if removal fails. Use [`try_remove_entry`](Self::try_remove_entry)
    /// for fallible operation.
    #[inline(always)]
    #[expect(
        clippy::panic,
        reason = "Convenience wrapper; use try_remove_entry for fallible version"
    )]
    pub fn remove_entry(self) -> Option<(Vec<u8>, S::Output)> {
        match self.try_remove_entry() {
            Ok(result) => result,

            Err(e) => panic!("OccupiedEntry::remove_entry failed: {e:?}"),
        }
    }

    /// Fallible version of [`remove_entry`](Self::remove_entry).
    #[inline(always)]
    pub fn try_remove_entry(self) -> RemoveEntryResult<S::Output> {
        let key_owned: Vec<u8> = self.key.to_vec();

        (self.tree.remove_with_guard(self.key, self.guard)?)
            .map_or_else(|| Ok(None), |value: S::Output| Ok(Some((key_owned, value))))
    }
}

// ============================================================================
//  VacantEntry Implementation
// ============================================================================

impl<S, L, A> VacantEntry<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Gets a reference to the key that would be used when inserting.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> &[u8] {
        self.key
    }

    /// Consumes the entry and returns the key as an owned [`Vec<u8>`]
    #[must_use]
    #[inline(always)]
    pub fn into_key(self) -> Vec<u8> {
        self.key.to_vec()
    }

    /// Inserts the value into the vacant entry and returns it.
    ///
    /// # Performance
    ///
    /// Single tree traversal. The output is created and cloned before insertion,
    /// eliminating the need for a second lookup.
    ///
    /// # Concurrency Warning
    ///
    /// If another thread inserted a value for this key between `entry_with_guard()`
    /// and `insert()`, this method will **overwrite** that value. This is
    /// "last-writer-wins" semantics.
    #[inline(always)]
    pub fn insert(self, value: S::Value) -> S::Output {
        // Convert to output before insert, clone for return value
        let output = S::into_output(value);
        let return_output = output.clone();

        // Insert using internal method that accepts S::Output
        // The return value (old value if any) is ignored since we're in a VacantEntry
        let _old = self
            .tree
            .insert_output_with_guard(self.key, output, self.guard);
        return_output
    }
}

// ============================================================================
//  Debug Implementations
// ============================================================================

impl<S, L, A> Debug for Entry<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Debug,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        match self {
            Entry::Occupied(o) => f.debug_tuple("Occupied").field(&o.value).finish(),

            Entry::Vacant(v) => f.debug_tuple("Vacant").field(&v.key).finish(),
        }
    }
}

impl<S, L, A> Debug for OccupiedEntry<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Debug,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", &self.key)
            .field("value", &self.value)
            .finish()
    }
}

impl<S, L, A> Debug for VacantEntry<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Debug,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("VacantEntry")
            .field("key", &self.key)
            .finish()
    }
}

// ============================================================================
//  Constructor (internal)
// ============================================================================

impl<'t, 'e, S, L, A> Entry<'t, 'e, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Create an Entry by looking up the key.
    ///
    /// This is called by `MassTreeGeneric::entry_with_guard`.
    ///
    /// # Performance
    ///
    /// One tree traversal for the initial lookup. Key is borrowed,
    /// not copied (zero allocation).
    #[inline]
    pub(crate) fn new(
        tree: &'t MassTreeGeneric<S, L, A>,
        key: &'e [u8],
        guard: &'e LocalGuard<'t>,
    ) -> Self {
        tree.get_with_guard(key, guard).map_or_else(
            || Entry::Vacant(VacantEntry { key, tree, guard }),
            |value: S::Output| {
                Entry::Occupied(OccupiedEntry {
                    key,
                    value,
                    tree,
                    guard,
                })
            },
        )
    }
}

#[cfg(test)]
mod unit_tests;
