//! Entry API for conditional key access
//!
//! Provides [`Entry`], [`OccupiedEntry`], and [`VacantEntry`] types for
//! ergonomic conditional insertion and modification patterns.

use std::fmt::{self as StdFmt, Debug, Formatter};

use seize::LocalGuard;

use crate::{MassTreeGeneric, TreeAllocator, ValueRef, policy::LeafPolicy, tree::RemoveError};

use crate::{key::Key, leaf15::LeafNode15};

use super::insert_strategy::{GenericInsert, WriteThroughInsert};

/// An optimistic location, kept alive by the containing Entry's guard.
/// The version, full key identity, and leaf membership must be rechecked under
/// lock before use. `key_offset` preserves the position in a promoted sublayer.
pub(super) struct EntryLeafHint<P: LeafPolicy> {
    pub(super) leaf_ptr: *mut LeafNode15<P>,
    pub(super) version: u32,
    pub(super) slot: Option<usize>,
    pub(super) key_offset: usize,
}

impl<P: LeafPolicy> EntryLeafHint<P> {
    #[inline(always)]
    pub(super) const fn capture<const CAPTURE: bool>(
        hint: &mut Option<Self>,
        leaf_ptr: *mut LeafNode15<P>,
        version: u32,
        slot: Option<usize>,
        key: &Key<'_>,
    ) {
        if CAPTURE {
            *hint = Some(Self {
                leaf_ptr,
                version,
                slot,
                key_offset: key.len() - key.current_len(),
            });
        }
    }
}

type EntryUpdate<'g, V> = (ValueRef<'g, V>, Option<ValueRef<'g, V>>);

/// Result type for [`OccupiedEntry::try_remove_entry`].
#[allow(clippy::type_complexity)]
pub type RemoveEntryResult<'g, V> = Result<Option<(Vec<u8>, ValueRef<'g, V>)>, RemoveError>;

/// A view into a single entry in a tree, which may either be vacant or occupied.
///
/// # Concurrency
///
/// Classification (Occupied vs Vacant) is a point-in-time snapshot taken
/// without holding a lock. A concurrent insert or remove between
/// classification and the subsequent mutation can cause:
///
/// - `and_modify`: lost updates (computation based on stale value).
/// - `VacantEntry::insert`: silent overwrite of a concurrently inserted key.
///
/// For atomic read-modify-write, use `insert_with_guard` together with
/// external synchronization, or use the tree's write-through update path.
///
/// # Example
///
/// ```rust
/// use masstree::MassTree;
///
/// let tree: MassTree<u64> = MassTree::new();
/// let guard = tree.guard();
///
/// // Insert if absent, get value either way
/// let _value = tree.entry_with_guard(b"counter", &guard)
///     .or_insert(0);
///
/// // Modify existing or insert default
/// let _value = tree.entry_with_guard(b"counter", &guard)
///     .and_modify(|v| v + 1)
///     .or_insert(0);
/// ```
pub enum Entry<'t, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// An occupied entry (key exists in tree at classification time).
    Occupied(OccupiedEntry<'t, 'e, P, A>),

    /// A vacant entry (key does not exist classification time).
    Vacant(VacantEntry<'t, 'e, P, A>),
}

/// A view into an occupied entry in a tree.
pub struct OccupiedEntry<'t, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// The key (borrowed from caller, zero allocation).
    key: &'e [u8],

    /// The current value (guard-scoped snapshot at entry creation time).
    value: ValueRef<'e, P::Value>,

    /// Reference to the tree for insertion.
    tree: &'t MassTreeGeneric<P, A>,

    /// Guard for concurrent access.
    guard: &'e LocalGuard<'t>,

    hint: Option<EntryLeafHint<P>>,
}

/// A view into a vacant entry in a tree.
pub struct VacantEntry<'t, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// The key (borrowed from caller, zero alloc)
    key: &'e [u8],

    /// Reference to the tree for insertion.
    tree: &'t MassTreeGeneric<P, A>,

    /// Guard for concurrent access.
    guard: &'e LocalGuard<'t>,

    hint: Option<EntryLeafHint<P>>,
}

// ============================================================================
//  Entry Implementation
// ============================================================================

impl<'t, 'e, P, A> Entry<'t, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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

    /// Inserts a default value if the entry is vacant, then returns
    /// the entry's value.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use masstree::MassTree;
    /// # let tree: MassTree<u64> = MassTree::new();
    /// # let guard = tree.guard();
    /// let value = tree.entry_with_guard(b"key", &guard).or_insert(42);
    /// ```
    /// Returns the entry's value if occupied, or inserts the default and returns it.
    #[inline(always)]
    pub fn or_insert(self, default: P::Value) -> ValueRef<'e, P::Value> {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// Returns the entry's value if occupied, or computes and inserts a default.
    #[inline(always)]
    pub fn or_insert_with<F>(self, default: F) -> ValueRef<'e, P::Value>
    where
        F: FnOnce() -> P::Value,
    {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => v.insert(default()),
        }
    }

    /// Returns the entry's value if occupied, or computes a default from the key.
    #[inline(always)]
    pub fn or_insert_with_key<F>(self, default: F) -> ValueRef<'e, P::Value>
    where
        F: FnOnce(&[u8]) -> P::Value,
    {
        match self {
            Entry::Occupied(o) => o.into_value(),
            Entry::Vacant(v) => {
                let value: P::Value = default(v.key());
                v.insert(value)
            }
        }
    }

    /// Inserts the default value if vacant, returns the entry's value.
    pub fn or_default(self) -> ValueRef<'e, P::Value>
    where
        P::Value: Default,
    {
        self.or_insert(Default::default())
    }

    /// Modifies the value if occupied using the provided function.
    ///
    /// The function receives a snapshot of the current value. Under concurrent
    /// writes, the snapshot may be stale and the resulting store can overwrite
    /// a newer value. See [`Entry`] concurrency notes.
    #[must_use]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&P::Value) -> P::Value,
    {
        match self {
            Entry::Occupied(mut o) => {
                let new_value: P::Value = f(&o.value);
                o.insert(new_value);

                Entry::Occupied(o)
            }

            Entry::Vacant(v) => Entry::Vacant(v),
        }
    }

    /// Inserts a value and returns an `OccupiedEntry`.
    #[inline]
    pub fn insert_entry(self, value: P::Value) -> OccupiedEntry<'t, 'e, P, A> {
        match self {
            Entry::Occupied(mut o) => {
                o.insert(value);
                o
            }

            Entry::Vacant(v) => {
                let key: &[u8] = v.key;
                let tree: &MassTreeGeneric<P, A> = v.tree;
                let guard: &LocalGuard<'_> = v.guard;

                let (value, _) = tree.insert_for_entry(key, value, v.hint.as_ref(), guard);
                OccupiedEntry {
                    key,
                    value,
                    tree,
                    guard,
                    hint: None,
                }
            }
        }
    }

    /// Returns a reference to the value if occupied, or `None` if vacant.
    #[must_use]
    #[inline(always)]
    pub fn get(&self) -> Option<&P::Value> {
        match self {
            Entry::Occupied(o) => Some(o.get()),

            Entry::Vacant(_) => None,
        }
    }
}

// ============================================================================
//  OccupiedEntry Implementation
// ============================================================================

impl<'e, P, A> OccupiedEntry<'_, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Gets a reference to the key in the entry.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> &[u8] {
        self.key
    }

    /// Gets a reference to the value in the entry.
    #[inline(always)]
    pub fn get(&self) -> &P::Value {
        &self.value
    }

    /// Converts the [`OccupiedEntry`] into the value
    #[inline(always)]
    pub fn into_value(self) -> ValueRef<'e, P::Value> {
        self.value
    }

    /// Sets the value of the entry, and returns the old value.
    #[inline(always)]
    pub fn insert(&mut self, value: P::Value) -> Option<ValueRef<'e, P::Value>> {
        let (new_value, old) =
            self.tree
                .insert_for_entry(self.key, value, self.hint.take().as_ref(), self.guard);
        self.value = new_value;
        old
    }

    /// Removes the entry from the tree and returns the actually removed value.
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
    pub fn remove(self) -> Option<ValueRef<'e, P::Value>> {
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
    pub fn try_remove(self) -> Result<Option<ValueRef<'e, P::Value>>, RemoveError> {
        self.tree.remove_with_guard(self.key, self.guard)
    }

    /// Removes the entry from the tree and returns the key and removed value.
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
    #[allow(clippy::type_complexity)]
    pub fn remove_entry(self) -> Option<(Vec<u8>, ValueRef<'e, P::Value>)> {
        match self.try_remove_entry() {
            Ok(result) => result,

            Err(e) => panic!("OccupiedEntry::remove_entry failed: {e:?}"),
        }
    }

    /// Fallible version of [`remove_entry`](Self::remove_entry).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tree operation encounters a failure.
    #[inline(always)]
    pub fn try_remove_entry(self) -> RemoveEntryResult<'e, P::Value> {
        let key_owned: Vec<u8> = self.key.to_vec();

        match self.tree.remove_with_guard(self.key, self.guard)? {
            Some(value) => Ok(Some((key_owned, value))),
            None => Ok(None),
        }
    }
}

// ============================================================================
//  VacantEntry Implementation
// ============================================================================

impl<'e, P, A> VacantEntry<'_, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
    #[inline(always)]
    pub fn insert(self, value: P::Value) -> ValueRef<'e, P::Value> {
        self.tree
            .insert_for_entry(self.key, value, self.hint.as_ref(), self.guard)
            .0
    }
}

// ============================================================================
//  Debug Implementations
// ============================================================================

impl<P, A> Debug for Entry<'_, '_, P, A>
where
    P: LeafPolicy,
    P::Value: Send + Sync + Debug,
    A: TreeAllocator<P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        match self {
            Entry::Occupied(o) => f.debug_tuple("Occupied").field(&o.value).finish(),

            Entry::Vacant(v) => f.debug_tuple("Vacant").field(&v.key).finish(),
        }
    }
}

impl<P, A> Debug for OccupiedEntry<'_, '_, P, A>
where
    P: LeafPolicy,
    P::Value: Send + Sync + Debug,
    A: TreeAllocator<P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", &self.key)
            .field("value", &self.value)
            .finish()
    }
}

impl<P, A> Debug for VacantEntry<'_, '_, P, A>
where
    P: LeafPolicy,
    P::Value: Send + Sync + Debug,
    A: TreeAllocator<P>,
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

impl<'t, 'e, P, A> Entry<'t, 'e, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Create an Entry by looking up the key.
    #[inline]
    pub(crate) fn new(
        tree: &'t MassTreeGeneric<P, A>,
        key: &'e [u8],
        guard: &'e LocalGuard<'t>,
    ) -> Self {
        let mut hint = None;
        let output = tree.get_output_with_hint::<true>(key, guard, &mut hint);
        match output {
            Some(output) => Entry::Occupied(OccupiedEntry {
                key,
                // SAFETY: The shared lookup validated the output under this
                // guard. Boxed borrows stay alive, and write-through values
                // are copied atomically rather than borrowed.
                value: unsafe { P::output_to_value_ref(&output) },
                tree,
                guard,
                hint,
            }),
            None => Entry::Vacant(VacantEntry {
                key,
                tree,
                guard,
                hint,
            }),
        }
    }
}

impl<P: LeafPolicy, A: TreeAllocator<P>> MassTreeGeneric<P, A> {
    /// Shared Entry mutation: retain a snapshot of the new value without a
    /// post-insert lookup, and return the actual overwritten value separately.
    fn insert_for_entry<'g>(
        &self,
        key: &[u8],
        value: P::Value,
        hint: Option<&EntryLeafHint<P>>,
        guard: &'g LocalGuard<'_>,
    ) -> EntryUpdate<'g, P::Value> {
        if P::can_write_through() {
            // SAFETY: The sealed policy enables write-through only for primitive
            // atomic-word types with no ownership or Drop. Copying the input
            // retains the new-value snapshot without allocating a second Box.
            let copied = unsafe { std::ptr::read(&raw const value) };
            let old = self.insert_at_entry_hint::<WriteThroughInsert>(key, value, hint, guard);
            return (ValueRef::Owned(copied), old.map(ValueRef::Owned));
        }

        let output = P::into_output(value);
        // SAFETY: Non-write-through output is uniquely owned here and will be
        // published below. The Entry guard protects its allocation thereafter;
        // inline policies return an owned copy.
        let new_value = unsafe { P::output_to_value_ref(&output) };
        let old = self.insert_at_entry_hint::<GenericInsert>(key, output, hint, guard);
        let old_value = old.map(|output| {
            // SAFETY: The insertion retired this old allocation, but the guard
            // keeps it alive until after the returned borrow is no longer used.
            unsafe { P::output_to_value_ref(&output) }
        });
        (new_value, old_value)
    }
}

#[cfg(test)]
mod unit_tests;
