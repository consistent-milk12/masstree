//! `MassTreeIndex` - convenience wrapper for copyable values.
//!
//! Provides a simpler API that returns `V` directly instead of `Arc<V>`.
//! Best for small, copyable values like `u64`, handles, or pointers.

use std::fmt as StdFmt;
use std::sync::Arc;

use crate::alloc::{NodeAllocator, SeizeAllocator};
use crate::leaf::LeafValue;

use super::{InsertError, MassTree};

/// Convenience wrapper for index-style workloads with copyable values.
///
/// Provides a simpler API that returns `V` directly instead of `Arc<V>`.
/// Best for small, copyable values like `u64`, handles, or pointers.
///
/// # Implementation Note
///
/// **This is currently a wrapper around `MassTree<V>`, NOT true inline storage.**
/// Values are still stored as `Arc<V>` internally; this type simply copies
/// the value out on read. True inline storage is planned for a future release.
///
/// For performance-critical code where Arc overhead matters, use `MassTree<V>`
/// directly and manage the Arc yourself.
///
/// # Type Parameters
///
/// * `V` - The value type to store (must be `Copy`)
/// * `WIDTH` - Node width (default: 15, max: 15)
/// * `A` - Node allocator (default: `SeizeAllocator`)
///
/// # Example
///
/// ```ignore
/// use masstree::MassTreeIndex;
///
/// let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();
/// tree.insert(b"hello", 42).unwrap();
///
/// let value = tree.get(b"hello");
/// assert_eq!(value, Some(42));
/// ```
pub struct MassTreeIndex<
    V: Copy,
    const WIDTH: usize = 15,
    A: NodeAllocator<LeafValue<V>, WIDTH> = SeizeAllocator<LeafValue<V>, WIDTH>,
> {
    /// Wraps `MassTree` internally. True inline storage is planned for future.
    pub(crate) inner: MassTree<V, WIDTH, A>,
}

impl<V: Copy, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> StdFmt::Debug
    for MassTreeIndex<V, WIDTH, A>
{
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTreeIndex")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<V: Copy, const WIDTH: usize> MassTreeIndex<V, WIDTH, SeizeAllocator<LeafValue<V>, WIDTH>> {
    /// Create a new empty `MassTreeIndex` with the default seize allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: MassTree::new(),
        }
    }
}

impl<V: Copy, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>>
    MassTreeIndex<V, WIDTH, A>
{
    /// Create a new empty `MassTreeIndex` with a custom allocator.
    #[must_use]
    #[inline(always)]
    pub fn with_allocator(allocator: A) -> Self {
        Self {
            inner: MassTree::with_allocator(allocator),
        }
    }

    /// Check if the tree is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the number of keys in the tree.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Look up a value by key.
    ///
    /// Returns a copy of the value, or None if not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice, up to 256 bytes)
    ///
    /// # Returns
    ///
    /// * `Some(V)` - If the key was found (value is copied)
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        // For index mode, we dereference the Arc and copy
        self.inner.get(key).map(|arc: Arc<V>| *arc)
    }

    /// Insert a key-value pair into the tree.
    ///
    /// If the key already exists, the value is updated and the old value returned.
    ///
    /// # Arguments
    ///
    /// * `key` - The key as a byte slice (up to 256 bytes)
    /// * `value` - The value to insert (copied)
    ///
    /// # Returns
    ///
    /// * `Ok(Some(old_value))` - Key existed, old value returned (copied)
    /// * `Ok(None)` - Key inserted (new key)
    ///
    /// # Errors
    ///
    /// * [`InsertError::LeafFull`] - Internal error (shouldn't happen with layer support)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();
    ///
    /// // Insert new key
    /// assert_eq!(tree.insert(b"hello", 42)?, None);
    ///
    /// // Update existing key
    /// assert_eq!(tree.insert(b"hello", 100)?, Some(42));
    /// ```
    #[inline(always)]
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>, InsertError> {
        // Use inner tree's insert, convert Arc<V> to V
        self.inner
            .insert(key, value)
            .map(|opt: Option<Arc<V>>| opt.map(|arc: Arc<V>| *arc))
    }
}

impl<V: Copy, const WIDTH: usize> Default
    for MassTreeIndex<V, WIDTH, SeizeAllocator<LeafValue<V>, WIDTH>>
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
