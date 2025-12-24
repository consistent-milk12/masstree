//! `MassTreeIndex` - convenience wrapper for copyable values.
//!
//! Provides a simpler API that returns `V` directly instead of `Arc<V>`.
//! Best for small, copyable values like `u64`, handles, or pointers.

use std::fmt as StdFmt;
use std::sync::Arc;

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
pub struct MassTreeIndex<V: Copy + Send + Sync + 'static> {
    /// Wraps `MassTree` internally. True inline storage is planned for future.
    pub(crate) inner: MassTree<V>,
}

impl<V: Copy + Send + Sync + 'static> StdFmt::Debug for MassTreeIndex<V> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTreeIndex")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<V: Copy + Send + Sync + 'static> MassTreeIndex<V> {
    /// Create a new empty `MassTreeIndex`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: MassTree::new(),
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
    #[must_use]
    #[inline(always)]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        self.inner.get(key).map(|arc: Arc<V>| *arc)
    }

    /// Insert a key-value pair into the tree.
    ///
    /// If the key already exists, the value is updated and the old value returned.
    ///
    /// # Errors
    ///
    /// Returns error on a failed insert.
    #[inline(always)]
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>, InsertError> {
        self.inner
            .insert(key, value)
            .map(|opt: Option<Arc<V>>| opt.map(|arc: Arc<V>| *arc))
    }
}

impl<V: Copy + Send + Sync + 'static> Default for MassTreeIndex<V> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
