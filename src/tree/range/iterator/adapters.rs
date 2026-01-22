//! Filepath: src/tree/range/iterator/adapters.rs
//!
//! Key-only and value-only iterator adapters.

use std::fmt::{self as StdFmt, Debug, Formatter};
use std::iter::FusedIterator;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::{LayerCapableLeaf, TreeLeafNode};
use crate::slot::ValueSlot;

use super::RangeIter;

// ============================================================================
//  KeysIter
// ============================================================================

/// Iterator adapter that yields only keys.
pub struct KeysIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    pub(super) inner: RangeIter<'a, 'g, S, L, A>,
}

impl<S, L, A> Debug for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("KeysIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<S, L, A> Iterator for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = Vec<u8>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.key)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S, L, A> FusedIterator for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
}

// ============================================================================
//  ValuesIter
// ============================================================================

/// Iterator adapter that yields only values.
pub struct ValuesIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    pub(super) inner: RangeIter<'a, 'g, S, L, A>,
}

impl<S, L, A> Debug for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("ValuesIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<S, L, A> Iterator for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = S::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S, L, A> FusedIterator for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
}
