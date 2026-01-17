//! Key search algorithms for `MassTree`.
//!
//! Provides:
//! - Binary/linear search for upper bound in internodes (routing to children)
//! - Scalar ikey matching for leaves (find all slots with target ikey)
//!
//! # Reference
//! Based on `ksearch.hh` from the C++ Masstree implementation.

use crate::ksearch::scalar::Scalar;
use crate::leaf_trait::TreeInternode;
use crate::permuter::Permuter;
use std::cmp::Ordering;

pub(crate) mod scalar;

// ============================================================================
//  KeyIndexPosition
// ============================================================================

/// Result of a key search operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyIndexPosition {
    /// Logical position in sorted order.
    pub i: usize,
    /// Physical slot index, or `NOT_FOUND` if key not present.
    pub p: usize,
}

impl KeyIndexPosition {
    /// Sentinel value indicating key was not found.
    pub const NOT_FOUND: usize = usize::MAX;

    /// Creates a found result with logical position `i` and physical slot `p`.
    #[must_use]
    #[inline(always)]
    pub const fn found(i: usize, p: usize) -> Self {
        Self { i, p }
    }

    /// Creates a not-found result at logical position `i`.
    #[must_use]
    #[inline(always)]
    pub const fn not_found(i: usize) -> Self {
        Self {
            i,
            p: Self::NOT_FOUND,
        }
    }

    /// Returns `true` if the key was found.
    #[must_use]
    #[inline(always)]
    pub const fn is_found(&self) -> bool {
        self.p != Self::NOT_FOUND
    }

    /// Returns the physical slot index. Panics in debug if not found.
    #[must_use]
    #[inline(always)]
    pub fn slot(&self) -> usize {
        debug_assert!(self.is_found(), "slot() called on not-found position");
        self.p
    }

    /// Returns the physical slot index if found, `None` otherwise.
    #[must_use]
    #[inline(always)]
    pub const fn try_slot(&self) -> Option<usize> {
        if self.p == Self::NOT_FOUND {
            None
        } else {
            Some(self.p)
        }
    }
}

impl Default for KeyIndexPosition {
    fn default() -> Self {
        Self::not_found(0)
    }
}

// ============================================================================
//  Binary Search
// ============================================================================

/// Binary search lower bound with custom comparator.
#[inline(always)]
pub fn lower_bound_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> KeyIndexPosition
where
    F: Fn(usize) -> Ordering,
{
    let (mut l, mut r) = (0, size);

    while l < r {
        let m = (l + r) >> 1;
        let mp = perm.get(m);

        match compare(mp) {
            Ordering::Less => r = m,
            Ordering::Equal => return KeyIndexPosition::found(m, mp),
            Ordering::Greater => l = m + 1,
        }
    }

    KeyIndexPosition::not_found(l)
}

/// Binary search upper bound with custom comparator.
#[inline(always)]
pub fn upper_bound_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> usize
where
    F: Fn(usize) -> Ordering,
{
    let (mut l, mut r) = (0, size);

    while l < r {
        let m = (l + r) >> 1;
        match compare(perm.get(m)) {
            Ordering::Less => r = m,
            Ordering::Equal => return m + 1,
            Ordering::Greater => l = m + 1,
        }
    }

    l
}

// ============================================================================
//  Linear Search (for tests/reference)
// ============================================================================

/// Linear search lower bound - same semantics as `lower_bound_by`.
#[inline(always)]
pub fn lower_bound_linear_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> KeyIndexPosition
where
    F: Fn(usize) -> Ordering,
{
    for i in 0..size {
        let p = perm.get(i);
        match compare(p) {
            Ordering::Less => return KeyIndexPosition::not_found(i),
            Ordering::Equal => return KeyIndexPosition::found(i, p),
            Ordering::Greater => {}
        }
    }
    KeyIndexPosition::not_found(size)
}

/// Linear search upper bound - same semantics as `upper_bound_by`.
#[inline(always)]
pub fn upper_bound_linear_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> usize
where
    F: Fn(usize) -> Ordering,
{
    for i in 0..size {
        match compare(perm.get(i)) {
            Ordering::Less => return i,
            Ordering::Equal => return i + 1,
            Ordering::Greater => {}
        }
    }
    size
}

// ============================================================================
//  Internode Search
// ============================================================================

/// Upper bound search for internode routing.
///
/// Returns child index (0 to nkeys) to follow for the given ikey.
#[inline]
pub fn upper_bound_internode_generic<I: TreeInternode>(search_ikey: u64, node: &I) -> usize {
    Scalar::upper_bound_internode_scalar(search_ikey, node)
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod unit_tests;
