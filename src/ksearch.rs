//! Key search algorithms for `MassTree`.
//!
//! Provides:
//! - Binary/linear search for upper bound in internodes (routing to children)
//! - Scalar ikey matching for leaves (find all slots with target ikey)
//!
//! # Reference
//! Based on `ksearch.hh` from the C++ Masstree implementation.

use crate::internode::InternodeNode;
use crate::leaf_trait::TreeInternode;
use crate::leaf24::LeafNode24;
use crate::permuter::Permuter;
use crate::slot;
use std::cmp::Ordering;

// ============================================================================
//  KeyIndexPosition
// ============================================================================

/// Result of a key search operation.
///
/// Contains both the logical position (where the key is or should be) and
/// the physical slot (if the key was found).
///
/// # Fields
/// * `i` - Logical position (0 to size). For `lower_bound`, this is the insertion point.
/// * `p` - Physical slot index. `NOT_FOUND` if key not present.
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

    /// Create a new position for a found key.
    #[must_use]
    #[inline(always)]
    pub const fn found(i: usize, p: usize) -> Self {
        Self { i, p }
    }

    /// Create a new position for a not-found key.
    #[must_use]
    #[inline(always)]
    pub const fn not_found(i: usize) -> Self {
        Self {
            i,
            p: Self::NOT_FOUND,
        }
    }

    /// Check if the key was found.
    #[must_use]
    #[inline(always)]
    pub const fn is_found(&self) -> bool {
        self.p != Self::NOT_FOUND
    }

    /// Get the physical slot, panicking if not found.
    ///
    /// # Panics
    ///
    /// Panics if the key was not found.
    #[must_use]
    #[inline(always)]
    pub fn slot(&self) -> usize {
        assert!(self.is_found(), "slot() called on not-found position");
        self.p
    }

    /// Get the physical slot as Option.
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
    #[inline(always)]
    fn default() -> Self {
        Self::not_found(0)
    }
}

// ============================================================================
//  Generic Binary Search
// ============================================================================

/// Binary search lower bound with custom comparator.
///
/// Searches for a key in a node using the provided comparator function.
/// The comparator receives a physical slot index and returns:
/// - `Ordering::Less` if `search_key < key_at_slot`
/// - `Ordering::Equal` if `search_key == key_at_slot`
/// - `Ordering::Greater` if `search_key > key_at_slot`
///
/// # Arguments
/// * `size` - Number of keys in the node
/// * `perm` - Permutation mapping logical → physical indices
/// * `compare` - Comparator function `|physical_slot| -> Ordering`
///
/// # Returns
/// `KeyIndexPosition` with logical position and physical slot (if found).
pub fn lower_bound_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> KeyIndexPosition
where
    F: Fn(usize) -> Ordering,
{
    let mut l: usize = 0;
    let mut r: usize = size;

    while l < r {
        let m: usize = (l + r) >> 1;
        let mp: usize = perm.get(m); // Physical slot at logical position m

        match compare(mp) {
            Ordering::Less => {
                // search_key < key_at_slot, narrow to left half
                r = m;
            }

            Ordering::Equal => {
                // Exact match! Return both logical and physical
                return KeyIndexPosition::found(m, mp);
            }

            Ordering::Greater => {
                // search_key > key_at_slot, narrow to right half
                l = m + 1;
            }
        }
    }

    // Not found, l is the insertion point
    KeyIndexPosition::not_found(l)
}

/// Binary search upper bound with custom comparator.
///
/// Returns the index of the first key greater than the search key,
/// or `size` if all keys are ≤ search key.
///
/// Used for internode routing: the returned index is the child to follow.
///
/// # Arguments
/// * `size` - Number of keys in the node
/// * `perm` - Permutation mapping logical → physical indices
/// * `compare` - Comparator function `|physical_slot| -> Ordering`
///
/// # Returns
/// Child index (0 to size).
#[inline]
pub fn upper_bound_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> usize
where
    F: Fn(usize) -> Ordering,
{
    let mut l: usize = 0;
    let mut r: usize = size;

    while l < r {
        let m: usize = (l + r) >> 1;
        let mp: usize = perm.get(m);

        match compare(mp) {
            Ordering::Less => {
                r = m;
            }

            Ordering::Equal => {
                // On exact match, route to RIGHT child (m + 1)
                return m + 1;
            }

            Ordering::Greater => {
                l = m + 1;
            }
        }
    }

    l
}

// ============================================================================
//  Linear Search (for reference/small nodes)
// ============================================================================

/// Linear search lower bound with custom comparator.
///
/// Simpler than binary search, potentially faster for very small nodes.
/// Same semantics as `lower_bound_by`.
#[inline]
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
            Ordering::Less => {
                // search_key < key_at_slot, found insertion point
                return KeyIndexPosition::not_found(i);
            }

            Ordering::Equal => {
                // Exact match
                return KeyIndexPosition::found(i, p);
            }

            Ordering::Greater => {
                // search_key > key_at_slot, continue
            }
        }
    }

    // search_key > all keys
    KeyIndexPosition::not_found(size)
}

/// Linear search upper bound with custom comparator.
///
/// Same semantics as `upper_bound_by`.
#[inline]
pub fn upper_bound_linear_by<const WIDTH: usize, F>(
    size: usize,
    perm: Permuter<WIDTH>,
    compare: F,
) -> usize
where
    F: Fn(usize) -> Ordering,
{
    for i in 0..size {
        let p = perm.get(i);
        match compare(p) {
            Ordering::Less => {
                return i;
            }

            Ordering::Equal => {
                return i + 1;
            }

            Ordering::Greater => {
                // Continue
            }
        }
    }

    size
}

// ============================================================================
//  Specialized Search Functions for Internodes
// ============================================================================

/// Upper bound search in an internode.
///
/// Returns the child index to follow for routing.
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to route
/// * `node` - The internode to search
///
/// # Returns
/// Child index (0 to nkeys). Use `node.child(result)` to get the child pointer.
///
/// # Example
///
/// ```ignore
/// let child_idx = upper_bound_internode(ikey, &internode);
/// let child_ptr = internode.child(child_idx);
/// // Follow child_ptr to continue traversal
/// ```
#[inline]
pub fn upper_bound_internode(search_ikey: u64, node: &InternodeNode) -> usize {
    let size: usize = node.size();

    // Internodes don't use permutation, keys are in physical order
    // Create identity permutation for the generic function (WIDTH=15)
    let perm = Permuter::<15>::make_sorted(size);

    upper_bound_by(size, perm, |slot| {
        let node_ikey: u64 = node.ikey(slot);
        search_ikey.cmp(&node_ikey)
    })
}

/// Upper bound search in an internode (direct version).
///
/// Optimized version that doesn't create a permutation.
#[inline]
pub fn upper_bound_internode_direct(search_ikey: u64, node: &InternodeNode) -> usize {
    let size: usize = node.size();
    let mut l: usize = 0;
    let mut r: usize = size;

    while l < r {
        let m: usize = (l + r) >> 1;
        let node_ikey: u64 = node.ikey(m);

        match search_ikey.cmp(&node_ikey) {
            Ordering::Less => {
                r = m;
            }

            Ordering::Equal => {
                return m + 1;
            }

            Ordering::Greater => {
                l = m + 1;
            }
        }
    }

    l
}

/// Upper bound search in an internode (generic version).
///
/// Works with any internode type implementing [`TreeInternode`].
/// Used by `MassTreeGeneric` for WIDTH-agnostic traversal.
///
/// # Algorithm
/// Uses **optimized linear search** for small nodes (WIDTH ≤ 16), matching C++ Masstree.
/// Linear search is faster than binary for small nodes due to:
/// - No branch mispredictions from binary search pattern
/// - Sequential memory access (better hardware prefetching)
/// - Simpler loop with no midpoint calculation
///
/// Optimizations:
/// - Loop unrolling (4 at a time)
/// - Early exit on match
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to route
/// * `node` - The internode to search (any type implementing [`TreeInternode`] )
///
/// # Returns
/// Child index (0 to nkeys). Use `node.child(result)` to get the child pointer.
/// Find the upper bound position for a search key in an internode.
///
/// Uses optimized linear search with loop unrolling. Linear search outperforms
/// binary search for small nodes (WIDTH ≤ 16) due to predictable branches and
/// cache-friendly sequential access.
///
/// Returns the child index to follow: the first position where `ikey[i] >= search_ikey`,
/// or `nkeys` if the search key is greater than all keys.
#[inline(always)]
pub fn upper_bound_internode_generic<I: TreeInternode>(search_ikey: u64, node: &I) -> usize {
    let size: usize = node.nkeys();
    let mut l: usize = 0;

    // Unrolled loop: process 4 keys per iteration
    while l + 4 <= size {
        let k0: u64 = node.ikey(l);
        if search_ikey < k0 {
            return l;
        }
        if search_ikey == k0 {
            return l + 1;
        }

        let k1: u64 = node.ikey(l + 1);
        if search_ikey < k1 {
            return l + 1;
        }
        if search_ikey == k1 {
            return l + 2;
        }

        let k2: u64 = node.ikey(l + 2);
        if search_ikey < k2 {
            return l + 2;
        }
        if search_ikey == k2 {
            return l + 3;
        }

        let k3: u64 = node.ikey(l + 3);
        if search_ikey < k3 {
            return l + 3;
        }
        if search_ikey == k3 {
            return l + 4;
        }

        l += 4;
    }

    // Handle remainder (0-3 keys)
    while l < size {
        let node_ikey: u64 = node.ikey(l);
        if search_ikey < node_ikey {
            return l;
        }
        if search_ikey == node_ikey {
            return l + 1;
        }
        l += 1;
    }

    l
}

// ============================================================================
//  Leaf ikey Matching (Scalar)
// ============================================================================

/// Find all slots in a leaf where `ikey == target_ikey`.
///
/// Returns a bitmask where bit `i` is set if `leaf.ikey(i) == target_ikey`.
///
/// # Arguments
/// * `target_ikey` - The 8-byte key slice to search for
/// * `leaf` - The leaf node to search
///
/// # Returns
/// A `u32` bitmask with bits set for matching slots (0-23 for WIDTH=24).
#[inline]
#[must_use]
pub fn find_ikey_matches_leaf24<S: slot::ValueSlot>(target_ikey: u64, leaf: &LeafNode24<S>) -> u32 {
    use crate::leaf24::WIDTH_24;

    let mut mask: u32 = 0;

    for i in 0..WIDTH_24 {
        if leaf.ikey(i) == target_ikey {
            mask |= 1 << i;
        }
    }

    mask
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod unit_tests;
