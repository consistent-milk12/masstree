//! Key search algorithms for `MassTree`.
//!
//! Provides binary search for:
//! - Lower bound in leaves (finding keys or insertion points)
//! - Upper bound in internodes (routing to children)
//!
//! # Submodules
//!
//! - [`simd`]: SIMD-accelerated key comparison primitives
//! - [`simd_search`]: High-level SIMD search functions for tree nodes
//!
//! # Reference
//! Based on `ksearch.hh` from the C++ Masstree implementation.

pub mod simd;
pub mod simd_search;

// Re-export SIMD search functions for convenience
pub use simd_search::find_ikey_matches_leaf;
pub use simd_search::upper_bound_internode_simd;

use crate::internode::InternodeNode;
use crate::leaf::LeafNode;
use crate::leaf_trait::TreeInternode;
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
//  Specialized Search Functions
// ============================================================================

/// Lower bound search in a leaf node.
///
/// Searches for a key by comparing both `ikey` and `keylenx`.
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to search for
/// * `search_keylenx` - The key length/type encoding
/// * `node` - The leaf node to search
///
/// # Returns
/// `KeyIndexPosition` with logical position and physical slot (if found).
///
/// # Example
///
/// ```ignore
/// let pos = lower_bound_leaf(ikey, keylenx, &leaf);
/// if pos.is_found() {
///     let slot = pos.slot();
///     // Key found at physical slot
/// } else {
///     let insert_pos = pos.i;
///     // Key not found, would insert at logical position i
/// }
/// ```
#[inline]
pub fn lower_bound_leaf<S: slot::ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    search_keylenx: u8,
    node: &LeafNode<S, WIDTH>,
) -> KeyIndexPosition {
    let perm: Permuter<WIDTH> = node.permutation();
    let size: usize = perm.size();

    lower_bound_by(size, perm, |slot: usize| {
        // Compare ikey first, then keylenx only if ikeys match
        let node_ikey: u64 = node.ikey(slot);
        match search_ikey.cmp(&node_ikey) {
            Ordering::Equal => search_keylenx.cmp(&node.keylenx(slot)),
            other => other,
        }
    })
}

/// Lower bound search in a leaf node by ikey only.
///
/// Simpler version that only compares ikeys, ignoring keylenx.
/// Useful for finding the first slot with a given ikey prefix.
#[inline]
pub fn lower_bound_leaf_ikey<S: slot::ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    node: &LeafNode<S, WIDTH>,
) -> KeyIndexPosition {
    let perm: Permuter<WIDTH> = node.permutation();
    let size: usize = perm.size();

    lower_bound_by(size, perm, |slot: usize| {
        let node_ikey: u64 = node.ikey(slot);

        search_ikey.cmp(&node_ikey)
    })
}

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
pub fn upper_bound_internode<S: slot::ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    node: &InternodeNode<S, WIDTH>,
) -> usize {
    let size: usize = node.size();

    // Internodes don't use permutation, keys are in physical order
    // Create identity permutation for the generic function
    let perm = Permuter::<WIDTH>::make_sorted(size);

    upper_bound_by(size, perm, |slot| {
        let node_ikey: u64 = node.ikey(slot);
        search_ikey.cmp(&node_ikey)
    })
}

/// Upper bound search in an internode (direct version).
///
/// Optimized version that doesn't create a permutation.
#[inline]
pub fn upper_bound_internode_direct<S: slot::ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    node: &InternodeNode<S, WIDTH>,
) -> usize {
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
/// # Arguments
/// * `search_ikey` - The 8-byte key to route
/// * `node` - The internode to search (any type implementing [`TreeInternode`] )
///
/// # Returns
/// Child index (0 to nkeys). Use `node.child(result)` to get the child pointer.
#[inline]
pub fn upper_bound_internode_generic<S: slot::ValueSlot, I: TreeInternode<S>>(
    search_ikey: u64,
    node: &I,
) -> usize {
    let size: usize = node.nkeys();
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

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod tests {
    use super::*;

    // Helper to create a sorted permutation
    fn sorted_perm<const W: usize>(size: usize) -> Permuter<W> {
        Permuter::make_sorted(size)
    }

    // ========================================================================
    //  KeyIndexPosition Tests
    // ========================================================================

    #[test]
    fn test_key_index_position_found() {
        let pos = KeyIndexPosition::found(3, 5);

        assert!(pos.is_found());
        assert_eq!(pos.i, 3);
        assert_eq!(pos.p, 5);
        assert_eq!(pos.slot(), 5);
        assert_eq!(pos.try_slot(), Some(5));
    }

    #[test]
    fn test_key_index_position_not_found() {
        let pos = KeyIndexPosition::not_found(7);

        assert!(!pos.is_found());
        assert_eq!(pos.i, 7);
        assert_eq!(pos.p, KeyIndexPosition::NOT_FOUND);
        assert_eq!(pos.try_slot(), None);
    }

    #[test]
    #[should_panic(expected = "slot() called on not-found")]
    fn test_key_index_position_slot_panics() {
        let pos = KeyIndexPosition::not_found(0);
        let _ = pos.slot();
    }

    // ========================================================================
    //  Generic Binary Search Tests
    // ========================================================================

    #[test]
    fn test_lower_bound_empty() {
        let perm: Permuter<15> = sorted_perm(0);
        let keys: [u64; 0] = [];

        let pos: KeyIndexPosition = lower_bound_by(0, perm, |slot| 100u64.cmp(&keys[slot]));

        assert!(!pos.is_found());
        assert_eq!(pos.i, 0);
    }

    #[test]
    fn test_lower_bound_single_less() {
        let perm: Permuter<15> = sorted_perm(1);
        let keys: [u64; 1] = [50];

        let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 25u64.cmp(&keys[slot]));

        assert!(!pos.is_found());
        assert_eq!(pos.i, 0); // Insert before the only key
    }

    #[test]
    fn test_lower_bound_single_equal() {
        let perm: Permuter<15> = sorted_perm(1);
        let keys: [u64; 1] = [50];

        let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 50u64.cmp(&keys[slot]));

        assert!(pos.is_found());
        assert_eq!(pos.i, 0);
        assert_eq!(pos.slot(), 0);
    }

    #[test]
    fn test_lower_bound_single_greater() {
        let perm: Permuter<15> = sorted_perm(1);
        let keys: [u64; 1] = [50];

        let pos: KeyIndexPosition = lower_bound_by(1, perm, |slot| 75u64.cmp(&keys[slot]));

        assert!(!pos.is_found());
        assert_eq!(pos.i, 1); // Insert after the only key
    }

    #[test]
    fn test_lower_bound_multiple_exact_match() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        // Find middle element
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot| 30u64.cmp(&keys[slot]));

        assert!(pos.is_found());
        assert_eq!(pos.i, 2);
        assert_eq!(pos.slot(), 2);
    }

    #[test]
    fn test_lower_bound_multiple_not_found() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        // Search for value between 20 and 30
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot| 25u64.cmp(&keys[slot]));

        assert!(!pos.is_found());
        assert_eq!(pos.i, 2); // Would insert at position 2 (before 30)
    }

    #[test]
    fn test_upper_bound_empty() {
        let perm: Permuter<15> = sorted_perm(0);
        let keys: [u64; 0] = [];

        let idx: usize = upper_bound_by(0, perm, |slot| 100u64.cmp(&keys[slot]));

        assert_eq!(idx, 0);
    }

    #[test]
    fn test_upper_bound_exact_match() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        // Exact match at index 2 (key 30) returns 3 (right child)
        let idx: usize = upper_bound_by(5, perm, |slot| 30u64.cmp(&keys[slot]));

        assert_eq!(idx, 3);
    }

    #[test]
    fn test_upper_bound_between_keys() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        // Search for 25 (between 20 and 30) returns 2
        let idx: usize = upper_bound_by(5, perm, |slot| 25u64.cmp(&keys[slot]));

        assert_eq!(idx, 2);
    }

    #[test]
    fn test_upper_bound_less_than_all() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        let idx: usize = upper_bound_by(5, perm, |slot| 5u64.cmp(&keys[slot]));

        assert_eq!(idx, 0); // Route to leftmost child
    }

    #[test]
    fn test_upper_bound_greater_than_all() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        let idx: usize = upper_bound_by(5, perm, |slot| 100u64.cmp(&keys[slot]));

        assert_eq!(idx, 5); // Route to rightmost child
    }

    // ========================================================================
    //  Linear Search Tests (verify same results as binary)
    // ========================================================================

    #[test]
    fn test_linear_matches_binary() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        for search in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] {
            let binary: KeyIndexPosition = lower_bound_by(5, perm, |slot| search.cmp(&keys[slot]));
            let linear: KeyIndexPosition =
                lower_bound_linear_by(5, perm, |slot| search.cmp(&keys[slot]));

            assert_eq!(
                binary, linear,
                "Mismatch for search key {search}: binary={binary:?}, linear={linear:?}"
            );
        }
    }

    #[test]
    fn test_linear_upper_bound_matches_binary() {
        let perm: Permuter<15> = sorted_perm(5);
        let keys: [u64; 5] = [10, 20, 30, 40, 50];

        for search in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] {
            let binary: usize = upper_bound_by(5, perm, |slot: usize| search.cmp(&keys[slot]));
            let linear: usize =
                upper_bound_linear_by(5, perm, |slot: usize| search.cmp(&keys[slot]));

            assert_eq!(
                binary, linear,
                "Upper bound mismatch for search key {search}: binary={binary}, linear={linear}"
            );
        }
    }

    // ========================================================================
    //  Permutation-Aware Tests
    // ========================================================================

    #[test]
    fn test_lower_bound_with_permutation() {
        // Test with make_sorted which creates an identity permutation.
        // Physical slots: [10, 20, 30, 40, 50] (already sorted)
        // Permutation: identity [0, 1, 2, 3, 4]
        // Logical order: [10, 20, 30, 40, 50]

        let keys: [u64; 5] = [10, 20, 30, 40, 50]; // Physical order (sorted)

        // Create sorted permutation
        let perm: Permuter<15> = Permuter::make_sorted(5);

        // Search for 30: logical position 2, physical slot 2
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 30u64.cmp(&keys[slot]));

        assert!(pos.is_found());
        assert_eq!(pos.i, 2); // Logical position
        assert_eq!(pos.slot(), 2); // Physical slot (same as logical for identity permutation)

        // Search for 10: logical position 0, physical slot 0
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 10u64.cmp(&keys[slot]));
        assert!(pos.is_found());
        assert_eq!(pos.i, 0);
        assert_eq!(pos.slot(), 0);

        // Search for 50: logical position 4, physical slot 4
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 50u64.cmp(&keys[slot]));
        assert!(pos.is_found());
        assert_eq!(pos.i, 4);
        assert_eq!(pos.slot(), 4);

        // Search for non-existent key (25): should be not found, insert at position 2
        let pos: KeyIndexPosition = lower_bound_by(5, perm, |slot: usize| 25u64.cmp(&keys[slot]));
        assert!(!pos.is_found());
        assert_eq!(pos.i, 2); // Would insert before 30
    }
}
