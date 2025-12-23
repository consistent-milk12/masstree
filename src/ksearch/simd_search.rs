//! SIMD-accelerated search functions for tree nodes.
//!
//! This module provides high-level search functions that use SIMD primitives
//! from the [`simd`](super::simd) module when beneficial, falling back to
//! binary search for larger nodes.
//!
//! # Design Philosophy
//!
//! For small nodes (≤8 keys), SIMD linear scan can outperform binary search:
//! - Binary search: O(log n) comparisons, but each comparison has data dependency
//! - SIMD scan: O(n/4) or O(n/8) comparisons, all in parallel
//!
//! For larger nodes, binary search remains optimal because the cost of loading
//! all keys for SIMD outweighs the parallel comparison benefit.
//!
//! # Usage
//!
//! ```ignore
//! use crate::ksearch::simd_search::{upper_bound_internode_simd, find_ikey_matches_leaf};
//!
//! // Internode search
//! let child_idx = upper_bound_internode_simd(target_ikey, &internode);
//! let child = internode.child(child_idx);
//!
//! // Leaf ikey search (returns bitmask of matching slots)
//! let matches = find_ikey_matches_leaf(target_ikey, &leaf);
//! for i in 0..perm.size() {
//!     let slot = perm.get(i);
//!     if matches & (1 << slot) != 0 {
//!         // This slot has matching ikey, check keylenx...
//!     }
//! }
//! ```

use crate::internode::InternodeNode;
use crate::leaf::LeafNode;
use crate::slot::ValueSlot;

use super::simd;
use super::upper_bound_internode_direct;

/// Threshold for using SIMD vs binary search.
///
/// For nodes with ≤ this many keys, use SIMD linear scan.
/// For larger nodes, binary search is more efficient.
const SIMD_THRESHOLD: usize = 8;

/// SIMD-accelerated upper bound search in an internode.
///
/// Uses SIMD comparison for small nodes (≤8 keys), falling back to
/// binary search for larger nodes.
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to route
/// * `node` - The internode to search
///
/// # Returns
/// Child index (0 to nkeys). Use `node.child(result)` to get the child pointer.
///
/// # Performance
///
/// For WIDTH=15 (default):
/// - Nodes with ≤8 keys: Uses SIMD, ~2x faster than binary search
/// - Nodes with >8 keys: Uses binary search, optimal for larger nodes
#[inline(always)]
#[expect(
    clippy::needless_range_loop,
    clippy::indexing_slicing,
    reason = "i < size < WIDTH <= 15 < 16, cannot use enumerate due to ikey() call"
)]
pub fn upper_bound_internode_simd<S: ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    node: &InternodeNode<S, WIDTH>,
) -> usize {
    let size = node.size();

    // For small nodes, SIMD is beneficial
    if size <= SIMD_THRESHOLD {
        // Load ikeys into contiguous array for SIMD
        let mut keys = [u64::MAX; 16]; // Max WIDTH is 15, pad with MAX
        for i in 0..size {
            keys[i] = node.ikey(i);
        }

        // Use SIMD to count keys ≤ target
        // This gives us the upper bound (first key > target)
        simd::count_le_u64(&keys, size, search_ikey)
    } else {
        // Fall back to binary search for larger nodes
        upper_bound_internode_direct(search_ikey, node)
    }
}

/// Find exact ikey match in an internode using SIMD.
///
/// Returns the slot index if the exact ikey is found, or `None` otherwise.
/// This is useful for verifying a key exists before following a child pointer.
///
/// # Performance
///
/// Always uses SIMD for exact match (no threshold), as SIMD equality
/// comparison is always efficient regardless of node size.
#[inline]
#[allow(dead_code)] // Will be used in future optimizations
#[expect(
    clippy::needless_range_loop,
    clippy::indexing_slicing,
    reason = "i < size < WIDTH <= 15 < 16, cannot use enumerate due to ikey() call"
)]
pub fn find_exact_internode<S: ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    node: &InternodeNode<S, WIDTH>,
) -> Option<usize> {
    let size = node.size();
    if size == 0 {
        return None;
    }

    // Load ikeys for SIMD comparison
    let mut keys = [0u64; 16];
    for i in 0..size {
        keys[i] = node.ikey(i);
    }

    simd::find_exact_u64(&keys[..size], search_ikey)
}

// ============================================================================
//  Leaf SIMD Search
// ============================================================================

/// Find all slots in a leaf that have matching ikey, returning a bitmask.
///
/// Uses SIMD to compare all slots in parallel. Returns a `u16` bitmask where
/// bit `i` is set if `leaf.ikey(i) == target_ikey`.
///
/// # Performance
///
/// This pre-loads all ikeys and uses SIMD comparison, trading O(WIDTH) loads
/// for O(WIDTH/4) SIMD comparisons (with AVX2). Beneficial when the leaf
/// has many keys and multiple comparisons would be needed.
///
/// # Usage
///
/// ```ignore
/// let matches = find_ikey_matches_leaf(target_ikey, &leaf);
/// for i in 0..perm.size() {
///     let slot = perm.get(i);
///     if matches & (1 << slot) != 0 {
///         // This slot has matching ikey, check keylenx...
///     }
/// }
/// ```
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to search for
/// * `leaf` - The leaf node to search
///
/// # Returns
/// Bitmask where bit `i` is set if slot `i` has matching ikey.
#[inline]
#[must_use]
pub fn find_ikey_matches_leaf<S: ValueSlot, const WIDTH: usize>(
    search_ikey: u64,
    leaf: &LeafNode<S, WIDTH>,
) -> u16 {
    // Load all ikeys into a contiguous buffer
    let ikeys: [u64; WIDTH] = leaf.load_all_ikeys();

    // Use SIMD to find all matching slots
    simd::find_all_matches_u64(&ikeys, WIDTH, search_ikey)
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internode::InternodeNode;
    use crate::leaf::LeafValue;

    // Create a test internode with given keys
    fn make_internode(keys: &[u64]) -> Box<InternodeNode<LeafValue<()>, 15>> {
        let node = InternodeNode::<LeafValue<()>, 15>::new(0);
        for (i, &key) in keys.iter().enumerate() {
            node.set_ikey(i, key);
        }
        #[expect(clippy::cast_possible_truncation, reason = "keys.len() <= WIDTH <= 15")]
        node.set_nkeys(keys.len() as u8);
        node
    }

    #[test]
    fn test_upper_bound_empty() {
        let node = make_internode(&[]);
        assert_eq!(upper_bound_internode_simd(100, &node), 0);
    }

    #[test]
    fn test_upper_bound_single() {
        let node = make_internode(&[50]);
        assert_eq!(upper_bound_internode_simd(25, &node), 0); // Less than
        assert_eq!(upper_bound_internode_simd(50, &node), 1); // Equal (route right)
        assert_eq!(upper_bound_internode_simd(75, &node), 1); // Greater
    }

    #[test]
    fn test_upper_bound_multiple() {
        let node = make_internode(&[10, 20, 30, 40, 50]);
        assert_eq!(upper_bound_internode_simd(5, &node), 0);
        assert_eq!(upper_bound_internode_simd(10, &node), 1);
        assert_eq!(upper_bound_internode_simd(15, &node), 1);
        assert_eq!(upper_bound_internode_simd(30, &node), 3);
        assert_eq!(upper_bound_internode_simd(50, &node), 5);
        assert_eq!(upper_bound_internode_simd(100, &node), 5);
    }

    #[test]
    fn test_upper_bound_matches_binary() {
        // Verify SIMD matches binary search for various sizes
        for size in 1..=15 {
            #[expect(clippy::cast_sign_loss)]
            let keys: Vec<u64> = (0..size).map(|i| (i as u64 + 1) * 10).collect();
            let node = make_internode(&keys);

            for target in [5, 10, 15, 25, 35, 45, 100, 150] {
                let simd_result = upper_bound_internode_simd(target, &node);
                let binary_result = upper_bound_internode_direct(target, &node);
                assert_eq!(
                    simd_result, binary_result,
                    "Mismatch for size={size}, target={target}"
                );
            }
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact() {
        let node = make_internode(&[10, 20, 30, 40, 50]);
        assert_eq!(find_exact_internode(10, &node), Some(0));
        assert_eq!(find_exact_internode(30, &node), Some(2));
        assert_eq!(find_exact_internode(50, &node), Some(4));
        assert_eq!(find_exact_internode(25, &node), None);
    }

    // ========================================================================
    //  Leaf SIMD Search Tests
    // ========================================================================

    use crate::leaf::LeafNode;

    fn make_leaf_with_ikeys(ikeys: &[u64]) -> Box<LeafNode<LeafValue<()>, 15>> {
        let leaf = LeafNode::<LeafValue<()>, 15>::new();
        for (i, &ikey) in ikeys.iter().enumerate() {
            leaf.set_ikey(i, ikey);
        }
        leaf
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_empty() {
        let leaf = make_leaf_with_ikeys(&[]);
        // Even though leaf has no actual keys in permutation,
        // all slots are loaded. Slot 0..15 have default value 0.
        // But we search for 42, so no matches.
        let matches = find_ikey_matches_leaf(42, &leaf);
        assert_eq!(matches, 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_single() {
        let leaf = make_leaf_with_ikeys(&[42]);
        let matches = find_ikey_matches_leaf(42, &leaf);
        assert_eq!(matches & 1, 1); // Slot 0 matches
        assert_eq!(find_ikey_matches_leaf(43, &leaf) & 1, 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_multiple() {
        // Set up slots 0, 2, 5 with same ikey
        let leaf = LeafNode::<LeafValue<()>, 15>::new();
        leaf.set_ikey(0, 100);
        leaf.set_ikey(1, 200);
        leaf.set_ikey(2, 100);
        leaf.set_ikey(3, 300);
        leaf.set_ikey(4, 400);
        leaf.set_ikey(5, 100);

        let matches = find_ikey_matches_leaf(100, &leaf);
        // Bits 0, 2, 5 should be set
        assert_eq!(matches & (1 << 0), 1 << 0);
        assert_eq!(matches & (1 << 2), 1 << 2);
        assert_eq!(matches & (1 << 5), 1 << 5);
        // Bits 1, 3, 4 should not be set
        assert_eq!(matches & (1 << 1), 0);
        assert_eq!(matches & (1 << 3), 0);
        assert_eq!(matches & (1 << 4), 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_all_slots() {
        let leaf = LeafNode::<LeafValue<()>, 15>::new();
        for i in 0..15 {
            leaf.set_ikey(i, 42);
        }

        let matches = find_ikey_matches_leaf(42, &leaf);
        // All 15 slots should match
        assert_eq!(matches, 0x7FFF); // bits 0-14 set
    }
}
