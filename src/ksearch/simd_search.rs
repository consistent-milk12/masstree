//! SIMD-accelerated search functions for tree nodes.
//!
//! This module provides high-level search functions that use SIMD primitives
//! from the [`simd`](super::simd) module when beneficial.
//!
//! # Design Philosophy
//!
//! For leaf nodes with WIDTH=24, loading all ikeys upfront and using SIMD
//! comparison is faster than sequential per-slot atomic loads:
//! - **Sequential:** Up to 24 atomic loads with branch misprediction
//! - **SIMD:** 24 loads (batched) + 6 AVX2 comparisons (parallel)
//!
//! The key insight is that we trade sequential loads-with-comparison for
//! bulk loading followed by parallel comparison.

use super::simd;
use crate::leaf24::{LeafNode24, WIDTH_24};
use crate::slot::ValueSlot;

// ============================================================================
//  Leaf SIMD Search (WIDTH=24)
// ============================================================================

/// Find all slots in a WIDTH=24 leaf that have matching ikey, returning a bitmask.
///
/// Uses SIMD to compare all 24 slots in parallel. Returns a `u32` bitmask where
/// bit `i` is set if `leaf.ikey(i) == target_ikey`.
///
/// # Performance
///
/// This pre-loads all 24 ikeys via `load_all_ikeys()` and uses SIMD comparison.
/// With AVX2, this requires only 6 comparison operations instead of up to 24
/// sequential comparisons.
///
/// # Usage
///
/// ```ignore
/// let matches: u32 = find_ikey_matches_leaf24(target_ikey, &leaf);
/// for i in 0..perm.size() {
///     let slot: usize = perm.get(i);
///     if (matches & (1 << slot)) != 0 {
///         // This slot has matching ikey, check keylenx...
///     }
/// }
/// ```
///
/// # Arguments
/// * `search_ikey` - The 8-byte key to search for
/// * `leaf` - The WIDTH=24 leaf node to search
///
/// # Returns
/// Bitmask where bit `i` is set if slot `i` has matching ikey (bits 0-23).
#[inline]
#[must_use]
pub fn find_ikey_matches_leaf24<S: ValueSlot>(search_ikey: u64, leaf: &LeafNode24<S>) -> u32 {
    // Load all ikeys into a contiguous buffer (24 Acquire loads)
    let ikeys: [u64; WIDTH_24] = leaf.load_all_ikeys();

    // Use SIMD to find all matching slots
    simd::find_all_matches_u64(&ikeys, WIDTH_24, search_ikey)
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    fn make_leaf_with_ikeys(ikeys: &[u64]) -> Box<LeafNode24<LeafValue<()>>> {
        let leaf = LeafNode24::<LeafValue<()>>::new();
        for (i, &ikey) in ikeys.iter().enumerate() {
            leaf.set_ikey(i, ikey);
        }
        leaf
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_empty() {
        let leaf = make_leaf_with_ikeys(&[]);
        // All slots have default value 0, search for 42 finds nothing
        let matches = find_ikey_matches_leaf24(42, &leaf);
        assert_eq!(matches, 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_single() {
        let leaf = make_leaf_with_ikeys(&[42]);
        let matches = find_ikey_matches_leaf24(42, &leaf);
        assert_eq!(matches & 1, 1); // Slot 0 matches
        assert_eq!(find_ikey_matches_leaf24(43, &leaf) & 1, 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_multiple() {
        // Set up slots 0, 2, 5 with same ikey
        let leaf = LeafNode24::<LeafValue<()>>::new();
        leaf.set_ikey(0, 100);
        leaf.set_ikey(1, 200);
        leaf.set_ikey(2, 100);
        leaf.set_ikey(3, 300);
        leaf.set_ikey(4, 400);
        leaf.set_ikey(5, 100);

        let matches = find_ikey_matches_leaf24(100, &leaf);
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
        let leaf = LeafNode24::<LeafValue<()>>::new();
        for i in 0..24 {
            leaf.set_ikey(i, 42);
        }

        let matches = find_ikey_matches_leaf24(42, &leaf);
        // All 24 slots should match (bits 0-23)
        assert_eq!(matches, 0x00FF_FFFF);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_ikey_matches_high_slots() {
        // Test that high slot indices (20-23) work correctly
        let leaf = LeafNode24::<LeafValue<()>>::new();
        leaf.set_ikey(20, 999);
        leaf.set_ikey(21, 999);
        leaf.set_ikey(22, 999);
        leaf.set_ikey(23, 999);

        let matches = find_ikey_matches_leaf24(999, &leaf);
        // Bits 20-23 should be set
        assert_eq!(matches & (1 << 20), 1 << 20);
        assert_eq!(matches & (1 << 21), 1 << 21);
        assert_eq!(matches & (1 << 22), 1 << 22);
        assert_eq!(matches & (1 << 23), 1 << 23);
        // Lower bits should not be set (they have default value 0)
        assert_eq!(matches & 0x000F_FFFF, 0);
    }
}
