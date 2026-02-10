//! Unit tests for [`LeafNode15`] methods.

use super::LeafNode15;
use crate::policy::ArcPolicy;

// ========================================================================
//  Adaptive Prefetch Tests
// ========================================================================

#[test]
fn test_prefetch_adaptive() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = LeafNode15::new_boxed();

    // Test all size ranges - these should not panic or crash
    leaf.prefetch_for_search_adaptive(0);
    leaf.prefetch_for_search_adaptive(4);
    leaf.prefetch_for_search_adaptive(8);
    leaf.prefetch_for_search_adaptive(12);
    leaf.prefetch_for_search_adaptive(15);
}

#[test]
fn test_prefetch_for_search() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = LeafNode15::new_boxed();

    // Should not panic or crash
    leaf.prefetch_for_search();
}

#[test]
fn test_prefetch_adaptive_empty_node() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = LeafNode15::new_boxed();

    // For empty nodes, adaptive prefetch should skip CL 2-3
    // This is purely a "doesn't crash" test - we can't verify
    // that the CPU honored the prefetch hint
    assert_eq!(leaf.size(), 0);
    leaf.prefetch_for_search_adaptive(0);
}

#[test]
fn test_prefetch_adaptive_small_node() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = LeafNode15::new_boxed();

    // For size <= 8, adaptive prefetch should skip CL 3
    // Test boundary conditions
    leaf.prefetch_for_search_adaptive(1);
    leaf.prefetch_for_search_adaptive(7);
    leaf.prefetch_for_search_adaptive(8);
}

#[test]
fn test_prefetch_adaptive_large_node() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = LeafNode15::new_boxed();

    // For size > 8, adaptive prefetch should include CL 3
    leaf.prefetch_for_search_adaptive(9);
    leaf.prefetch_for_search_adaptive(14);
    leaf.prefetch_for_search_adaptive(15);
}
