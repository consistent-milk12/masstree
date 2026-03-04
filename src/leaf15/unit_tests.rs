//! Unit tests for [`LeafNode15`] methods.

use super::LeafNode15;
use crate::leaf15::{MODSTATE_DELETED_LAYER, MODSTATE_EMPTY, MODSTATE_INSERT, MODSTATE_REMOVE};
use crate::policy::BoxPolicy;

// ========================================================================
//  Adaptive Prefetch Tests
// ========================================================================

#[test]
fn test_prefetch_adaptive() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new_boxed();

    // Test all size ranges - these should not panic or crash
    leaf.prefetch_for_search_adaptive(0);
    leaf.prefetch_for_search_adaptive(4);
    leaf.prefetch_for_search_adaptive(8);
    leaf.prefetch_for_search_adaptive(12);
    leaf.prefetch_for_search_adaptive(15);
}

#[test]
fn test_prefetch_for_search() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new_boxed();

    // Should not panic or crash
    leaf.prefetch_for_search();
}

#[test]
fn test_prefetch_adaptive_empty_node() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new_boxed();

    // For empty nodes, adaptive prefetch should skip CL 2-3
    // This is purely a "doesn't crash" test - we can't verify
    // that the CPU honored the prefetch hint
    assert_eq!(leaf.size(), 0);
    leaf.prefetch_for_search_adaptive(0);
}

#[test]
fn test_prefetch_adaptive_small_node() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new_boxed();

    // For size <= 8, adaptive prefetch should skip CL 3
    // Test boundary conditions
    leaf.prefetch_for_search_adaptive(1);
    leaf.prefetch_for_search_adaptive(7);
    leaf.prefetch_for_search_adaptive(8);
}

#[test]
fn test_prefetch_adaptive_large_node() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new_boxed();

    // For size > 8, adaptive prefetch should include CL 3
    leaf.prefetch_for_search_adaptive(9);
    leaf.prefetch_for_search_adaptive(14);
    leaf.prefetch_for_search_adaptive(15);
}

// ========================================================================
//  Modstate Queued Bit Masking Tests
// ========================================================================

type TestLeaf = LeafNode15<BoxPolicy<u64>>;

#[test]
fn test_modstate_queued_bit_lifecycle() {
    let leaf: Box<TestLeaf> = LeafNode15::new_boxed();

    // Initial state: INSERT, not queued.
    assert!(!leaf.is_queued());
    assert!(!leaf.is_empty_state());
    assert!(!leaf.is_removing());
    assert!(!leaf.deleted_layer());

    // Mark queued. First call returns true (was not queued).
    assert!(leaf.try_mark_queued());
    assert!(leaf.is_queued());
    // Second call returns false (already queued).
    assert!(!leaf.try_mark_queued());
    assert!(leaf.is_queued());

    // Lifecycle checks still work with queued bit set.
    assert!(!leaf.is_empty_state());
    assert!(!leaf.is_removing());
    assert!(!leaf.deleted_layer());
}

#[test]
fn test_modstate_transitions_preserve_queued_bit() {
    let leaf: Box<TestLeaf> = LeafNode15::new_boxed();

    // Set queued bit first.
    assert!(leaf.try_mark_queued());

    // REMOVE transition preserves queued.
    leaf.mark_remove();
    assert!(leaf.is_removing());
    assert!(leaf.is_queued());

    // EMPTY transition preserves queued.
    leaf.mark_empty();
    assert!(leaf.is_empty_state());
    assert!(leaf.is_queued());
    assert!(!leaf.is_removing());

    // DELETED_LAYER transition preserves queued.
    leaf.mark_deleted_layer();
    assert!(leaf.deleted_layer());
    assert!(leaf.is_queued());
    assert!(!leaf.is_empty_state());
}

#[test]
fn test_modstate_clear_empty_clears_queued() {
    let leaf: Box<TestLeaf> = LeafNode15::new_boxed();

    leaf.mark_empty();
    assert!(leaf.try_mark_queued());
    assert!(leaf.is_queued());
    assert!(leaf.is_empty_state());

    // clear_empty_state stores MODSTATE_INSERT (0), clearing queued bit.
    leaf.clear_empty_state();
    assert!(!leaf.is_queued());
    assert!(!leaf.is_empty_state());
}

#[test]
fn test_modstate_clear_queued_preserves_value() {
    let leaf: Box<TestLeaf> = LeafNode15::new_boxed();

    leaf.mark_empty();
    assert!(leaf.try_mark_queued());
    assert!(leaf.is_empty_state());
    assert!(leaf.is_queued());

    // clear_queued only strips the bit, value unchanged.
    leaf.clear_queued();
    assert!(!leaf.is_queued());
    assert!(leaf.is_empty_state());
}

#[test]
fn test_modstate_value_constants() {
    // Verify constants don't overlap with queued bit.
    assert_eq!(MODSTATE_INSERT & 0x04, 0);
    assert_eq!(MODSTATE_REMOVE & 0x04, 0);
    assert_eq!(MODSTATE_DELETED_LAYER & 0x04, 0);
    assert_eq!(MODSTATE_EMPTY & 0x04, 0);

    // Verify all values fit in 2 bits.
    const { assert!(MODSTATE_INSERT < 4) };
    const { assert!(MODSTATE_REMOVE < 4) };
    const { assert!(MODSTATE_DELETED_LAYER < 4) };
    const { assert!(MODSTATE_EMPTY < 4) };
}
