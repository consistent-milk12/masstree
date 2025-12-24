//! Pointer marking utilities for concurrent split operations.
//!
//! Provides provenance-safe pointer marking using the LSB for split signaling.
//! Reference: `reference/btree_leaflink.hh:39-56`

const MARK_BIT: usize = 1;

/// Set mark bit (provenance-safe).
#[inline(always)]
pub fn mark_ptr<T>(p: *mut T) -> *mut T {
    p.map_addr(|a| a | MARK_BIT)
}

/// Clear mark bit (provenance-safe).
#[inline(always)]
pub fn unmark_ptr<T>(p: *mut T) -> *mut T {
    p.map_addr(|a| a & !MARK_BIT)
}

/// Check if marked.
#[inline(always)]
pub fn is_marked<T>(p: *mut T) -> bool {
    p.addr() & MARK_BIT != 0
}
