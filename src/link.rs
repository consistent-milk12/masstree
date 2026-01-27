//! Pointer marking utilities for concurrent split operations.
//!
//! Provides provenance-safe pointer marking using the LSB for split signaling.
//! Reference: `reference/btree_leaflink.hh:39-56`

const MARK_BIT: usize = 1;

/// Pointer marking/linking utilities for concurrent split operations.
#[derive(Debug)]
pub struct Linker;

impl Linker {
    /// Set the mark bit to signal split-in-progress (provenance-safe).
    ///
    /// # Arguments
    ///
    /// * `p` - Pointer to mark
    ///
    /// # Returns
    ///
    /// Marked pointer with LSB set. Preserves provenance via `map_addr`.
    #[inline(always)]
    pub fn mark_ptr<T>(p: *mut T) -> *mut T {
        p.map_addr(|a: usize| a | MARK_BIT)
    }

    /// Clear the mark bit (provenance-safe).
    ///
    /// # Arguments
    ///
    /// * `p` - Pointer to unmark
    ///
    /// # Returns
    ///
    /// Clean pointer with LSB cleared. Preserves provenance via `map_addr`.
    #[inline(always)]
    pub fn unmark_ptr<T>(p: *mut T) -> *mut T {
        p.map_addr(|a: usize| a & !MARK_BIT)
    }

    /// Check if pointer has mark bit set (split-in-progress).
    ///
    /// # Arguments
    ///
    /// * `p` - Pointer to check
    ///
    /// # Returns
    ///
    /// `true` if LSB is set, indicating concurrent split on this boundary.
    #[inline(always)]
    pub fn is_marked<T>(p: *mut T) -> bool {
        p.addr() & MARK_BIT != 0
    }

    /// Check for split and return cleaned pointer if safe.
    ///
    /// Combines `is_marked` + `unmark_ptr` for the common scan pattern.
    ///
    /// # Arguments
    ///
    /// * `next_raw` - Raw `next` pointer from leaf node
    ///
    /// # Returns
    ///
    /// * `Some(ptr)` - Unmarked pointer, safe to follow after version check
    /// * `None` - Split detected, caller should wait and retry
    #[inline(always)]
    pub fn check_split<T>(next_raw: *mut T) -> Option<*mut T> {
        if Self::is_marked(next_raw) {
            return None;
        }

        Some(Self::unmark_ptr(next_raw))
    }
}
