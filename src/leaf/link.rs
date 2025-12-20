//! Concurrent leaf-link operations with CAS+mark protocol.
//!
//! Reference: `reference/btree_leaflink.hh:39-56`

use std::ptr as StdPtr;

use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
use crate::slot::ValueSlot;

use super::LeafNode;

const MARK_BIT: usize = 1;

/// Set mark bit (provenance-safe).
#[inline]
pub fn mark_ptr<T>(p: *mut T) -> *mut T {
    p.map_addr(|a| a | MARK_BIT)
}

/// Clear mark bit (provenance-safe).
#[inline]
pub fn unmark_ptr<T>(p: *mut T) -> *mut T {
    p.map_addr(|a| a & !MARK_BIT)
}

/// Check if marked.
#[inline]
pub fn is_marked<T>(p: *mut T) -> bool {
    p.addr() & MARK_BIT != 0
}

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    /// Compare-and-swap the next pointer.
    ///
    /// # Errors
    ///
    /// Returns `Err(current_value)` if the CAS failed because the current
    /// value didn't match `current`.
    #[inline]
    pub fn cas_next(&self, current: *mut Self, new: *mut Self) -> Result<*mut Self, *mut Self> {
        self.next
            .compare_exchange(current, new, CAS_SUCCESS, CAS_FAILURE)
    }

    /// Check if next pointer is marked.
    #[inline]
    pub fn is_next_marked(&self) -> bool {
        is_marked(self.next_raw())
    }

    /// Wait for split to complete (spin on mark).
    #[inline]
    pub fn wait_for_split(&self) {
        while self.is_next_marked() {
            std::hint::spin_loop();
        }
    }

    /// Link new sibling after split.
    ///
    /// Returns true if successful, false if CAS failed (retry).
    ///
    /// # Safety
    ///
    /// - `new_sibling` must be a valid pointer to a newly allocated leaf.
    /// - The caller must hold the lock on `self` during the link operation.
    /// - No other thread may be modifying `self.next` concurrently.
    pub unsafe fn link_split(&self, new_sibling: *mut Self) -> bool {
        let old_next = self.safe_next();

        // P0.5 FIX: Never mark a null pointer.
        if old_next.is_null() {
            unsafe {
                (*new_sibling).set_next(StdPtr::null_mut());
                (*new_sibling).set_prev(StdPtr::from_ref(self).cast_mut());
            }

            return self.cas_next(old_next, new_sibling).is_ok();
        }

        // Mark before modifying.
        let marked = mark_ptr(old_next);
        if self.cas_next(old_next, marked).is_err() {
            return false;
        }

        unsafe {
            (*new_sibling).set_next(old_next);
            (*new_sibling).set_prev(StdPtr::from_ref(self).cast_mut());
        }

        match self.cas_next(marked, new_sibling) {
            Ok(_) => {
                unsafe {
                    (*old_next).set_prev(new_sibling);
                }
                true
            }
            Err(_) => false,
        }
    }
}
