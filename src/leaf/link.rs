//! Concurrent leaf-link operations with CAS+mark protocol.
//!
//! Reference: `reference/btree_leaflink.hh:39-56`

use std::ptr as StdPtr;

use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
use crate::slot::ValueSlot;

use super::LeafNode;

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

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    /// Compare-and-swap the next pointer.
    ///
    /// # Errors
    ///
    /// Returns `Err(current_value)` if the CAS failed because the current
    /// value didn't match `current`.
    #[inline(always)]
    pub fn cas_next(&self, current: *mut Self, new: *mut Self) -> Result<*mut Self, *mut Self> {
        self.next
            .compare_exchange(current, new, CAS_SUCCESS, CAS_FAILURE)
    }

    /// Check if next pointer is marked.
    #[inline(always)]
    pub fn is_next_marked(&self) -> bool {
        is_marked(self.next_raw())
    }

    /// Wait for split to complete (spin on mark) with bounded wait.
    ///
    /// Uses progressive backoff: spin briefly, then wait on `version.stable()`.
    /// The splitter must have set [`SPLITTING_BIT`] before marking the pointer,
    /// so `stable()` will wait for the split critical section to complete.
    pub fn wait_for_split(&self) {
        const MAX_SPIN_ITERS: u32 = 64;
        const MAX_STABLE_RETRIES: u32 = 100;

        let mut stable_retries: u32 = 0;

        if !self.is_next_marked() {
            return; // Fast path: not marked
        }

        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            next_ptr = ?self.next_raw(),
            "wait_for_split: ENTER - next pointer is marked"
        );

        while self.is_next_marked() {
            // Brief spin with backoff
            for _ in 0..MAX_SPIN_ITERS {
                std::hint::spin_loop();
                if !self.is_next_marked() {
                    #[cfg(feature = "tracing")]
                    tracing::trace!(
                        leaf_ptr = ?std::ptr::from_ref(self),
                        stable_retries,
                        "wait_for_split: EXIT - mark cleared during spin"
                    );
                    return;
                }
            }

            // Wait for split to complete via version.stable()
            #[cfg(feature = "tracing")]
            tracing::trace!(
                leaf_ptr = ?std::ptr::from_ref(self),
                stable_retries,
                version = self.version().value(),
                "wait_for_split: calling stable() - waiting for split"
            );
            let _ = self.version().stable();
            stable_retries += 1;

            // Bounded wait: if we've waited too long, the mark should have been
            // cleared. If not, there may be a bug, but we avoid infinite hang.
            if stable_retries >= MAX_STABLE_RETRIES {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    stable_retries,
                    leaf_ptr = ?std::ptr::from_ref(self),
                    next_ptr = ?self.next_raw(),
                    version = self.version().value(),
                    "wait_for_split: TIMEOUT - exceeded max retries, proceeding anyway"
                );
                return;
            }
        }

        #[cfg(feature = "tracing")]
        tracing::trace!(
            leaf_ptr = ?std::ptr::from_ref(self),
            stable_retries,
            "wait_for_split: EXIT - mark cleared"
        );
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

        // FIXED: Never mark a null pointer.
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

        if self.cas_next(marked, new_sibling).is_ok() {
            unsafe {
                (*old_next).set_prev(new_sibling);
            }
            true
        } else {
            // OPTIMIZE: Unmark the pointer before returning false.
            // If we leave it marked, wait_for_split() will spin forever.
            // This CAS should not fail if locking is correct, but we must
            // handle it gracefully to avoid infinite loops.
            let _ = self.cas_next(marked, old_next);
            false
        }
    }
}
