//! Heap-allocated suffix storage sidecar.

use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering};

use seize::Collector;

use super::{InlineSuffixBag, SuffixBag};

/// Utility functions for suffix sidecar operations.
#[derive(Debug)]
pub struct SideCarUtils;

impl SideCarUtils {
    /// Cleanup function for retiring external suffix bags.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid [`SuffixBag`] pointer that was allocated via [`Box`].
    pub unsafe fn retire_suffix_bag(ptr: *mut SuffixBag, _collector: &Collector) {
        // SAFETY: Caller guarantees ptr is a valid Box-allocated `SuffixBag`
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Heap-allocated suffix storage for leaves with long keys.
///
/// Contains inline storage plus an atomic pointer to external overflow.
/// Production reads and writes go through `suffix_ops` free functions;
/// the methods below are retained for unit tests only.
#[derive(Debug)]
#[repr(C)]
pub struct SuffixSidecar {
    /// Inline suffix storage.
    pub(crate) inline: InlineSuffixBag,

    /// External overflow for large suffixes.
    pub(crate) external: AtomicPtr<SuffixBag>,
}

impl SuffixSidecar {
    /// Create a new empty sidecar.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inline: InlineSuffixBag::new(),
            external: AtomicPtr::new(StdPtr::null_mut()),
        }
    }
}

// Test-only convenience methods (production code uses suffix_ops).
#[cfg(test)]
impl SuffixSidecar {
    /// Get suffix for slot, checking both inline and external.
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        if let Some(suffix) = self.inline.get(slot) {
            return Some(suffix);
        }

        let external: *mut SuffixBag = self.external.load(Ordering::Acquire);

        if external.is_null() {
            None
        } else {
            // SAFETY: external is valid if non-null (we own it)
            unsafe { &*external }.get(slot)
        }
    }

    /// Get or create external storage.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock.
    pub unsafe fn ensure_external(&self) -> *mut SuffixBag {
        let ptr: *mut SuffixBag = self.external.load(Ordering::Acquire);

        if !ptr.is_null() {
            return ptr;
        }

        let new_external: Box<SuffixBag> = Box::default();
        let new_ptr: *mut SuffixBag = Box::into_raw(new_external);
        self.external.store(new_ptr, Ordering::Release);

        new_ptr
    }

    /// Check if slot has a suffix (inline or external).
    pub fn has_suffix(&self, slot: usize) -> bool {
        if self.inline.has_suffix(slot) {
            return true;
        }

        let external: *mut SuffixBag = self.external.load(Ordering::Acquire);

        if external.is_null() {
            false
        } else {
            // SAFETY: external is valid if non-null
            unsafe { &*external }.has_suffix(slot)
        }
    }
}

impl Default for SuffixSidecar {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SuffixSidecar {
    fn drop(&mut self) {
        // SAFETY: self.external is either null (no external ever created) or
        // the most recent bag (all previous ones were retired separately via
        // defer_retire during drain-and-rebuild).
        let external: *mut SuffixBag = self.external.load(Ordering::Acquire);

        if !external.is_null() {
            // SAFETY: We own this external bag exclusively during drop.
            unsafe {
                drop(Box::from_raw(external));
            }
        }
    }
}

// SAFETY: `SuffixSidecar` is `Send` if `SuffixBag` is `Send`.
// The `AtomicPtr` provides thread-safe access to the external bag.
// Concurrent access is serialized by the leaf lock.
unsafe impl Send for SuffixSidecar {}

// SAFETY: `SuffixSidecar` is `Sync` if `SuffixBag` is `Sync`.
// Read access is safe as part of the leaf's OCC protocol: readers validate the
// leaf version after reads, and writers only mutate under the leaf lock while
// the leaf is marked dirty (INSERTING/SPLITTING), so `stable()` readers won't
// race with writes.
unsafe impl Sync for SuffixSidecar {}
