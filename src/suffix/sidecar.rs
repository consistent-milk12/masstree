//! Heap-allocated suffix storage sidecar

use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering};

use seize::Collector;

use super::{InlineSuffixBag, SuffixBag};
use crate::alloc_common::BoxAllocator;
use crate::error::{AllocKind, AllocResult};

/// Utility functions for suffix sidecar operations.
#[derive(Debug)]
pub struct SideCarUtils;

impl SideCarUtils {
    /// Cleanup function for retiring external suffix bags.
    ///
    /// Used with `BatchedRetire::defer_custom` for safe deferred reclamation.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid [`SuffixBag<WIDTH>`] pointer that was allocated via [`Box`].
    pub unsafe fn retire_suffix_bag<const WIDTH: usize>(
        ptr: *mut SuffixBag<WIDTH>,
        _collector: &Collector,
    ) {
        // SAFETY: Caller guarantees ptr is a valid Box-allocated `SuffixBag`
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Heap-allocated suffix storage for leaves with long keys.
///
/// Created lazily on first suffix assignment. Contains both inline
/// storage (fast path) and external overflow (large suffixes).
///
/// # Memory Layout
///
/// ```text
/// SuffixSidecar<15> = 328 bytes
/// ├── inline: InlineSuffixBag<15, 256>  // 320 bytes
/// └── external: AtomicPtr<SuffixBag>    // 8 bytes
/// ```
#[derive(Debug)]
#[repr(C)]
pub struct SuffixSidecar<const WIDTH: usize> {
    /// Inline suffix storage (256 bytes capacity)
    pub(crate) inline: InlineSuffixBag<WIDTH, 256>,

    /// External overflow for large suffixes.
    pub(crate) external: AtomicPtr<SuffixBag<WIDTH>>,
}

impl<const WIDTH: usize> SuffixSidecar<WIDTH> {
    /// Create a new empty sidecar.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inline: InlineSuffixBag::new(),
            external: AtomicPtr::new(StdPtr::null_mut()),
        }
    }

    /// Get suffix for slot from inline storage only.
    #[inline]
    pub fn get_inline(&self, slot: usize) -> Option<&[u8]> {
        self.inline.get(slot)
    }

    /// Get suffix for slot, checking both inline and external.
    ///
    /// Tries inline first (common case), then falls back to external.
    #[inline(always)]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        // Try inline first (common case)
        if let Some(suffix) = self.inline.get(slot) {
            return Some(suffix);
        }

        // Check external
        let external: *mut SuffixBag<WIDTH> = self.external.load(Ordering::Acquire);

        if external.is_null() {
            None
        } else {
            // SAFETY: external is valid if non-null (we own it)
            unsafe { &*external }.get(slot)
        }
    }

    /// Get or create external storage (fallible).
    ///
    /// Returns a raw pointer to avoid aliasing issues, the caller is
    /// responsible for ensuring exclusive access when dereferencing.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock. The returned pointer is valid for the
    /// lifetime of the sidecar (unitl [`Drop`] or the next swap).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](crate::alloc_common) if external bag allocation fails.
    #[inline(always)]
    pub unsafe fn ensure_external(&self) -> AllocResult<*mut SuffixBag<WIDTH>> {
        let ptr: *mut SuffixBag<WIDTH> = self.external.load(Ordering::Acquire);

        if !ptr.is_null() {
            return Ok(ptr);
        }

        // Allocate new external bag with tracking
        let new_external: Box<SuffixBag<WIDTH>> =
            BoxAllocator::try_box_with_kind(SuffixBag::new(), AllocKind::Suffix)?;
        let new_ptr: *mut SuffixBag<WIDTH> = Box::into_raw(new_external);
        self.external.store(new_ptr, Ordering::Release);

        Ok(new_ptr)
    }

    /// Check if slot has a suffix (inline or external).
    #[inline(always)]
    pub fn has_suffix(&self, slot: usize) -> bool {
        if self.inline.has_suffix(slot) {
            return true;
        }

        let external: *mut SuffixBag<WIDTH> = self.external.load(Ordering::Acquire);

        if external.is_null() {
            false
        } else {
            // SAFETY: external is valid if non-null
            unsafe { &*external }.has_suffix(slot)
        }
    }
}

impl<const WIDTH: usize> Default for SuffixSidecar<WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize> Drop for SuffixSidecar<WIDTH> {
    fn drop(&mut self) {
        // SAFETY: When external bags are swapped (via drain_to_external), the old
        // bag is retired via BatchedRetire::defer_custom, and a new bag is stored
        // in self.external. This Drop only ever sees the current (most recent)
        // external bag, never the old one that was retired.
        //
        // The sequence during swap is:
        // 1. Create new external bag with all suffixes
        // 2. atomic_swap() stores new ptr, returns old ptr
        // 3. Old ptr is retired via BatchedRetire (deferred drop)
        // 4. self.external now holds new ptr
        //
        // So when Drop runs, self.external is either:
        // - null (no external ever created), or
        // - the most recent bag (all previous ones were retired separately)
        let external: *mut SuffixBag<WIDTH> = self.external.load(Ordering::Acquire);

        if !external.is_null() {
            // SAFETY: We own this external bag exclusively during drop.
            // Any previously swaped-out bags were retired via BatchedRetire.
            unsafe {
                drop(Box::from_raw(external));
            }
        }
    }
}

// SAFETY: `SuffixSIdecar` is `Send` if `SuffixBag` is `Send`.
// The `AtomicPtr` provides thread-safe access to the external bag.
// Concurrent access is serialized by the leaf lock.
unsafe impl<const WIDTH: usize> Send for SuffixSidecar<WIDTH> {}

// SAFETY: `SuffixSidecar` is `Sync` if `SuffixBag` is `Sync`.
// Read access is safe without synchronization (immutable after publication).
// Write access requires the leaf lock.
unsafe impl<const WIDTH: usize> Sync for SuffixSidecar<WIDTH> {}
