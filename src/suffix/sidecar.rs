//! Heap-allocated suffix storage sidecar.
//!
//! # Key Invariant: Suffix Immutability
//!
//! Once a suffix is assigned to a slot via [`LeafNode15::assign_ksuf`], the suffix
//! bytes are **NEVER modified**. The only operations are:
//!
//! 1. **Add**: Assign a new suffix to a slot that doesn't have one
//! 2. **Remove**: Clear a suffix when the entire key is removed
//!
//! This invariant enables the "orphaned inline" optimization: when inline suffixes
//! are drained to external storage, we do NOT clear the inline data. Readers may
//! observe inline data for slots that have been drained, but since suffixes are
//! immutable, this data is still valid.
//!
//! # Publication Protocol
//!
//! Writers must follow this sequence:
//! 1. Allocate sidecar (if needed)
//! 2. Write suffix bytes to inline or external storage
//! 3. Store sidecar pointer with Release ordering (if newly allocated)
//! 4. Store `keylenx = KSUF_KEYLENX` with Release ordering (publication point)
//!
//! Readers observe via:
//! 1. Load `keylenx` with Acquire ordering
//! 2. Load sidecar pointer with Acquire ordering
//! 3. Read suffix from inline or external
//!
//! The `keylenx` store is the linearization point for suffix visibility.
//!
//! [`LeafNode15::assign_ksuf`]: crate::leaf15::LeafNode15::assign_ksuf

use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering};

use seize::Collector;

use super::{InlineSuffixBag, SuffixBag};
use crate::alloc_common::BoxAllocator;
// Note: AllocError/AllocResult removed - allocations are now infallible

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
    pub fn new() -> Self {
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
    /// Aborts on allocation failure (standard Rust OOM behavior).
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock. The returned pointer is valid for the
    /// lifetime of the sidecar (until [`Drop`] or the next swap).
    #[inline(always)]
    pub unsafe fn ensure_external(&self) -> *mut SuffixBag<WIDTH> {
        let ptr: *mut SuffixBag<WIDTH> = self.external.load(Ordering::Acquire);

        if !ptr.is_null() {
            return ptr;
        }

        // Allocate new external bag (aborts on OOM)
        let new_external: Box<SuffixBag<WIDTH>> = BoxAllocator::boxed(SuffixBag::new());
        let new_ptr: *mut SuffixBag<WIDTH> = Box::into_raw(new_external);
        self.external.store(new_ptr, Ordering::Release);

        new_ptr
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
