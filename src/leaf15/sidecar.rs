use std::sync::atomic::Ordering as AtomicOrdering;

use crate::{
    AllocKind, AllocResult, LeafNode15, ValueSlot, WIDTH_15, alloc_common::BoxAllocator,
    suffix::SuffixSidecar,
};

impl<S> LeafNode15<S>
where
    S: ValueSlot,
{
    /// Get or create suffix sidecar (fallible).
    ///
    /// Uses [`AllocKind::Suffix`] for consistent allocation tracking.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock for write operations.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](crate::AllocError) if sidecar allocation fails.
    #[inline]
    pub unsafe fn ensure_sidecar(&self) -> AllocResult<*mut SuffixSidecar<WIDTH_15>> {
        let ptr: *mut SuffixSidecar<15> = self.suffix_sidecar.load(AtomicOrdering::Acquire);

        if !ptr.is_null() {
            return Ok(ptr);
        }

        // Caller holds lock, no race possible
        // Use AllocKind::Suffix for consistent allocation tracking
        let new_sidecar: Box<SuffixSidecar<15>> =
            BoxAllocator::try_box_with_kind(SuffixSidecar::new(), AllocKind::Suffix)?;

        let ptr: *mut SuffixSidecar<15> = Box::into_raw(new_sidecar);
        self.suffix_sidecar.store(ptr, AtomicOrdering::Release);

        Ok(ptr)
    }

    /// Get or create suffix sidecar (infallible, panics on OOM).
    ///
    /// Used by [`Self::assign_ksuf`] which maintains infallible semantics
    /// matching existing API.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock for write operations.
    ///
    /// # Panics
    ///
    /// Panics if sidecar allocation fails (OOM).
    #[inline]
    #[expect(clippy::expect_used, reason = "Intentional panic")]
    pub unsafe fn ensure_sidecar_infallible(&self) -> *mut SuffixSidecar<WIDTH_15> {
        unsafe {
            self.ensure_sidecar()
                .expect("sidecar allocation failed (OOM)")
        }
    }

    /// Check if leaf has any suffixes allocated.
    #[inline(always)]
    pub fn has_suffix_storage(&self) -> bool {
        !self.suffix_sidecar.load(AtomicOrdering::Acquire).is_null()
    }
}
