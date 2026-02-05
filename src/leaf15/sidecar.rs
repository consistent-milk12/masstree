use std::sync::atomic::Ordering as AtomicOrdering;

use crate::{alloc_common::BoxAllocator, suffix::SuffixSidecar, LeafNode15, ValueSlot};

impl<S> LeafNode15<S>
where
    S: ValueSlot,
{
    /// Get or create suffix sidecar.
    ///
    /// Aborts on allocation failure (standard Rust OOM behavior).
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock for write operations.
    #[inline]
    pub unsafe fn ensure_sidecar(&self) -> *mut SuffixSidecar {
        let ptr: *mut SuffixSidecar = self.suffix_sidecar.load(AtomicOrdering::Acquire);

        if !ptr.is_null() {
            return ptr;
        }

        // Caller holds lock, no race possible
        let new_sidecar: Box<SuffixSidecar> = BoxAllocator::boxed(SuffixSidecar::new());

        let ptr: *mut SuffixSidecar = Box::into_raw(new_sidecar);
        self.suffix_sidecar.store(ptr, AtomicOrdering::Release);

        ptr
    }

    /// Check if leaf has any suffixes allocated.
    #[inline(always)]
    pub fn has_suffix_storage(&self) -> bool {
        !self.suffix_sidecar.load(AtomicOrdering::Acquire).is_null()
    }
}
