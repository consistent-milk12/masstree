//! Lazy-allocated sidecar suffix storage.

use std::cmp::Ordering;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::{Collector, Guard, LocalGuard};

use crate::TreePermutation;
use crate::suffix::{SideCarUtils, SuffixBag, SuffixSidecar};

use super::SuffixStore;

// ============================================================================
//  SidecarSuffix
// ============================================================================

/// Suffix storage via lazy-allocated heap sidecar.
#[repr(C)]
pub struct SidecarSuffix {
    sidecar: AtomicPtr<SuffixSidecar>,
}

// SAFETY: SidecarSuffix is Send+Sync.
// The AtomicPtr provides thread-safe access to the sidecar pointer.
// Sidecar contents are protected by:
// - OCC protocol for reads (version-validated)
// - Leaf lock for writes
unsafe impl Send for SidecarSuffix {}
unsafe impl Sync for SidecarSuffix {}

impl SidecarSuffix {
    /// Get the sidecar pointer (may be null if never allocated).
    #[inline(always)]
    fn sidecar_ptr(&self) -> *mut SuffixSidecar {
        self.sidecar.load(AtomicOrdering::Acquire)
    }

    /// Get or create the sidecar.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock for write operations.
    #[inline]
    unsafe fn ensure_sidecar(&self) -> *mut SuffixSidecar {
        let ptr: *mut SuffixSidecar = self.sidecar.load(AtomicOrdering::Acquire);

        if !ptr.is_null() {
            return ptr;
        }

        let new_sidecar: Box<SuffixSidecar> = Box::default();
        let ptr: *mut SuffixSidecar = Box::into_raw(new_sidecar);
        self.sidecar.store(ptr, AtomicOrdering::Release);

        ptr
    }
}

impl SuffixStore for SidecarSuffix {
    #[inline(always)]
    fn new() -> Self {
        Self {
            sidecar: AtomicPtr::new(StdPtr::null_mut()),
        }
    }

    // ========================================================================
    //  Read Operations
    // ========================================================================

    #[inline(always)]
    fn get(&self, slot: usize) -> Option<&[u8]> {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return None;
        }

        // SAFETY: ptr is non-null and was allocated by us. The sidecar
        // is never deallocated while the leaf is alive (only in Drop).
        let sidecar: &SuffixSidecar = unsafe { &*ptr };
        super::suffix_ops::get_suffix(&sidecar.inline, &sidecar.external, slot)
    }

    #[inline(always)]
    fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return false;
        }

        let sidecar: &SuffixSidecar = unsafe { &*ptr };
        super::suffix_ops::suffix_equals(&sidecar.inline, &sidecar.external, slot, suffix)
    }

    #[inline(always)]
    fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return None;
        }

        let sidecar: &SuffixSidecar = unsafe { &*ptr };
        super::suffix_ops::suffix_compare(&sidecar.inline, &sidecar.external, slot, suffix)
    }

    #[inline(always)]
    fn has_external(&self) -> bool {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return false;
        }

        let sidecar: &SuffixSidecar = unsafe { &*ptr };
        super::suffix_ops::has_external(&sidecar.external)
    }

    // ========================================================================
    //  Write Operations
    // ========================================================================

    #[inline(always)]
    unsafe fn assign(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
        _guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };

        // SAFETY: Caller holds leaf lock.
        unsafe {
            super::suffix_ops::assign_suffix(&sidecar.inline, &sidecar.external, slot, suffix, perm)
        }
    }

    #[inline(always)]
    unsafe fn assign_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };

        // SAFETY: Caller holds leaf lock, guard is valid.
        unsafe {
            super::suffix_ops::assign_suffix_init(
                &sidecar.inline,
                &sidecar.external,
                slot,
                suffix,
                guard,
            );
        }
    }

    #[inline(always)]
    unsafe fn assign_prealloc(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
        _guard: &LocalGuard<'_>,
        prealloc: Vec<u8>,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };

        // SAFETY: Caller holds leaf lock.
        unsafe {
            super::suffix_ops::assign_suffix_prealloc(
                &sidecar.inline,
                &sidecar.external,
                slot,
                suffix,
                perm,
                prealloc,
            )
        }
    }

    #[inline(always)]
    unsafe fn ensure_external(&self) -> *mut SuffixBag {
        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };
        unsafe { super::suffix_ops::ensure_external_bag(&sidecar.external) }
    }

    #[inline(always)]
    unsafe fn clear(&self, slot: usize, _guard: &LocalGuard<'_>) {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return;
        }

        let sidecar: &SuffixSidecar = unsafe { &*ptr };

        // SAFETY: Caller holds leaf lock.
        unsafe { super::suffix_ops::clear_suffix(&sidecar.inline, &sidecar.external, slot) }
    }

    #[inline(always)]
    unsafe fn retire_bag_ptr(ptr: *mut u8, guard: &LocalGuard<'_>) {
        if ptr.is_null() {
            return;
        }

        // SAFETY: ptr came from assign() and is a valid SuffixBag pointer.
        unsafe {
            guard.defer_retire(
                ptr.cast::<SuffixBag>(),
                |ptr: *mut SuffixBag, collector: &Collector| {
                    SideCarUtils::retire_suffix_bag(ptr, collector);
                },
            );
        }
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    unsafe fn drop_storage(&mut self) {
        let ptr: *mut SuffixSidecar = self.sidecar.load(AtomicOrdering::Acquire);

        if !ptr.is_null() {
            // SAFETY: We have exclusive access (&mut self from Drop).
            // The sidecar's Drop impl handles cleaning up the external bag.
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    unsafe fn init_at_zero(_ptr: *mut Self) {
        // No-op: AtomicPtr<SuffixSidecar> is correctly zero (null).
    }
}
