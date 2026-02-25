//! Lazy-allocated sidecar suffix storage.

use std::cmp::Ordering;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::{Collector, Guard, LocalGuard};

use crate::TreePermutation;
use crate::ordering::RELAXED;
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
// - OCC protocol for reads (immutable after publication)
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
        // Suffix bytes are immutable after publication.
        unsafe { &*ptr }.get(slot)
    }

    #[inline(always)]
    fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot) == Some(suffix)
    }

    #[inline(always)]
    fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        self.get(slot).map(|stored| stored.cmp(suffix))
    }

    #[inline(always)]
    fn has_external(&self) -> bool {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return false;
        }

        // SAFETY: ptr is non-null and valid.
        let sidecar: &SuffixSidecar = unsafe { &*ptr };
        !sidecar.external.load(AtomicOrdering::Acquire).is_null()
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
        if suffix.is_empty() {
            return StdPtr::null_mut();
        }

        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };

        // SAFETY: sidecar_ptr is valid (just ensured).
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };

        if sidecar.inline.try_assign(slot, suffix) {
            return StdPtr::null_mut(); // No retirement needed.
        }

        let ext_ptr: *mut SuffixBag = sidecar.external.load(RELAXED);

        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
            if bag.try_assign_in_place(slot, suffix) {
                sidecar.inline.clear(slot);
                return StdPtr::null_mut();
            }
        }

        // SAFETY: Caller holds leaf lock.
        unsafe { self.drain_and_rebuild(sidecar, slot, suffix, perm) }
    }

    #[inline(always)]
    unsafe fn assign_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        if suffix.is_empty() {
            return;
        }

        // SAFETY: Caller holds leaf lock. During init, slots are filled
        // sequentially, so we don't need the permutation.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };

        if sidecar.inline.try_assign(slot, suffix) {
            return;
        }

        let ext_ptr: *mut SuffixBag = sidecar.external.load(RELAXED);

        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
            if bag.try_assign_in_place(slot, suffix) {
                return;
            }
        }

        // SAFETY: Caller holds leaf lock, guard is valid.
        unsafe { self.assign_init_slow(sidecar, slot, suffix, guard) }
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

        if sidecar.inline.try_assign(slot, suffix) {
            return StdPtr::null_mut();
        }

        let ext_ptr: *mut SuffixBag = sidecar.external.load(RELAXED);
        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };

            if bag.try_assign_in_place(slot, suffix) {
                sidecar.inline.clear(slot);
                return StdPtr::null_mut();
            }
        }

        // SAFETY: Caller holds leaf lock.
        unsafe { self.drain_and_rebuild_prealloc(sidecar, slot, suffix, perm, prealloc) }
    }

    #[inline(always)]
    unsafe fn ensure_external(&self) -> *mut SuffixBag {
        // SAFETY: Caller holds leaf lock.
        let sidecar_ptr: *mut SuffixSidecar = unsafe { self.ensure_sidecar() };
        let sidecar: &SuffixSidecar = unsafe { &*sidecar_ptr };
        unsafe { sidecar.ensure_external() }
    }

    #[inline(always)]
    unsafe fn clear(&self, slot: usize, _guard: &LocalGuard<'_>) {
        let ptr: *mut SuffixSidecar = self.sidecar_ptr();

        if ptr.is_null() {
            return; // No sidecar, nothing to clear.
        }

        // SAFETY: ptr is valid, caller holds lock.
        let sidecar: &SuffixSidecar = unsafe { &*ptr };

        // Clear inline slot metadata.
        sidecar.inline.clear(slot);

        // Clear external slot if present.
        let external: *mut SuffixBag = sidecar.external.load(AtomicOrdering::Acquire);

        if !external.is_null() {
            // SAFETY: external is valid, caller holds lock.
            let external_ref: &mut SuffixBag = unsafe { &mut *external };
            external_ref.clear(slot);
        }
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

// ============================================================================
//  Private Helpers
// ============================================================================

impl SidecarSuffix {
    /// Drain inline suffixes to a new external bag, assign the new suffix,
    /// merge old external entries, and install the new bag.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock.
    #[cold]
    #[inline(never)]
    #[expect(clippy::unused_self, reason = "Required by SuffixStore trait")]
    unsafe fn drain_and_rebuild(
        &self,
        sidecar: &SuffixSidecar,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock. Sidecar fields are valid.
        unsafe {
            super::drain_rebuild::drain_and_rebuild(
                &sidecar.inline,
                &sidecar.external,
                slot,
                suffix,
                perm,
            )
        }
    }

    /// Same as [`drain_and_rebuild`](Self::drain_and_rebuild) but uses a
    /// pre-allocated `Vec<u8>` buffer to reduce allocation inside the
    /// critical section.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock.
    #[cold]
    #[inline(never)]
    #[expect(clippy::unused_self, reason = "Required by SuffixStore trait")]
    unsafe fn drain_and_rebuild_prealloc(
        &self,
        sidecar: &SuffixSidecar,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
        prealloc: Vec<u8>,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock.
        unsafe {
            super::drain_rebuild::drain_and_rebuild_prealloc(
                &sidecar.inline,
                &sidecar.external,
                slot,
                suffix,
                perm,
                prealloc,
            )
        }
    }

    /// Slow path for suffix assignment during node initialization.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock. Guard must come from this tree's collector.
    #[cold]
    #[inline(never)]
    #[expect(clippy::unused_self, reason = "Required by SuffixStore trait")]
    unsafe fn assign_init_slow(
        &self,
        sidecar: &SuffixSidecar,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: Caller holds leaf lock, guard is valid.
        unsafe {
            super::drain_rebuild::drain_and_rebuild_init(
                &sidecar.inline,
                &sidecar.external,
                slot,
                suffix,
                guard,
            );
        }
    }
}
