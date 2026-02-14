//! Embedded suffix storage (always-present, no sidecar indirection).
//!
//! Contains `InlineSuffixBag` embedded directly in the leaf struct plus
//! `AtomicPtr<SuffixBag>` for heap overflow. No lazy allocation, no extra
//! pointer chase for suffix reads.
//!
//! Optimal for string-key workloads with frequent suffixes.
//!
//! This is the suffix storage backend for [`InlinePolicy<V>`].
//!
//! # Size
//!
//! - Default (`CAPACITY = 512`): 584 bytes (inline bag: 576 + external ptr: 8)
//! - `small-suffix-capacity` (`CAPACITY = 256`): 328 bytes (inline bag: 320 + external ptr: 8)
//!
//! [`InlinePolicy<V>`]: super::InlinePolicy

use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::{Guard, LocalGuard};

use crate::suffix::{InlineSuffixBag, SideCarUtils, SuffixBag};
use crate::TreePermutation;

use super::SuffixStore;

// ============================================================================
//  EmbeddedSuffix
// ============================================================================

/// Suffix storage embedded directly in the leaf struct.
///
/// # Layout
///
/// ```text
/// EmbeddedSuffix {
///     inline_ksuf:   UnsafeCell<InlineSuffixBag>  // 320 or 576 bytes
///     external_ksuf: AtomicPtr<SuffixBag>          // 8 bytes
/// }
/// ```
///
/// Total: 584 bytes (default) or 328 bytes (`small-suffix-capacity`).
///
/// # Advantages over `SidecarSuffix`
///
/// - No pointer dereference for inline suffix access (embedded in leaf)
/// - No lazy allocation overhead (always present)
/// - Better cache locality for string-key workloads
///
/// # Tradeoff
///
/// Leaf size increases by 320/576 bytes compared to `SidecarSuffix` (8 bytes).
/// This is acceptable for inline-value leaves where the value array already
/// adds 120 bytes (`[AtomicU64; 15]`), making the leaf 896/1152 bytes total.
#[repr(C)]
pub struct EmbeddedSuffix {
    /// Inline suffix storage (256 or 512 bytes data capacity).
    ///
    /// Uses `UnsafeCell` for interior mutability. Writers hold the leaf lock.
    /// Readers use OCC validation. Suffix bytes are immutable after publication.
    inline_ksuf: UnsafeCell<InlineSuffixBag>,

    /// External overflow for large/many suffixes.
    ///
    /// Null until inline capacity is exhausted.
    external_ksuf: AtomicPtr<SuffixBag>,
}

// SAFETY: EmbeddedSuffix is Send+Sync.
// - UnsafeCell<InlineSuffixBag>: protected by leaf lock (writes) and OCC (reads).
//   Suffix bytes are immutable after publication, so concurrent reads are safe.
// - AtomicPtr<SuffixBag>: thread-safe atomic access.
unsafe impl Send for EmbeddedSuffix {}
unsafe impl Sync for EmbeddedSuffix {}

impl EmbeddedSuffix {
    /// Get a reference to the inline suffix bag.
    ///
    /// # Safety
    ///
    /// For read operations: safe under OCC (suffix bytes are immutable).
    /// For write operations: caller must hold the leaf lock.
    #[inline(always)]
    fn inline_bag(&self) -> &InlineSuffixBag {
        // SAFETY: InlineSuffixBag uses internal atomics for metadata.
        // Suffix bytes are immutable after publication. Read access
        // is safe under OCC validation.
        unsafe { &*self.inline_ksuf.get() }
    }

    /// Get the external bag pointer (may be null).
    #[inline(always)]
    fn external_ptr(&self) -> *mut SuffixBag {
        self.external_ksuf.load(AtomicOrdering::Acquire)
    }
}

impl SuffixStore for EmbeddedSuffix {
    #[inline(always)]
    fn new() -> Self {
        Self {
            inline_ksuf: UnsafeCell::new(InlineSuffixBag::new()),
            external_ksuf: AtomicPtr::new(StdPtr::null_mut()),
        }
    }

    // ========================================================================
    //  Read Operations
    // ========================================================================

    #[inline(always)]
    fn get(&self, slot: usize) -> Option<&[u8]> {
        // Try inline first (common case, no pointer dereference).
        if let Some(suffix) = self.inline_bag().get(slot) {
            return Some(suffix);
        }

        // Check external overflow.
        let external: *mut SuffixBag = self.external_ptr();

        if external.is_null() {
            None
        } else {
            // SAFETY: external is non-null and valid (we own it).
            // Suffix bytes are immutable after publication.
            unsafe { &*external }.get(slot)
        }
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
        !self.external_ptr().is_null()
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
        // Empty suffix is accepted and results in a no-op, consistent with
        // the policy-wide suffix contract.
        if suffix.is_empty() {
            return StdPtr::null_mut();
        }

        let inline: &InlineSuffixBag = self.inline_bag();

        // Fast path: try inline assignment.
        if inline.try_assign(slot, suffix) {
            return StdPtr::null_mut(); // No retirement needed.
        }

        // Middle path: try in-place assignment on existing external.
        // This avoids drain-and-rebuild when the external bag can accommodate
        // the suffix by reusing or compacting existing space (no Vec growth).
        // Matches existing leaf15_true.rs 3-tier pattern.
        let old_ext: *mut SuffixBag = self.external_ptr();
        if !old_ext.is_null() {
            // SAFETY: old_ext is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                // Suffix is now in external; clear the orphaned inline entry
                // so that get() returns the new external version.
                inline.clear(slot);
                return StdPtr::null_mut();
            }
        }

        // Slow path: drain inline to new external, merge old external.
        // SAFETY: Caller holds leaf lock.
        unsafe { self.drain_and_rebuild(slot, suffix, perm) }
    }

    #[inline(always)]
    unsafe fn assign_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        if suffix.is_empty() {
            return;
        }

        let inline: &InlineSuffixBag = self.inline_bag();

        // Fast path: inline assignment.
        if inline.try_assign(slot, suffix) {
            return;
        }

        // Middle path: try in-place on existing external.
        let old_ext: *mut SuffixBag = self.external_ptr();
        if !old_ext.is_null() {
            // SAFETY: old_ext is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                return;
            }
        }

        // Slow path: drain and rebuild.
        // SAFETY: Caller holds leaf lock, guard is valid.
        unsafe { self.assign_init_slow(slot, suffix, guard) }
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
        let inline: &InlineSuffixBag = self.inline_bag();

        // Fast path: try inline.
        if inline.try_assign(slot, suffix) {
            return StdPtr::null_mut();
        }

        // Middle path: try in-place on existing external.
        let old_ext: *mut SuffixBag = self.external_ptr();
        if !old_ext.is_null() {
            // SAFETY: old_ext is valid, caller holds lock.
            let bag: &mut SuffixBag = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                return StdPtr::null_mut();
            }
        }

        // Slow path with pre-allocated buffer.
        // SAFETY: Caller holds leaf lock.
        unsafe { self.drain_and_rebuild_prealloc(slot, suffix, perm, prealloc) }
    }

    #[inline(always)]
    unsafe fn ensure_external(&self) -> *mut SuffixBag {
        unsafe { self.ensure_external_inner() }
    }

    #[inline(always)]
    unsafe fn clear(&self, slot: usize, _guard: &LocalGuard<'_>) {
        // Clear inline slot metadata.
        self.inline_bag().clear(slot);

        // Clear external slot if present.
        let external: *mut SuffixBag = self.external_ptr();

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
            guard.defer_retire(ptr.cast::<SuffixBag>(), |ptr, collector| {
                SideCarUtils::retire_suffix_bag(ptr, collector);
            });
        }
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    unsafe fn drop_storage(&mut self) {
        // External bag: drop if allocated.
        let external: *mut SuffixBag = self.external_ksuf.load(AtomicOrdering::Acquire);

        if !external.is_null() {
            // SAFETY: We have exclusive access (&mut self from Drop).
            // Any previously swapped-out bags were retired separately.
            unsafe {
                drop(Box::from_raw(external));
            }
        }

        // Inline bag: no heap allocation to free (embedded in struct).
        // InlineSuffixBag's data is in UnsafeCell<[u8; CAPACITY]>, not heap.
    }

    unsafe fn init_at_zero(ptr: *mut Self) {
        // SAFETY: InlineSuffixBag needs header initialization.
        // The caller zeroed the memory, but the bag header may need
        // non-zero initial values.
        unsafe {
            let bag_ptr: *mut InlineSuffixBag = StdPtr::addr_of_mut!((*ptr).inline_ksuf).cast();
            StdPtr::write(bag_ptr, InlineSuffixBag::new());
        }
    }
}

// ============================================================================
//  Private Helpers
// ============================================================================

impl EmbeddedSuffix {
    /// Get or create external overflow storage.
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock.
    #[inline(always)]
    unsafe fn ensure_external_inner(&self) -> *mut SuffixBag {
        let ptr: *mut SuffixBag = self.external_ksuf.load(AtomicOrdering::Acquire);

        if !ptr.is_null() {
            return ptr;
        }

        let new_external: Box<SuffixBag> = Box::new(SuffixBag::new());
        let new_ptr: *mut SuffixBag = Box::into_raw(new_external);
        self.external_ksuf.store(new_ptr, AtomicOrdering::Release);

        new_ptr
    }

    /// Drain inline+external suffixes to a new external bag, assign the
    /// new suffix, and install it.
    ///
    /// Returns the old external bag pointer for deferred retirement
    /// (null if no previous external bag existed).
    ///
    /// # Safety
    ///
    /// Caller must hold leaf lock.
    #[cold]
    #[inline(never)]
    unsafe fn drain_and_rebuild(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock. inline_bag() and external_ksuf are valid.
        unsafe {
            super::drain_rebuild::drain_and_rebuild(
                self.inline_bag(),
                &self.external_ksuf,
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
    unsafe fn drain_and_rebuild_prealloc(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl TreePermutation,
        prealloc: Vec<u8>,
    ) -> *mut u8 {
        // SAFETY: Caller holds leaf lock.
        unsafe {
            super::drain_rebuild::drain_and_rebuild_prealloc(
                self.inline_bag(),
                &self.external_ksuf,
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
    unsafe fn assign_init_slow(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Caller holds leaf lock, guard is valid.
        unsafe {
            super::drain_rebuild::drain_and_rebuild_init(
                self.inline_bag(),
                &self.external_ksuf,
                slot,
                suffix,
                guard,
            );
        }
    }
}
