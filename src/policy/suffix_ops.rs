//! Shared suffix operations for [`EmbeddedSuffix`] and [`SidecarSuffix`].
//!
//! Both suffix stores use the same core algorithms for reads (inline-first
//! fallback to external) and writes (try inline, try external in-place, then
//! drain-and-rebuild). This module provides free functions that operate on
//! the common `(&InlineSuffixBag, &AtomicPtr<SuffixBag>)` pair.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::{Guard, LocalGuard};

use crate::TreePermutation;
use crate::suffix::{InlineSuffixBag, SuffixBag};

// ============================================================================
//  Read Operations (safe, lock-free via OCC)
// ============================================================================

/// Get suffix for a slot, checking inline first then external.
#[inline(always)]
pub(super) fn get_suffix<'a>(
    inline: &'a InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
) -> Option<&'a [u8]> {
    if let Some(suffix) = inline.get(slot) {
        return Some(suffix);
    }

    let ext_ptr: *mut SuffixBag = external.load(AtomicOrdering::Acquire);

    if ext_ptr.is_null() {
        None
    } else {
        // SAFETY: ext_ptr is non-null and valid (owned by the suffix store).
        // Readers rely on leaf OCC validation; writers mutate under lock.
        unsafe { &*ext_ptr }.get(slot)
    }
}

/// Check if a slot's suffix equals the given bytes.
#[inline(always)]
#[allow(dead_code, reason = "SuffixStore trait plumbing, not all impls call through here")]
pub(super) fn suffix_equals(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
) -> bool {
    get_suffix(inline, external, slot) == Some(suffix)
}

/// Compare a slot's suffix with the given bytes.
#[inline(always)]
#[allow(dead_code, reason = "SuffixStore trait plumbing, not all impls call through here")]
pub(super) fn suffix_compare(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
) -> Option<Ordering> {
    get_suffix(inline, external, slot).map(|stored: &[u8]| stored.cmp(suffix))
}

/// Check if external (overflow) storage has been allocated.
#[inline(always)]
pub(super) fn has_external(external: &AtomicPtr<SuffixBag>) -> bool {
    !external.load(AtomicOrdering::Acquire).is_null()
}

// ============================================================================
//  Write Operations (unsafe, require leaf lock)
// ============================================================================

/// Assign a suffix to a slot, using the full fast+slow path.
///
/// Fast path: try inline, then try external in-place.
/// Slow path: drain-and-rebuild.
///
/// Returns a pointer to the old external bag (for retirement), or null.
///
/// # Safety
///
/// Caller must hold the leaf lock.
#[inline(always)]
pub(super) unsafe fn assign_suffix(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
) -> *mut u8 {
    if suffix.is_empty() {
        return std::ptr::null_mut();
    }

    if inline.try_assign(slot, suffix) {
        return std::ptr::null_mut();
    }

    let ext_ptr: *mut SuffixBag = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
        if bag.try_assign_in_place(slot, suffix) {
            inline.clear(slot);
            return std::ptr::null_mut();
        }
    }

    // SAFETY: Caller holds leaf lock.
    unsafe { drain_and_rebuild(inline, external, slot, suffix, perm) }
}

/// Assign a suffix during node initialization (sequential slots 0..slot).
///
/// # Safety
///
/// Caller must hold leaf lock. Guard must come from this tree's collector.
#[inline(always)]
pub(super) unsafe fn assign_suffix_init(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    guard: &LocalGuard<'_>,
) {
    if suffix.is_empty() {
        return;
    }

    if inline.try_assign(slot, suffix) {
        return;
    }

    let ext_ptr: *mut SuffixBag = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
        if bag.try_assign_in_place(slot, suffix) {
            inline.clear(slot);
            return;
        }
    }

    // SAFETY: Caller holds leaf lock, guard is valid.
    unsafe { drain_and_rebuild_init(inline, external, slot, suffix, guard) }
}

/// Assign a suffix using a pre-allocated buffer.
///
/// # Safety
///
/// Caller must hold the leaf lock.
#[inline(always)]
pub(super) unsafe fn assign_suffix_prealloc(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
    prealloc: Vec<u8>,
) -> *mut u8 {
    if inline.try_assign(slot, suffix) {
        return std::ptr::null_mut();
    }

    let ext_ptr: *mut SuffixBag = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
        if bag.try_assign_in_place(slot, suffix) {
            inline.clear(slot);
            return std::ptr::null_mut();
        }
    }

    // SAFETY: Caller holds leaf lock.
    unsafe { drain_and_rebuild_prealloc(inline, external, slot, suffix, perm, prealloc) }
}

/// Clear a slot's suffix in both inline and external stores.
///
/// # Safety
///
/// Caller must hold the leaf lock.
#[inline(always)]
pub(super) unsafe fn clear_suffix(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBag>,
    slot: usize,
) {
    inline.clear(slot);

    let ext_ptr: *mut SuffixBag = external.load(AtomicOrdering::Acquire);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        let bag: &mut SuffixBag = unsafe { &mut *ext_ptr };
        bag.clear(slot);
    }
}

/// Get or create external overflow storage.
///
/// # Safety
///
/// Caller must hold the leaf lock.
#[inline(always)]
pub(super) unsafe fn ensure_external_bag(external: &AtomicPtr<SuffixBag>) -> *mut SuffixBag {
    let ptr: *mut SuffixBag = external.load(AtomicOrdering::Acquire);

    if !ptr.is_null() {
        return ptr;
    }

    let new_external: Box<SuffixBag> = Box::default();
    let new_ptr: *mut SuffixBag = Box::into_raw(new_external);
    external.store(new_ptr, AtomicOrdering::Release);

    new_ptr
}

// ============================================================================
//  Drain-and-Rebuild (cold paths)
// ============================================================================

/// Drain inline suffixes + new suffix into a new external bag, merge old
/// external entries, and install the new bag.
///
/// # Safety
///
/// - Caller must hold the leaf lock.
/// - `inline` must point to a valid `InlineSuffixBag`.
/// - `external_slot` must be the `AtomicPtr<SuffixBag>` that stores the
///   external bag pointer for this suffix store.
#[cold]
#[inline(never)]
unsafe fn drain_and_rebuild(
    inline: &InlineSuffixBag,
    external_slot: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
) -> *mut u8 {
    let mut new_bag: SuffixBag = inline.drain_to_external(perm, slot, suffix);

    let old_external: *mut SuffixBag = external_slot.load(AtomicOrdering::Relaxed);
    merge_old_external_perm(&mut new_bag, old_external, slot, perm);

    let new_ptr: *mut SuffixBag = Box::into_raw(Box::new(new_bag));
    external_slot.store(new_ptr, AtomicOrdering::Release);

    old_external.cast::<u8>()
}

/// Same as [`drain_and_rebuild`] but uses a pre-allocated `Vec<u8>` buffer
/// to reduce allocation inside the critical section.
///
/// # Safety
///
/// Same as [`drain_and_rebuild`].
#[cold]
#[inline(never)]
unsafe fn drain_and_rebuild_prealloc(
    inline: &InlineSuffixBag,
    external_slot: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
    prealloc: Vec<u8>,
) -> *mut u8 {
    let mut new_bag: SuffixBag = inline.drain_to_external_with_vec(perm, slot, suffix, prealloc);

    let old_external: *mut SuffixBag = external_slot.load(AtomicOrdering::Relaxed);
    merge_old_external_perm(&mut new_bag, old_external, slot, perm);

    let new_ptr: *mut SuffixBag = Box::into_raw(Box::new(new_bag));
    external_slot.store(new_ptr, AtomicOrdering::Release);

    old_external.cast::<u8>()
}

/// Slow path for suffix assignment during node initialization.
///
/// # Safety
///
/// Caller must hold leaf lock. Guard must come from this tree's collector.
#[cold]
#[inline(never)]
unsafe fn drain_and_rebuild_init(
    inline: &InlineSuffixBag,
    external_slot: &AtomicPtr<SuffixBag>,
    slot: usize,
    suffix: &[u8],
    guard: &LocalGuard<'_>,
) {
    let mut new_bag: SuffixBag = inline.drain_to_external_init(slot, suffix);

    let old_external: *mut SuffixBag = external_slot.load(AtomicOrdering::Relaxed);
    merge_old_external_init(&mut new_bag, old_external, slot);

    let new_ptr: *mut SuffixBag = Box::into_raw(Box::new(new_bag));
    external_slot.store(new_ptr, AtomicOrdering::Release);

    if !old_external.is_null() {
        // SAFETY: old_external was a valid Box<SuffixBag> from a prior drain.
        unsafe {
            guard.defer_retire(old_external, |ptr, _| {
                drop(Box::from_raw(ptr));
            });
        }
    }
}

// ============================================================================
//  Merge Helpers
// ============================================================================

/// Merge active suffixes from old external bag into new bag (permutation-based).
fn merge_old_external_perm(
    new_bag: &mut SuffixBag,
    old_external: *mut SuffixBag,
    skip_slot: usize,
    perm: &impl TreePermutation,
) {
    if old_external.is_null() {
        return;
    }

    // SAFETY: old_external is valid, caller holds the lock.
    let old_ref: &SuffixBag = unsafe { &*old_external };

    for i in 0..perm.size() {
        let phys: usize = perm.get(i);

        if phys != skip_slot
            && !new_bag.has_suffix(phys)
            && let Some(s) = old_ref.get(phys)
        {
            new_bag.assign(phys, s);
        }
    }
}

/// Merge active suffixes from old external bag into new bag (sequential init).
fn merge_old_external_init(new_bag: &mut SuffixBag, old_external: *mut SuffixBag, slot: usize) {
    if old_external.is_null() {
        return;
    }

    // SAFETY: old_external is valid, caller holds the lock.
    let old_ref: &SuffixBag = unsafe { &*old_external };

    for s in 0..slot {
        if !new_bag.has_suffix(s)
            && let Some(ext_suffix) = old_ref.get(s)
        {
            new_bag.assign(s, ext_suffix);
        }
    }
}
