//! Shared drain-and-rebuild logic for suffix stores.
//!
//! Both [`EmbeddedSuffix`] and [`SidecarSuffix`] use the same core algorithm
//! for their slow-path suffix assignment: drain inline suffixes into a new
//! external bag, merge entries from an old external bag, and install the result.

use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::{Guard, LocalGuard};

use crate::TreePermutation;
use crate::suffix::{InlineSuffixBag, SuffixBag};

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
pub(super) unsafe fn drain_and_rebuild(
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
pub(super) unsafe fn drain_and_rebuild_prealloc(
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
pub(super) unsafe fn drain_and_rebuild_init(
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
