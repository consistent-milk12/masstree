//! Shared suffix operations for [`EmbeddedSuffix`] and [`SidecarSuffix`].
//!
//! Both suffix stores use the same core algorithms for reads (inline-first
//! fallback to external) and writes (try inline, try external in-place, then
//! drain-and-rebuild). This module provides free functions that operate on
//! the common `(&InlineSuffixBag, &AtomicPtr<SuffixBagCell>)` pair.

use std::cmp::Ordering;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use arrayvec::ArrayVec;
use seize::{Guard, LocalGuard};

use crate::TreePermutation;
use crate::suffix::{InlineSuffixBag, SuffixBag, SuffixBagCell};

// ============================================================================
//  Read Operations (safe, lock-free via OCC)
// ============================================================================

/// Fetch inline metadata/data and the external bag's actual byte buffer.
/// The caller protects the leaf and retired bags with its tree guard, just as
/// for `get_suffix`. The no-prefetch feature also removes pointer chasing.
#[inline(always)]
pub(super) fn prefetch_suffix(inline: &InlineSuffixBag, external: &AtomicPtr<SuffixBagCell>) {
    if cfg!(feature = "no-prefetch") {
        return;
    }
    let inline_ptr = StdPtr::from_ref(inline).cast::<u8>();
    crate::prefetch::prefetch_read(inline_ptr);
    crate::prefetch::prefetch_read(inline_ptr.wrapping_add(64));

    // This load is followed by a dereference, so use the same protected-load
    // ordering as get_suffix; a relaxed hint-only load is not sufficient.
    let external_ptr = external.load(AtomicOrdering::SeqCst);
    if !external_ptr.is_null() {
        crate::prefetch::prefetch_read(external_ptr);
        // SAFETY: The caller's active guard keeps this cell and its immutable
        // buffer alive after replacement. The pointer load acquires publication.
        unsafe { (*external_ptr).as_ref() }.prefetch_data();
    }
}

/// Get suffix for a slot, checking inline first then external.
#[inline(always)]
pub(super) fn get_suffix<'a>(
    inline: &'a InlineSuffixBag,
    external: &'a AtomicPtr<SuffixBagCell>,
    slot: usize,
) -> Option<&'a [u8]> {
    if let Some(suffix) = inline.get(slot) {
        return Some(suffix);
    }

    // SeqCst participates in seize's protected-load total order; the cell is
    // dereferenced below and may be retired by a concurrent update.
    let ext_ptr: *mut SuffixBagCell = external.load(AtomicOrdering::SeqCst);

    if ext_ptr.is_null() {
        None
    } else {
        // SAFETY: ext_ptr is non-null and valid (owned by the suffix store).
        // Readers rely on leaf OCC validation; writers mutate under lock.
        // SuffixBagCell::as_ref() forms &SuffixBag through UnsafeCell,
        // avoiding &mut/& aliasing UB.
        unsafe { (*ext_ptr).as_ref() }.get(slot)
    }
}

/// Check if a slot's suffix equals the given bytes.
#[inline(always)]
#[allow(
    dead_code,
    reason = "SuffixStore trait plumbing, not all impls call through here"
)]
pub(super) fn suffix_equals(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
) -> bool {
    get_suffix(inline, external, slot) == Some(suffix)
}

/// Compare a slot's suffix with the given bytes.
#[inline(always)]
#[allow(
    dead_code,
    reason = "SuffixStore trait plumbing, not all impls call through here"
)]
pub(super) fn suffix_compare(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
) -> Option<Ordering> {
    get_suffix(inline, external, slot).map(|stored: &[u8]| stored.cmp(suffix))
}

/// Check if external (overflow) storage has been allocated.
#[inline(always)]
pub(super) fn has_external(external: &AtomicPtr<SuffixBagCell>) -> bool {
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
/// Returns a pointer to the old external cell (for retirement), or null.
///
/// # Safety
///
/// Caller must hold the leaf lock.
#[inline(always)]
pub(super) unsafe fn assign_suffix(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
) -> *mut u8 {
    if suffix.is_empty() {
        return StdPtr::null_mut();
    }

    if inline.try_assign(slot, suffix) {
        return StdPtr::null_mut();
    }

    let ext_ptr: *mut SuffixBagCell = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock. `as_writer` gives
        // exclusive logical write access; mutation goes through the bag's
        // atomic/interior-mutable fields. Append-only never overwrites
        // existing suffix bytes, so concurrent OCC readers see stable data.
        let bag: &SuffixBag = unsafe { (*ext_ptr).as_writer() };

        if bag.try_assign_append_only(slot, suffix) {
            inline.clear(slot);
            return StdPtr::null_mut();
        }
    }

    // SAFETY: Caller holds leaf lock.
    unsafe { drain_and_rebuild(inline, external, slot, suffix, perm) }
}

/// Assign a suffix during node initialization (sequential slots 0..slot).
///
/// Init paths operate on unpublished leaves with no concurrent readers,
/// so we can use direct `assign` (which may compact/grow) instead of the
/// append-only path. This avoids quadratic drain-and-rebuild loops when
/// filling a leaf with external suffixes.
///
/// # Safety
///
/// Caller must hold leaf lock. Guard must come from this tree's collector.
#[inline(always)]
pub(super) unsafe fn assign_suffix_init(
    inline: &InlineSuffixBag,
    external: &AtomicPtr<SuffixBagCell>,
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

    let ext_ptr: *mut SuffixBagCell = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock. No concurrent readers
        // on unpublished leaves, so direct assign (with compaction) is safe.
        let bag: &SuffixBag = unsafe { (*ext_ptr).as_writer() };

        // SAFETY: the destination is an unpublished/exclusively owned bag
        // (init path), so compaction/reallocation cannot invalidate readers.
        unsafe { bag.assign_locked(slot, suffix) };
        inline.clear(slot);
        return;
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
    external: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
    prealloc: Vec<u8>,
) -> *mut u8 {
    if inline.try_assign(slot, suffix) {
        return StdPtr::null_mut();
    }

    let ext_ptr: *mut SuffixBagCell = external.load(AtomicOrdering::Relaxed);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        // Append-only: safe for concurrent OCC readers.
        let bag: &SuffixBag = unsafe { (*ext_ptr).as_writer() };
        if bag.try_assign_append_only(slot, suffix) {
            inline.clear(slot);
            return StdPtr::null_mut();
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
    external: &AtomicPtr<SuffixBagCell>,
    slot: usize,
) {
    inline.clear(slot);

    // SeqCst participates in seize's protected-load total order; the cell is
    // dereferenced below and may be retired by a concurrent update.
    let ext_ptr: *mut SuffixBagCell = external.load(AtomicOrdering::SeqCst);

    if !ext_ptr.is_null() {
        // SAFETY: ext_ptr is valid, caller holds lock.
        let bag: &SuffixBag = unsafe { (*ext_ptr).as_writer() };

        // SAFETY: caller holds the leaf lock; clear only updates atomics.
        unsafe { bag.clear_locked(slot) };
    }
}

/// Reserve total capacity before filling an unpublished split destination.
///
/// # Safety
///
/// Caller exclusively owns this store and all its bytes, with no readers.
#[inline]
pub(super) unsafe fn reserve_external_bag(external: &AtomicPtr<SuffixBagCell>, capacity: usize) {
    let ptr: *mut SuffixBagCell = external.load(AtomicOrdering::Acquire);

    if !ptr.is_null() {
        // SAFETY: The unpublished destination and its buffer are exclusively
        // owned; no reader slices can be invalidated by growth.
        let bag = unsafe { (*ptr).as_writer() };
        // SAFETY: Same exclusive ownership as above.
        unsafe { bag.reserve_locked(capacity.saturating_sub(bag.used())) };
        return;
    }

    let new_cell = Box::new(SuffixBagCell::from_bag(SuffixBag::with_capacity(capacity)));
    let new_ptr: *mut SuffixBagCell = Box::into_raw(new_cell);
    external.store(new_ptr, AtomicOrdering::Release);
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
/// - `external_slot` must be the `AtomicPtr<SuffixBagCell>` that stores the
///   external cell pointer for this suffix store.
#[cold]
#[inline(never)]
unsafe fn drain_and_rebuild(
    inline: &InlineSuffixBag,
    external_slot: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
) -> *mut u8 {
    // SAFETY: The leaf lock keeps both stores stable while we copy active bytes.
    unsafe { rebuild(inline, external_slot, slot, suffix, perm, None) }
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
    external_slot: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
    prealloc: Vec<u8>,
) -> *mut u8 {
    // SAFETY: The leaf lock keeps both stores stable while we copy active bytes.
    unsafe { rebuild(inline, external_slot, slot, suffix, perm, Some(prealloc)) }
}

/// Gather both stores before allocating, so merging old external suffixes
/// cannot consume the append slack reserved for subsequent writes.
///
/// # Safety
///
/// Caller holds the leaf lock, which keeps the external cell alive and prevents
/// concurrent writers. Existing readers only observe the old immutable bytes.
unsafe fn build_replacement(
    inline: &InlineSuffixBag,
    old_external: *mut SuffixBagCell,
    active_slots: impl Iterator<Item = usize>,
    slot: usize,
    suffix: &[u8],
    buffer: Option<Vec<u8>>,
) -> SuffixBag {
    let old_bag = if old_external.is_null() {
        None
    } else {
        // SAFETY: The locked leaf still owns this cell; it is only read here.
        Some(unsafe { (*old_external).as_ref() })
    };
    let mut active: ArrayVec<(usize, &[u8]), 15> = ArrayVec::new();
    let mut required: usize = suffix.len();
    for s in active_slots.filter(|&s| s != slot) {
        if let Some(bytes) = inline.get(s).or_else(|| old_bag.and_then(|bag| bag.get(s))) {
            required += bytes.len();
            active.push((s, bytes));
        }
    }

    // Also accommodate filling the leaf, but always leave slack beyond the
    // complete live set, including suffixes that were already external.
    let full_leaf: usize = 15 * (required / (active.len() + 1));
    let capacity: usize = (required * 2).max(required + 512).max(full_leaf);
    let mut bag = match buffer {
        Some(vec) => {
            let mut bag = SuffixBag::from_vec(vec);
            bag.reserve(capacity);
            bag
        }
        None => SuffixBag::with_capacity(capacity),
    };
    for (s, bytes) in active {
        bag.assign(s, bytes);
    }
    bag.assign(slot, suffix);
    bag
}

/// Rebuild and publish under the leaf lock; the caller retires the old cell.
///
/// # Safety
///
/// Caller holds the leaf lock and `perm` identifies its active slots.
unsafe fn rebuild(
    inline: &InlineSuffixBag,
    external_slot: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    perm: &impl TreePermutation,
    buffer: Option<Vec<u8>>,
) -> *mut u8 {
    let old_external: *mut SuffixBagCell = external_slot.load(AtomicOrdering::Relaxed);
    // SAFETY: The lock keeps the old cell alive and its suffixes stable.
    let bag = unsafe {
        build_replacement(
            inline,
            old_external,
            (0..perm.size()).map(|i| perm.get(i)),
            slot,
            suffix,
            buffer,
        )
    };
    let new_ptr = Box::into_raw(Box::new(SuffixBagCell::from_bag(bag)));
    external_slot.store(new_ptr, AtomicOrdering::Release);

    // Publish before clearing the old inline copy, so readers can always find
    // the suffix. The replaced slot must not shadow its new external bytes.
    inline.clear(slot);
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
    external_slot: &AtomicPtr<SuffixBagCell>,
    slot: usize,
    suffix: &[u8],
    guard: &LocalGuard<'_>,
) {
    let old_external: *mut SuffixBagCell = external_slot.load(AtomicOrdering::Relaxed);
    // SAFETY: Initialization owns the unpublished destination exclusively.
    let new_bag = unsafe { build_replacement(inline, old_external, 0..slot, slot, suffix, None) };

    let new_cell: SuffixBagCell = SuffixBagCell::from_bag(new_bag);
    let new_ptr: *mut SuffixBagCell = Box::into_raw(Box::new(new_cell));
    external_slot.store(new_ptr, AtomicOrdering::Release);

    if !old_external.is_null() {
        // SAFETY: old_external was a valid Box<SuffixBagCell> from a prior drain.
        unsafe {
            guard.defer_retire(old_external, |ptr, _| {
                drop(Box::from_raw(ptr));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_during_external_bag_replacement() {
        use seize::Collector;
        use std::sync::Barrier;

        let collector = Collector::new();
        let inline = InlineSuffixBag::new();
        let external = AtomicPtr::new(StdPtr::null_mut());
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let guard = collector.enter();
                for byte in 1..=16 {
                    let mut bag = SuffixBag::with_capacity(128);
                    bag.assign(0, &[byte; 128]);
                    let new = Box::into_raw(Box::new(SuffixBagCell::from_bag(bag)));
                    barrier.wait();
                    let old = external.swap(new, AtomicOrdering::AcqRel);
                    if !old.is_null() {
                        // SAFETY: The swap removed the owning pointer. Reader
                        // guards keep the old cell/buffer alive until quiescence.
                        unsafe {
                            guard.defer_retire(
                                old,
                                crate::suffix::SideCarUtils::retire_suffix_bag_cell,
                            );
                        }
                    }
                    guard.flush();
                    barrier.wait();
                }
            });
            let mut valid = true;
            for byte in 1..=16 {
                let guard = collector.enter();
                barrier.wait();
                prefetch_suffix(&inline, &external);
                if let Some(bytes) = get_suffix(&inline, &external, 0) {
                    valid &= bytes == [byte; 128] || bytes == [byte - 1; 128];
                }
                barrier.wait();
                valid &= get_suffix(&inline, &external, 0) == Some([byte; 128].as_slice());
                drop(guard);
            }
            // Check after all barrier rounds so failure cannot strand a peer.
            assert!(valid);
        });
        // SAFETY: Both threads have exited their guards. Reclaim retired bags
        // and free the final still-owned cell exactly once, with no readers.
        unsafe {
            collector.reclaim_all();
            drop(Box::from_raw(external.into_inner()));
        }
    }

    #[test]
    fn rebuild_sizes_the_complete_live_set_and_preserves_precedence() {
        let mut old = SuffixBag::with_capacity(3_000);
        for slot in 0..15 {
            old.assign(slot, &[1; 200]);
        }
        let old = SuffixBagCell::from_bag(old);
        let inline = InlineSuffixBag::new();
        assert!(inline.try_assign(0, &[2; 200]));
        for buffer in [None, Some(Vec::with_capacity(128))] {
            // SAFETY: The stack-owned cell stays alive and has no concurrent
            // writers. Rebuilding only reads its immutable suffix bytes.
            let bag = unsafe {
                build_replacement(
                    &inline,
                    StdPtr::from_ref(&old).cast_mut(),
                    0..14,
                    13,
                    &[3; 248],
                    buffer,
                )
            };
            assert_eq!(bag.get(0), Some([2; 200].as_slice()));
            for slot in 1..13 {
                assert_eq!(bag.get(slot), Some([1; 200].as_slice()));
            }
            assert_eq!(bag.get(13), Some([3; 248].as_slice()));
            assert_eq!(bag.get(14), None, "inactive external slot was copied");
            assert_eq!(bag.used(), 13 * 200 + 248);
            assert!(bag.capacity() >= bag.used() * 2);
        }
    }
}
