//! Thread-local node pools using size-class buckets.
//!
//! Based on C++ Masstree's `threadinfo::pool_allocate()` (`kvthread.hh:226-259`).
//! Pools are bucketed by cache-line count (1–20), not concrete type, enabling
//! reuse across node types within the same size class.
//!
//! Pool access is thread-local (no synchronization on fast path). Nodes may
//! return to a different thread's pool due to seize's batch reclamation.
//!
//! ## Divergences from C++
//!
//! - **Capacity-bounded**: each bucket caps at 256 entries, spilling to the
//!   global allocator. C++ freelists are unbounded.
//! - **Individual refill**: allocates 64 blocks individually via `alloc()`.
//!   C++ carves a single 2 MB `posix_memalign` slab (with optional hugepages).
//! - **OOM**: aborts via `handle_alloc_error`. C++ returns null.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::cell::UnsafeCell;
use std::ptr as StdPtr;

use seize::Collector;

use crate::inline::bits::InlineBits;
use crate::inline::leaf15_true::LeafNode15TrueInline;
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::slot::ValueSlot;

// ============================================================================
//  Constants
// ============================================================================

const CACHE_LINE: usize = 64;

/// C++ uses 20 → nodes up to 1280 bytes.
const MAX_SIZE_CLASSES: usize = 20;

/// Max cached nodes per size class per thread.
const POOL_CAPACITY: usize = 256;

/// Batch refill count from global allocator.
const REFILL_BATCH: usize = 64;

// ============================================================================
//  Size Class Computation
// ============================================================================

/// Size class (1-indexed cache-line count), or `None` if too large.
#[inline]
const fn size_class(layout: Layout) -> Option<usize> {
    let nl: usize = layout.size().div_ceil(CACHE_LINE);

    if nl == 0 || nl > MAX_SIZE_CLASSES {
        None
    } else {
        Some(nl)
    }
}

/// `nl * CACHE_LINE` layout with cache-line alignment.
///
/// # Safety
///
/// `nl` must be in `[1, MAX_SIZE_CLASSES]`.
#[inline]
unsafe fn bucket_layout(nl: usize) -> Layout {
    debug_assert!(nl > 0 && nl <= MAX_SIZE_CLASSES);

    // SAFETY: nl in valid range → non-zero size, power-of-2 alignment
    unsafe { Layout::from_size_align_unchecked(nl * CACHE_LINE, CACHE_LINE) }
}

// ============================================================================
//  Freelist
// ============================================================================

/// Intrusive freelist — first 8 bytes of each freed block store the next pointer.
struct Freelist {
    head: *mut u8,
    count: usize,
}

impl Freelist {
    const fn new() -> Self {
        Self {
            head: StdPtr::null_mut(),
            count: 0,
        }
    }

    #[inline(always)]
    const fn pop(&mut self) -> Option<*mut u8> {
        if self.head.is_null() {
            return None;
        }

        let ptr: *mut u8 = self.head;

        // SAFETY: ptr from our freelist, first 8 bytes are next ptr
        self.head = unsafe { StdPtr::read(ptr.cast::<*mut u8>()) };
        self.count -= 1;

        Some(ptr)
    }

    /// # Safety
    ///
    /// `ptr` must be valid, cache-line-aligned memory of at least 8 bytes.
    #[inline(always)]
    const unsafe fn push(&mut self, ptr: *mut u8) {
        unsafe { StdPtr::write(ptr.cast::<*mut u8>(), self.head) };
        self.head = ptr;
        self.count += 1;
    }

    /// Slow path: batch-allocate `REFILL_BATCH` blocks.
    #[cold]
    fn refill(&mut self, layout: Layout) {
        for _ in 0..REFILL_BATCH {
            // SAFETY: layout is a valid bucket layout
            let ptr: *mut u8 = unsafe { alloc(layout) };

            if ptr.is_null() {
                break;
            }

            // SAFETY: freshly allocated with correct layout
            unsafe { self.push(ptr) };
        }
    }

    fn drain(&mut self, layout: Layout) {
        while let Some(ptr) = self.pop() {
            // SAFETY: ptr was allocated with this layout
            unsafe { dealloc(ptr, layout) };
        }
    }

    #[inline(always)]
    const fn has_capacity(&self) -> bool {
        self.count < POOL_CAPACITY
    }
}

// ============================================================================
//  ThreadPool
// ============================================================================

/// Per-thread pool. Index `i` = size class `i + 1` (1-indexed, like C++).
struct ThreadPool {
    buckets: [Freelist; MAX_SIZE_CLASSES],
}

impl ThreadPool {
    const fn new() -> Self {
        const EMPTY: Freelist = Freelist::new();
        Self {
            buckets: [EMPTY; MAX_SIZE_CLASSES],
        }
    }

    /// Returns null on OOM; caller is responsible for aborting.
    #[inline]
    fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let Some(nl) = size_class(layout) else {
            return unsafe { alloc(layout) };
        };

        // SAFETY: nl in [1, MAX_SIZE_CLASSES]
        let bucket: &mut Freelist = unsafe { self.buckets.get_unchecked_mut(nl - 1) };
        let bucket_layout: Layout = unsafe { bucket_layout(nl) };

        if let Some(ptr) = bucket.pop() {
            return ptr;
        }

        bucket.refill(bucket_layout);
        bucket
            .pop()
            .unwrap_or_else(|| unsafe { alloc(bucket_layout) })
    }

    /// # Safety
    ///
    /// `ptr` must be valid memory allocated with a compatible layout.
    #[inline]
    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let Some(nl) = size_class(layout) else {
            unsafe { dealloc(ptr, layout) };
            return;
        };

        // SAFETY: nl in [1, MAX_SIZE_CLASSES]
        let bucket: &mut Freelist = unsafe { self.buckets.get_unchecked_mut(nl - 1) };

        if bucket.has_capacity() {
            unsafe { bucket.push(ptr) };
        } else {
            unsafe { dealloc(ptr, bucket_layout(nl)) };
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let nl: usize = i + 1;
            // SAFETY: nl in [1, MAX_SIZE_CLASSES]
            bucket.drain(unsafe { bucket_layout(nl) });
        }
    }
}

// ============================================================================
//  Thread-Local Access
// ============================================================================

thread_local! {
    static POOL: UnsafeCell<ThreadPool> = const { UnsafeCell::new(ThreadPool::new()) };
}

/// Allocate from the thread-local pool. Aborts on OOM.
#[inline]
#[must_use]
pub fn pool_alloc(layout: Layout) -> *mut u8 {
    POOL.with(|cell: &UnsafeCell<ThreadPool>| {
        // SAFETY: thread-local access is single-threaded
        let pool: &mut ThreadPool = unsafe { &mut *cell.get() };
        let ptr: *mut u8 = pool.alloc(layout);

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        ptr
    })
}

/// Return memory to the thread-local pool.
///
/// # Safety
///
/// - `ptr` must be valid memory with the given layout
/// - `layout.align()` must not exceed `CACHE_LINE` (64)
#[inline]
pub unsafe fn pool_dealloc(ptr: *mut u8, layout: Layout) {
    debug_assert!(
        layout.align() <= CACHE_LINE,
        "pool_dealloc: layout alignment ({}) exceeds CACHE_LINE ({})",
        layout.align(),
        CACHE_LINE,
    );

    POOL.with(|cell: &UnsafeCell<ThreadPool>| {
        let pool: &mut ThreadPool = unsafe { &mut *cell.get() };
        unsafe { pool.dealloc(ptr, layout) };
    });
}

// ============================================================================
//  Capture-Free Reclaimers (for `guard.defer_retire()`)
// ============================================================================

/// # Safety
///
/// `ptr` must point to a valid `LeafNode15<S>`.
#[inline]
pub unsafe fn reclaim_leaf15<S: ValueSlot>(ptr: *mut LeafNode15<S>, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    unsafe { pool_dealloc(ptr.cast(), Layout::new::<LeafNode15<S>>()) };
}

/// # Safety
///
/// `ptr` must point to a valid `InternodeNode`.
#[inline]
pub unsafe fn reclaim_internode(ptr: *mut InternodeNode, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    unsafe { pool_dealloc(ptr.cast(), Layout::new::<InternodeNode>()) };
}

/// # Safety
///
/// `ptr` must point to a valid `LeafNode15TrueInline<V>`.
#[inline]
pub unsafe fn reclaim_leaf15_true_inline<V: InlineBits>(
    ptr: *mut LeafNode15TrueInline<V>,
    _collector: &Collector,
) {
    unsafe { StdPtr::drop_in_place(ptr) };
    unsafe { pool_dealloc(ptr.cast(), Layout::new::<LeafNode15TrueInline<V>>()) };
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::similar_names)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
mod unit_tests;
