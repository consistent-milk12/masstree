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
//! - **OOM**: aborts via `handle_alloc_error`. C++ returns null.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::cell::UnsafeCell;
use std::ptr as StdPtr;

use seize::Collector;

use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;

// ============================================================================
//  Constants
// ============================================================================

const CACHE_LINE: usize = 64;

/// C++ uses 20 → nodes up to 1280 bytes.
const MAX_SIZE_CLASSES: usize = 20;

/// Slab size for batch refill (2 MB, matching C++ `posix_memalign` slab).
const SLAB_SIZE: usize = 2 * 1024 * 1024;

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
///
/// Blocks are carved from contiguous 2 MB slabs during refill, giving good spatial
/// locality and cache/prefetch behavior. Slabs are intentionally **never freed** —
/// blocks handed out may still be live nodes in the shared tree when a thread exits.
/// This matches C++ Masstree, which never frees pool memory.
struct Freelist {
    head: *mut u8,
}

impl Freelist {
    const fn new() -> Self {
        Self {
            head: StdPtr::null_mut(),
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

        Some(ptr)
    }

    /// # Safety
    ///
    /// `ptr` must be valid, cache-line-aligned memory of at least 8 bytes.
    #[inline(always)]
    const unsafe fn push(&mut self, ptr: *mut u8) {
        unsafe { StdPtr::write(ptr.cast::<*mut u8>(), self.head) };
        self.head = ptr;
    }

    /// Slow path: allocate a 2 MB slab and carve it into `block_size`-byte blocks.
    ///
    /// The slab is never freed — blocks may outlive this thread as live tree nodes.
    /// This matches C++ Masstree's pool behavior.
    #[cold]
    fn refill(&mut self, nl: usize) {
        let block_size: usize = nl * CACHE_LINE;
        let num_blocks: usize = SLAB_SIZE / block_size;

        // SAFETY: SLAB_SIZE > 0 and CACHE_LINE is a power of 2
        let slab_layout: Layout =
            unsafe { Layout::from_size_align_unchecked(num_blocks * block_size, CACHE_LINE) };

        // SAFETY: slab_layout has non-zero size and valid alignment
        let slab_ptr: *mut u8 = unsafe { alloc(slab_layout) };

        if slab_ptr.is_null() {
            return;
        }

        // Carve contiguous blocks from the slab.
        for i in 0..num_blocks {
            // SAFETY: i * block_size is within the slab allocation
            let block: *mut u8 = unsafe { slab_ptr.add(i * block_size) };

            // SAFETY: block is cache-line-aligned, at least 8 bytes (block_size >= CACHE_LINE)
            unsafe { self.push(block) };
        }
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

        bucket.refill(nl);
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

        // SAFETY: ptr is cache-line-aligned, at least 8 bytes (from a pooled size class)
        unsafe { bucket.push(ptr) };
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

/// Reclaim a `LeafNode15<P>` — runs Drop, returns memory to pool.
///
/// # Safety
///
/// `ptr` must point to a valid `LeafNode15<P>`.
#[inline]
pub unsafe fn reclaim_leaf15<P: LeafPolicy>(ptr: *mut LeafNode15<P>, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    unsafe { pool_dealloc(ptr.cast(), Layout::new::<LeafNode15<P>>()) };
}

/// # Safety
///
/// `ptr` must point to a valid `InternodeNode`.
#[inline]
pub unsafe fn reclaim_internode(ptr: *mut InternodeNode, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    unsafe { pool_dealloc(ptr.cast(), Layout::new::<InternodeNode>()) };
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::similar_names)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
mod unit_tests;
