//! Thread-local node pools using size-class buckets.
//!
//! This module provides thread-local caches for node allocation, matching
//! C++ Masstree's `threadinfo::pool_allocate()` pattern.
//!
//! # Design
//!
//! Pools are organized by **size class** (number of cache lines), not by
//! concrete type. This allows safe reuse across different node types as
//! long as they fit within the same size class.
//!
//! # C++ Reference
//!
//! See `reference/kvthread.hh:226-259` for the original implementation:
//! - `pool_max_nlines = 20` (max 20 cache lines = 1280 bytes)
//! - `nl = ceil((sz + overhead) / CACHE_LINE_SIZE)`
//! - `pool_[nl - 1]` is the freelist for that size class
//!
//! # Thread Safety
//!
//! - Pool access is thread-local (no synchronization on fast path)
//! - Nodes return to the *reclaiming* thread's pool (not necessarily
//!   the allocating thread), due to seize's batch reclamation semantics

use std::alloc::{Layout, alloc, dealloc};
use std::cell::UnsafeCell;
use std::ptr as StdPtr;

use seize::Collector;

use crate::inline::bits::InlineBits;
use crate::inline::leaf15_true::LeafNode15TrueInline;
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;

// ============================================================================
//  Constants (CODE_066 optimized)
// ============================================================================

/// Cache line size for alignment and bucketing.
const CACHE_LINE: usize = 64;

/// Maximum size classes (cache lines). C++ uses 20.
/// Supports nodes up to 20 * 64 = 1280 bytes.
const MAX_SIZE_CLASSES: usize = 20;

/// Maximum nodes to cache per size class per thread.
/// Increased from 64 to 256 to reduce fallback to global allocator.
const POOL_CAPACITY: usize = 256;

/// Batch size when refilling from global allocator.
/// Increased from 16 to 64 to amortize allocation overhead.
const REFILL_BATCH: usize = 64;

// ============================================================================
//  Size Class Computation
// ============================================================================

/// Compute the size class (1-indexed cache line count) for a layout.
///
/// Returns `None` if the layout is too large for pooling.
#[inline]
const fn size_class(layout: Layout) -> Option<usize> {
    // Round up to cache line boundary
    let size = layout.size();
    let nl = size.div_ceil(CACHE_LINE);

    if nl == 0 || nl > MAX_SIZE_CLASSES {
        None
    } else {
        Some(nl)
    }
}

/// Compute the bucket layout for a size class.
///
/// The bucket layout is `nl * CACHE_LINE` bytes with `CACHE_LINE` alignment.
#[inline]
fn bucket_layout(nl: usize) -> Layout {
    debug_assert!(nl > 0 && nl <= MAX_SIZE_CLASSES);
    // SAFETY: size is non-zero and alignment is valid power of 2
    unsafe { Layout::from_size_align_unchecked(nl * CACHE_LINE, CACHE_LINE) }
}

// ============================================================================
//  Freelist
// ============================================================================

/// Intrusive freelist for a single size class.
///
/// Uses the first 8 bytes of each freed block to store the next pointer.
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
        // SAFETY: ptr is valid (from our freelist), first 8 bytes are next ptr
        self.head = unsafe { StdPtr::read(ptr.cast::<*mut u8>()) };
        self.count -= 1;
        Some(ptr)
    }

    /// # Safety
    /// - `ptr` must be valid memory of at least 8 bytes
    /// - `ptr` must be cache-line aligned
    #[inline(always)]
    const unsafe fn push(&mut self, ptr: *mut u8) {
        unsafe { StdPtr::write(ptr.cast::<*mut u8>(), self.head) };
        self.head = ptr;
        self.count += 1;
    }

    /// Refill freelist from global allocator.
    ///
    /// Marked cold since this is the slow path - the hot path is `pop()`.
    #[cold]
    fn refill(&mut self, layout: Layout) {
        for _ in 0..REFILL_BATCH {
            // SAFETY: layout is valid bucket layout
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                break; // OOM
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

/// Per-thread pool with size-class buckets.
///
/// Index `i` holds the freelist for size class `i + 1` (1-indexed like C++).
struct ThreadPool {
    /// Size-class buckets. Index 0 = 1 cache line, index 19 = 20 cache lines.
    buckets: [Freelist; MAX_SIZE_CLASSES],
}

impl ThreadPool {
    const fn new() -> Self {
        // const array initialization
        const EMPTY: Freelist = Freelist::new();
        Self {
            buckets: [EMPTY; MAX_SIZE_CLASSES],
        }
    }

    /// Allocate from the pool for the given layout.
    ///
    /// Returns null if the layout is too large for pooling or OOM.
    #[inline]
    fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let Some(nl) = size_class(layout) else {
            // Too large for pool, fall back to global allocator
            return unsafe { alloc(layout) };
        };

        // SAFETY: size_class() returns nl in [1, MAX_SIZE_CLASSES], so nl-1 is in [0, 19]
        let bucket = unsafe { self.buckets.get_unchecked_mut(nl - 1) };
        let bucket_layout = bucket_layout(nl);

        if let Some(ptr) = bucket.pop() {
            return ptr;
        }

        // Refill and try again
        bucket.refill(bucket_layout);
        bucket.pop().unwrap_or_else(|| {
            // Refill failed, try direct allocation
            unsafe { alloc(bucket_layout) }
        })
    }

    /// Return memory to the pool.
    ///
    /// # Safety
    /// - `ptr` must be valid memory originally from this pool or compatible allocator
    /// - `layout` must match what was used for allocation
    #[inline]
    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let Some(nl) = size_class(layout) else {
            // Was too large for pool, use global deallocator
            unsafe { dealloc(ptr, layout) };
            return;
        };

        // SAFETY: size_class() returns nl in [1, MAX_SIZE_CLASSES], so nl-1 is in [0, 19]
        let bucket: &mut Freelist = unsafe { self.buckets.get_unchecked_mut(nl - 1) };

        if bucket.has_capacity() {
            unsafe { bucket.push(ptr) };
        } else {
            // Pool full, free directly
            unsafe { dealloc(ptr, bucket_layout(nl)) };
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let nl: usize = i + 1;
            bucket.drain(bucket_layout(nl));
        }
    }
}

// ============================================================================
//  Thread-Local Access
// ============================================================================

thread_local! {
    /// Thread-local pool storage.
    static POOL: UnsafeCell<ThreadPool> = const { UnsafeCell::new(ThreadPool::new()) };
}

/// Allocate memory from the thread-local pool.
///
/// Falls back to global allocator if layout is too large for pooling.
#[inline]
#[must_use]
pub fn pool_alloc(layout: Layout) -> *mut u8 {
    POOL.with(|cell| {
        // SAFETY: thread_local access is single-threaded
        let pool = unsafe { &mut *cell.get() };
        pool.alloc(layout)
    })
}

/// Return memory to the thread-local pool.
///
/// # Safety
/// - `ptr` must be valid memory with the given layout
/// - `ptr` must not be used after this call
#[inline]
pub unsafe fn pool_dealloc(ptr: *mut u8, layout: Layout) {
    POOL.with(|cell: &UnsafeCell<ThreadPool>| {
        let pool: &mut ThreadPool = unsafe { &mut *cell.get() };
        unsafe { pool.dealloc(ptr, layout) };
    });
}

/// Pre-allocate freelist entries for common node size classes.
///
/// Call this on each worker thread before benchmarking to eliminate
/// first-allocation overhead. This fills the freelist so the hot path
/// never needs to call the allocator.
///
/// # Common size classes for Masstree
///
/// - 8 cache lines (512 bytes): `InternodeNode`
/// - 12 cache lines (768 bytes): `LeafNode15`
/// - 16 cache lines (1024 bytes): `LeafNode24`
pub fn warmup_pool() {
    // Size classes commonly used by Masstree nodes
    const WARMUP_SIZES: &[usize] = &[
        8 * CACHE_LINE,  // 512 bytes - InternodeNode
        12 * CACHE_LINE, // 768 bytes - LeafNode15
        16 * CACHE_LINE, // 1024 bytes - LeafNode24
    ];

    for &size in WARMUP_SIZES {
        // SAFETY: size is non-zero and alignment is power of 2
        let layout = unsafe { Layout::from_size_align_unchecked(size, CACHE_LINE) };

        // Allocate REFILL_BATCH nodes to trigger refill, then return all
        let mut ptrs = Vec::with_capacity(REFILL_BATCH);
        for _ in 0..REFILL_BATCH {
            let ptr = pool_alloc(layout);
            if !ptr.is_null() {
                ptrs.push(ptr);
            }
        }
        // Return all to freelist
        for ptr in ptrs {
            // SAFETY: ptr was just allocated with this layout
            unsafe { pool_dealloc(ptr, layout) };
        }
    }
}

// ============================================================================
//  Capture-Free Reclaimers
// ============================================================================

/// Reclaim a `LeafNode15` to the thread-local pool.
///
/// This is a capture-free reclaimer for use with `guard.defer_retire()`.
///
/// # Safety
/// - `ptr` must point to a valid `LeafNode15<S>`
#[inline]
pub unsafe fn reclaim_leaf15<S: ValueSlot>(ptr: *mut LeafNode15<S>, _collector: &Collector) {
    // Drop the leaf contents
    unsafe { StdPtr::drop_in_place(ptr) };

    // Return raw memory to pool
    let layout = Layout::new::<LeafNode15<S>>();
    unsafe { pool_dealloc(ptr.cast(), layout) };
}

/// Reclaim a `LeafNode24` to the thread-local pool.
///
/// # Safety
/// - `ptr` must point to a valid `LeafNode24<S>`
#[inline]
pub unsafe fn reclaim_leaf24<S: ValueSlot>(ptr: *mut LeafNode24<S>, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    let layout = Layout::new::<LeafNode24<S>>();
    unsafe { pool_dealloc(ptr.cast(), layout) };
}

/// Reclaim an `InternodeNode` to the thread-local pool.
///
/// # Safety
/// - `ptr` must point to a valid `InternodeNode`
#[inline]
pub unsafe fn reclaim_internode(ptr: *mut InternodeNode, _collector: &Collector) {
    unsafe { StdPtr::drop_in_place(ptr) };
    let layout = Layout::new::<InternodeNode>();
    unsafe { pool_dealloc(ptr.cast(), layout) };
}

/// Reclaim a `LeafNode15TrueInline` to the thread-local pool.
///
/// # Safety
/// - `ptr` must point to a valid `LeafNode15TrueInline<V>`
#[inline]
pub unsafe fn reclaim_leaf15_true_inline<V: InlineBits>(
    ptr: *mut LeafNode15TrueInline<V>,
    _collector: &Collector,
) {
    unsafe { std::ptr::drop_in_place(ptr) };
    let layout = Layout::new::<LeafNode15TrueInline<V>>();
    unsafe { pool_dealloc(ptr.cast(), layout) };
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::similar_names)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_computation() {
        // 64 bytes = 1 cache line
        let layout1 = Layout::from_size_align(64, 8).unwrap();
        assert_eq!(size_class(layout1), Some(1));

        // 65 bytes = 2 cache lines
        let layout2 = Layout::from_size_align(65, 8).unwrap();
        assert_eq!(size_class(layout2), Some(2));

        // 768 bytes = 12 cache lines (LeafNode15 size)
        let layout12 = Layout::from_size_align(768, 64).unwrap();
        assert_eq!(size_class(layout12), Some(12));

        // 1280 bytes = 20 cache lines (max)
        let layout20 = Layout::from_size_align(1280, 64).unwrap();
        assert_eq!(size_class(layout20), Some(20));

        // 1281 bytes = too large
        let layout_big = Layout::from_size_align(1281, 64).unwrap();
        assert_eq!(size_class(layout_big), None);
    }

    #[test]
    fn test_alloc_dealloc_roundtrip() {
        let layout = Layout::from_size_align(128, 64).unwrap();

        let ptr = pool_alloc(layout);
        assert!(!ptr.is_null());

        // Write to verify it's usable
        unsafe { ptr.write(0xAB) };

        unsafe { pool_dealloc(ptr, layout) };

        // Allocate again - should get the same pointer back (from pool)
        let ptr2 = pool_alloc(layout);
        assert_eq!(ptr, ptr2);

        unsafe { pool_dealloc(ptr2, layout) };
    }

    #[test]
    fn test_different_layouts_same_class() {
        // Both fit in 2 cache lines
        let layout_a = Layout::from_size_align(100, 8).unwrap();
        let layout_b = Layout::from_size_align(120, 8).unwrap();

        assert_eq!(size_class(layout_a), size_class(layout_b));

        let ptr_a = pool_alloc(layout_a);
        unsafe { pool_dealloc(ptr_a, layout_a) };

        // Should reuse from same bucket
        let ptr_b = pool_alloc(layout_b);
        assert_eq!(ptr_a, ptr_b);

        unsafe { pool_dealloc(ptr_b, layout_b) };
    }
}
