//! Common allocation utilities.
//!
//! This module provides helper functions for allocating and initializing
//! nodes. Allocations are infallible (abort on OOM like standard Rust).

use core::alloc::Layout;
use core::ptr as CorePtr;
use std::alloc as StdAlloc;

/// Generic Allocator.
#[derive(Debug)]
pub struct GenericAllocator;

impl GenericAllocator {
    /// Allocate memory for a type using the global allocator.
    ///
    /// Returns a properly aligned, uninitialized pointer on success.
    /// Aborts on allocation failure (standard Rust OOM behavior).
    #[inline(always)]
    pub fn alloc<T>() -> *mut T {
        let layout = Layout::new::<T>();

        // SAFETY: Layout is valid (derived from type)
        let ptr: *mut u8 = unsafe { StdAlloc::alloc(layout) };

        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: alloc returns properly aligned pointer for the layout
        ptr.cast::<T>()
    }

    /// Allocate zeroed memory for a type.
    ///
    /// Returns a properly aligned, zero-initialized pointer on success.
    /// Aborts on allocation failure (standard Rust OOM behavior).
    #[inline(always)]
    #[allow(dead_code, reason = "utility function used in tests")]
    pub fn alloc_zeroed<T>() -> *mut T {
        let layout: Layout = Layout::new::<T>();

        // SAFETY: Layout is valid (derived from type)
        let ptr: *mut u8 = unsafe { StdAlloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        ptr.cast::<T>()
    }

    /// Deallocate memory previously allocated with `alloc`.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by `alloc::<T>()` or `alloc_zeroed::<T>()`
    /// - `ptr` must not have been deallocated yet.
    /// - `ptr` must not be used after this call.
    #[inline(always)]
    #[allow(dead_code, reason = "utility function used in tests")]
    pub unsafe fn dealloc<T>(ptr: *mut T) {
        let layout: Layout = Layout::new::<T>();

        // SAFETY: Caller guarantees ptr was allocated with this layout
        unsafe { StdAlloc::dealloc(ptr.cast(), layout) };
    }
}

/// Allocate [`Box`] wrapped types.
#[derive(Debug)]
pub struct BoxAllocator;

impl BoxAllocator {
    /// Create a [`Box`] containing the value.
    ///
    /// Aborts on allocation failure (standard Rust OOM behavior).
    #[inline(always)]
    pub fn boxed<T>(value: T) -> Box<T> {
        let ptr: *mut T = GenericAllocator::alloc::<T>();

        // SAFETY: ptr is valid and properly aligned
        unsafe {
            CorePtr::write(ptr, value);
            Box::from_raw(ptr)
        }
    }
}

#[cfg(test)]
mod unit_tests;
