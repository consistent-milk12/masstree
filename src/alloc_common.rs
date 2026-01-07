//! Common allocation utilities for fallible allocation.
//!
//! This module provides helper functions for allocating and initializing
//! nodes with proper error handling.

#![allow(dead_code)]

use core::alloc::Layout;
use core::ptr as CorePtr;
use std::alloc as StdAlloc;

use crate::error::{AllocError, AllocKind, AllocResult};

/// Generic Allocator.
#[derive(Debug)]
pub struct GenericAllocator;

impl GenericAllocator {
    /// Try to allocate memory for a type with a specified allocation kind.
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_alloc_with_kind<T>(kind: AllocKind) -> AllocResult<*mut T> {
        let layout = Layout::new::<T>();

        // SAFETY: Layout is valid (derived from type)
        let ptr: *mut u8 = unsafe { StdAlloc::alloc(layout) };

        if ptr.is_null() {
            return Err(AllocError::new(layout.size(), layout.align(), kind));
        }

        // SAFETY: alloc returns properly aligned pointer for the layout
        Ok(ptr.cast::<T>())
    }

    /// Try to allocate memory for a type using the global allocator.
    ///
    /// Returns a properly aligned, uninitialized pointer on success,
    /// or [`AllocError`] on failure.
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_alloc<T>() -> AllocResult<*mut T> {
        Self::try_alloc_with_kind::<T>(AllocKind::Other)
    }

    /// Try to alllocate zeroed memory for a type.
    ///
    /// Returns a properly aligned, zero-initialized pointer on success,
    /// or [`AllocError`] on failure.
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_alloc_zeroed_with_kind<T>(kind: AllocKind) -> AllocResult<*mut T> {
        let layout: Layout = Layout::new::<T>();

        // SAFETY: Layout is valid (derived from type)
        let ptr: *mut u8 = unsafe { StdAlloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(AllocError::new(layout.size(), layout.align(), kind));
        }

        Ok(ptr.cast::<T>())
    }

    /// Try to allocate zeroed memory for a type.
    ///
    /// Returns a properly aligned, zero-initialized pointer on success,
    /// or [`AllocError`] on failure.
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_alloc_zeroed<T>() -> AllocResult<*mut T> {
        Self::try_alloc_zeroed_with_kind::<T>(AllocKind::Other)
    }

    /// Deallocate memory previously allocated with `try_alloc`.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by `try_alloc::<T>()` or `try_alloc_zeroed::<T>()`
    /// - `ptr` must not have been deallocated yet.
    /// - `ptr` must not be used after this call.
    #[inline(always)]
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
    /// Try to create a [`Box`] with specified allocation kind.
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_box_with_kind<T>(value: T, kind: AllocKind) -> AllocResult<Box<T>> {
        let ptr: *mut T = GenericAllocator::try_alloc_with_kind::<T>(kind)?;

        // SAFETY: ptr is valid and properly aligned
        unsafe {
            CorePtr::write(ptr, value);

            Ok(Box::from_raw(ptr))
        }
    }

    /// Try to create a [`Box`], returning an error on allocation failure.
    ///
    /// This is a fallible version of [`Box::new`]
    ///
    /// # Errors
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_box<T>(value: T) -> AllocResult<Box<T>> {
        Self::try_box_with_kind(value, AllocKind::Other)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "Fail fast with reason in tests")]
mod unit_tests;
