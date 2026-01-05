//! Common allocation utilities for fallible allocation.
//!
//! This module provides helper functions for allocating and initializing
//! nodes with proper error handling.

use std::alloc::Layout;
use std::ptr as StdPtr;

use crate::error::{AllocError, AllocKind, AllocResult};

/// Try to allocate memory for a type with a specified allocation kind.
///
/// # Errors
/// Upon allocation failure.
#[inline]
pub fn try_alloc_with_kind<T>(kind: AllocKind) -> AllocResult<*mut T> {
    let layout = Layout::new::<T>();

    // SAFETY: Layout is valid (derived from type)
    let ptr = unsafe { std::alloc::alloc(layout) };

    if ptr.is_null() {
        return Err(AllocError::new(layout.size(), layout.align(), kind));
    }

    // SAFETY: alloc returns properly aligned pointer for the layout
    Ok(ptr.cast::<T>())
}
