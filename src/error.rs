//! Error types for `MassTree` operations.
//!
//! This module defines internal error types used throughout the crate.
//! User facing errors are re-exported from `src/tree.rs`.

use std::fmt as StdFmt;
use std::mem as StdMem;

/// Kind of allocation that failed.
///
/// Used for debugging and policy decisions (like, different handling for
/// structural vs data allocs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocKind {
    /// Leaf node allocation (split sibling, layer twig, etc etc..)
    Leaf,

    /// Internode allocation (should not appear in current planned phase (Tier 1))
    Internode,

    /// Suffix bag storage (external bag, capacity growth)
    Suffix,

    /// Value storage (Box for `LeafValueIndex`)
    Value,

    /// Allocator tracking vector growth
    AllocatorTracking,

    /// Currently unspecified allocation
    Other,
}

impl StdFmt::Display for AllocKind {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::Leaf => write!(f, "leaf"),

            Self::Internode => write!(f, "internode"),

            Self::Suffix => write!(f, "suffix"),

            Self::Value => write!(f, "value"),

            Self::AllocatorTracking => write!(f, "allocator tracking"),

            Self::Other => write!(f, "other"),
        }
    }
}

/// Error returned when memory allocation fails.
///
/// This is an internal error type used by allocator implementations.
/// It gets converted to [`InsertError::AllocationFailed`] at API boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocError {
    /// Approximate size of the failed allocation in bytes.
    pub size: usize,

    /// Alignment requirement of the failed allocation.
    pub align: usize,

    /// Kind of allocation that failed.
    pub kind: AllocKind,
}

impl AllocError {
    /// Create a new allocation error with full details.
    #[inline]
    #[must_use]
    pub const fn new(size: usize, align: usize, kind: AllocKind) -> Self {
        Self { size, align, kind }
    }

    /// Create an allocation error from a type with specified kind.
    #[inline]
    #[must_use]
    pub const fn for_type_with_kind<T>(kind: AllocKind) -> Self {
        Self {
            size: StdMem::size_of::<T>(),
            align: StdMem::align_of::<T>(),
            kind,
        }
    }

    /// Create an allocation error for a type (default to `Other` kind)
    #[inline]
    #[must_use]
    pub const fn for_type<T>() -> Self {
        Self::for_type_with_kind::<T>(AllocKind::Other)
    }

    /// Create an allocation error for a leaf node.
    #[inline]
    #[must_use]
    pub const fn for_leaf<T>() -> Self {
        Self::for_type_with_kind::<T>(AllocKind::Leaf)
    }

    /// Create an allocation error for suffix storage.
    #[inline]
    #[must_use]
    pub const fn for_suffix(size: usize) -> Self {
        Self::new(size, 1, AllocKind::Suffix)
    }

    /// Create an allocation error for value boxing.
    #[inline]
    #[must_use]
    pub const fn for_value<T>() -> Self {
        Self::for_type_with_kind::<T>(AllocKind::Value)
    }

    /// Create an allocation error for allocator tracking.
    #[inline]
    #[must_use]
    pub const fn for_tracking(size: usize) -> Self {
        Self::new(
            size,
            StdMem::align_of::<*mut u8>(),
            AllocKind::AllocatorTracking,
        )
    }
}

impl StdFmt::Display for AllocError {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(
            f,
            "{} allocation of {} bytes (align {}) failed",
            self.kind, self.size, self.align
        )
    }
}

impl std::error::Error for AllocError {}

/// Result type alias for fallible allocations.
pub type AllocResult<T> = Result<T, AllocError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_error_new() {
        let err: AllocError = AllocError::new(1024, 8, AllocKind::Leaf);

        assert_eq!(err.size, 1024);
        assert_eq!(err.align, 8);
        assert_eq!(err.kind, AllocKind::Leaf);
    }

    #[test]
    fn test_alloc_error_for_type() {
        let err: AllocError = AllocError::for_type::<u64>();

        assert_eq!(err.size, 8);
        assert_eq!(err.align, 8);
        assert_eq!(err.kind, AllocKind::Other);
    }

    #[test]
    fn test_alloc_error_for_leaf() {
        let err: AllocError = AllocError::for_leaf::<[u8; 256]>();

        assert_eq!(err.size, 256);
        assert_eq!(err.kind, AllocKind::Leaf);
    }

    #[test]
    fn test_alloc_error_display() {
        let err = AllocError::new(4096, 64, AllocKind::Suffix);
        let msg: String = format!("{err}");

        assert!(msg.contains("suffix"));
        assert!(msg.contains("4096"));
        assert!(msg.contains("64"));
    }
}
