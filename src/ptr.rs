//! Type-safe pointer wrappers for tree nodes.
//!
//! This module provides newtype wrappers to distinguish between:
//! - [`NodePtr`]: Type-erased pointer to a tree node (leaf or internode)
//! - Value pointers (`*mut u8` from `Arc::into_raw`) - NOT wrapped
//!
//! # Motivation
//!
//! The codebase uses `*mut u8` for both tree node pointers and value storage
//! pointers. These have completely different semantics:
//! - Node pointers require `NodeVersion` checks and proper casting
//! - Value pointers are managed by Arc/Box reference counting
//!
//! Mixing them up causes undefined behavior. These newtypes make the
//! distinction explicit at the type level.
//!
//! # Usage
//!
//! ```ignore
//! // Old (ambiguous):
//! fn set_parent(&self, parent: *mut u8);
//!
//! // New (explicit):
//! fn set_parent(&self, parent: NodePtr);
//! ```

use std::ptr::{self as StdPtr, NonNull};

use crate::leaf_trait::{TreeInternode, TreeLeafNode};
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;

// ============================================================================
//  NodePtr - Type-erased tree node pointer
// ============================================================================

/// A type-erased pointer to a tree node (leaf or internode).
///
/// This wrapper provides type safety by distinguishing tree structure pointers
/// from value storage pointers. The wrapped pointer can point to either a
/// `LeafNode` or `InternodeNode`, determined at runtime via [`NodeVersion::is_leaf()`].
///
/// # Invariants
///
/// - The pointer must be properly aligned for `NodeVersion` (first field of both types)
/// - The pointed-to node must have a valid `NodeVersion` as its first field
/// - The pointer must remain valid for its intended lifetime (protected by guard or lock)
///
/// # Null Handling
///
/// `NodePtr` can represent null via [`NodePtr::null()`]. Use [`NodePtr::is_null()`]
/// to check, or [`NodePtr::as_non_null()`] to convert to `Option<NonNull<u8>>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodePtr(*mut u8);

impl NodePtr {
    /// Create a null node pointer.
    #[inline(always)]
    #[must_use]
    pub const fn null() -> Self {
        Self(StdPtr::null_mut())
    }

    /// Create a `NodePtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must either be null or point to a valid tree node
    /// (`LeafNode` or `InternodeNode`) with `NodeVersion` as the first field.
    #[inline(always)]
    #[must_use]
    pub const unsafe fn from_raw(ptr: *mut u8) -> Self {
        Self(ptr)
    }

    /// Create a `NodePtr` from a leaf pointer.
    #[inline(always)]
    #[must_use]
    pub const fn from_leaf<S: ValueSlot, L: TreeLeafNode<S>>(leaf: *mut L) -> Self {
        Self(leaf.cast::<u8>())
    }

    /// Create a `NodePtr` from an internode pointer.
    #[inline(always)]
    #[must_use]
    pub const fn from_internode<S: ValueSlot, I: TreeInternode<S>>(internode: *mut I) -> Self {
        Self(internode.cast::<u8>())
    }

    /// Get the raw pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_raw(self) -> *mut u8 {
        self.0
    }

    /// Check if this pointer is null.
    #[inline(always)]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Convert to `NonNull`, returning None if null.
    #[inline(always)]
    #[must_use]
    pub const fn as_non_null(self) -> Option<NonNull<u8>> {
        NonNull::new(self.0)
    }

    /// Check if this node is a leaf (vs internode).
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and point to a valid node.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Caller guarantees proper alignment"
    )]
    pub unsafe fn is_leaf(self) -> bool {
        debug_assert!(!self.is_null(), "is_leaf called on null NodePtr");
        // SAFETY: Caller guarantees pointer is valid and properly aligned.
        // NodeVersion is the first field of both LeafNode and InternodeNode.
        unsafe { (*self.0.cast::<NodeVersion>()).is_leaf() }
    }

    /// Get a reference to the `NodeVersion`.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and point to a valid node.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Caller guarantees proper alignment"
    )]
    pub unsafe fn version(self) -> &'static NodeVersion {
        debug_assert!(!self.is_null(), "version called on null NodePtr");
        // SAFETY: Caller guarantees pointer is valid. NodeVersion is first field.
        unsafe { &*self.0.cast::<NodeVersion>() }
    }

    /// Cast to a leaf pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `LeafNode` of type L.
    /// Use [`is_leaf()`](Self::is_leaf) to verify before casting.
    #[inline(always)]
    #[must_use]
    pub const unsafe fn cast_leaf<S: ValueSlot, L: TreeLeafNode<S>>(self) -> *mut L {
        self.0.cast::<L>()
    }

    /// Cast to an internode pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `InternodeNode` of type I.
    /// Use [`is_leaf()`](Self::is_leaf) to verify this is NOT a leaf before casting.
    #[inline(always)]
    #[must_use]
    pub const unsafe fn cast_internode<S: ValueSlot, I: TreeInternode<S>>(self) -> *mut I {
        self.0.cast::<I>()
    }

    /// Cast to a const pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_const(self) -> *const u8 {
        self.0.cast_const()
    }
}

impl std::fmt::Debug for NodePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodePtr({:?})", self.0)
    }
}

impl std::fmt::Pointer for NodePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Pointer::fmt(&self.0, f)
    }
}

impl Default for NodePtr {
    fn default() -> Self {
        Self::null()
    }
}

// SAFETY: NodePtr is just a pointer wrapper. The pointed-to data has its own
// synchronization via NodeVersion locks and epoch-based reclamation.
unsafe impl Send for NodePtr {}
unsafe impl Sync for NodePtr {}

// ============================================================================
//  Conversions
// ============================================================================

impl From<*mut u8> for NodePtr {
    /// Convert from raw pointer.
    ///
    /// # Note
    ///
    /// This conversion exists for gradual migration. Prefer explicit
    /// [`NodePtr::from_leaf`] or [`NodePtr::from_internode`] when the
    /// concrete type is known.
    #[inline(always)]
    fn from(ptr: *mut u8) -> Self {
        Self(ptr)
    }
}

impl From<NodePtr> for *mut u8 {
    #[inline(always)]
    fn from(ptr: NodePtr) -> Self {
        ptr.0
    }
}

impl From<NodePtr> for *const u8 {
    #[inline(always)]
    fn from(ptr: NodePtr) -> Self {
        ptr.0.cast_const()
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null() {
        let ptr = NodePtr::null();
        assert!(ptr.is_null());
        assert!(ptr.as_non_null().is_none());
    }

    #[test]
    fn test_from_raw() {
        let value: u64 = 42;
        let raw = &raw const value as *mut u8;

        // SAFETY: Just testing the wrapper, not dereferencing
        let ptr = unsafe { NodePtr::from_raw(raw) };

        assert!(!ptr.is_null());
        assert_eq!(ptr.as_raw(), raw);
    }

    #[test]
    fn test_debug_format() {
        let ptr = NodePtr::null();
        let debug = format!("{ptr:?}");
        assert!(debug.contains("NodePtr"));
    }

    #[test]
    fn test_pointer_format() {
        let ptr = NodePtr::null();
        let formatted = format!("{ptr:p}");
        assert!(formatted.contains("0x"));
    }

    #[test]
    fn test_default_is_null() {
        let ptr = NodePtr::default();
        assert!(ptr.is_null());
    }

    #[test]
    fn test_equality() {
        let ptr1 = NodePtr::null();
        let ptr2 = NodePtr::null();
        assert_eq!(ptr1, ptr2);

        // Use real allocated memory for strict provenance compatibility
        let mut value: u8 = 42;
        let raw: *mut u8 = &raw mut value;

        // SAFETY: raw points to valid stack memory
        let ptr3 = unsafe { NodePtr::from_raw(raw) };
        let ptr4 = unsafe { NodePtr::from_raw(raw) };

        assert_eq!(ptr3, ptr4);
        assert_ne!(ptr1, ptr3);
    }
}
