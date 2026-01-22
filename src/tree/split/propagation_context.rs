//! Unified-lifetime lock context for split propagation.
//!
//! # Problem
//!
//! [`LockGuard<'_>`] has a lifetime tied to the node it was acquired from.
//! When transitioning from leaf to internode level, we need to transfer the lock
//! (`left_lock = parent_lock`), but Rust lifetimes prevent this assignment.
//!
//! # Solution
//!
//! [`PropagationContext`] unifies all guard lifetimes to `'op` (the reclamation
//! guard's lifetime). Since all nodes remain valid for `'op`, extending any
//! [`LockGuard<'a>`] to [`LockGuard<'op>`] is safe.
//!
//! Previous approaches considered:
//! - No-drop guard type: Lock leaks on panic, hard to audit
//! - Current approach: RAII-safe, auditable, panic-safe

use crate::nodeversion::{LockGuard, NodeVersion};
use seize::LocalGuard;
use std::marker::PhantomData;

/// Context for hand-over-hand split propagation with unified lifetimes.
///
/// Provides panic-safe RAII locking by binding all [`LockGuard`]s to lifetime `'op`.
///
/// # Why This Works
///
/// All nodes remain valid for `'op` because the [`LocalGuard`] prevents reclamation.
/// Therefore, extending [`LockGuard<'a>`] to [`LockGuard<'op>`] is sound.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = PropagationContext::new(guard);
/// let mut left_lock: LockGuard<'op> = unsafe { ctx.unify_guard(leaf_lock) };
///
/// loop {
///     let parent_lock: LockGuard<'op> = unsafe { ctx.lock_node(parent_version_ptr) };
///     drop(left_lock);
///     left_lock = parent_lock; // Same lifetime allows assignment
/// }
/// // left_lock auto-unlocks on drop
/// ```
pub struct PropagationContext<'op> {
    /// Binds `'op` to the reclamation guard's lifetime.
    _marker: PhantomData<&'op LocalGuard<'op>>,
}

impl<'op> PropagationContext<'op> {
    /// Create a context tied to the reclamation guard's lifetime.
    #[inline]
    pub const fn new(_guard: &'op LocalGuard<'op>) -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Lock a node and return a [`LockGuard`] with unified lifetime `'op`.
    ///
    /// Prefer [`Self::lock_node_yielding`] for high-contention scenarios.
    ///
    /// # Safety
    /// `version_ptr` must point to a valid node protected by the reclamation guard.
    #[inline(always)]
    #[allow(dead_code, reason = "Low-contention API; prefer lock_node_yielding")]
    #[expect(clippy::unused_self, reason = "Binds output lifetime to context's 'op")]
    pub unsafe fn lock_node(&self, version_ptr: *const NodeVersion) -> LockGuard<'op> {
        // SAFETY: Caller guarantees validity for 'op.
        let version: &'op NodeVersion = unsafe { &*version_ptr };
        version.lock()
    }

    /// Lock a node with yield-on-contention for high thread counts.
    ///
    /// # Safety
    /// Same as [`Self::lock_node`].
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "Binds output lifetime to context's 'op")]
    pub unsafe fn lock_node_yielding(&self, version_ptr: *const NodeVersion) -> LockGuard<'op> {
        // SAFETY: Caller guarantees validity for 'op.
        let version: &'op NodeVersion = unsafe { &*version_ptr };
        version.lock_with_yield()
    }

    /// Extend a [`LockGuard`]'s lifetime to the unified `'op`.
    ///
    /// # Safety
    /// The guard's underlying node must remain valid for `'op`.
    /// This is guaranteed when the node is protected by the reclamation guard.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "Binds output lifetime to context's 'op")]
    pub unsafe fn unify_guard<'a>(&self, guard: LockGuard<'a>) -> LockGuard<'op> {
        // SAFETY: Reclamation guard prevents deallocation for 'op.
        unsafe { std::mem::transmute(guard) }
    }
}
