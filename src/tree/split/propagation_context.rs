//! Filepath: `src/tree/split/propagation_context`.
//!
//! The core insight: Rust's [`LockGuard<'_>`] has a lifetime parameter tried to the node
//! it wwas acquired from. When we transition from leaf level to internode level, we
//! need to "transefer" the lock from the parent to become the new left , but Rust
//! lifetimes prevent `left_lock = parent_lock`.
//!
//! The solution proposed here is a refinement over the two previous solutions.
//! I originally came up with the following:
//!
//! Solution 1: The curent broken system.
//! Solution 2: [`PropagationLock`] - A guard type with no lifetime parameter and no [`Drop`].
//! I implemented this on my side before realizing it had a major flaw:
//! Lock leaks on panic, excruciatingly hard to audit. And after my current experience debugging
//! this EXTREMELY complex [`MassTree`] implementation, I just knew that I had to come up with
//! something better
//!
//! **Refined Current**: [`PropagationContext`] Unify all guard lifetimes via a context struct.
//! The key is that all nodes remain valid for 'op (the reclamation guard's lifetime),
//! so we can safely extend any [`LockGuard<'a>`] to [`LockGuard<'op>`].

use crate::nodeversion::{LockGuard, NodeVersion};
use seize::LocalGuard;
use std::marker::PhantomData;

/// Context for hand-over-hand split propagation with unified lifetimes.
///
/// This type provides panic-safe RAII locking by ensuring all [`LockGuard`]'s,
/// have the same lifetime parameter `'op` tied to the reclamation guard.
///
/// # Why This Works
///
/// All nodes in the tree remain valid for the duration of `'op` because:
/// 1. The [`LocalGuard`] prevents memory reclamation
/// 2. Nodes are never deallocated while a reclamation guard in active
///
/// Therefore, extending a [`LockGuard<'shorter'>`] to [`LockGuard<'op>`] is safe.
///
/// # Usage
///
/// ```rust,ignore
/// let ctx = PropagationContext::new(guard);
/// let mut left_lock: LockGuard<'op> = unsafe { ctx.unify_guard(leaf_lock) };
///
/// loop {
///     let parent_lock: LockGuard<'op> = unsafe { ctx.lock_node(parent_version_ptr) };
///     // ... work ...
///
///     drop(left_lock); // RAII unlock
///     left_lock = parent_lock; // This works: same lifetime 'op
/// }
///
/// // left_lock auto-unlocks on drop
/// ```
pub struct PropagationContext<'op> {
    /// Marker to bind the 'op lifetime to the reclamation guard.
    /// The [`LocalGuard`] is stored externally, we just need the lifetime.
    _marker: PhantomData<&'op LocalGuard<'op>>,
}

impl<'op> PropagationContext<'op> {
    /// Create a new propagation context.
    ///
    /// The `'op` lifetime is tied to the provided reclamation guard.
    #[inline]
    pub const fn new(_guard: &'op LocalGuard<'op>) -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Lock a [`NodeVersion`] and return a [`LockGuard`] with unified lifetime `'op`.
    ///
    /// # Safety
    /// `version_ptr` must point to a valid [`NodeVersion`] that remains valid
    /// for the duration of `'op`. This is guaranteed by the reclamation guard.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub unsafe fn lock_node(&self, version_ptr: *const NodeVersion) -> LockGuard<'op> {
        // Create a reference with the unified lifetime.
        // SAFETY: Caller guarantees version_ptr is valid for 'op.
        let version: &'op NodeVersion = unsafe { &*version_ptr };

        version.lock()
    }

    /// Lock a [`NodeVersion`] using [`NodeVersion::lock_with_yield`] for high contention.
    ///
    /// # Safety
    /// Same requirements as `lock_node`.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub unsafe fn lock_node_yielding(&self, version_ptr: *const NodeVersion) -> LockGuard<'op> {
        // SAFETY: Caller guarantees version_ptr is valid for 'op.
        let version: &'op NodeVersion = unsafe { &*version_ptr };

        version.lock_with_yield()
    }

    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub unsafe fn unify_guard<'a>(&self, guard: LockGuard<'a>) -> LockGuard<'op> {
        // SAFETY: Both 'a and 'op are within the reclamation guard's protection.
        // The node cannot be freed while any guard exists, so extending
        // the lifetime is safe.
        unsafe { std::mem::transmute(guard) }
    }
}
