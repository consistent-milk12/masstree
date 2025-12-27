//! Hand-over-hand split propagation.
//!
//! This module implements TRUE hand-over-hand split propagation matching
//! C++ `tcursor::make_split()` from `masstree_split.hh:179-297`.
//!
//! # Key Invariant
//! The left node remains locked while we lock its parent and install the
//! split sibling. This prevents the "split-but-uninstalled" window that
//! causes permanent routing inconsistencies.
//!
//! # Design Choice: [`PropagationContext`]
//! This implementation [`PropagationContext<'op>`] - a unified-lifetime
//! context that allows all [`LockGuard<'op>`] instances to have the same
//! lifetime parameter. This enables lock transer across loop iterations
//! while preserving RAII (panic-safe automatic unlocking).
//!
//! Unlike the previous `PropagationLock` concept:
//! - Guards auto-unlock on drop (RAII preserved)
//! - No `mem::forget` patterns
//! - All locks use existing `NodeVersion::lock()` implementation
//!
//! # RAII Helpers
//!
//! Two complementary RAII patterns are available:
//!
//! - [`PropagationContext`]: Unified-lifetime context for lock guards
//!   (complex lock transfer across loop iterations)
//! - [`ExitGuard`]: Simple scope-exit callbacks (cleanup, rollback)
//!
//! # Module Organization
//!
//! - [`PropagationContext`]: Unified-lifetime context for RAII guards
//! - [`Propagation`]: Core hand-over-hand propagation loop
//! - [`RootCreation`]: Root and layer-root creation helpers
//! - [`ParentLocking`]: Membership validation helpers

mod exit_guard;
mod parent_locking;
mod propagation;
mod propagation_context;
mod root_creation;

pub use exit_guard::ExitGuard;
pub use propagation::Propagation;
