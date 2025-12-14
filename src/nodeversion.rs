//! Filepath: src/nodeversion.rs
//!
//! Node version for optimistic concurrency control.
//!
//! [`NodeVersion`] combines lock state, version counters, and metadata flags
//! in a single `u32`. Readers use optimistic validation, writers acquire locks.
//!
//! # Concurrency Model
//! 1. Readers: Call `stable()` to get version, perform read, call `has_changed()`
//! 2. Writers: Call `lock()` to get a [`LockGuard`], modify node, let guard drop.
//!
//! # Type-State Pattern
//! The [`LockGuard`] type provides compile-time verification that the lock is held.
//! Operations that require the lock take `&mut LockGuard` as proof. The guard
//! automatically unlocks on drop (panic-safe).
//!
//! ```rust,ignore
//! let mut guard = version.lock();
//! gaurd.mark_insert();
//! ```
