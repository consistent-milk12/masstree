//! Filepath: src/tree/range/iterator/mod.rs
//!
//! Range iterator implementation for Masstree.
//!
//! Provides [`RangeIter`], an iterator over key-value pairs in lexicographic order.
//! The iterator yields [`ScanEntry`] items containing owned keys and values.
//!
//! # State Machine
//!
//! The iterator is driven by a state machine that handles Masstree's layered
//! structure (trie of B+ trees) and optimistic concurrency control.

// ============================================================================
//  Submodule declarations
// ============================================================================

mod adapters;
mod batch_forward;
mod batch_reverse;
pub(super) mod iter_flags;
mod range_bound;
mod scan_entry;

#[cfg(test)]
mod unit_tests;

// ============================================================================
//  Public re-exports
// ============================================================================

pub use adapters::{KeysIter, ValuesIter};
pub use range_bound::{CursorBound, RangeBound};
pub use scan_entry::ScanEntry;

// ============================================================================
//  Internal imports
// ============================================================================

use std::fmt::{self as StdFmt, Debug, Formatter};
use std::iter::FusedIterator;
use std::marker::PhantomData;

use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;

use crate::policy::LeafPolicy;
use crate::tree::MassTreeGeneric;

use super::cursor_key::CursorKey;
use super::forward_ctx::ForwardScanCtx;
use super::reverse_ctx::ReverseScanCtx;
use super::scan_state::ScanState;

// ============================================================================
//  RangeIter
// ============================================================================

/// Iterator over a key range in a [`crate::MassTree`].
///
/// Yields entries in lexicographic key order. The iterator maintains internal
/// state for the scan position and handles concurrent modifications via the
/// optimistic concurrency control protocol.
///
/// Implements both [`Iterator`] and [`DoubleEndedIterator`], allowing forward
/// iteration with `next()` and reverse iteration with `next_back()` or `.rev()`.
/// # Example
///
/// ```no_run
/// # use masstree::{MassTree, RangeBound};
/// # let tree: MassTree<u64> = MassTree::new();
/// let guard = tree.guard();
/// let mut count = 0;
///
/// for entry in tree.range(
///     RangeBound::Included(b"prefix:"),
///     RangeBound::Excluded(b"prefix;"), // ';' is after ':' in ASCII
///     &guard
/// ) {
///     count += 1;
/// }
///
/// println!("Found {} entries", count);
/// ```
pub struct RangeIter<'a, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    // ========================================================================
    //  Forward iteration state (bundled into ForwardScanCtx)
    // ========================================================================
    /// Memory reclamation guard.
    pub(super) guard: &'g LocalGuard<'a>,

    /// Forward scan context — owns stack, `layer_stack`, `cursor_key`, state,
    /// snapshot, `last_output`, forward flags, and debug fields.
    pub(super) fwd: ForwardScanCtx<P>,

    // ========================================================================
    //  Reverse iteration state (bundled into ReverseScanCtx)
    // ========================================================================
    /// Reverse scan context — `None` for forward-only iterators.
    ///
    /// Lazily initialized when `next_back()`/`.rev()` is first requested.
    /// `Option` avoids initializing unused reverse state; its storage is still
    /// part of the iterator's size.
    pub(super) rev: Option<ReverseScanCtx<P>>,

    // ========================================================================
    //  Shared state
    // ========================================================================
    /// Tree root pointer (needed for back initialization).
    pub(super) tree_root: *const u8,

    /// Start bound for the range (needed for back bound checking).
    pub(super) start_bound: RangeBound<'a>,

    /// End bound for the range (needed for forward bound checking).
    pub(super) end_bound: RangeBound<'a>,

    /// Marker for lifetime and type parameter covariance.
    _marker: PhantomData<&'a A>,
}

impl<P, A> Debug for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        let mut s = f.debug_struct("RangeIter");
        s.field("exhausted", &self.fwd.flags.exhausted())
            .field("initialized", &self.fwd.flags.initialized())
            .field("state", &self.fwd.state);

        if let Some(rev) = &self.rev {
            s.field("back_exhausted", &rev.flags.exhausted())
                .field("back_initialized", &rev.flags.initialized())
                .field("back_state", &rev.state);
        } else {
            s.field("reverse", &"forward-only");
        }

        s.finish_non_exhaustive()
    }
}

impl<'a, 'g, P, A> RangeIter<'a, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Create a forward-only range iterator (no backward state initialization).
    ///
    /// This is cheaper than [`new`](Self::new) because it skips initializing
    /// the reverse cursor, signed leaf position and layer stack.
    /// Use this when you know only forward batch
    /// methods will be called (`for_each`, `for_each_intra_leaf_batch`,
    /// `for_each_values_batch`).
    ///
    /// If reverse iteration is requested later (`next_back` / `.rev()`), reverse
    /// state is lazily initialized on first use to preserve correctness.
    pub(crate) fn new_forward_only(
        tree: &'a MassTreeGeneric<P, A>,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> Self {
        let (start_key, emit_equal) = start.to_start_params();
        let cursor_key = CursorKey::from_slice(start_key);
        let root = tree.load_root_ptr_generic(guard);

        Self {
            guard,
            fwd: ForwardScanCtx::new(root, cursor_key, emit_equal),
            rev: None,
            tree_root: root,
            start_bound: start,
            end_bound: end,
            _marker: PhantomData,
        }
    }

    /// Create a forward-only range iterator rooted at a specific sublayer.
    ///
    /// The `cursor_key` must already be prepared for that layer (offset/len set
    /// as if descent had already occurred).
    pub(crate) fn new_forward_only_from_root(
        layer_root: *const u8,
        cursor_key: CursorKey,
        start_bound: RangeBound<'a>,
        end_bound: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> Self {
        Self {
            guard,
            fwd: ForwardScanCtx::new(layer_root, cursor_key, true),
            rev: None,
            tree_root: layer_root,
            start_bound,
            end_bound,
            _marker: PhantomData,
        }
    }

    /// Create a new range iterator.
    pub(crate) fn new(
        tree: &'a MassTreeGeneric<P, A>,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> Self {
        let (start_key, emit_equal) = start.to_start_params();
        let cursor_key = CursorKey::from_slice(start_key);
        let root = tree.load_root_ptr_generic(guard);

        let back_emit_equal = match &end {
            RangeBound::Unbounded | RangeBound::Included(_) => true,
            RangeBound::Excluded(_) => false,
        };

        Self {
            guard,
            fwd: ForwardScanCtx::new(root, cursor_key, emit_equal),
            rev: Some(ReverseScanCtx::new_with_bound(
                root,
                CursorKey::for_reverse_scan(&end),
                back_emit_equal,
            )),
            tree_root: root,
            start_bound: start,
            end_bound: end,
            _marker: PhantomData,
        }
    }

    // ========================================================================
    //  Helper methods for Option<ReverseScanCtx> access
    // ========================================================================

    /// Returns `true` if the back iterator is exhausted.
    #[inline(always)]
    const fn back_exhausted(&self) -> bool {
        match &self.rev {
            Some(rev) => rev.flags.exhausted(),
            None => false,
        }
    }

    /// Returns `true` if the back iterator has been initialized.
    #[inline(always)]
    const fn back_initialized(&self) -> bool {
        match &self.rev {
            Some(rev) => rev.flags.initialized(),
            None => false,
        }
    }

    /// Mark the back iterator as exhausted (if present).
    #[inline(always)]
    const fn mark_back_exhausted(&mut self) {
        if let Some(rev) = &mut self.rev {
            rev.flags.mark_exhausted();
        }
    }

    /// Initialize the iterator (lazy initialization on first `next()` call).
    pub(super) fn initialize(&mut self) {
        if self.fwd.flags.initialized() {
            return;
        }
        self.fwd.flags.mark_initialized();

        // Handle empty tree
        if self.fwd.stack.root().is_null() {
            self.fwd.flags.mark_exhausted();
            return;
        }

        // Run initial descent loop
        loop {
            let (state, snapshot) = self.fwd.find_initial(
                self.fwd.stack.root(),
                self.fwd.flags.emit_equal(),
                self.guard,
            );

            match state {
                ScanState::Down => {
                    self.fwd.handle_down();
                }

                ScanState::Retry => {
                    // Version conflict, retry from current root
                }

                _ => {
                    // Ready to iterate (Emit, Find, Up)
                    self.fwd.state = state;
                    self.fwd.snapshot = snapshot;
                    break;
                }
            }
        }
    }

    /// Apply the forward bound and double-ended meeting check to a shared scan step.
    #[inline]
    fn advance(&mut self) -> Option<ScanEntry<P::Output>> {
        let (key, value) = self.fwd.advance_no_alloc(&self.end_bound, self.guard)?;
        if let Some(rev) = &mut self.rev
            && rev.flags.initialized()
            && !rev.flags.exhausted()
            && key >= rev.cursor_key.full_key()
        {
            rev.flags.mark_exhausted();
            self.fwd.flags.mark_exhausted();
            return None;
        }
        Some(ScanEntry::new(key.to_vec(), value))
    }

    // ========================================================================
    //  Reverse Iteration (DoubleEndedIterator support)
    // ========================================================================

    /// Initialize the back cursor for reverse iteration.
    ///
    /// Called lazily on the first `next_back()` call.
    pub(super) fn initialize_back(&mut self) {
        if self.back_initialized() {
            return;
        }

        // Forward-only constructor path: lazily materialize reverse state
        // when callers request `next_back`/`.rev()`.
        if self.rev.is_none() {
            let back_emit_equal = matches!(
                self.end_bound,
                RangeBound::Unbounded | RangeBound::Included(_)
            );

            self.rev = Some(ReverseScanCtx::new_with_bound(
                self.tree_root,
                CursorKey::for_reverse_scan(&self.end_bound),
                back_emit_equal,
            ));
        }

        // SAFETY: rev is always Some after the block above
        let rev = unsafe { self.rev.as_mut().unwrap_unchecked() };
        let back_emit_equal = rev.flags.emit_equal();

        rev.flags.mark_initialized();

        // Handle empty tree
        if self.tree_root.is_null() {
            rev.flags.mark_exhausted();
            return;
        }

        // For unbounded end, set upper_bound so lower_reverse returns last slot
        if self.end_bound.is_unbounded() {
            rev.flags.set_upper_bound();
        }

        // Run initial reverse descent loop
        loop {
            let (state, snapshot) =
                rev.find_initial_reverse(rev.stack.root(), back_emit_equal, self.guard);

            match state {
                ScanState::Down => {
                    // Descend into sublayer
                    rev.handle_down_back();

                    // Continue loop to call find_initial_reverse with new layer root
                }

                ScanState::Retry => {
                    // Version conflict, retry
                }

                _ => {
                    // Ready to iterate (Emit, Find, Up)
                    rev.state = state;
                    rev.snapshot = snapshot;

                    // Clear upper_bound after first successful positioning
                    if rev.state.is_emit() {
                        rev.flags.clear_upper_bound();
                    }

                    break;
                }
            }
        }
    }

    /// Apply the reverse bound and double-ended meeting check to a shared scan step.
    #[inline]
    fn advance_back(&mut self) -> Option<ScanEntry<P::Output>> {
        let rev = self.rev.as_mut()?;
        let (key, value) = rev.advance_no_alloc(&self.start_bound, self.guard)?;
        if self.fwd.flags.initialized()
            && !self.fwd.flags.exhausted()
            && key <= self.fwd.cursor_key.full_key()
        {
            rev.flags.mark_exhausted();
            self.fwd.flags.mark_exhausted();
            return None;
        }
        Some(ScanEntry::new(key.to_vec(), value))
    }

    /// Convert to a keys-only iterator.
    pub const fn keys(self) -> KeysIter<'a, 'g, P, A> {
        KeysIter { inner: self }
    }

    /// Convert to a values-only iterator.
    pub const fn values(self) -> ValuesIter<'a, 'g, P, A> {
        ValuesIter { inner: self }
    }
}

// ============================================================================
//  Iterator Trait Implementations
// ============================================================================

impl<P, A> Iterator for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    type Item = ScanEntry<P::Output>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd.flags.exhausted() {
            // FusedIterator: once one direction is exhausted the whole
            // double-ended iterator is exhausted.
            self.mark_back_exhausted();
            return None;
        }

        // Lazy initialization
        if !self.fwd.flags.initialized() {
            self.initialize();

            if self.fwd.flags.exhausted() {
                self.mark_back_exhausted();
                return None;
            }
        }

        // Meeting detection: if back is initialized, check if we've crossed
        if self.back_initialized() && !self.back_exhausted() {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let front_key: &[u8] = unsafe { self.fwd.cursor_key.full_key_unchecked() };
            // SAFETY: rev is Some when back_initialized() is true
            let back_key: &[u8] = unsafe {
                self.rev
                    .as_ref()
                    .unwrap_unchecked()
                    .cursor_key
                    .full_key_unchecked()
            };

            if front_key >= back_key {
                // Mark both as exhausted when they meet
                self.fwd.flags.mark_exhausted();
                self.mark_back_exhausted();
                return None;
            }
        }

        let item: Option<Self::Item> = self.advance();

        if item.is_none() {
            self.mark_back_exhausted();
        }

        item
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.fwd.flags.exhausted() {
            (0, Some(0))
        } else {
            // We can't know the exact count without iterating
            (0, None)
        }
    }
}

impl<P, A> DoubleEndedIterator for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // Check if back cursor is exhausted
        if self.back_exhausted() {
            self.fwd.flags.mark_exhausted();
            return None;
        }

        // Lazy initialization of back cursor
        if !self.back_initialized() {
            self.initialize_back();

            if self.back_exhausted() {
                self.fwd.flags.mark_exhausted();
                return None;
            }
        }

        // Check meeting condition: if front has advanced past where back would be
        if self.fwd.flags.initialized() && !self.fwd.flags.exhausted() {
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let front_key: &[u8] = unsafe { self.fwd.cursor_key.full_key_unchecked() };

            // SAFETY: rev is Some after initialize_back
            let back_key: &[u8] = unsafe {
                self.rev
                    .as_ref()
                    .unwrap_unchecked()
                    .cursor_key
                    .full_key_unchecked()
            };

            if back_key <= front_key {
                // Mark both as exhausted when they meet
                self.mark_back_exhausted();
                self.fwd.flags.mark_exhausted();

                return None;
            }
        }

        let item: Option<Self::Item> = self.advance_back();

        if item.is_none() {
            self.fwd.flags.mark_exhausted();
        }

        item
    }
}

impl<P, A> FusedIterator for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
}

// No manual Drop needed — `last_output: Option<P::Output>` drops automatically.
