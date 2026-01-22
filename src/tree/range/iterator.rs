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
//!
//! ## States
//!
//! | State | Description |
//! |-------|-------------|
//! | **`Emit`** | Valid entry found; ready to yield to caller |
//! | **`FindNext`** | Scanning current leaf for the next valid slot |
//! | **`Down`** | Layer pointer encountered; must descend into sublayer |
//! | **`Up`** | Current layer exhausted; must return to parent layer |
//! | **`Retry`** | Repositioning after version conflict or layer transition |
//!
//! | From | To | Trigger |
//! |------|-----|---------|
//! | `FindNext` | `Emit` | Found valid entry passing duplicate filter |
//! | `FindNext` | `FindNext` | Advanced to sibling leaf via B-link pointer |
//! | `FindNext` | `Down` | Slot contains layer pointer (`keylenx >= LAYER`) |
//! | `FindNext` | `Up` | Leaf exhausted with no next sibling |
//! | `FindNext` | `Retry` | Version validation failed (concurrent modification) |
//! | `Emit` | `FindNext` | Entry yielded; `ki` advanced to next slot |
//! | `Down` | `Retry` | After `shift_clear()` pushes prefix and clears suffix |
//! | `Up` | `FindNext` | After `unshift()` restores parent context (or Done if stack empty) |
//! | `Retry` | `FindNext` | After `find_retry()` repositions using key prefix stack |
//!
//! ## Why `Down` Always Transitions to `Retry`
//!
//! When descending into a sublayer, `shift_clear()` updates the cursor's key prefix
//! but the scan position is unknown in the new layer. `find_retry()` performs a fresh
//! lookup using the accumulated key prefix to find the correct starting position.
//!
//! ## Version Conflict Recovery
//!
//! On version mismatch (detected via OCC protocol), the iterator transitions to `Retry`.
//! The `find_retry()` function uses the cursor's key prefix stack to reposition to the
//! correct location, ensuring no entries are skipped or duplicated.
//!
//! # Usage
//!
//! ```no_run
//! # use masstree::{MassTree, RangeBound};
//! let tree: MassTree<u64> = MassTree::new();
//! let guard = tree.guard();
//!
//! // Full iteration (all keys)
//! for entry in tree.iter(&guard) {
//!     println!("{:?} -> {:?}", entry.key, entry.value);
//! }
//!
//! // Bounded range: keys in ["start", "end")
//! for entry in tree.range(
//!     RangeBound::Included(b"start"),
//!     RangeBound::Excluded(b"end"),
//!     &guard
//! ) {
//!     // entry.key is >= b"start" and < b"end"
//! }
//!
//! // Reverse iteration
//! for entry in tree.iter(&guard).rev() {
//!     // keys in descending order
//! }
//! ```

// ============================================================================
//  Submodule declarations
// ============================================================================

mod adapters;
mod batch_forward;
mod batch_reverse;
mod cleanup_guard;
mod iter_flags;
mod range_bound;
mod scan_entry;

#[cfg(test)]
mod unit_tests;

// ============================================================================
//  Public re-exports
// ============================================================================

pub use adapters::{KeysIter, ValuesIter};
pub use range_bound::RangeBound;
pub use scan_entry::ScanEntry;

// ============================================================================
//  Internal imports
// ============================================================================

use std::fmt::{self as StdFmt, Debug, Formatter};
use std::iter::FusedIterator;
use std::marker::PhantomData;

use arrayvec::ArrayVec;
use seize::LocalGuard;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::key::IKEY_SIZE;
use crate::leaf_trait::{LayerCapableLeaf, TreeLeafNode};
use crate::slot::ValueSlot;
use crate::tree::MassTreeGeneric;

use super::cursor_key::CursorKey;
use super::find::{
    find_initial, find_next, find_next_with_duplicate_check, find_retry, handle_down, handle_up,
};
use super::find_rev::ReverseScan;
use super::helper::ReverseScanHelper;
use super::scan_state::{
    BackStackElement, LayerContext, LayerStack, ScanSnapshot, ScanStackElement, ScanState,
    ScanStateBack,
};

use iter_flags::IterFlags;

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
///
/// # Thread Safety
///
/// The iterator holds a reference to the tree and the guard. The guard must
/// remain alive for the duration of iteration to protect pointers from
/// garbage collection.
///
/// # Consistency
///
/// Range scans are **weakly consistent**:
/// - Keys are yielded in sorted order
/// - May see some concurrent inserts and miss others
/// - No torn reads (partial key/value data)
/// - Duplicates filtered via cursor key tracking (may rarely occur under extreme contention)
///
/// # Performance
///
/// The iterator allocates:
/// - `Vec<u8>` for each key (unavoidable for owned keys)
/// - `SmallVec` for layer stack (usually inline, up to 4 layers)
/// - No per-item allocation for value cloning (Arc refcount bump or Copy)
///
/// For higher performance, use the batch methods: [`for_each`](Self::for_each),
/// [`for_each_ref`](Self::for_each_ref), or [`for_each_intra_leaf_batch_ref`](Self::for_each_intra_leaf_batch_ref).
///
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
pub struct RangeIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // ========================================================================
    //  Forward iteration state
    // ========================================================================
    /// Memory reclamation guard.
    pub(super) guard: &'g LocalGuard<'a>,

    /// Current scan position (forward).
    pub(super) stack: ScanStackElement<L, S>,

    /// Parent layer stack for sublayer navigation (forward).
    pub(super) layer_stack: LayerStack<L>,

    /// Cursor tracking current key position (forward).
    pub(super) cursor_key: CursorKey,

    /// Current state machine state (forward).
    pub(super) state: ScanState,

    /// Captured snapshot for current entry (forward, if in Emit state).
    pub(super) snapshot: Option<ScanSnapshot<S>>,

    /// Tracks the output pointer from `initialize()`'s snapshot.
    ///
    /// This field is **only** used for the first entry case in `advance_no_alloc_ref`,
    /// where we convert the `ScanSnapshot` from `initialize()` to a raw pointer.
    ///
    /// For `LeafValueIndex<V>` (Copy types), `output_to_raw` allocates a Box
    /// to provide a stable pointer. This field tracks that allocation so we
    /// can clean it up when:
    /// - Advancing to the next entry (previous pointer no longer needed)
    /// - Dropping the iterator
    ///
    /// For `LeafValue<V>` (Arc types), this tracks the cloned Arc that needs
    /// to be decremented when no longer needed.
    ///
    /// # Why Only First Entry?
    ///
    /// After the first entry, `find_next_ptr` returns `ScanSnapshotPtr` with
    /// pointers directly into the leaf node (protected by guard), so no
    /// allocation tracking is needed. Only the `initialize()` snapshot path
    /// requires allocation tracking because it converts `S::Output` → raw pointer.
    pub(super) last_output_ptr: Option<*mut u8>,

    // ========================================================================
    //  Reverse iteration state (for DoubleEndedIterator)
    // ========================================================================
    /// Current scan position (backward).
    pub(super) back_stack: BackStackElement<L, S>,

    /// Parent layer stack for sublayer navigation (backward).
    pub(super) back_layer_stack: LayerStack<L>,

    /// Cursor tracking current key position (backward).
    pub(super) back_cursor_key: CursorKey,

    /// Reverse scan helper (tracks `upper_bound` state).
    pub(super) back_helper: ReverseScanHelper,

    /// Current state machine state (backward).
    pub(super) back_state: ScanStateBack,

    /// Captured snapshot for current entry (backward, if in Emit state).
    pub(super) back_snapshot: Option<ScanSnapshot<S>>,

    // ========================================================================
    //  Shared state
    // ========================================================================
    /// Tree root pointer (needed for back initialization).
    pub(super) tree_root: *const u8,

    /// Start bound for the range (needed for back bound checking).
    pub(super) start_bound: RangeBound<'a>,

    /// End bound for the range (needed for forward bound checking).
    pub(super) end_bound: RangeBound<'a>,

    /// Packed boolean flags.
    ///
    /// Contains: exhausted, initialized, `emit_equal`, `needs_duplicate_check`, `single_layer_mode`,
    /// `back_initialized`, `back_exhausted`, `back_emit_equal`
    pub(super) flags: IterFlags,

    /// Marker for lifetime and type parameter covariance.
    _marker: PhantomData<&'a A>,
}

impl<S, L, A> Debug for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("RangeIter")
            .field("exhausted", &self.flags.exhausted())
            .field("initialized", &self.flags.initialized())
            .field("state", &self.state)
            .field("back_exhausted", &self.flags.back_exhausted())
            .field("back_initialized", &self.flags.back_initialized())
            .field("back_state", &self.back_state)
            .finish_non_exhaustive()
    }
}

impl<'a, 'g, S, L, A> RangeIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Create a new range iterator.
    ///
    /// # Arguments
    ///
    /// - `tree`: The tree to iterate over
    /// - `start`: Start bound of the range
    /// - `end`: End bound of the range
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// A new iterator that will yield entries in the specified range.
    pub(crate) fn new(
        tree: &'a MassTreeGeneric<S, L, A>,
        start: RangeBound<'a>,
        end: RangeBound<'a>,
        guard: &'g LocalGuard<'a>,
    ) -> Self {
        // Convert start bound to cursor key and emit_equal flag
        let (start_key, emit_equal) = start.to_start_params();
        let cursor_key = CursorKey::from_slice(start_key);

        // Get root pointer
        let root = tree.load_root_ptr_generic(guard);

        // Create initial stack element
        let stack = ScanStackElement::new(root);

        // Determine if we can use single-layer fast path.
        // Single-layer mode is valid when both bounds fit within a single ikey.
        // If we encounter a layer pointer during iteration, we fall back.
        //
        // Note: Unbounded end bounds are considered "ok" because the fallback
        // mechanism (setting `single_layer_mode = false` on Down) handles
        // unexpected layer pointers gracefully.
        let single_layer_mode = {
            let start_ok = start_key.len() <= IKEY_SIZE;
            let end_ok = match &end {
                RangeBound::Unbounded => true,
                RangeBound::Included(k) | RangeBound::Excluded(k) => k.len() <= IKEY_SIZE,
            };
            start_ok && end_ok
        };

        // Determine back_emit_equal from end bound
        // For reverse iteration starting at end bound:
        // - Included: emit_equal = true (include the boundary key)
        // - Excluded: emit_equal = false (exclude the boundary key)
        // - Unbounded: emit_equal = true (start from max, emit everything)
        let back_emit_equal = match &end {
            RangeBound::Unbounded | RangeBound::Included(_) => true,
            RangeBound::Excluded(_) => false,
        };

        Self {
            // Forward iteration state
            guard,
            stack,
            layer_stack: ArrayVec::new(),
            cursor_key,
            state: ScanState::FindNext, // Will be set properly in first iteration
            snapshot: None,
            last_output_ptr: None,

            // Reverse iteration state (lazily initialized)
            back_stack: BackStackElement::new(root),
            back_layer_stack: ArrayVec::new(),
            back_cursor_key: CursorKey::for_reverse_scan(&end),
            back_helper: ReverseScanHelper::new(),
            back_state: ScanStateBack::FindPrev,
            back_snapshot: None,

            // Shared state
            tree_root: root,
            start_bound: start,
            end_bound: end,
            flags: IterFlags::with_both_bounds(emit_equal, single_layer_mode, back_emit_equal),

            _marker: PhantomData,
        }
    }

    /// Initialize the iterator (lazy initialization on first `next()` call).
    ///
    /// # State Machine Initialization
    ///
    /// This function handles the initial descent from the tree root to the
    /// starting position. It may descend through multiple layers if the
    /// start key has a layer pointer prefix.
    ///
    /// The loop handles:
    /// - `Down`: Descend into sublayer, shift cursor key
    /// - `Retry`: Re-traverse after version conflict
    /// - Other: Ready to iterate
    pub(super) fn initialize(&mut self) {
        if self.flags.initialized() {
            return;
        }
        self.flags.mark_initialized();

        // Handle empty tree
        if self.stack.root().is_null() {
            self.flags.mark_exhausted();
            return;
        }

        // Run initial descent loop
        loop {
            // Save parent ROOT before find_initial modifies it.
            // We need the original root for the LayerContext when descending,
            // since find_initial overwrites stack.root with the layer pointer.
            // Note: stack.leaf remains valid (points to parent leaf with layer pointer).
            let parent_root: *const u8 = self.stack.root();

            let (state, snapshot) = find_initial(
                self.stack.root(),
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                self.flags.emit_equal(),
                self.guard,
            );

            match state {
                ScanState::Down => {
                    // Start key descends into a sublayer.
                    // - parent_root: saved before find_initial modified stack.root
                    // - stack.leaf_ptr(): still points to the parent leaf (not modified)
                    // find_initial already set stack.root to the layer pointer.
                    self.layer_stack
                        .push(LayerContext::new(parent_root, self.stack.leaf_ptr()));

                    // If the key has more bytes (suffix), shift to use them.
                    // Otherwise, the prefix exactly matches the layer pointer's ikey,
                    // so we shift_clear to scan all keys in the sublayer.
                    if self.cursor_key.has_suffix() {
                        self.cursor_key.shift();
                    } else {
                        self.cursor_key.shift_clear();
                    }

                    // Continue loop to descend further into the new layer
                }

                ScanState::Retry => {
                    // Version conflict, retry from current root
                }

                _ => {
                    // Ready to iterate (Emit, FindNext, Up)
                    self.state = state;
                    self.snapshot = snapshot;
                    break;
                }
            }
        }
    }

    /// Advance the iterator state machine.
    #[inline]
    fn advance(&mut self) -> Option<ScanEntry<S::Output>> {
        loop {
            match self.state {
                ScanState::Emit => {
                    // Check end bound
                    let key = self.cursor_key.full_key();

                    if !self.end_bound.contains(key) {
                        self.flags.mark_exhausted();
                        return None;
                    }

                    // Check meeting condition: front caught up to back
                    if self.flags.back_initialized() && !self.flags.back_exhausted() {
                        let back_key = self.back_cursor_key.full_key();

                        if key >= back_key {
                            self.flags.mark_exhausted();
                            self.flags.mark_back_exhausted();
                            return None;
                        }
                    }

                    // Take snapshot (should always be Some when in Emit state)
                    debug_assert!(
                        self.snapshot.is_some(),
                        "Emit state entered without snapshot - state machine bug"
                    );

                    let snapshot = self.snapshot.take()?;

                    // Build entry
                    let entry = ScanEntry::new(key.to_vec(), snapshot.value);

                    // Transition to FindNext
                    self.state = ScanState::FindNext;

                    return Some(entry);
                }

                ScanState::FindNext => {
                    // OPTIMIZATION: Only check for duplicates after a Retry,
                    // not in normal forward iteration
                    let (new_state, snapshot) = if self.flags.needs_duplicate_check() {
                        self.flags.clear_duplicate_check();
                        find_next_with_duplicate_check(
                            &mut self.stack,
                            &mut self.cursor_key,
                            &mut self.layer_stack,
                            self.guard,
                        )
                    } else {
                        find_next(
                            &mut self.stack,
                            &mut self.cursor_key,
                            &mut self.layer_stack,
                            self.guard,
                        )
                    };

                    self.state = new_state;
                    self.snapshot = snapshot;
                }

                ScanState::Down => {
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    // After layer descent, we need duplicate check
                    self.flags.require_duplicate_check();
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        // No parent layer, scan complete
                        self.flags.mark_exhausted();
                        return None;
                    }

                    self.state = ScanState::FindNext;

                    // After layer ascent, we need duplicate check
                    self.flags.require_duplicate_check();
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    // After retry, we need duplicate check on next FindNext
                    self.flags.require_duplicate_check();
                }
            }
        }
    }

    // ========================================================================
    //  Reverse Iteration (DoubleEndedIterator support)
    // ========================================================================

    /// Initialize the back cursor for reverse iteration.
    ///
    /// Called lazily on the first `next_back()` call.
    pub(super) fn initialize_back(&mut self) {
        if self.flags.back_initialized() {
            return;
        }
        self.flags.mark_back_initialized();

        // Handle empty tree
        if self.tree_root.is_null() {
            self.flags.mark_back_exhausted();
            return;
        }

        // For unbounded end, set upper_bound so lower_reverse returns last slot
        if self.end_bound.is_unbounded() {
            self.back_helper.upper_bound = true;
        }

        // Run initial reverse descent loop
        loop {
            let (state, snapshot) = ReverseScan::find_initial_reverse(
                self.back_stack.get_root(),
                &mut self.back_stack,
                &mut self.back_cursor_key,
                &mut self.back_layer_stack,
                self.flags.back_emit_equal(),
                &mut self.back_helper,
                self.guard,
            );

            match state {
                ScanStateBack::Down => {
                    // Descend into sublayer
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);

                    // Continue loop to call find_initial_reverse with new layer root
                }

                ScanStateBack::Retry => {
                    // Version conflict, retry
                }

                _ => {
                    // Ready to iterate (Emit, FindPrev, Up)
                    self.back_state = state;
                    self.back_snapshot = snapshot;

                    // Clear upper_bound after first successful positioning
                    if self.back_state.is_emit() {
                        self.back_helper.mark_key_complete();
                    }

                    break;
                }
            }
        }
    }

    /// Advance the back iterator state machine.
    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "State machine benefits from unified logic"
    )]
    fn advance_back(&mut self) -> Option<ScanEntry<S::Output>> {
        loop {
            match self.back_state {
                ScanStateBack::Emit => {
                    // Check start bound (reverse of end bound check)
                    let key: &[u8] = self.back_cursor_key.full_key();

                    if !self.start_bound.contains_reverse(key) {
                        self.flags.mark_back_exhausted();
                        return None;
                    }

                    // Check meeting condition: back caught up to front
                    if self.flags.initialized() && !self.flags.exhausted() {
                        let front_key: &[u8] = self.cursor_key.full_key();

                        if key <= front_key {
                            self.flags.mark_back_exhausted();
                            self.flags.mark_exhausted();

                            return None;
                        }
                    }

                    // Take snapshot
                    let snapshot: ScanSnapshot<S> = self.back_snapshot.take()?;

                    // Build entry
                    let entry = ScanEntry::new(key.to_vec(), snapshot.value);

                    // CRITICAL: Clear upper_bound on every emission
                    self.back_helper.mark_key_complete();

                    // Transition to FindPrev
                    self.back_state = ScanStateBack::FindPrev;

                    return Some(entry);
                }

                ScanStateBack::FindPrev => {
                    // OPTIMIZATION: Only check for duplicates after a Retry,
                    // not in normal reverse iteration
                    let (new_state, snapshot) = if self.flags.back_needs_duplicate_check() {
                        self.flags.clear_back_duplicate_check();

                        ReverseScan::find_prev_with_duplicate_check(
                            &mut self.back_stack,
                            &mut self.back_cursor_key,
                            &mut self.back_layer_stack,
                            &mut self.back_helper,
                            self.guard,
                        )
                    } else {
                        ReverseScan::find_prev(
                            &mut self.back_stack,
                            &mut self.back_cursor_key,
                            &mut self.back_layer_stack,
                            &mut self.back_helper,
                            self.guard,
                        )
                    };

                    self.back_state = new_state;
                    self.back_snapshot = snapshot;
                }

                ScanStateBack::Down => {
                    // Handle sublayer descent
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);

                    // Call find_initial_reverse for the sublayer
                    let (state, snapshot) = ReverseScan::find_initial_reverse(
                        self.back_stack.get_root(),
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_layer_stack,
                        false, // emit_equal: false for scan-discovered descent
                        &mut self.back_helper,
                        self.guard,
                    );

                    self.back_state = state;
                    self.back_snapshot = snapshot;

                    // After layer descent, we need duplicate check
                    self.flags.require_back_duplicate_check();
                }

                ScanStateBack::Up => {
                    if !ReverseScan::handle_up_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_layer_stack,
                        &mut self.back_helper,
                        self.guard,
                    ) {
                        // No parent layer, scan complete
                        self.flags.mark_back_exhausted();

                        return None;
                    }

                    self.back_state = ScanStateBack::FindPrev;

                    // After layer ascent, we need duplicate check
                    self.flags.require_back_duplicate_check();
                }

                ScanStateBack::Retry => {
                    let (new_state, _) = ReverseScan::reposition_back(
                        &mut self.back_stack,
                        &mut self.back_cursor_key,
                        &mut self.back_helper,
                        self.guard,
                    );

                    self.back_state = new_state;

                    // After retry, we need duplicate check on next FindPrev
                    self.flags.require_back_duplicate_check();
                }
            }
        }
    }

    /// Convert to a keys-only iterator.
    pub const fn keys(self) -> KeysIter<'a, 'g, S, L, A> {
        KeysIter { inner: self }
    }

    /// Convert to a values-only iterator.
    pub const fn values(self) -> ValuesIter<'a, 'g, S, L, A> {
        ValuesIter { inner: self }
    }
}

// ============================================================================
//  Iterator Trait Implementations
// ============================================================================

impl<S, L, A> Iterator for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = ScanEntry<S::Output>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.flags.exhausted() {
            return None;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();

            if self.flags.exhausted() {
                return None;
            }
        }

        // Meeting detection: if back is initialized, check if we've crossed
        if self.flags.back_initialized() && !self.flags.back_exhausted() {
            let front_key: &[u8] = self.cursor_key.full_key();
            let back_key: &[u8] = self.back_cursor_key.full_key();

            if front_key >= back_key {
                // Mark both as exhausted when they meet
                self.flags.mark_exhausted();
                self.flags.mark_back_exhausted();
                return None;
            }
        }

        self.advance()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.flags.exhausted() {
            (0, Some(0))
        } else {
            // We can't know the exact count without iterating
            (0, None)
        }
    }
}

impl<S, L, A> DoubleEndedIterator for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // Check if back cursor is exhausted
        if self.flags.back_exhausted() {
            return None;
        }

        // Lazy initialization of back cursor
        if !self.flags.back_initialized() {
            self.initialize_back();

            if self.flags.back_exhausted() {
                return None;
            }
        }

        // Check meeting condition: if front has advanced past where back would be
        if self.flags.initialized() && !self.flags.exhausted() {
            let front_key: &[u8] = self.cursor_key.full_key();
            let back_key: &[u8] = self.back_cursor_key.full_key();

            if back_key <= front_key {
                // Mark both as exhausted when they meet
                self.flags.mark_back_exhausted();
                self.flags.mark_exhausted();

                return None;
            }
        }

        self.advance_back()
    }
}

impl<S, L, A> FusedIterator for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
}

impl<S, L, A> Drop for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn drop(&mut self) {
        // Clean up any outstanding output pointer from advance_no_alloc_ref.
        // This pointer was created by output_to_raw and must be freed.
        if let Some(ptr) = self.last_output_ptr.take() {
            // SAFETY: ptr was created by S::output_to_raw and has not been cleaned up yet.
            // We only create one pointer at a time and track it in last_output_ptr.
            unsafe { S::cleanup_output_raw(ptr) };
        }
    }
}
