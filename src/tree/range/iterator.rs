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

use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;
use crate::key::IKEY_SIZE;

use crate::policy::LeafPolicy;
use crate::tree::MassTreeGeneric;

#[cfg(debug_assertions)]
use super::cursor_key::CursorDebugState;

use super::cursor_key::CursorKey;
use super::find::{
    find_initial, find_next, find_next_with_duplicate_check, find_retry, handle_down, handle_up,
};
use super::find_rev::{ReverseScan, find_prev_single_layer};
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
/// - `LayerStack` for layer stack (inline up to 6 layers, heap spillover for deeper keys)
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
pub struct RangeIter<'a, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    // ========================================================================
    //  Forward iteration state
    // ========================================================================
    /// Memory reclamation guard.
    pub(super) guard: &'g LocalGuard<'a>,

    /// Current scan position (forward).
    pub(super) stack: ScanStackElement<P>,

    /// Parent layer stack for sublayer navigation (forward).
    pub(super) layer_stack: LayerStack<P>,

    /// Cursor tracking current key position (forward).
    pub(super) cursor_key: CursorKey,

    /// Current state machine state (forward).
    pub(super) state: ScanState,

    /// Captured snapshot for current entry (forward, if in Emit state).
    pub(super) snapshot: Option<ScanSnapshot<P>>,

    /// Tracks the output from `initialize()`'s snapshot.
    ///
    /// This field is **only** used for the first entry case in `advance_no_alloc_ref`,
    /// where we keep the `P::Output` alive so that `output_as_ref` can return
    /// a borrowed `&P::Value` from it.
    ///
    /// For `BoxPolicy<V>` (Arc types), this tracks the cloned Arc that needs
    /// to be decremented when no longer needed.
    ///
    /// # Why Only First Entry?
    ///
    /// After the first entry, `find_next_ptr` returns `ScanSnapshotPtr` with
    /// pointers directly into the leaf node (protected by guard), so no
    /// allocation tracking is needed. Only the `initialize()` snapshot path
    /// requires allocation tracking because it holds the `P::Output` alive.
    pub(super) last_output: Option<P::Output>,

    // ========================================================================
    //  Reverse iteration state (for DoubleEndedIterator)
    // ========================================================================
    /// Current scan position (backward).
    pub(super) back_stack: BackStackElement<P>,

    /// Parent layer stack for sublayer navigation (backward).
    pub(super) back_layer_stack: LayerStack<P>,

    /// Cursor tracking current key position (backward).
    pub(super) back_cursor_key: CursorKey,

    /// Reverse scan helper (tracks `upper_bound` state).
    pub(super) back_helper: ReverseScanHelper,

    /// Current state machine state (backward).
    pub(super) back_state: ScanStateBack,

    /// Captured snapshot for current entry (backward, if in Emit state).
    pub(super) back_snapshot: Option<ScanSnapshot<P>>,

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
    /// `back_initialized`, `back_exhausted`, `back_emit_equal`, `forward_only`
    pub(super) flags: IterFlags,

    // ========================================================================
    //  Debug-only fields for ordering violation detection
    // ========================================================================
    /// Last emitted key for forward iteration (debug builds only).
    ///
    /// Used to assert that keys are emitted in strictly increasing order.
    /// This catches ordering violations at the exact point they occur.
    #[cfg(debug_assertions)]
    pub(super) debug_last_emitted_key: Option<Vec<u8>>,

    /// Last emitted key for backward iteration (debug builds only).
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(super) debug_last_emitted_key_back: Option<Vec<u8>>,

    /// Cursor state at last emission (debug builds only).
    ///
    /// Captures the full cursor state when a key is emitted, useful for
    /// diagnosing what went wrong when an ordering violation is detected.
    #[cfg(debug_assertions)]
    pub(super) debug_last_cursor_state: Option<CursorDebugState>,

    /// Ring buffer of recent state transitions for debugging (debug builds only).
    ///
    /// Stores the last N state transitions (Retry, Down, Up) to help diagnose
    /// what happened before an ordering violation.
    #[cfg(debug_assertions)]
    pub(super) debug_transition_history: Vec<String>,

    /// Marker for lifetime and type parameter covariance.
    _marker: PhantomData<&'a A>,
}

impl<P, A> Debug for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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

impl<'a, 'g, P, A> RangeIter<'a, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Create a forward-only range iterator (no backward state initialization).
    ///
    /// This is cheaper than [`new`](Self::new) because it skips initializing
    /// backward iteration state (~300 bytes of `CursorKey`, `BackStackElement`,
    /// `ReverseScanHelper`, etc.). Use this when you know only forward batch
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
        let stack = ScanStackElement::new(root);

        let single_layer_mode = {
            let start_ok = start_key.len() <= IKEY_SIZE;
            let end_ok = match &end {
                RangeBound::Unbounded => true,
                RangeBound::Included(k) | RangeBound::Excluded(k) => k.len() <= IKEY_SIZE,
            };
            start_ok && end_ok
        };

        Self {
            // Forward iteration state (fully initialized)
            guard,
            stack,
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanState::FindNext,
            snapshot: None,
            last_output: None,

            // Backward state: cheap defaults (never accessed by forward batch methods)
            back_stack: BackStackElement::default(),
            back_layer_stack: LayerStack::new(),
            back_cursor_key: CursorKey::empty(),
            back_helper: ReverseScanHelper::new(),
            back_state: ScanStateBack::FindPrev,
            back_snapshot: None,

            // Shared state
            tree_root: root,
            start_bound: start,
            end_bound: end,
            flags: IterFlags::with_forward_only(emit_equal, single_layer_mode),

            #[cfg(debug_assertions)]
            debug_last_emitted_key: None,
            #[cfg(debug_assertions)]
            debug_last_emitted_key_back: None,
            #[cfg(debug_assertions)]
            debug_last_cursor_state: None,
            #[cfg(debug_assertions)]
            debug_transition_history: Vec::with_capacity(32),

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
        let stack = ScanStackElement::new(layer_root);

        Self {
            // Forward iteration state (fully initialized except lazy initialize())
            guard,
            stack,
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanState::FindNext,
            snapshot: None,
            last_output: None,

            // Backward state: cheap defaults (forward path)
            back_stack: BackStackElement::default(),
            back_layer_stack: LayerStack::new(),
            back_cursor_key: CursorKey::empty(),
            back_helper: ReverseScanHelper::new(),
            back_state: ScanStateBack::FindPrev,
            back_snapshot: None,

            // Shared state
            tree_root: layer_root,
            start_bound,
            end_bound,
            flags: IterFlags::with_forward_only(true, false),

            #[cfg(debug_assertions)]
            debug_last_emitted_key: None,
            #[cfg(debug_assertions)]
            debug_last_emitted_key_back: None,
            #[cfg(debug_assertions)]
            debug_last_cursor_state: None,
            #[cfg(debug_assertions)]
            debug_transition_history: Vec::with_capacity(32),

            _marker: PhantomData,
        }
    }

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
        tree: &'a MassTreeGeneric<P, A>,
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
            layer_stack: LayerStack::new(),
            cursor_key,
            state: ScanState::FindNext, // Will be set properly in first iteration
            snapshot: None,
            last_output: None,

            // Reverse iteration state (lazily initialized)
            back_stack: BackStackElement::new(root),
            back_layer_stack: LayerStack::new(),
            back_cursor_key: CursorKey::for_reverse_scan(&end),
            back_helper: ReverseScanHelper::new(),
            back_state: ScanStateBack::FindPrev,
            back_snapshot: None,

            // Shared state
            tree_root: root,
            start_bound: start,
            end_bound: end,
            flags: IterFlags::with_both_bounds(emit_equal, single_layer_mode, back_emit_equal),

            // Debug-only fields for ordering violation detection
            #[cfg(debug_assertions)]
            debug_last_emitted_key: None,

            #[cfg(debug_assertions)]
            debug_last_emitted_key_back: None,

            #[cfg(debug_assertions)]
            debug_last_cursor_state: None,

            #[cfg(debug_assertions)]
            debug_transition_history: Vec::with_capacity(32),

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
                    // Entering a sublayer invalidates single-layer assumptions.
                    self.flags.disable_single_layer_mode();

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
    ///
    /// Uses a fused FindNext+Emit optimization: when `find_next()` returns `Emit`,
    /// the emit logic is processed inline rather than looping back through the
    /// state machine. This eliminates one loop iteration per entry.
    ///
    /// # Performance
    ///
    /// The fused approach reuses existing `find_next()` logic (no code duplication)
    /// while reducing per-entry overhead by ~15-20% on dense scans.
    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "State machine with fused optimization and debug instrumentation"
    )]
    fn advance(&mut self) -> Option<ScanEntry<P::Output>> {
        loop {
            match self.state {
                ScanState::Emit => {
                    // Check end bound
                    // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                    let key = unsafe { self.cursor_key.full_key_unchecked() };

                    if !self.end_bound.contains(key) {
                        self.flags.mark_exhausted();
                        return None;
                    }

                    // Check meeting condition: front caught up to back
                    if self.flags.back_initialized() && !self.flags.back_exhausted() {
                        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                        let back_key = unsafe { self.back_cursor_key.full_key_unchecked() };

                        if key >= back_key {
                            self.flags.mark_exhausted();
                            self.flags.mark_back_exhausted();
                            return None;
                        }
                    }

                    // DEBUG: Assert strict ordering - key must be greater than last emitted
                    #[cfg(debug_assertions)]
                    #[allow(
                        clippy::panic,
                        reason = "Intentional panic for debug-only ordering violation detection"
                    )]
                    {
                        if let Some(ref last_key) = self.debug_last_emitted_key
                            && key <= last_key.as_slice()
                        {
                            // Capture current state for diagnosis
                            let current_state = self.cursor_key.debug_state();
                            let last_state = self.debug_last_cursor_state.as_ref();

                            // Log detailed information before panicking
                            eprintln!("\n=== ORDERING VIOLATION DETECTED ===");
                            eprintln!("Current key:  {:?}", String::from_utf8_lossy(key));
                            eprintln!("Last key:     {:?}", String::from_utf8_lossy(last_key));
                            eprintln!("Current key bytes: {key:?}");
                            eprintln!("Last key bytes:    {last_key:?}");
                            eprintln!("Current cursor: {current_state}");
                            if let Some(last) = last_state {
                                eprintln!("Last cursor:    {last}");
                            }
                            eprintln!("=== END ORDERING VIOLATION ===\n");

                            panic!(
                                "Scan ordering violation: emitted key {:?} is not > last emitted key {:?}",
                                String::from_utf8_lossy(key),
                                String::from_utf8_lossy(last_key)
                            );
                        }

                        // Update tracking state
                        self.debug_last_emitted_key = Some(key.to_vec());
                        self.debug_last_cursor_state = Some(self.cursor_key.debug_state());
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
                    // Call find_next (with or without duplicate check)
                    let (new_state, snapshot) = if self.flags.needs_duplicate_check() {
                        self.flags.clear_duplicate_check();
                        find_next_with_duplicate_check(
                            &mut self.stack,
                            &mut self.cursor_key,
                            &mut self.layer_stack,
                            self.guard,
                        )
                        .into_parts()
                    } else {
                        find_next(
                            &mut self.stack,
                            &mut self.cursor_key,
                            &mut self.layer_stack,
                            self.guard,
                        )
                        .into_parts()
                    };

                    // FUSED OPTIMIZATION: If find_next returns Emit, process inline
                    // instead of looping back. This eliminates one state machine
                    // iteration per entry (the common case).
                    if new_state == ScanState::Emit {
                        // Emit logic inline (same as Emit arm above)
                        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                        let key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                        if !self.end_bound.contains(key) {
                            self.flags.mark_exhausted();
                            return None;
                        }

                        if self.flags.back_initialized() && !self.flags.back_exhausted() {
                            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                            let back_key: &[u8] =
                                unsafe { self.back_cursor_key.full_key_unchecked() };

                            if key >= back_key {
                                self.flags.mark_exhausted();
                                self.flags.mark_back_exhausted();
                                return None;
                            }
                        }

                        #[cfg(debug_assertions)]
                        #[allow(
                            clippy::panic,
                            reason = "Intentional panic for debug-only ordering violation detection"
                        )]
                        {
                            if let Some(ref last_key) = self.debug_last_emitted_key
                                && key <= last_key.as_slice()
                            {
                                let current_state: CursorDebugState = self.cursor_key.debug_state();
                                let last_state: Option<&CursorDebugState> =
                                    self.debug_last_cursor_state.as_ref();

                                eprintln!("\n=== ORDERING VIOLATION DETECTED (FUSED) ===");
                                eprintln!("Current key:  {:?}", String::from_utf8_lossy(key));
                                eprintln!("Last key:     {:?}", String::from_utf8_lossy(last_key));
                                eprintln!("Current key bytes: {key:?}");
                                eprintln!("Last key bytes:    {last_key:?}");
                                eprintln!("Current cursor: {current_state}");
                                if let Some(last) = last_state {
                                    eprintln!("Last cursor:    {last}");
                                }
                                eprintln!("=== END ORDERING VIOLATION ===\n");

                                panic!(
                                    "Scan ordering violation: emitted key {:?} is not > last emitted key {:?}",
                                    String::from_utf8_lossy(key),
                                    String::from_utf8_lossy(last_key)
                                );
                            }

                            self.debug_last_emitted_key = Some(key.to_vec());
                            self.debug_last_cursor_state = Some(self.cursor_key.debug_state());
                        }

                        // snapshot is guaranteed Some when new_state == Emit
                        // (find_next only returns Emit with Some(snapshot))
                        let snapshot: ScanSnapshot<P> = snapshot?;
                        return Some(ScanEntry::new(key.to_vec(), snapshot.value));
                    }

                    // Not Emit - continue state machine normally
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

        // Forward-only constructor path: lazily materialize reverse state
        // when callers request `next_back`/`.rev()`.
        let mut back_emit_equal = self.flags.back_emit_equal();
        if self.flags.forward_only() {
            back_emit_equal = matches!(
                self.end_bound,
                RangeBound::Unbounded | RangeBound::Included(_)
            );
            self.back_stack = BackStackElement::new(self.tree_root);
            self.back_layer_stack.clear();
            self.back_cursor_key = CursorKey::for_reverse_scan(&self.end_bound);
            self.back_helper = ReverseScanHelper::new();
            self.back_state = ScanStateBack::FindPrev;
            self.back_snapshot = None;
            self.flags.clear_forward_only();
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
                back_emit_equal,
                &mut self.back_helper,
                self.guard,
            );

            match state {
                ScanStateBack::Down => {
                    // Entering a sublayer invalidates single-layer assumptions.
                    self.flags.disable_single_layer_mode();

                    // Descend into sublayer
                    ReverseScan::handle_down_back(&mut self.back_cursor_key, &mut self.back_helper);

                    // Continue loop to call find_initial_reverse with new layer root
                }

                ScanStateBack::Retry => {
                    // Version conflict, retry
                }

                _ => {
                    // `find_initial_reverse` performs iterative layer descent internally.
                    // If it descended, `back_layer_stack` is non-empty even though we
                    // did not observe an explicit `Down` state here.
                    if !self.back_layer_stack.is_empty() {
                        self.flags.disable_single_layer_mode();
                    }

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
    fn advance_back(&mut self) -> Option<ScanEntry<P::Output>> {
        loop {
            match self.back_state {
                ScanStateBack::Emit => {
                    // Check start bound (reverse of end bound check)
                    // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                    let key: &[u8] = unsafe { self.back_cursor_key.full_key_unchecked() };

                    if !self.start_bound.contains_reverse(key) {
                        self.flags.mark_back_exhausted();
                        return None;
                    }

                    // Check meeting condition: back caught up to front
                    if self.flags.initialized() && !self.flags.exhausted() {
                        // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
                        let front_key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };

                        if key <= front_key {
                            self.flags.mark_back_exhausted();
                            self.flags.mark_exhausted();

                            return None;
                        }
                    }

                    // Take snapshot
                    let snapshot: ScanSnapshot<P> = self.back_snapshot.take()?;

                    // Build entry
                    let entry = ScanEntry::new(key.to_vec(), snapshot.value);

                    // CRITICAL: Clear upper_bound on every emission
                    self.back_helper.mark_key_complete();

                    // Transition to FindPrev
                    self.back_state = ScanStateBack::FindPrev;

                    return Some(entry);
                }

                ScanStateBack::FindPrev => {
                    // ================================================================
                    // Single-layer fast path (keys ≤ 8 bytes)
                    // ================================================================
                    if self.flags.single_layer_mode() {
                        let needs_dup_check = self.flags.back_needs_duplicate_check();
                        if needs_dup_check {
                            self.flags.clear_back_duplicate_check();
                        }

                        let (new_state, snapshot) = find_prev_single_layer(
                            &mut self.back_stack,
                            &mut self.back_cursor_key,
                            &mut self.back_helper,
                            self.guard,
                            needs_dup_check,
                        );

                        self.back_state = new_state;
                        self.back_snapshot = snapshot;

                        match new_state {
                            ScanStateBack::FindPrev => {
                                // Check for exhausted (null stack in single-layer = done)
                                if self.back_stack.get_leaf_ptr().is_null() {
                                    self.flags.mark_back_exhausted();
                                    return None;
                                }
                            }
                            ScanStateBack::Down => {
                                // Encountered suffix key or layer pointer - fall back to
                                // multi-layer mode and re-process this slot with find_prev
                                self.flags.disable_single_layer_mode();
                                self.back_state = ScanStateBack::FindPrev;
                                // Don't require duplicate check - slot hasn't been emitted
                            }
                            _ => {}
                        }

                        continue;
                    }

                    // ================================================================
                    // Multi-layer path (full feature set)
                    // ================================================================
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
                    // Disable single-layer mode when descending into sublayer
                    self.flags.disable_single_layer_mode();

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
                        &self.back_cursor_key,
                        self.back_helper,
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
    pub const fn keys(self) -> KeysIter<'a, 'g, P, A> {
        KeysIter { inner: self }
    }

    /// Convert to a values-only iterator.
    pub const fn values(self) -> ValuesIter<'a, 'g, P, A> {
        ValuesIter { inner: self }
    }

    /// Assert that keys are emitted in strictly increasing order (debug builds only).
    ///
    /// This catches ordering violations at the exact point they occur, providing
    /// detailed diagnostic information about the cursor state.
    #[cfg(debug_assertions)]
    #[inline]
    #[allow(
        clippy::panic,
        reason = "Intentional panic for debug-only ordering violation detection"
    )]
    pub(super) fn assert_ordering(&mut self, key: &[u8]) {
        if let Some(ref last_key) = self.debug_last_emitted_key
            && key <= last_key.as_slice()
        {
            // Capture current state for diagnosis
            let current_state: CursorDebugState = self.cursor_key.debug_state();
            let last_state: Option<&CursorDebugState> = self.debug_last_cursor_state.as_ref();

            // Log detailed information before panicking
            eprintln!("\n=== ORDERING VIOLATION DETECTED (batch path) ===");
            eprintln!("Current key:  {:?}", String::from_utf8_lossy(key));
            eprintln!("Last key:     {:?}", String::from_utf8_lossy(last_key));
            eprintln!("Current key bytes: {key:?}");
            eprintln!("Last key bytes:    {last_key:?}");
            eprintln!("Current cursor: {current_state}");
            if let Some(last) = last_state {
                eprintln!("Last cursor:    {last}");
            }

            // Print recent state transitions
            eprintln!("\n--- Recent state transitions ---");
            for (i, transition) in self.debug_transition_history.iter().enumerate() {
                eprintln!("[{i}] {transition}");
            }
            eprintln!("--- End transitions ---");
            eprintln!("=== END ORDERING VIOLATION ===\n");

            panic!(
                "Scan ordering violation: emitted key {:?} is not > last emitted key {:?}",
                String::from_utf8_lossy(key),
                String::from_utf8_lossy(last_key)
            );
        }

        // Update tracking state and record emission
        self.debug_last_emitted_key = Some(key.to_vec());
        self.debug_last_cursor_state = Some(self.cursor_key.debug_state());
        self.record_transition(format!(
            "EMIT: {:?} cursor={}",
            String::from_utf8_lossy(key),
            self.cursor_key.debug_state()
        ));
    }

    /// Record a state transition for debugging (debug builds only).
    #[cfg(debug_assertions)]
    #[inline]
    pub(super) fn record_transition(&mut self, description: String) {
        // Keep last 32 transitions
        if self.debug_transition_history.len() >= 32 {
            self.debug_transition_history.remove(0);
        }
        self.debug_transition_history.push(description);
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
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let front_key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let back_key: &[u8] = unsafe { self.back_cursor_key.full_key_unchecked() };

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

impl<P, A> DoubleEndedIterator for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let front_key: &[u8] = unsafe { self.cursor_key.full_key_unchecked() };
            // SAFETY: CursorKey invariant guarantees offset + len <= MAX_KEY_LENGTH
            let back_key: &[u8] = unsafe { self.back_cursor_key.full_key_unchecked() };

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

impl<P, A> FusedIterator for RangeIter<'_, '_, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
}

// No manual Drop needed — `last_output: Option<P::Output>` drops automatically.
