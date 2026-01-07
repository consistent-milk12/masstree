//! Filepath: src/tree/range/iterator.rs
//!
//! Range iterator implementation.
//!
//! Provides [`RangeIter`], an iterator over key-value pairs in lexicographic order.
//! The iterator yields [`ScanEntry`] items containing owned keys and values.
//!
//! # State Machine
//!
//! The iterator is implemented as an explicit state machine with the following states:
//!
//! ```text
//!                     ┌────────────────────────────────────────────┐
//!                     │                                            │
//!                     ▼                                            │
//!               ┌──────────┐                                       │
//!          ┌───▶│  Emit    │──────── yield entry ─────────────────▶│
//!          │    └──────────┘                                       │
//!          │         │                                             │
//!          │         │ advance ki                                  │
//!          │         ▼                                             │
//!          │    ┌──────────┐                                       │
//!          ├────│ FindNext │◀──────────────────────────────────────┤
//!          │    └────┬─────┘                                       │
//!          │         │                                             │
//!          │    ┌────┼────────┬────────────┬────────────┐          │
//!          │    ▼    │        ▼            ▼            ▼          │
//!          │ found   │     layer_ptr    exhausted   version_fail   │
//!          │    │    │        │            │            │          │
//!          │    │    │   ┌────┴────┐  ┌────┴────┐       │          │
//!          │    │    │   │  Down   │  │   Up    │       │          │
//!          │    │    │   └────┬────┘  └────┬────┘       │          │
//!          │    │    │        │            │            │          │
//!          │    │    │ shift_clear()   unshift()        │          │
//!          │    │    │        │            │            │          │
//!          │    │    │        └─────┬──────┴────────────┘          │
//!          │    │    │              ▼                              │
//!          │    │    │       ┌──────────┐                          │
//!          │    │    └──────▶│  Retry   │                          │
//!          │    │            └────┬─────┘                          │
//!          │    │                 │                                │
//!          │    │           find_retry()                           │
//!          │    │                 │                                │
//!          └────┴─────────────────┴────────────────────────────────┘
//! ```
//!
//! ## State Descriptions
//!
//! - **`Emit`**: Ready to yield a key-value pair to the caller
//! - **`FindNext`**: Searching for the next entry in the current leaf
//! - **`Down`**: Descending into a sublayer (encountered a layer pointer)
//! - **`Up`**: Ascending to parent layer (current layer exhausted)
//! - **`Retry`**: Repositioning after version conflict or layer transition
//!
//! ## Key Transitions
//!
//! | From | To | Trigger |
//! |------|-----|---------|
//! | `FindNext` | `Emit` | Found valid entry |
//! | `FindNext` | `Down` | Encountered layer pointer |
//! | `FindNext` | `Up` | Leaf exhausted, no next leaf |
//! | `FindNext` | `Retry` | Version changed |
//! | `Emit` | `FindNext` | Entry yielded |
//! | `Down` | `Retry` | After `shift_clear()` |
//! | `Up` | `FindNext` | After `unshift()` and state refresh |
//! | `Retry` | `FindNext` | After `find_retry()` repositioning |
//!
//! # Usage
//!
//! ```ignore
//! let guard = tree.guard();
//!
//! // Full iteration
//! for entry in tree.iter(&guard) {
//!     println!("{:?} -> {:?}", entry.key, entry.value);
//! }
//!
//! // Range iteration
//! for entry in tree.range(
//!     RangeBound::Included(b"start"),
//!     RangeBound::Excluded(b"end"),
//!     &guard
//! ) {
//!     // ...
//! }
//!```

use std::marker::PhantomData;
use std::ops::Bound;

use arrayvec::ArrayVec;
use seize::LocalGuard;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::key::IKEY_SIZE;
use crate::leaf_trait::{LayerCapableLeaf, TreeLeafNode};
use crate::slot::ValueSlot;
use crate::tree::MassTreeGeneric;

use super::cursor_key::CursorKey;
use super::find::{
    find_initial, find_next, find_next_ptr, find_next_single_layer_ptr,
    find_next_with_duplicate_check, find_next_with_duplicate_check_ptr, find_retry, handle_down,
    handle_up,
};
use super::scan_state::{LayerContext, LayerStack, ScanSnapshot, ScanStackElement, ScanState};

// ============================================================================
//  RangeBound
// ============================================================================

/// Range bound for scan operations.
///
/// Specifies the start or end of a key range for scanning.
///
/// # Variants
///
/// - [`Unbounded`](RangeBound::Unbounded): No bound (all keys)
/// - [`Included`](RangeBound::Included): Include the specified key
/// - [`Excluded`](RangeBound::Excluded): Exclude the specified key
///
/// # Example
///
/// ```ignore
/// // Scan from "aaa" (inclusive) to "zzz" (exclusive)
/// let start = RangeBound::Included(b"aaa");
/// let end = RangeBound::Excluded(b"zzz");
///
/// for entry in tree.range(start, end, &guard) {
///     // entry.key will be >= "aaa" and < "zzz"
/// }
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeBound<'a> {
    /// No bound - start from minimum or continue to maximum.
    Unbounded,

    /// Inclusive bound - include the specified key.
    Included(&'a [u8]),

    /// Exclusive bound - exclude the specified key.
    Excluded(&'a [u8]),
}

impl<'a> RangeBound<'a> {
    /// Check if a key is within this bound (for end bound checking).
    ///
    /// For end bounds:
    /// - `Unbounded`: all keys are within
    /// - `Included(k)`: keys <= k are within
    /// - `Excluded(k)`: keys < k are within
    ///
    /// # Arguments
    ///
    /// - `key`: The key to check
    ///
    /// # Returns
    ///
    /// `true` if the key is within the bound.
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, key: &[u8]) -> bool {
        match self {
            RangeBound::Unbounded => true,
            RangeBound::Included(bound) => key <= *bound,
            RangeBound::Excluded(bound) => key < *bound,
        }
    }

    /// Convert to `(start_key, emit_equal)` parameters for `find_initial`.
    ///
    /// For start bounds:
    /// - `Unbounded`: empty key, `emit_equal = true`
    /// - `Included(k)`: key k, `emit_equal = true`
    /// - `Excluded(k)`: key k, `emit_equal = false`
    ///
    /// # Returns
    ///
    /// Tuple of (key bytes, `emit_equal` flag).
    #[must_use]
    #[inline(always)]
    pub const fn to_start_params(&self) -> (&'a [u8], bool) {
        match self {
            RangeBound::Unbounded => (&[], true),
            RangeBound::Included(k) => (*k, true),
            RangeBound::Excluded(k) => (*k, false),
        }
    }

    /// Check if this is an unbounded bound.
    #[must_use]
    #[inline(always)]
    pub const fn is_unbounded(&self) -> bool {
        matches!(self, RangeBound::Unbounded)
    }

    /// Get the bound key if this is a bounded bound.
    ///
    /// # Note
    ///
    /// This method is provided for API completeness and may be useful for
    /// external callers who need to inspect bounds programmatically.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> Option<&'a [u8]> {
        match self {
            RangeBound::Unbounded => None,
            RangeBound::Included(k) | RangeBound::Excluded(k) => Some(*k),
        }
    }
}

// Conversion from std::ops::Bound
impl<'a> From<Bound<&'a [u8]>> for RangeBound<'a> {
    fn from(bound: Bound<&'a [u8]>) -> Self {
        match bound {
            Bound::Unbounded => RangeBound::Unbounded,
            Bound::Included(k) => RangeBound::Included(k),
            Bound::Excluded(k) => RangeBound::Excluded(k),
        }
    }
}

// ============================================================================
//  ScanEntry
// ============================================================================

/// Entry returned by the range iterator.
///
/// Contains an owned copy of the key and the value output.
///
/// # Type Parameters
///
/// - `O`: The output type (e.g., `Arc<V>` for `MassTree<V>`, `V` for `MassTree24Inline<V>`)
///
/// # Example
///
/// ```ignore
/// for entry in tree.iter(&guard) {
///     let key: &[u8] = &entry.key;
///     let value: &V = entry.value.as_ref(); // For Arc<V>
/// }
///
#[derive(Debug, Clone)]
pub struct ScanEntry<O> {
    /// The key as owned bytes.
    pub key: Vec<u8>,

    /// The value output.
    ///
    /// For `MassTree<V>`: `Arc<V>` (shared reference)
    /// For `MassTree24Inline<V>`: `V` (copy)
    pub value: O,
}

impl<O> ScanEntry<O> {
    /// Create a new scan entry.
    #[inline(always)]
    pub const fn new(key: Vec<u8>, value: O) -> Self {
        Self { key, value }
    }

    /// Get the key bytes.
    #[inline(always)]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get a reference to the value.
    #[inline(always)]
    pub const fn value(&self) -> &O {
        &self.value
    }

    /// Consume the entry and return (key, value).
    #[inline(always)]
    pub fn into_parts(self) -> (Vec<u8>, O) {
        (self.key, self.value)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IterFlags(u8);

#[allow(dead_code, reason = "API Completeness")]
impl IterFlags {
    // Bit Positions
    const EXHAUSTED: u8 = 1 << 0;
    const INITIALIZED: u8 = 1 << 1;
    const EMIT_EQUAL: u8 = 1 << 2;
    const NEEDS_DUPLICATE_CHECK: u8 = 1 << 3;
    const SINGLE_LAYER_MODE: u8 = 1 << 4;

    /// Create new flags with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Create flags with initial values.
    #[inline(always)]
    pub const fn with_values(emit_equal: bool, single_layer_mode: bool) -> Self {
        let mut bits: u8 = 0;

        if emit_equal {
            bits |= Self::EMIT_EQUAL;
        }

        if single_layer_mode {
            bits |= Self::SINGLE_LAYER_MODE;
        }

        Self(bits)
    }

    // ========================================================================
    //  Getters
    // ========================================================================

    #[inline(always)]
    pub const fn exhausted(self) -> bool {
        self.0 & Self::EXHAUSTED != 0
    }

    #[inline(always)]
    pub const fn initialized(self) -> bool {
        self.0 & Self::INITIALIZED != 0
    }

    #[inline(always)]
    pub const fn emit_equal(self) -> bool {
        self.0 & Self::EMIT_EQUAL != 0
    }

    #[inline(always)]
    pub const fn needs_duplicate_check(self) -> bool {
        self.0 & Self::NEEDS_DUPLICATE_CHECK != 0
    }

    #[inline(always)]
    pub const fn single_layer_mode(self) -> bool {
        self.0 & Self::SINGLE_LAYER_MODE != 0
    }

    // ========================================================================
    //  Setters
    // ========================================================================

    #[inline(always)]
    pub const fn set_exhausted(&mut self, value: bool) {
        if value {
            self.0 |= Self::EXHAUSTED;
        } else {
            self.0 &= !Self::EXHAUSTED;
        }
    }

    #[inline(always)]
    pub const fn set_initialized(&mut self, value: bool) {
        if value {
            self.0 |= Self::INITIALIZED;
        } else {
            self.0 &= !Self::INITIALIZED;
        }
    }

    #[inline(always)]
    pub const fn set_emit_equal(&mut self, value: bool) {
        if value {
            self.0 |= Self::EMIT_EQUAL;
        } else {
            self.0 &= !Self::EMIT_EQUAL;
        }
    }

    #[inline(always)]
    pub const fn set_needs_duplicate_check(&mut self, value: bool) {
        if value {
            self.0 |= Self::NEEDS_DUPLICATE_CHECK;
        } else {
            self.0 &= !Self::NEEDS_DUPLICATE_CHECK;
        }
    }

    #[inline(always)]
    pub const fn set_single_layer_mode(&mut self, value: bool) {
        if value {
            self.0 |= Self::SINGLE_LAYER_MODE;
        } else {
            self.0 &= !Self::SINGLE_LAYER_MODE;
        }
    }

    // ========================================================================
    //  Convenience methods
    // ========================================================================

    /// Mark as exhausted.
    #[inline(always)]
    pub const fn mark_exhausted(&mut self) {
        self.0 |= Self::EXHAUSTED;
    }

    /// Mark as initialized.
    #[inline(always)]
    pub const fn mark_initialized(&mut self) {
        self.0 |= Self::INITIALIZED;
    }

    /// Clear `needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn clear_duplicate_check(&mut self) {
        self.0 &= !Self::NEEDS_DUPLICATE_CHECK;
    }

    /// Set `needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn require_duplicate_check(&mut self) {
        self.0 |= Self::NEEDS_DUPLICATE_CHECK;
    }

    /// Disable single-layer mode (fall back to multi-layer).
    #[inline(always)]
    pub const fn disable_single_layer_mode(&mut self) {
        self.0 &= !Self::SINGLE_LAYER_MODE;
    }
}

// ============================================================================
//  RangeIter
// ============================================================================

/// Iterator over a key range in a [`crate::MassTree`].
///
/// Yields entries in lexicographic key order. The iterator maintains internal
/// state for the scan position and handles concurrent modifications via the
/// optimistic concurrency control protocol.
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
/// - Best-effort no duplicates (via duplicate filtering)
///
/// # Performance
///
/// The iterator allocates:
/// - `Vec<u8>` for each key (unavoidable for owned keys)
/// - `SmallVec` for layer stack (usually inline, up to 4 layers)
/// - No per-item allocation for value cloning (Arc refcount bump or Copy)
///
/// # Example
///
/// ```ignore
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
    /// Memory reclamation guard.
    guard: &'g LocalGuard<'a>,

    /// Current scan position.
    stack: ScanStackElement<L, S>,

    /// Parent layer stack for sublayer navigation.
    layer_stack: LayerStack<L>,

    /// Cursor tracking current key position.
    cursor_key: CursorKey,

    /// End bound for the range.
    end_bound: RangeBound<'a>,

    /// Current state machine state.
    state: ScanState,

    /// Captured snapshot for current entry (if in Emit state).
    snapshot: Option<ScanSnapshot<S>>,

    /// Packed boolean flags.
    ///
    /// Contains: exhausted, initialized, `emit_equal`, `needs_duplicate_check`, `single_layer_mode`
    flags: IterFlags,

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
    last_output_ptr: Option<*mut u8>,

    /// Marker for lifetime and type parameter covariance.
    _marker: PhantomData<&'a A>,
}

impl<S, L, A> std::fmt::Debug for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeIter")
            .field("exhausted", &self.flags.exhausted())
            .field("initialized", &self.flags.initialized())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

// SAFETY: Use a scope guard to ensure cleanup on panic.
// If visitor() panics, the guard's Drop will run and clean up ptr.
struct CleanupGuard<S: ValueSlot> {
    ptr: *mut u8,
    _marker: PhantomData<S>,
}

impl<S: ValueSlot> Drop for CleanupGuard<S> {
    fn drop(&mut self) {
        // SAFETY: ptr was created by S::output_to_raw
        unsafe { S::cleanup_output_raw(self.ptr) };
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

        Self {
            guard,
            stack,
            layer_stack: ArrayVec::new(),
            cursor_key,
            end_bound: end,
            state: ScanState::FindNext, // Will be set properly in first iteration
            snapshot: None,
            flags: IterFlags::with_values(emit_equal, single_layer_mode),
            last_output_ptr: None,
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
    fn initialize(&mut self) {
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
            // IMPORTANT: Save parent ROOT before find_initial, because
            // find_initial may update stack.root to the layer pointer.
            // However, stack.leaf is still valid after find_initial returns
            // (it points to the parent leaf where the layer pointer was found).
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

                    // Take snapshot
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
}

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

impl<S, L, A> RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Zero-allocation iteration with a visitor closure.
    ///
    /// This is significantly faster than the `Iterator` trait because it:
    /// - Avoids allocating `Vec<u8>` for each key
    /// - Uses references directly from internal buffers
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], S::Output)`. Return `true` to continue,
    ///   `false` to stop early.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Performance
    ///
    /// ~3-5x faster than using the Iterator trait for large scans.
    #[inline]
    pub fn for_each<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], S::Output) -> bool,
    {
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Fast path: process current entry without allocation
            if let Some(entry) = self.advance_no_alloc() {
                count += 1;
                if !visitor(entry.0, entry.1) {
                    break 'l;
                }
            } else {
                break 'l;
            }
        }

        count
    }

    /// Advance without allocating key Vec.
    ///
    /// Returns `(&[u8], S::Output)` where the key slice is borrowed from
    /// the internal `cursor_key` buffer.
    ///
    /// # Performance: Inlined Hot Path
    ///
    /// This function inlines the common case `(FindNext → Emit)` to avoid:
    /// - State machine dispatch overhead
    /// - Function call overhead to `find_next()`
    ///
    /// Only rare cases (Down, Up, Retry) use function calls.
    #[inline(always)]
    fn advance_no_alloc(&mut self) -> Option<(&[u8], S::Output)> {
        // Fast path: if we have a pending emit, process it first
        if self.state == ScanState::Emit
            && let Some(snapshot) = self.snapshot.take()
        {
            let key = self.cursor_key.full_key();

            if !self.end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            self.state = ScanState::FindNext;
            return Some((key, snapshot.value));
        }

        loop {
            // Handle rare states first (will break out of loop on Emit)
            match self.state {
                ScanState::Down => {
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();
                        return None;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Main hot path: FindNext (inlined from find_next)
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

            // Fast path: if Emit, return immediately without another loop iteration
            if new_state == ScanState::Emit
                && let Some(snap) = snapshot
            {
                let key = self.cursor_key.full_key();

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return None;
                }

                self.state = ScanState::FindNext;
                return Some((key, snap.value));
            }

            self.snapshot = snapshot;
        }
    }

    /// Zero-copy iteration with borrowed value references.
    ///
    /// Unlike [`Self::for_each`] which clones values (Arc increment for `LeafValue`),
    /// this returns `&S::Value` references tied to the guard lifetime.
    ///
    /// # Performance
    ///
    /// Eliminates 2 atomic operations per entry (Arc increment + decrement),
    /// which can improve scan throughput by 2-3x for Arc-based trees.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &S::Value)`. Return `true` to continue,
    ///   `false` to stop early.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    ///
    /// # Safety
    ///
    /// The value references are valid only during the callback. Do not store them.
    /// The guard ensures the underlying data isn't deallocated during iteration.
    #[inline]
    pub fn for_each_ref<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        'l: loop {
            // Use the zero-copy advance method
            if let Some((key, value_ref)) = self.advance_no_alloc_ref() {
                count += 1;
                if !visitor(key, value_ref) {
                    break 'l;
                }
            } else {
                break 'l;
            }
        }

        count
    }

    /// Batch iteration with zero-copy value references and reduced dispatch overhead.
    ///
    /// This is the highest-performance iteration method. It eliminates state machine
    /// dispatch overhead while maintaining identical correctness to [`Self::for_each_ref`].
    ///
    /// # Correctness
    ///
    /// Unlike approaches that validate only once per leaf, this method:
    /// - Uses per-entry OCC validation (same as `for_each_ref`)
    /// - Properly updates cursor key for duplicate filtering
    /// - Handles layer transitions correctly (dynamically switches from single-layer
    ///   to multi-layer mode when `Down` is encountered)
    ///
    /// # Performance
    ///
    /// Expected 1.3-1.5x improvement over `for_each_ref` for large scans.
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &S::Value)`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[expect(clippy::too_many_lines, reason = "Complex state management logic")]
    pub fn for_each_batch_ref<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization - reuses existing RangeIter::initialize()
        // which correctly handles start-bound descent (shift vs shift_clear)
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // NOTE: We don't use advance_no_alloc_ref here because it has issues
        // with multi-layer keys. Instead, we use the batch loop for all entries
        // which correctly handles cursor_key updates via find_next_ptr.

        // If state is Emit with a snapshot from initialize(), handle it specially
        // by extracting the snapshot and emitting directly
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                let key: &[u8] = self.cursor_key.full_key();

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return 0;
                }

                // Convert snapshot to reference
                let ptr: *mut u8 = S::output_to_raw(&snapshot.value);

                let guard = CleanupGuard::<S> {
                    ptr,
                    _marker: PhantomData,
                };

                let value_ref: &S::Value = unsafe { &*ptr.cast::<S::Value>() };

                count += 1;
                let should_continue = visitor(key, value_ref);

                // Explicitly drop guard to clean up (also runs on panic)
                drop(guard);

                if !should_continue {
                    return count;
                }
            }
            self.state = ScanState::FindNext;
        }

        // Main batch loop - uses find_next_ptr which correctly updates cursor_key

        loop {
            // ================================================================
            // Handle rare states (layer transitions, retries, exhaustion)
            // ================================================================

            // Handle pending state transitions first (like advance_no_alloc_ref)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();
                        return count;
                    }
                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }
                self.state = ScanState::Up;
                continue;
            }

            // Check leaf deletion
            let leaf: &L = unsafe { self.stack.leaf_ref() };
            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // Main hot path: FindNext → Emit (inlined)
            // ================================================================

            let (new_state, snapshot_ptr) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                find_next_with_duplicate_check_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            } else {
                find_next_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            };

            self.state = new_state;

            match new_state {
                ScanState::Emit => {
                    if let Some(snap) = snapshot_ptr {
                        let key: &[u8] = self.cursor_key.full_key();

                        // Check end bound
                        if !self.end_bound.contains(key) {
                            self.flags.mark_exhausted();
                            return count;
                        }

                        // SAFETY: find_next_ptr validated version, guard protects pointer
                        let value_ref: &S::Value = unsafe { &*snap.value_ptr };

                        count += 1;
                        self.state = ScanState::FindNext;

                        if !visitor(key, value_ref) {
                            return count;
                        }
                    }
                    // Continue to next entry
                }

                // Other states are handled at the top of the loop
                ScanState::FindNext | ScanState::Down | ScanState::Up | ScanState::Retry => {}
            }
        }
    }

    /// Intra-leaf batch iteration with maximum performance.
    ///
    /// This is the highest-performance iteration method. It processes entire
    /// leaves in tight loops, minimizing per-entry overhead.
    ///
    /// # Performance Characteristics
    ///
    /// - Processes all entries in a leaf before moving to next leaf
    /// - Single OCC validation per leaf (vs per-entry in `for_each_batch_ref`)
    /// - No function call overhead per entry within a leaf
    /// - Falls back to state machine for layer transitions
    ///
    /// Expected 2-3x improvement over `for_each_batch_ref` for large scans
    /// with many entries per leaf (typical case).
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &S::Value)`. Return `true` to continue.
    ///
    /// # Returns
    ///
    /// Number of entries visited.
    #[inline]
    #[expect(clippy::too_many_lines)]
    pub fn for_each_intra_leaf_batch_ref<F>(mut self, mut visitor: F) -> usize
    where
        F: FnMut(&[u8], &S::Value) -> bool,
    {
        use super::find::{
            LeafBatchResult, advance_leaf_ptr, find_retry, handle_down, handle_up,
            process_leaf_batch_ptr,
        };

        if self.flags.exhausted() {
            return 0;
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return 0;
            }
        }

        let mut count: usize = 0;

        // Handle initial Emit state from initialize() if present
        if self.state == ScanState::Emit {
            if let Some(snapshot) = self.snapshot.take() {
                let key: &[u8] = self.cursor_key.full_key();

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return 0;
                }

                let ptr: *mut u8 = S::output_to_raw(&snapshot.value);
                let guard = CleanupGuard::<S> {
                    ptr,
                    _marker: PhantomData,
                };
                let value_ref: &S::Value = unsafe { &*ptr.cast::<S::Value>() };

                count += 1;
                let should_continue = visitor(key, value_ref);
                drop(guard);

                if !should_continue {
                    return count;
                }
            }
            self.state = ScanState::FindNext;
        }

        loop {
            // Handle rare states (layer transitions, retries)
            match self.state {
                ScanState::Down => {
                    self.flags.disable_single_layer_mode();
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();
                        return count;
                    }
                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Check for null stack (layer exhausted)
            if self.stack.is_null() {
                if self.layer_stack.is_empty() {
                    self.flags.mark_exhausted();
                    return count;
                }
                self.state = ScanState::Up;
                continue;
            }

            // Check leaf deletion
            let leaf: &L = unsafe { self.stack.leaf_ref() };
            if leaf.version().is_deleted() {
                self.state = ScanState::Retry;
                continue;
            }

            // ================================================================
            // INTRA-LEAF BATCH: Process all remaining entries in this leaf
            // ================================================================

            let result = process_leaf_batch_ptr(
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                &self.end_bound,
                &mut visitor,
                &mut count,
            );

            match result {
                LeafBatchResult::LeafExhausted => {
                    // Advance to next leaf
                    let (state, _) =
                        advance_leaf_ptr(&mut self.stack, &self.cursor_key, self.guard);
                    self.state = state;
                }
                LeafBatchResult::LayerEncountered => {
                    self.state = ScanState::Down;
                }
                LeafBatchResult::VersionChanged => {
                    self.state = ScanState::Retry;
                }
                LeafBatchResult::Stopped => {
                    return count;
                }
                LeafBatchResult::EndBoundExceeded => {
                    self.flags.mark_exhausted();
                    return count;
                }
            }
        }
    }

    /// Fallible iteration with zero-copy value references.
    ///
    /// Like [`Self::for_each_ref`], but the visitor can return an error to stop
    /// iteration early. This is useful when processing entries might fail (e.g.,
    /// serialization, validation, I/O).
    ///
    /// # Arguments
    ///
    /// - `visitor`: Closure receiving `(&[u8], &S::Value)`. Return `Ok(true)` to
    ///   continue, `Ok(false)` to stop early, or `Err(E)` to stop with an error.
    ///
    /// # Returns
    ///
    /// - `Ok(count)`: Number of entries successfully visited
    /// - `Err(e)`: The error returned by the visitor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = tree.iter(&guard).try_for_each_ref(|key, value| {
    ///     if key.len() > MAX_KEY_LEN {
    ///         return Err(ValidationError::KeyTooLong);
    ///     }
    ///     writer.write_entry(key, value)?;
    ///     Ok(true)
    /// });
    ///
    /// match result {
    ///     Ok(count) => println!("Wrote {} entries", count),
    ///     Err(e) => eprintln!("Failed: {}", e),
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if fails to advance
    #[inline]
    pub fn try_for_each_ref<F, E>(mut self, mut visitor: F) -> Result<usize, E>
    where
        F: FnMut(&[u8], &S::Value) -> Result<bool, E>,
    {
        if self.flags.exhausted() {
            return Ok(0);
        }

        // Lazy initialization
        if !self.flags.initialized() {
            self.initialize();
            if self.flags.exhausted() {
                return Ok(0);
            }
        }

        let mut count: usize = 0;

        loop {
            // Use the zero-copy advance method
            if let Some((key, value_ref)) = self.advance_no_alloc_ref() {
                count += 1;
                match visitor(key, value_ref) {
                    Ok(true) => {}

                    Ok(false) => return Ok(count),

                    Err(e) => return Err(e),
                }
            } else {
                return Ok(count);
            }
        }
    }

    /// Advance without cloning values.
    ///
    /// Returns `(&[u8], &S::Value)` where both are borrowed references.
    /// The value is obtained by dereferencing the raw pointer directly,
    /// avoiding Arc clone overhead.
    ///
    /// # Note on Initial Entry
    ///
    /// After `initialize()`, there may be a pending emit in `self.snapshot`.
    /// For the first entry, we convert the Output to a raw pointer and dereference.
    /// This requires that `S::Output` is dereferenceable to `S::Value`.
    ///
    /// # Safety
    ///
    /// The returned references are valid because:
    /// 1. The guard prevents deallocation during iteration
    /// 2. Version validation ensures the slot hasn't been modified
    #[inline(always)]
    #[expect(clippy::too_many_lines, reason = "Complex allocation logic")]
    fn advance_no_alloc_ref(&mut self) -> Option<(&[u8], &S::Value)> {
        // Handle pending emit from initialize() - first entry case
        if self.state == ScanState::Emit && self.snapshot.is_some() {
            let key = self.cursor_key.full_key();

            if !self.end_bound.contains(key) {
                self.flags.mark_exhausted();
                return None;
            }

            // Take the snapshot to get the value
            let snapshot = self.snapshot.take()?;

            // Transition to FindNext for next call
            self.state = ScanState::FindNext;

            // Clean up the previous output pointer before creating a new one.
            // For LeafValueIndex (Copy types), output_to_raw allocates a Box
            // that must be freed. For LeafValue (Arc types), this decrements
            // the cloned Arc's refcount.
            if let Some(old_ptr) = self.last_output_ptr.take() {
                // SAFETY: old_ptr was created by output_to_raw in a previous call
                unsafe { S::cleanup_output_raw(old_ptr) };
            }

            // Convert the output to a raw pointer and dereference.
            // For Arc<V>: output_to_raw gives us the Arc's data pointer
            // For Copy types: output_to_raw gives us the Box's data pointer
            let ptr: *mut u8 = S::output_to_raw(&snapshot.value);

            // Track this pointer so we can clean it up later
            self.last_output_ptr = Some(ptr);

            // SAFETY: We just created this pointer from a valid Output.
            // The guard protects the underlying data.
            let value_ref: &S::Value = unsafe { &*ptr.cast::<S::Value>() };

            return Some((key, value_ref));
        }

        loop {
            // ================================================================
            // Single-layer fast path (keys ≤ 8 bytes)
            // ================================================================
            if self.flags.single_layer_mode() {
                // Retry handling in single-layer mode
                if self.state == ScanState::Retry {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                let (new_state, snapshot_ptr) = find_next_single_layer_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    self.guard,
                    self.flags.needs_duplicate_check(),
                );

                if self.flags.needs_duplicate_check() {
                    self.flags.clear_duplicate_check();
                }

                self.state = new_state;

                match new_state {
                    ScanState::Emit => {
                        if let Some(snap) = snapshot_ptr {
                            let key = self.cursor_key.full_key();

                            if !self.end_bound.contains(key) {
                                self.flags.mark_exhausted();
                                return None;
                            }

                            self.state = ScanState::FindNext;
                            let value_ref: &S::Value = unsafe { &*snap.value_ptr };

                            return Some((key, value_ref));
                        }
                    }

                    ScanState::FindNext => {
                        if self.stack.is_null() {
                            self.flags.mark_exhausted();
                            return None;
                        }

                        continue;
                    }

                    ScanState::Retry => continue,

                    ScanState::Down => {
                        // Encountered layer pointer - fall back to multi-layer
                        self.flags.disable_single_layer_mode();

                        // Push PARENT context to layer_stack before setting new root.
                        // find_next_single_layer_ptr already stored the ikey to cursor.
                        self.layer_stack
                            .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));

                        // Read the layer pointer from current slot and set as new root.
                        // Stack position is still at the layer pointer slot.
                        // SAFETY: find_next_single_layer_ptr verified leaf is valid
                        let Some(slot) = self.stack.kp() else {
                            // Defensive: shouldn't happen, but if slot is somehow invalid,
                            // fall back to multi-layer retry path
                            debug_assert!(false, "Down state should have valid slot");
                            self.state = ScanState::Retry;
                            continue;
                        };
                        let leaf: &L = unsafe { self.stack.leaf_ref() };
                        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                        self.stack.set_root(layer_ptr);

                        // Don't continue; fall through to handle Down below
                    }

                    ScanState::Up => {
                        self.flags.mark_exhausted();
                        return None;
                    }
                }
            }

            // ================================================================
            // Multi-layer path (handles Down/Up transitions)
            // ================================================================

            // Handle rare states first
            match self.state {
                ScanState::Down => {
                    handle_down(&mut self.stack, &mut self.cursor_key);
                    self.state = ScanState::Retry;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        self.flags.mark_exhausted();
                        return None;
                    }

                    self.state = ScanState::FindNext;
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    self.flags.require_duplicate_check();
                    continue;
                }

                ScanState::Emit | ScanState::FindNext => {}
            }

            // Use zero-copy find_next variants
            let (new_state, snapshot_ptr) = if self.flags.needs_duplicate_check() {
                self.flags.clear_duplicate_check();
                find_next_with_duplicate_check_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            } else {
                find_next_ptr(
                    &mut self.stack,
                    &mut self.cursor_key,
                    &mut self.layer_stack,
                    self.guard,
                )
            };

            self.state = new_state;

            // If Emit, return the reference directly
            if new_state == ScanState::Emit
                && let Some(snap) = snapshot_ptr
            {
                let key = self.cursor_key.full_key();

                if !self.end_bound.contains(key) {
                    self.flags.mark_exhausted();
                    return None;
                }

                self.state = ScanState::FindNext;

                // SAFETY: Guard prevents deallocation, version was validated
                // in find_next_inner_ptr before returning the pointer.
                // We dereference the raw pointer directly (not via snap.value_ref())
                // because the reference must outlive the local `snap` variable.
                let value_ref: &S::Value = unsafe { &*snap.value_ptr };
                return Some((key, value_ref));
            }

            // All non-Emit states (Up, Down, Retry, FindNext) continue the loop.
            // Exhaustion is detected by stack.is_null() or handle_up() returning false.
        }
    }
}

impl<S, L, A> std::iter::FusedIterator for RangeIter<'_, '_, S, L, A>
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

// ============================================================================
//  Key/Value Only Iterators
// ============================================================================

/// Iterator adapter that yields only keys.
pub struct KeysIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    inner: RangeIter<'a, 'g, S, L, A>,
}

impl<S, L, A> std::fmt::Debug for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeysIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<S, L, A> Iterator for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = Vec<u8>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.key)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S, L, A> std::iter::FusedIterator for KeysIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
}

/// Iterator adapter that yields only values.
pub struct ValuesIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    inner: RangeIter<'a, 'g, S, L, A>,
}

impl<S, L, A> std::fmt::Debug for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValuesIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<S, L, A> Iterator for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = S::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S, L, A> std::iter::FusedIterator for ValuesIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
}

// ============================================================================
//  Factory Functions (used by api.rs)
// ============================================================================

impl<'a, 'g, S, L, A> RangeIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
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
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
