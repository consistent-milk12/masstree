//! Filepath: src/tree/range/iterator.rs
//!
//! Range iterator implementation.
//!
//! Provides [`RangeIter`], an iterator over key-value pairs in lexicographic order.
//! The iterator yields [`ScanEntry`] items containing owned keys and values.
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

use seize::LocalGuard;
use smallvec::SmallVec;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::{LayerCapableLeaf, TreeLeafNode};
use crate::slot::ValueSlot;
use crate::tree::MassTreeGeneric;

use super::cursor_key::CursorKey;
use super::find::{
    find_initial, find_next, find_next_with_duplicate_check, find_retry, handle_down, handle_up,
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
impl<'a> From<std::ops::Bound<&'a [u8]>> for RangeBound<'a> {
    fn from(bound: std::ops::Bound<&'a [u8]>) -> Self {
        match bound {
            std::ops::Bound::Unbounded => RangeBound::Unbounded,
            std::ops::Bound::Included(k) => RangeBound::Included(k),
            std::ops::Bound::Excluded(k) => RangeBound::Excluded(k),
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

// ============================================================================
//  RangeIter
// ============================================================================

/// Iterator over a key range in a [`MassTree`].
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
#[expect(clippy::struct_excessive_bools)]
pub struct RangeIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Reference to the tree.
    tree: &'a MassTreeGeneric<S, L, A>,

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

    /// Whether the scan has been exhausted.
    exhausted: bool,

    /// Whether initial positioning has been done.
    initialized: bool,

    /// Whether to emit exact matches at start bound.
    emit_equal: bool,

    /// Whether the next `find_next` call needs duplicate checking.
    ///
    /// This is set to true after a Retry state, where we may have been
    /// repositioned to a slot we already emitted. In normal forward
    /// iteration, duplicates can't occur because `stack.next()` advances
    /// past the previous entry.
    needs_duplicate_check: bool,

    /// Marker for lifetime covariance.
    _marker: PhantomData<&'a ()>,
}

impl<S, L, A> std::fmt::Debug for RangeIter<'_, '_, S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeIter")
            .field("exhausted", &self.exhausted)
            .field("initialized", &self.initialized)
            .field("state", &self.state)
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

        Self {
            tree,
            guard,
            stack,
            layer_stack: SmallVec::new(),
            cursor_key,
            end_bound: end,
            state: ScanState::FindNext, // Will be set properly in first iteration
            snapshot: None,
            exhausted: false,
            initialized: false,
            emit_equal,
            needs_duplicate_check: false,
            _marker: PhantomData,
        }
    }

    /// Initialize the iterator (lazy initialization on first `next()` call).
    fn initialize(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;

        // Handle empty tree
        if self.stack.root().is_null() {
            self.exhausted = true;
            return;
        }

        // Run initial descent loop
        loop {
            let (state, snapshot) = find_initial(
                self.stack.root(),
                &mut self.stack,
                &mut self.cursor_key,
                &mut self.layer_stack,
                self.emit_equal,
                self.guard,
            );

            match state {
                ScanState::Down => {
                    // Start key descends into a sublayer
                    // Push current context and shift key
                    self.layer_stack
                        .push(LayerContext::new(self.stack.root(), self.stack.leaf_ptr()));

                    // Key shift (not shift_clear, using start key bytes)
                    self.cursor_key.shift();

                    // Update root to layer pointer
                    // (find_initial would have set this in a real impl)
                    // Continue loop to descend further
                }

                ScanState::Retry => {
                    // Version conflict, retry
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
    fn advance(&mut self) -> Option<ScanEntry<S::Output>> {
        loop {
            match self.state {
                ScanState::Emit => {
                    // Check end bound
                    let key = self.cursor_key.full_key();
                    if !self.end_bound.contains(key) {
                        self.exhausted = true;
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
                    let (new_state, snapshot) = if self.needs_duplicate_check {
                        self.needs_duplicate_check = false;
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
                    self.needs_duplicate_check = true;
                }

                ScanState::Up => {
                    if !handle_up(
                        &mut self.stack,
                        &mut self.cursor_key,
                        &mut self.layer_stack,
                        self.guard,
                    ) {
                        // No parent layer, scan complete
                        self.exhausted = true;
                        return None;
                    }
                    self.state = ScanState::FindNext;
                    // After layer ascent, we need duplicate check
                    self.needs_duplicate_check = true;
                }

                ScanState::Retry => {
                    self.state = find_retry(&mut self.stack, &self.cursor_key, self.guard);
                    // After retry, we need duplicate check on next FindNext
                    self.needs_duplicate_check = true;
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

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // Lazy initialization
        if !self.initialized {
            self.initialize();
            if self.exhausted {
                return None;
            }
        }

        self.advance()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.exhausted {
            (0, Some(0))
        } else {
            // We can't know the exact count without iterating
            (0, None)
        }
    }
}

impl<'a, 'g, S, L, A> std::iter::FusedIterator for RangeIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
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

impl<'a, 'g, S, L, A> Iterator for KeysIter<'a, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, 'g, S, L, A> std::iter::FusedIterator for KeysIter<'a, 'g, S, L, A>
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

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| entry.value)
    }

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
    pub fn keys(self) -> KeysIter<'a, 'g, S, L, A> {
        KeysIter { inner: self }
    }

    /// Convert to a values-only iterator.
    pub fn values(self) -> ValuesIter<'a, 'g, S, L, A> {
        ValuesIter { inner: self }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_bound_contains() {
        // Unbounded contains everything
        assert!(RangeBound::Unbounded.contains(b"anything"));
        assert!(RangeBound::Unbounded.contains(b""));

        // Included: key <= bound
        let included = RangeBound::Included(b"middle");
        assert!(included.contains(b"aaa"));
        assert!(included.contains(b"middle"));
        assert!(!included.contains(b"zzz"));

        // Excluded: key < bound
        let excluded = RangeBound::Excluded(b"middle");
        assert!(excluded.contains(b"aaa"));
        assert!(!excluded.contains(b"middle"));
        assert!(!excluded.contains(b"zzz"));
    }

    #[test]
    fn test_range_bound_to_start_params() {
        let (key, emit) = RangeBound::Unbounded.to_start_params();
        assert_eq!(key, b"");
        assert!(emit);

        let (key, emit) = RangeBound::Included(b"start").to_start_params();
        assert_eq!(key, b"start");
        assert!(emit);

        let (key, emit) = RangeBound::Excluded(b"start").to_start_params();
        assert_eq!(key, b"start");
        assert!(!emit);
    }

    #[test]
    fn test_range_bound_from_std_bound() {
        use std::ops::Bound;

        let rb: RangeBound = Bound::Unbounded.into();
        assert!(matches!(rb, RangeBound::Unbounded));

        let rb: RangeBound = Bound::Included(b"key".as_slice()).into();
        assert!(matches!(rb, RangeBound::Included(k) if k == b"key"));

        let rb: RangeBound = Bound::Excluded(b"key".as_slice()).into();
        assert!(matches!(rb, RangeBound::Excluded(k) if k == b"key"));
    }

    #[test]
    fn test_scan_entry() {
        let entry = ScanEntry::new(b"key".to_vec(), 42u64);

        assert_eq!(entry.key(), b"key");
        assert_eq!(*entry.value(), 42);

        let (key, value) = entry.into_parts();
        assert_eq!(key, b"key");
        assert_eq!(value, 42);
    }

    #[test]
    fn test_range_bound_is_unbounded() {
        assert!(RangeBound::Unbounded.is_unbounded());
        assert!(!RangeBound::Included(b"key").is_unbounded());
        assert!(!RangeBound::Excluded(b"key").is_unbounded());
    }

    #[test]
    fn test_range_bound_key() {
        assert!(RangeBound::Unbounded.key().is_none());
        assert_eq!(RangeBound::Included(b"key").key(), Some(b"key".as_slice()));
        assert_eq!(RangeBound::Excluded(b"key").key(), Some(b"key".as_slice()));
    }
}
