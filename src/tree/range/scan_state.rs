//! Filepath: `src/tree/range/scan_state.rs`
//!
//! Scan state machine types for range scan operations.
//!
//! # State Machine Design
//!
//! Range scans are implemented as an explicit state machine that transitions
//! between states as it navigates the trie structure. This design matches
//! the C++ reference (`masstree_scan.hh`) and handles:
//!
//! - Concurrent modifications (version-triggered retries)
//! - Multi-layer navigation (sublayer descent and ascent)
//! - B-link chain traversal
//!
//! # States
//!
//! ```text
//!                    ┌────────────────────────────────────────┐
//!                    │                                        │
//!                    ▼                                        │
//!              ┌──────────┐                                   │
//!         ┌───▶│  Emit    │───── emit entry ─────────────────▶│
//!         │    └──────────┘                                   │
//!         │         │                                         │
//!         │         │ advance ki                              │
//!         │         ▼                                         │
//!         │    ┌──────────┐                                   │
//!         ├────│ FindNext │◀──────────────────────────────────┤
//!         │    └──────────┘                                   │
//!         │         │                                         │
//!         │    ┌────┴────┬────────────┐                       │
//!         │    ▼         ▼            ▼                       │
//!         │  found    layer_ptr    exhausted                  │
//!         │    │         │            │                       │
//!         │    │    ┌────┴────┐  ┌────┴────┐                  │
//!         │    │    │  Down   │  │   Up    │                  │
//!         │    │    └────┬────┘  └────┬────┘                  │
//!         │    │         │            │                       │
//!         │    │  shift_clear()   unshift()                   │
//!         │    │         │            │                       │
//!         │    │         ▼            │                       │
//!         │    │    ┌──────────┐      │                       │
//!         │    │    │  Retry   │◀─────┘                       │
//!         │    │    └────┬─────┘                              │
//!         │    │         │                                    │
//!         │    │    find_retry()                              │
//!         │    │         │                                    │
//!         └────┴─────────┴────────────────────────────────────┘
//!
use std::marker::PhantomData;
use std::ptr::NonNull;

use smallvec::SmallVec;

use crate::leaf_trait::{TreeLeafNode, TreePermutation};
use crate::slot::ValueSlot;

// ============================================================================
//  ScanState Enum
// ============================================================================

/// State machine states for range scan operations.
///
/// Each state represents a phase in the scan algorithm:
///
/// - [`Emit`](ScanState::Emit): Ready to yield current entry to visitor
/// - [`FindNext`](ScanState::FindNext): Searching for next entry in current leaf
/// - [`Down`](ScanState::Down): Descending into a sublayer (encountered layer pointer)
/// - [`Up`](ScanState::Up): Ascending to parent layer (current layer exhausted)
/// - [`Retry`](ScanState::Retry): Repositioning after version change or layer transition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    /// Ready to emit the current entry to the visitor.
    ///
    /// The scan has identified a valid key-value pair that passes the
    /// duplicate filter and is within bounds. The caller should:
    /// 1. Check end bound
    /// 2. Extract key from `CursorKey::full_key()`
    /// 3. Clone value from captured snapshot
    /// 4. Advance `ki` and transition to `FindNext`
    Emit,

    /// Searching for the next entry within the current leaf/layer.
    ///
    /// This is the main iteration state. The scan will:
    /// 1. Check if current position has a valid slot
    /// 2. If yes: validate version, check duplicate, prepare for emit
    /// 3. If no: advance to next leaf or transition to `Up`
    FindNext,

    /// Descending into a sublayer (encountered a layer pointer).
    ///
    /// When a slot contains a layer pointer (`keylenx >= LAYER_KEYLENX`),
    /// the scan must descend to enumerate that sublayer's keys.
    ///
    /// The caller should:
    /// 1. Push current `(root, leaf)` to `LayerStack`
    /// 2. Set `root = layer_ptr`
    /// 3. Call `cursor_key.shift_clear()`
    /// 4. Transition to `Retry`
    Down,

    /// Ascending to parent layer (current layer exhausted).
    ///
    /// When the scan reaches the end of a sublayer (no more leaves in
    /// the B-link chain), it must return to the parent layer.
    ///
    /// The caller should:
    /// 1. If `LayerStack` is empty: scan is complete
    /// 2. Else: pop `(root, leaf)` from stack
    /// 3. Call `cursor_key.unshift()`
    /// 4. Refresh leaf state (version, perm)
    /// 5. Continue with `FindNext`
    Up,

    /// Repositioning after a conflict or layer transition.
    ///
    /// This state is entered after:
    /// - Version validation failure
    /// - Layer descent (`Down` → `shift_clear()`)
    /// - Layer ascent (`Up` → `unshift()`)
    ///
    /// The scan will call `find_retry()` to reposition at the correct
    /// leaf and slot for the current `CursorKey` state.
    Retry,
}

impl ScanState {
    /// Check if scan should continue (not in terminal state).
    ///
    /// Returns `false` only when scan is complete (after handling `Up`
    /// with empty layer stack).
    #[inline(always)]
    #[expect(clippy::unused_self)]
    pub const fn should_continue(self) -> bool {
        // All states continue; Up becomes terminal only when layer_stack is empty
        true
    }

    /// Check if this state will yield an entry.
    #[inline(always)]
    pub const fn is_emit(self) -> bool {
        matches!(self, Self::Emit)
    }

    /// Check if this state requires layer stack manipulation.
    #[inline(always)]
    pub const fn is_layer_transition(self) -> bool {
        matches!(self, Self::Down | Self::Up)
    }
}

// ============================================================================
//  ScanStackElement
// ============================================================================

/// Current layer position during a range scan.
///
/// Tracks the scanning state within a single trie layer:
/// - Current leaf node pointer
/// - Version snapshot for optimistic validation
/// - Cached permutation
/// - Logical position within permutation
///
/// # Thread Safety
///
/// The pointers stored here are only valid while the `LocalGuard` is held.
/// The guard prevents the garbage collector from reclaiming nodes.
///
/// # C++ Reference
///
/// Corresponds to `scanstackelt` in `masstree_scan.hh`.
pub struct ScanStackElement<L, S>
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    /// Current layer root (may be leaf or internode).
    ///
    /// Used for repositioning after retries.
    root: *const u8,

    /// Current leaf in this layer.
    ///
    /// Uses `Option<NonNull<L>>` for explicit null representation with niche
    /// optimization. The pointer is valid as long as the guard passed to scan
    /// APIs remains alive.
    leaf: Option<NonNull<L>>,

    /// Version snapshot for optimistic validation.
    ///
    /// Captured via `leaf.version().stable()`. After reading leaf fields,
    /// we check `leaf.version().has_changed(version)` to validate.
    version: u32,

    /// Cached permutation snapshot.
    ///
    /// Loaded via `leaf.permutation()`. Must be refreshed when version
    /// changes or when moving to a new leaf.
    perm: L::Perm,

    /// Current logical position in permutation.
    ///
    /// Range: `0..=perm.size()`.
    /// - `ki < perm.size()`: Valid position, physical slot = `perm.get(ki)`
    /// - `ki == perm.size()`: Leaf exhausted, need to advance or go up
    ki: usize,

    /// Marker for the slot type.
    _marker: PhantomData<S>,
}

impl<L, S> Clone for ScanStackElement<L, S>
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    fn clone(&self) -> Self {
        Self {
            root: self.root,
            leaf: self.leaf,
            version: self.version,
            perm: self.perm,
            ki: self.ki,
            _marker: PhantomData,
        }
    }
}

impl<L, S> std::fmt::Debug for ScanStackElement<L, S>
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanStackElement")
            .field("root", &self.root)
            .field("leaf", &self.leaf)
            .field("version", &self.version)
            .field("ki", &self.ki)
            .field("perm_size", &self.perm.size())
            .finish()
    }
}

impl<L, S> ScanStackElement<L, S>
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create a new stack element for a layer.
    ///
    /// The element starts uninitialized (null leaf, invalid version).
    /// Call `initialize()` or `set_leaf()` before using.
    #[must_use]
    pub fn new(root: *const u8) -> Self {
        Self {
            root,
            leaf: None,
            version: 0,
            perm: L::Perm::empty(),
            ki: 0,
            _marker: PhantomData,
        }
    }

    /// Create a stack element with all fields initialized.
    ///
    /// # Arguments
    ///
    /// - `leaf`: Raw pointer to the leaf node (may be null)
    #[must_use]
    pub const fn with_leaf(
        root: *const u8,
        leaf: *mut L,
        version: u32,
        perm: L::Perm,
        ki: usize,
    ) -> Self {
        Self {
            root,
            leaf: NonNull::new(leaf),
            version,
            perm,
            ki,
            _marker: PhantomData,
        }
    }

    // ========================================================================
    //  Accessors
    // ========================================================================

    /// Get the layer root pointer.
    #[inline(always)]
    pub const fn root(&self) -> *const u8 {
        self.root
    }

    /// Get the current leaf pointer as a raw pointer.
    ///
    /// Returns null if the leaf is not set.
    #[inline(always)]
    pub const fn leaf_ptr(&self) -> *mut L {
        match self.leaf {
            Some(nn) => nn.as_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Get a reference to the current leaf.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `self.leaf` is `Some` (not null)
    /// - The guard that protects this pointer is still alive
    #[inline(always)]
    pub unsafe fn leaf_ref(&self) -> &L {
        debug_assert!(self.leaf.is_some(), "leaf_ref called on null leaf");
        // SAFETY: Caller ensures leaf is valid and Some
        unsafe { self.leaf.unwrap_unchecked().as_ref() }
    }

    /// Get the cached version.
    #[inline(always)]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Get the cached permutation.
    #[inline(always)]
    pub const fn perm(&self) -> L::Perm {
        self.perm
    }

    /// Get the current logical position.
    #[inline(always)]
    pub const fn ki(&self) -> usize {
        self.ki
    }

    /// Get the physical slot at current position, or `None` if exhausted.
    ///
    /// Returns `Some(perm.get(ki))` if `ki < perm.size()`, else `None`.
    #[inline]
    pub fn kp(&self) -> Option<usize> {
        if self.ki < self.perm.size() {
            Some(self.perm.get(self.ki))
        } else {
            None
        }
    }

    /// Check if the leaf is exhausted (no more slots at current position).
    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.ki >= self.perm.size()
    }

    /// Check if the leaf pointer is null.
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.leaf.is_none()
    }

    // ========================================================================
    //  Mutation
    // ========================================================================

    /// Set the layer root pointer.
    #[inline(always)]
    pub const fn set_root(&mut self, root: *const u8) {
        self.root = root;
    }

    /// Set the leaf pointer from a raw pointer.
    ///
    /// Accepts null pointers (converts to `None` internally).
    #[inline(always)]
    pub const fn set_leaf(&mut self, leaf: *mut L) {
        self.leaf = NonNull::new(leaf);
    }

    /// Set the cached version.
    #[inline(always)]
    pub const fn set_version(&mut self, version: u32) {
        self.version = version;
    }

    /// Set the cached permutation.
    #[inline(always)]
    pub const fn set_perm(&mut self, perm: L::Perm) {
        self.perm = perm;
    }

    /// Set the logical position.
    #[inline(always)]
    pub const fn set_ki(&mut self, ki: usize) {
        self.ki = ki;
    }

    /// Advance to the next logical position.
    #[inline(always)]
    pub const fn next(&mut self) {
        self.ki += 1;
    }

    /// Update all leaf state (version, perm, ki).
    ///
    /// Used when moving to a new leaf or after version change.
    #[inline]
    pub const fn update_state(&mut self, version: u32, perm: L::Perm, ki: usize) {
        self.version = version;
        self.perm = perm;
        self.ki = ki;
    }

    /// Refresh state from the current leaf.
    ///
    /// Loads a fresh version and permutation from the leaf node.
    /// Sets `ki` to the provided value.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self.leaf` is `Some` and valid.
    ///
    /// # Returns
    ///
    /// `Ok(())` if permutation loaded successfully.
    /// `Err(())` if permutation is frozen (split in progress).
    #[expect(clippy::unnecessary_wraps, reason = "API Consistency")]
    pub unsafe fn refresh_from_leaf(&mut self, ki: usize) -> Result<(), ()> {
        debug_assert!(self.leaf.is_some(), "refresh_from_leaf called on null leaf");
        // SAFETY: Caller ensures leaf is valid
        let leaf: &L = unsafe { self.leaf.unwrap_unchecked().as_ref() };

        self.version = leaf.version().stable();

        self.perm = leaf.permutation();
        self.ki = ki;
        Ok(())
    }
}

// ============================================================================
//  LayerContext and LayerStack
// ============================================================================

/// Parent layer context stored during sublayer descent.
///
/// When the scan descends into a sublayer (encounters a layer pointer),
/// it saves the current layer's root and leaf pointers. After exhausting
/// the sublayer, these are restored to continue scanning the parent.
///
/// # Design Note
///
/// We only store `(root, leaf)` and recompute `(version, perm, ki)` on
/// ascent. This matches the C++ approach and is correct because:
///
/// 1. The parent leaf may have been modified while scanning the sublayer
/// 2. We need to reload version/perm anyway to validate
/// 3. We use `unshift()` to set `len=9` which makes `lower_bound` find
///    the position after the layer pointer slot
///
/// # Invariant
///
/// `leaf` is always non-null. A `LayerContext` is only created when
/// descending into a sublayer from a valid leaf position.
#[derive(Clone, Copy)]
pub struct LayerContext<L> {
    /// Parent layer root.
    pub root: *const u8,

    /// Parent leaf (where the layer pointer was found).
    ///
    /// Uses `NonNull` because a layer context is only created from a valid
    /// leaf - we never push a null leaf to the layer stack.
    pub leaf: NonNull<L>,
}

impl<L> std::fmt::Debug for LayerContext<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerContext")
            .field("root", &self.root)
            .field("leaf", &self.leaf)
            .finish()
    }
}

impl<L> LayerContext<L> {
    /// Create a new layer context.
    ///
    /// # Panics
    ///
    /// Debug-panics if `leaf` is null (indicates a bug in scan logic).
    #[inline(always)]
    pub fn new(root: *const u8, leaf: *mut L) -> Self {
        debug_assert!(!leaf.is_null(), "LayerContext::new called with null leaf");
        Self {
            root,
            // SAFETY: We just checked that leaf is non-null
            leaf: unsafe { NonNull::new_unchecked(leaf) },
        }
    }

    /// Get the leaf as a raw mutable pointer.
    #[inline(always)]
    pub const fn leaf_ptr(&self) -> *mut L {
        self.leaf.as_ptr()
    }
}

/// Stack of parent layer contexts for sublayer navigation.
///
/// Most keys need very few layers (32 bytes = 4 layers maximum in typical
/// use). We use `SmallVec` to avoid heap allocation for common cases.
///
/// # Capacity
///
/// - Inline storage: 4 elements (handles keys up to 32 bytes)
/// - Spills to heap for deeper nesting (rare)
///
/// # Usage
///
/// ```ignore
/// let mut layer_stack = LayerStack::<LeafNode24<S>>::new();
///
/// // On layer descent
/// layer_stack.push(LayerContext::new(current_root, current_leaf));
///
/// // On layer ascent
/// if let Some(parent) = layer_stack.pop() {
///     stack.set_root(parent.root);
///     stack.set_leaf(parent.leaf_ptr());
/// }
///
pub type LayerStack<L> = SmallVec<[LayerContext<L>; 4]>;

// ============================================================================
//  ScanSnapshot - Captured slot data for emission
// ============================================================================

/// Snapshot of slot data captured for emission.
///
/// After version validation succeeds, we capture the value output and
/// key length so the caller can emit them without re-reading the leaf.
///
/// # Lifetime
///
/// The `value` is cloned (for Arc types) or copied (for inline types)
/// at capture time. This makes it safe to use even if the leaf is
/// modified after validation.
#[derive(Debug, Clone)]
pub struct ScanSnapshot<S: ValueSlot> {
    /// The value output (Arc<V> clone or V copy).
    pub value: S::Output,

    /// The key length at current layer.
    ///
    /// For inline keys: 0-8 (actual length)
    /// For suffix keys:`8 + suffix.len()`
    pub key_len: usize,
}

// ============================================================================
//  ScanSnapshotPtr - Zero-copy snapshot for scan_ref
// ============================================================================

/// Snapshot of slot data for zero-copy emission.
///
/// Unlike [`ScanSnapshot`] which clones the value (Arc increment for `LeafValue`),
/// this stores a typed pointer and defers dereferencing to the caller.
///
/// # Type Parameter
///
/// - `V`: The value type stored in the tree (e.g., `u64` for `MassTree24<u64>`)
///
/// # Safety
///
/// The `value_ptr` is only valid while:
/// 1. The guard is held (prevents deallocation)
/// 2. The version hasn't changed (leaf not modified)
///
/// The caller must dereference the pointer immediately after receiving it,
/// within the same guard scope.
///
/// # Performance
///
/// This eliminates Arc atomic increment/decrement per scanned entry,
/// which can improve scan throughput by 2-3x for Arc-based trees.
#[derive(Debug, Clone, Copy)]
pub struct ScanSnapshotPtr<V> {
    /// Typed pointer to the value data.
    ///
    /// - For `LeafValue<V>`: Points to Arc<V>'s data
    /// - For `LeafValueIndex<V>`: Points to Box<V>'s data
    pub value_ptr: *const V,

    /// The key length at current layer (same as [`ScanSnapshot`]).
    pub key_len: usize,
}

impl<V> ScanSnapshotPtr<V> {
    /// Create a new zero-copy snapshot.
    #[inline(always)]
    pub const fn new(value_ptr: *const V, key_len: usize) -> Self {
        Self { value_ptr, key_len }
    }

    /// Create from a raw `*const u8` pointer by casting.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer actually points to a valid `V`.
    #[inline(always)]
    pub const fn from_raw(ptr: *const u8, key_len: usize) -> Self {
        Self {
            value_ptr: ptr.cast::<V>(),
            key_len,
        }
    }

    /// Get a reference to the value.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. The guard is still held
    /// 2. The version hasn't changed since the snapshot was created
    #[inline(always)]
    pub const unsafe fn value_ref(&self) -> &V {
        // SAFETY: Caller ensures pointer is valid
        unsafe { &*self.value_ptr }
    }
}

impl<S: ValueSlot> ScanSnapshot<S> {
    /// Create a new scan snapshot.
    #[inline(always)]
    pub const fn new(value: S::Output, key_len: usize) -> Self {
        Self { value, key_len }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
