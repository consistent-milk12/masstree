//! Filepath: `src/tree/range/scan_state.rs`
//!
//! Scan state machine types for range scan operations.

use crate::permuter::Permuter15;
use std::fmt::{self as StdFmt, Debug, Formatter};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::{self as StdPtr, NonNull};

use arrayvec::ArrayVec;

use crate::leaf15::LeafNode15;
use crate::ordering::READ_ORD;
use crate::policy::LeafPolicy;
use crate::policy::atomic_read_value;

use super::direction::Direction;

/// The shared scan state machine. Direction determines how `Find` advances.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    Emit,
    Find,
    Down,
    Up,
    Retry,
}

impl ScanState {
    #[inline(always)]
    pub const fn is_emit(self) -> bool {
        matches!(self, Self::Emit)
    }

    #[cfg(test)]
    pub const fn is_layer_transition(self) -> bool {
        matches!(self, Self::Down | Self::Up)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Continue,
    Ready,
    Exhausted,
}

/// Result of processing entries within one leaf in either direction.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafBatchResult {
    LeafExhausted,
    LayerEncountered,
    VersionChanged,
    Stopped,
    BoundExceeded,
}

/// A guarded leaf position. Signed indices cover both exhausted boundaries:
/// -1 before the first slot, and `perm.size()` after the last slot.
pub struct ScanPosition<P: LeafPolicy, D: Direction> {
    root: *const u8,
    leaf: Option<NonNull<LeafNode15<P>>>,
    version: u32,
    perm: Permuter15,
    ki: isize,
    last_ikey: u64,
    direction: PhantomData<D>,
}

impl<P: LeafPolicy, D: Direction> ScanPosition<P, D> {
    #[inline]
    pub const fn new(root: *const u8) -> Self {
        Self {
            root,
            leaf: None,
            version: 0,
            perm: Permuter15::empty(),
            ki: 0,
            last_ikey: D::INITIAL_IKEY,
            direction: PhantomData,
        }
    }
    #[inline(always)]
    pub const fn root(&self) -> *const u8 {
        self.root
    }
    #[inline(always)]
    pub const fn leaf_ptr(&self) -> *mut LeafNode15<P> {
        match self.leaf {
            Some(leaf) => leaf.as_ptr(),
            None => StdPtr::null_mut(),
        }
    }
    #[inline(always)]
    pub const fn version(&self) -> u32 {
        self.version
    }
    #[inline(always)]
    pub const fn perm(&self) -> Permuter15 {
        self.perm
    }
    #[inline(always)]
    pub const fn ki(&self) -> isize {
        self.ki
    }
    #[inline(always)]
    pub fn kp(&self) -> Option<usize> {
        (self.ki >= 0 && self.ki.cast_unsigned() < self.perm.size())
            .then(|| self.perm.get(self.ki.cast_unsigned()))
    }
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.leaf.is_none()
    }
    #[inline(always)]
    pub const fn set_root(&mut self, root: *const u8) {
        self.root = root;
    }
    #[inline(always)]
    pub const fn set_leaf(&mut self, leaf: *mut LeafNode15<P>) {
        self.leaf = NonNull::new(leaf);
    }
    #[inline(always)]
    pub const fn set_ki(&mut self, ki: isize) {
        self.ki = ki;
    }
    #[inline(always)]
    pub const fn last_ikey(&self) -> u64 {
        self.last_ikey
    }
    #[inline(always)]
    pub const fn set_last_ikey(&mut self, ikey: u64) {
        self.last_ikey = ikey;
    }
    #[inline(always)]
    pub fn next(&mut self) {
        self.ki = D::step(self.ki);
    }
    #[inline(always)]
    pub const fn update_state(&mut self, version: u32, perm: Permuter15, ki: isize) {
        self.version = version;
        self.perm = perm;
        self.ki = ki;
    }
}

// ============================================================================
//  LayerContext and LayerStack
// ============================================================================

/// Parent layer context stored during sublayer descent.
#[derive(Clone, Copy)]
pub struct LayerContext<P: LeafPolicy> {
    /// Parent layer root.
    pub root: *const u8,

    /// Parent leaf (where the layer pointer was found).
    pub leaf: NonNull<LeafNode15<P>>,
}

impl<P: LeafPolicy> Debug for LayerContext<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("LayerContext")
            .field("root", &self.root)
            .field("leaf", &self.leaf)
            .finish()
    }
}

impl<P: LeafPolicy> LayerContext<P> {
    /// Create a new layer context.
    ///
    /// # Panics
    ///
    /// Panics if `leaf` is null.
    #[track_caller]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Infallible")]
    pub const fn new(root: *const u8, leaf: *mut LeafNode15<P>) -> Self {
        Self {
            root,
            leaf: NonNull::new(leaf).expect("LayerContext requires non-null leaf"),
        }
    }

    /// Get the leaf as a raw mutable pointer.
    #[inline(always)]
    pub const fn leaf_ptr(&self) -> *mut LeafNode15<P> {
        self.leaf.as_ptr()
    }
}

/// Stack of parent layer contexts for sublayer navigation.
pub struct LayerStack<P: LeafPolicy> {
    /// Inline storage for the common case (up to 6 layers).
    inline: ArrayVec<LayerContext<P>, 6>,

    /// Heap spillover for deep keys (> 6 layers). `None` until first spill.
    overflow: Option<Vec<LayerContext<P>>>,
}

impl<P: LeafPolicy> LayerStack<P> {
    /// Create a new empty layer stack.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            inline: ArrayVec::new_const(),
            overflow: None,
        }
    }

    /// Push a layer context onto the stack.
    #[inline]
    pub fn push(&mut self, ctx: LayerContext<P>) {
        if let Some(ref mut overflow) = self.overflow {
            overflow.push(ctx);
        } else if let Err(err) = self.inline.try_push(ctx) {
            // Spill: move all inline elements to heap + the rejected element.
            let mut heap = Vec::with_capacity(8);

            heap.extend(self.inline.drain(..));
            heap.push(err.element());
            self.overflow = Some(heap);
        }
    }

    /// Pop the most recent layer context.
    #[inline]
    pub fn pop(&mut self) -> Option<LayerContext<P>> {
        if let Some(ref mut overflow) = self.overflow {
            let result = overflow.pop();
            // Reclaim heap when empty (shrink back to inline-only mode).
            if overflow.is_empty() {
                self.overflow = None;
            }
            result
        } else {
            self.inline.pop()
        }
    }

    /// Returns `true` if the stack has no elements.
    #[cfg(test)]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        match self.overflow {
            Some(ref overflow) => overflow.is_empty(),
            None => self.inline.is_empty(),
        }
    }

    /// Remove all elements from the stack.
    #[inline]
    #[allow(dead_code, reason = "API completeness")]
    pub fn clear(&mut self) {
        self.overflow = None;
        self.inline.clear();
    }

    /// Returns the total number of elements.
    #[cfg(test)]
    pub const fn len(&self) -> usize {
        match self.overflow.as_ref() {
            Some(v) => v.len(),
            None => self.inline.len(),
        }
    }
}

// ============================================================================
//  ScanSnapshot - Captured slot data for emission
// ============================================================================

/// Snapshot of slot data captured for emission.
#[derive(Debug, Clone)]
pub struct ScanSnapshot<P: LeafPolicy> {
    /// The value output (`ValuePtr<V>` clone or V copy).
    pub value: P::Output,

    /// The key length at current layer.
    #[allow(dead_code, reason = "Used in tests and part of snapshot API")]
    pub key_len: usize,
}

// ============================================================================
//  ScanSnapshotPtr - Zero-copy snapshot for scan_ref
// ============================================================================

/// Snapshot of slot data for zero-copy emission.
///
/// Unlike [`ScanSnapshot`] which clones the value,
/// this stores a typed pointer and defers dereferencing to the caller.
#[derive(Debug, Clone, Copy)]
pub struct ScanSnapshotPtr<V> {
    /// Typed pointer to the value data.
    pub value_ptr: *const V,

    /// The key length at current layer (same as [`ScanSnapshot`]).
    #[allow(dead_code, reason = "Used in tests")]
    pub key_len: usize,
}

#[allow(dead_code, reason = "Zero-copy scan snapshot API")]
impl<V> ScanSnapshotPtr<V> {
    /// Create a new zero-copy snapshot.
    #[inline(always)]
    pub const fn new(value_ptr: *const V, key_len: usize) -> Self {
        Self { value_ptr, key_len }
    }

    /// Create from a raw `*const u8` pointer by casting.
    #[inline(always)]
    pub const fn from_raw(ptr: *const u8, key_len: usize) -> Self {
        Self {
            value_ptr: ptr.cast::<V>(),
            key_len,
        }
    }

    /// Get a reference to the value.
    ///
    /// Only sound for non-write-through types. For `CAN_WRITE_THROUGH` types,
    /// use [`value_copy`](Self::value_copy) instead.
    ///
    /// # Safety
    ///
    /// Caller ensures pointer is valid, guard is held, and no concurrent
    /// write-through update can modify the pointed-to data.
    #[inline(always)]
    pub const unsafe fn value_ref(&self) -> &V {
        // SAFETY: Caller ensures pointer is valid and guard is held.
        unsafe { &*self.value_ptr }
    }

    /// Atomically read the value by copy.
    ///
    /// Sound for `CAN_WRITE_THROUGH` types where the pointed-to data may be
    /// concurrently modified by `write_through_update`.
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and the guard must be held.
    /// - `CAN_WRITE_THROUGH` must be true (size 1/2/4/8, natural alignment).
    #[inline(always)]
    pub unsafe fn value_copy(&self) -> V {
        // SAFETY: CAN_WRITE_THROUGH guarantees size 1/2/4/8 with natural
        // alignment. Atomic read avoids aliasing violation.
        unsafe { atomic_read_value::<V>(self.value_ptr.cast(), READ_ORD) }
    }

    /// Resolve a value reference, using atomic copy for write-through types.
    ///
    /// For write-through policies, atomically copies the value into `scratch`
    /// and returns a reference to it. For non-write-through policies, returns
    /// a direct reference to the pointed-to value.
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and protected by a guard.
    /// - For write-through: pointer must be valid for atomic read.
    /// - For non-write-through: no concurrent write-through update may modify
    ///   the pointed-to data.
    #[inline(always)]
    pub unsafe fn resolve_value_ref<'a, P: LeafPolicy<Value = V>>(
        &self,
        scratch: &'a mut MaybeUninit<V>,
    ) -> &'a V {
        if P::can_write_through() {
            // SAFETY: CAN_WRITE_THROUGH guarantees atomic read is sound.
            *scratch = MaybeUninit::new(unsafe { self.value_copy() });

            // SAFETY: Just initialized above.
            unsafe { scratch.assume_init_ref() }
        } else {
            // SAFETY: Caller ensures pointer is valid and no concurrent modification.
            unsafe { &*self.value_ptr }
        }
    }
}

impl<P: LeafPolicy> ScanSnapshot<P> {
    /// Create a new scan snapshot.
    #[inline(always)]
    pub const fn new(value: P::Output, key_len: usize) -> Self {
        Self { value, key_len }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;

#[cfg(test)]
#[path = "scan_state/reverse_tests.rs"]
mod reverse_tests;
