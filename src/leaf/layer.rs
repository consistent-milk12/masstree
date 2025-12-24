//! Extending [`LeafNode`] implementation for layer descent

use std::ptr as StdPtr;
use std::sync::Arc;

use crate::key::Key;
use crate::ordering::{READ_ORD, WRITE_ORD};
use crate::slot::ValueSlot;
use seize::LocalGuard;

use super::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode, LeafValue};

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    /// Convert this leaf into a layer root.
    ///
    /// Sets up the node to serve as the root of a sub-layer:
    /// - Sets parent to null
    /// - Marks version to root
    ///
    /// # Safety
    /// Caller must ensure this node is not currently part of another tree.
    pub fn make_layer_root(&self) {
        self.set_parent(StdPtr::null_mut());
        self.version.mark_root();
    }

    /// Create a new leaf node configured as a layer root.
    #[must_use]
    pub fn new_layer_root() -> Box<Self> {
        let node: Box<Self> = Box::new(Self::new_with_root(true));
        node.make_layer_root();
        node
    }

    /// Check if this node is layer root.
    #[must_use]
    #[inline(always)]
    pub fn is_layer_root(&self) -> bool {
        self.version.is_root() && self.parent().is_null()
    }

    /// Assign a slot for use in a layer (initialize for layer transition).
    ///
    /// This sets up a slot with an ikey but marks it as a layer pointer
    /// (no value yet, the layer will be assigned separately).
    ///
    /// # Arguments
    /// * `slot` - Physical slot index
    /// * `ikey` - The 8-byte key for this slot
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    pub fn assign_initialize_for_layer(&self, slot: usize, ikey: u64) {
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(LAYER_KEYLENX, WRITE_ORD);

        // Layer pointer set separately via set_layer
        self.leaf_values[slot].store(StdPtr::null_mut(), WRITE_ORD);
    }

    /// Set a slot's value to a layer pointer.
    ///
    /// # Arguments
    /// * `slot` - Physical slot index
    /// * `layer` - Pointer to the sublayer's root node
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    #[inline(always)]
    pub fn set_layer(&self, slot: usize, layer: *mut u8) {
        self.leaf_values[slot].store(layer, WRITE_ORD);
        self.keylenx[slot].store(LAYER_KEYLENX, WRITE_ORD);
    }

    /// Get the layer pointer from a slot.
    ///
    /// # Returns
    /// Some(pointer) if slot contains a layer, None otherwise.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    pub fn get_layer(&self, slot: usize) -> Option<*mut u8> {
        if self.keylenx[slot].load(READ_ORD) >= LAYER_KEYLENX {
            let ptr: *mut u8 = self.leaf_values[slot].load(READ_ORD);
            if ptr.is_null() { None } else { Some(ptr) }
        } else {
            None
        }
    }

    /// Assign a key-value pair from a [`Key`] iterator.
    ///
    /// This used when creating new layer entries. The key's current
    /// position determines the ikey and suffix.
    ///
    /// # Arguments
    /// * `slot` - Physical slot index (0..WIDTH)
    /// * `key` - The key containing ikey and suffix information
    /// * `arc` - The Arc-wrapped value. Wrapped in `Option`
    ///   for caller convenience (e.g., from `try_clone_arc()`), but `None` will panic.
    /// * `guard` - The seize guard for deferred retirement of old suffix
    ///
    /// # Panics
    /// Panics if `arc` is `None`. For layer pointer setup, use
    /// [`assign_initialize_for_layer`](Self::assign_initialize_for_layer) instead.
    ///
    /// # Safety
    ///
    /// Caller must ensure `guard` comes from this tree's collector.
    #[expect(
        clippy::indexing_slicing,
        reason = "slot is caller-provided index 0..WIDTH; bounds enforced by caller contract"
    )]
    pub unsafe fn assign_from_key<V>(
        &self,
        slot: usize,
        key: &Key<'_>,
        arc: Option<Arc<V>>,
        guard: &LocalGuard<'_>,
    ) {
        // Calculate inline length (0-8)
        #[expect(
            clippy::cast_possible_truncation,
            reason = "current_len() capped at slice length, min(8) ensures <= 8"
        )]
        let inline_len: u8 = key.current_len().min(8) as u8;

        // INVARIANT: arc must be Some - caller must ensure the source slot contains
        // a value, not a layer pointer. Layer creation only happens when an existing
        // VALUE conflicts with a new key, so try_clone_arc() should always succeed.
        #[expect(
            clippy::expect_used,
            reason = "invariant: source slot must contain value"
        )]
        let value: Arc<V> = arc.expect(
            "assign_from_key: arc cannot be None (source slot was not a value); \
             use assign_initialize_for_layer for layer pointer setup",
        );

        // Store the ikey
        self.set_ikey(slot, key.ikey());

        // Store the Arc as raw pointer
        let ptr: *mut u8 = Arc::into_raw(value).cast_mut().cast::<u8>();
        self.leaf_values[slot].store(ptr, WRITE_ORD);

        // Set keylenx - either inline length or KSUF_KEYLENX for suffix keys
        if key.has_suffix() {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            // SAFETY: guard requirement passed through from caller
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            self.keylenx[slot].store(inline_len, WRITE_ORD);
        }
    }
}

/// Arc-mode specific layer methods for `LeafNode<LeafValue<V>, WIDTH>`.
impl<V, const WIDTH: usize> LeafNode<LeafValue<V>, WIDTH> {
    /// Assigns a slot from a `Key` and optional Arc-wrapped value.
    ///
    /// Convenience method for Arc mode. For generic code, use `assign_from_key`.
    ///
    /// # Safety
    ///
    /// Caller must ensure `guard` comes from this tree's collector.
    pub unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &Key<'_>,
        value: Option<Arc<V>>,
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: guard requirement passed through from caller
        unsafe { self.assign_from_key(slot, key, value, guard) };
    }
}
