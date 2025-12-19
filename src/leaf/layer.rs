//! Extending [`LeafNode`] implementation for layer descent

use std::ptr as StdPtr;
use std::sync::Arc;

use crate::key::Key;
use crate::slot::ValueSlot;

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
    pub fn make_layer_root(&mut self) {
        self.parent = StdPtr::null_mut();
        self.version.mark_root();
    }

    /// Create a new leaf node configured as a layer root.
    #[must_use]
    pub fn new_layer_root() -> Box<Self> {
        let mut node: Box<Self> = Self::new();
        node.make_layer_root();

        node
    }

    /// Check if this node is layer root.
    #[inline]
    #[must_use]
    pub fn is_layer_root(&self) -> bool {
        self.version.is_root() && self.parent.is_null()
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
    pub fn assign_initialize_for_layer(&mut self, slot: usize, ikey: u64) {
        self.ikey0[slot] = ikey;
        self.keylenx[slot] = LAYER_KEYLENX;

        // Layer pointer set separately via set_layer
        self.leaf_values[slot] = S::default();
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
    pub fn set_layer(&mut self, slot: usize, layer: *mut u8) {
        self.leaf_values[slot] = S::layer(layer);
        self.keylenx[slot] = LAYER_KEYLENX;
    }

    /// Get the layer pointer from a slot.
    ///
    /// # Returns
    /// Some(pointer) if slot contains a layer, None otherwise.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    #[must_use]
    pub fn get_layer(&self, slot: usize) -> Option<*mut u8> {
        if self.keylenx[slot] >= LAYER_KEYLENX {
            self.leaf_values[slot].try_layer()
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
    /// * `output` - The output value (from `S::into_output`). Wrapped in `Option`
    ///   for caller convenience (e.g., from `try_clone_arc()`), but `None` will panic.
    ///
    /// # Panics
    /// Panics if `output` is `None`. For layer pointer setup, use
    /// [`assign_initialize_for_layer`](Self::assign_initialize_for_layer) instead.
    ///
    /// FIXED: Previously passed `KSUF_KEYLENX` to `assign_output()` which has
    /// `debug_assert!(key_len <= 8)`. Now we pass the inline length and set
    /// `keylenx` separately for suffix keys.
    pub fn assign_from_key(&mut self, slot: usize, key: &Key<'_>, output: Option<S::Output>) {
        // Calculate inline length (0-8)
        #[expect(
            clippy::cast_possible_truncation,
            reason = "current_len() capped at slice length, min(8) ensures <= 8"
        )]
        let inline_len: u8 = key.current_len().min(8) as u8;

        // INVARIANT: output must be Some - caller must ensure the source slot contains
        // a value, not a layer pointer. Layer creation only happens when an existing
        // VALUE conflicts with a new key, so try_clone_arc() should always succeed.
        #[expect(clippy::expect_used, reason = "invariant: source slot must contain value")]
        let value = output.expect(
            "assign_from_key: output cannot be None (source slot was not a value); \
             use assign_initialize_for_layer for layer pointer setup",
        );

        // Assign with inline length (satisfies assign_output's debug_assert)
        self.assign_output(slot, key.ikey(), inline_len, value);

        // If key has suffix, override keylenx and store suffix
        if key.has_suffix() {
            self.keylenx[slot] = KSUF_KEYLENX;
            self.assign_ksuf(slot, key.suffix());
        }
    }
}

/// Arc-mode specific layer methods for `LeafNode<LeafValue<V>, WIDTH>`.
impl<V, const WIDTH: usize> LeafNode<LeafValue<V>, WIDTH> {
    /// Assigns a slot from a `Key` and optional Arc-wrapped value.
    ///
    /// Convenience method for Arc mode. For generic code, use `assign_from_key`.
    pub fn assign_from_key_arc(&mut self, slot: usize, key: &Key<'_>, value: Option<Arc<V>>) {
        self.assign_from_key(slot, key, value);
    }
}
