//! Extending [`LeafNode`] implementation for layer descent

use std::ptr as StdPtr;
use std::sync::Arc;

use crate::key::Key;

use super::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode, LeafValue};

impl<V, const WIDTH: usize> LeafNode<V, WIDTH> {
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
    /// (no value yet, the layer will be assigned seperately).
    ///
    /// # Arguments
    /// * `slot` - Physical slot index
    /// * `layer` - Pointer to the sublayer's root node
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    pub fn assign_initialize_for_layer(&mut self, slot: usize, ikey: u64) {
        self.ikey0[slot] = ikey;
        self.keylenx[slot] = LAYER_KEYLENX;

        // Layer pointer set seperately
        self.leaf_values[slot] = LeafValue::Empty;
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
    pub fn set_layer(&mut self, slot: usize, layer: *mut u8) {
        self.leaf_values[slot] = LeafValue::Layer(layer);
        self.keylenx[slot] = LAYER_KEYLENX;
    }

    /// Get the layer pointer from a slot.
    ///
    /// # Returns
    /// Some(pointer) if slot contains a layer, None otherwise.
    #[expect(
        clippy::indexing_slicing,
        reason = "slot from permuter, valid by construction"
    )]
    #[must_use]
    pub const fn get_layer(&self, slot: usize) -> Option<*mut u8> {
        if self.keylenx[slot] >= LAYER_KEYLENX {
            self.leaf_values[slot].try_as_layer()
        } else {
            None
        }
    }

    /// Assigns a slot from a `Key` and optional value.
    ///
    /// If value is `Some`, assigns ikey, keylenx, value, and suffix (if any).
    /// If value is `None`, initializes the slot for layer descent.
    ///
    /// This is the canonical way to populate a slot from key state during
    /// layer creation or insertion.
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index (0..WIDTH)
    /// * `key` - The key containing ikey and suffix information
    /// * `value` - Optional Arc-wrapped value; `None` means this slot will be a layer pointer
    pub fn assign_from_key(&mut self, slot: usize, key: &Key<'_>, value: Option<Arc<V>>) {
        if let Some(val) = value {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "current_len() is at most 8 (single ikey slice)"
            )]
            let keylenx: u8 = if key.has_suffix() {
                KSUF_KEYLENX
            } else {
                key.current_len() as u8
            };
            self.assign_arc(slot, key.ikey(), keylenx, val);
            if key.has_suffix() {
                self.assign_ksuf(slot, key.suffix());
            }
        } else {
            self.assign_initialize_for_layer(slot, key.ikey());
        }
    }
}
