use std::sync::Arc;
use std::{cmp::Ordering, ptr as StdPtr};

use crate::permuter::Permuter;
use crate::{key::Key, leaf::LeafNode};

use super::MassTree;

impl<V, const WIDTH: usize> MassTree<V, WIDTH> {
    /// Create a new layer when two keys share the same ikey.
    ///
    /// This is called when inserting a key that has the same 8-byte prefix
    /// as an existing entry but a different suffix.
    ///
    /// # Arguments
    /// * `leaf` - The leaf containing the conflicting slot
    /// * `slot` - Physical slot with existing key
    /// * `new_key` - The key being inserted (already shifted to current layer)
    /// * `new_value` - The value to insert
    ///
    /// # Returns
    /// A tuple of (layer_root, insert_slot) where:
    /// - `layer_root` is the new sub-layer to continue insertion
    /// - `insert_slot` is the slot in that layer for the new key
    fn make_new_layer(
        &mut self,
        leaf: &mut LeafNode<V, WIDTH>,
        slot: usize,
        new_key: &mut Key<'_>,
        new_value: Arc<V>,
    ) -> (*mut LeafNode<V, WIDTH>, usize) {
        // 1. Extract existing key's suffix and create a Key from it
        let existing_suffix: &[u8] = leaf.ksuf(slot).unwrap_or(&[]);
        let mut existing_key: Key<'_> = Key::from_suffix(existing_suffix);

        // 2. Get existing value before we modify the slot
        let existing_value = leaf.leaf_values[slot].try_clone_arc();

        // 3. Shift both keys to compare their next 8-byte slices
        new_key.shift();
        existing_key.shift();

        let mut cmp = existing_key.ikey().cmp(&new_key.ikey());

        // 4. Create twig chain while ikeys match
        let mut twig_head: Option<*mut LeafNode<V, WIDTH>> = None;
        let mut twig_tail: *mut LeafNode<V, WIDTH> = StdPtr::null_mut();

        while cmp.eq(&Ordering::Equal) {
            // Create intermediate layer node (single entry)
            let mut twig = LeafNode::<V, WIDTH>::new_layer_root();
            twig.assign_initialize_for_layer(0, existing_key.ikey());
            twig.permutation = Permuter::make_sorted(1);

            let twig_ptr = self.allocate_leaf(twig);

            // Link to previous twig
            if let Some(head) = twig_head {
                // Previous twig points to this one
                // Safety twig_tail is valid, we just created it
                unsafe {
                    (*twig_tail).set_layer(0, twig_ptr as *mut u8);
                }
            } else {
            }
        }

        todo!()
    }

    /// Allocate a leaf node in the arena and return a raw pointer.
    fn allocate_leaf(&mut self, node: Box<LeafNode<V, WIDTH>>) -> *mut LeafNode<V, WIDTH> {
        todo!()
    }
}
