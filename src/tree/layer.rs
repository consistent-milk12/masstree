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
    ///
    /// * `leaf` - The leaf containing the conflicting slot
    /// * `slot` - Physical slot with the existing key
    /// * `new_key` - The key being inserted (already shifted to current layer)
    /// * `new_value` - The value to insert
    ///
    /// # Returns
    ///
    /// A tuple of (`layer_root`, `insert_slot`) where:
    /// - `layer_root` is the new sublayer to continue insertion
    /// - `insert_slot` is the slot in that layer for the new key
    #[expect(dead_code, reason = "Will be used in Phase 2 layer support")]
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
        //
        // Use leaf_value() accessor here.
        let existing_value: Option<Arc<V>> = leaf.leaf_value(slot).try_clone_arc();

        // 3. Shift both keys to compare their next 8-byte slices
        new_key.shift();
        existing_key.shift();

        let mut cmp: Ordering = existing_key.ikey().cmp(&new_key.ikey());

        // 4. Create twig chain while ikeys match
        let mut twig_head: Option<*mut LeafNode<V, WIDTH>> = None;
        let mut twig_tail: *mut LeafNode<V, WIDTH> = StdPtr::null_mut();

        while cmp.eq(&Ordering::Equal) {
            // Create intermediate layer node (single entry)
            let mut twig: Box<LeafNode<V, WIDTH>> = LeafNode::<V, WIDTH>::new_layer_root();
            twig.assign_initialize_for_layer(0, existing_key.ikey());
            twig.set_permutation(Permuter::make_sorted(1));

            let twig_ptr: *mut LeafNode<V, WIDTH> = self.store_leaf_in_arena(twig);

            // Link to previous twig
            if twig_head.is_some() {
                // Previous twig points to this one
                //  SAFETY: twig_tail is valid, we just created it.
                unsafe {
                    (*twig_tail).set_layer(0, twig_ptr.cast::<u8>());
                }
            } else {
                twig_head = Some(twig_ptr);
            }

            twig_tail = twig_ptr;

            // Shift both keys again
            existing_key.shift();
            new_key.shift();
            cmp = existing_key.ikey().cmp(&new_key.ikey());
        }

        // 5. Create final leaf with both keys (they now have different ikeys)
        let mut final_leaf = LeafNode::<V, WIDTH>::new_layer_root();

        // Determine ordering: existing vs new
        let (first_key, first_val, second_key, second_val, new_slot) = match cmp {
            Ordering::Less => {
                // existing < new : existing at slot 0, new at slot 1
                (existing_key, existing_value, *new_key, Some(new_value), 1)
            }

            Ordering::Greater => {
                // existing > new: new at slot 0, existing at slot 1
                (*new_key, Some(new_value), existing_key, existing_value, 0)
            }

            Ordering::Equal => {
                unreachable!("ikeys should differ after twig chain")
            }
        };

        // Assign both slots using the unified helper
        final_leaf.assign_from_key(0, &first_key, first_val);
        final_leaf.assign_from_key(1, &second_key, second_val);

        // Set up permutation (size 2, but slot for new key not yet "inserted")
        // The caller will call finish_insert to add the new slot to permutation
        let mut perm: Permuter<WIDTH> = Permuter::make_sorted(2);
        perm.remove_to_back(new_slot); // Remove new slot from "visible" permutation
        final_leaf.set_permutation(perm);

        let final_ptr: *mut LeafNode<V, WIDTH> = self.store_leaf_in_arena(final_leaf);

        // 6. Link twig chain to final leaf
        if twig_tail.is_null() {
            // No twig chain, final leaf is directly linked
            twig_head = Some(final_ptr);
        } else {
            // SAFETY: twig_tail is valid
            unsafe {
                (*twig_tail).set_layer(0, final_ptr.cast::<u8>());
            }
        }

        // 7. Update original slot to point to layer
        // INVARIANT: twig_head is always set because:
        // - Either the while loop ran at least once (setting twig_head), OR
        // - The loop didn't run and we set twig_head = Some(final_ptr) above
        #[expect(clippy::option_if_let_else, reason = "match is more readable for invariant")]
        let layer_root: *mut LeafNode<V, WIDTH> = match twig_head {
            Some(ptr) => ptr,
            None => unreachable!("twig_head is always set before this point"),
        };
        leaf.set_layer(slot, layer_root.cast::<u8>());

        (final_ptr, new_slot)
    }
}
