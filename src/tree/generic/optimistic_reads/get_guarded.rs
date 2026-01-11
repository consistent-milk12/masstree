use seize::LocalGuard;

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric, ValueSlot,
    key::Key,
    leaf_trait::LayerCapableLeaf,
    tree::generic::optimistic_reads::{LookupResult, search_leaf_multi_layer},
    value::traits::LeafValueLoad,
};

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Get a value by key, returning a clone of the output.
    ///
    /// This is the main read path. Uses optimistic concurrency control
    /// with version validation.
    #[inline]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<S::Output> {
        let mut key: Key = Key::new(key);

        // Find root
        let layer_root: *const u8 = self.load_root_ptr_generic(guard);

        if layer_root.is_null() {
            return None;
        }

        // Navigate from root to leaf (handles internode traversal)
        let mut leaf_ptr: *mut L =
            self.reach_leaf_concurrent_generic(layer_root, &key, false, guard);

        loop {
            // SAFETY: leaf_ptr is valid (from reach_leaf_concurrent_generic)
            let leaf: &L = unsafe { &*leaf_ptr };

            // P1-1: Prefetch ikeys/permutation while waiting for stable version
            leaf.prefetch_for_search();

            let v0: u32 = leaf.version().stable();
            let result: LookupResult = search_leaf_multi_layer::<S, L>(leaf, &key);

            match result {
                LookupResult::ValueSlot(slot) => {
                    let output: Option<S::Output> = leaf.try_load_output(slot);

                    if leaf.version().has_changed(v0) {
                        // Only do full B-link handling if split occurred
                        // For update-only, simple retry is faster
                        if leaf.version().has_split_no_compiler_fence(v0) {
                            let (new_ptr, _, _) = self.handle_version_change(leaf, &key, v0, guard);
                            leaf_ptr = new_ptr;
                        }
                        continue;
                    }

                    return output;
                }

                LookupResult::Layer(layer_ptr) => {
                    if leaf.version().has_changed(v0) {
                        if leaf.version().has_split_no_compiler_fence(v0) {
                            let (new_ptr, _, _) = self.handle_version_change(leaf, &key, v0, guard);
                            leaf_ptr = new_ptr;
                        }
                        continue;
                    }

                    key.shift();

                    // Descend into sublayer - need to reach leaf again
                    leaf_ptr = self.reach_leaf_concurrent_generic(layer_ptr, &key, true, guard);
                }

                LookupResult::NotFound => {
                    // Check if leaf is being modified - retry search
                    if leaf.version().is_dirty() {
                        continue;
                    }

                    // Check B-link chain for key (concurrent split may have moved it)
                    let target_ikey: u64 = key.ikey();
                    if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                        leaf_ptr = next_ptr;
                        continue;
                    }

                    return None;
                }
            }
        }
    }
}
