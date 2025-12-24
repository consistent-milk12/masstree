use std::{
    cmp::Ordering,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering, fence as atomicFence},
    },
};

use parking_lot::{Condvar, Mutex};
use seize::{Collector, Guard, LocalGuard};

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric, TreeInternode, TreePermutation,
    key::Key,
    leaf_trait::LayerCapableLeaf,
    leaf24::LAYER_KEYLENX,
    nodeversion::NodeVersion,
    tree::{CasInsertResultGeneric, InsertError, InsertSearchResultGeneric},
    value::LeafValue,
};

impl<V, L, A> MassTreeGeneric<V, L, A>
where
    V: Send + Sync + 'static,
    L: LayerCapableLeaf<V>,
    A: NodeAllocatorGeneric<LeafValue<V>, L>,
{
    /// Create a new empty `MassTreeGeneric` with the given allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn with_allocator(allocator: A) -> Self {
        // Create root leaf and register with allocator.
        let root_leaf: Box<L> = L::new_root_boxed();
        let root_ptr: *mut L = allocator.alloc_leaf(root_leaf);

        Self {
            collector: Collector::new(),
            allocator,
            root_ptr: AtomicPtr::new(root_ptr.cast::<u8>()),
            count: AtomicUsize::new(0),
            parent_set_condvar: Condvar::new(),
            parent_set_mutex: Mutex::new(()),
            _marker: PhantomData,
        }
    }

    /// Enter a protected region and return a guard.
    ///
    /// The guard protects any pointers loaded during its lifetime from being
    /// reclaimed. Call this before reading tree nodes or values.
    #[must_use]
    #[inline(always)]
    pub fn guard(&self) -> LocalGuard<'_> {
        self.collector.enter()
    }

    /// Get the number of keys in the tree.
    ///
    /// This is O(1) as we track the count incrementally.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load(AtomicOrdering::Relaxed)
    }

    /// Check if the tree is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.root_is_leaf_generic() {
            // SAFETY: root_is_leaf_generic confirmed this is a leaf
            let leaf_ptr: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            unsafe { (*leaf_ptr).is_empty() }
        } else {
            // Internode implies at least one key
            false
        }
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Notify all threads waiting for a parent pointer to be set.
    ///
    /// Called after setting a node's parent pointer during split propagation.
    /// This wakes up any threads waiting in the condvar.
    #[inline(always)]
    pub(crate) fn notify_parent_set(&self) {
        self.parent_set_condvar.notify_all();
    }

    /// Wait for a parent pointer to be set, with timeout.
    ///
    /// Returns `true` if notified (should recheck condition), `false` on timeout.
    #[inline(always)]
    fn wait_for_parent_set(&self, timeout: std::time::Duration) -> bool {
        !self
            .parent_set_condvar
            .wait_for(&mut self.parent_set_mutex.lock(), timeout)
            .timed_out()
    }

    /// Load the root pointer atomically.
    #[inline(always)]
    pub(crate) fn load_root_ptr_generic(&self, _guard: &LocalGuard<'_>) -> *const u8 {
        self.root_ptr.load(AtomicOrdering::Acquire)
    }

    /// Compare-and-swap the root pointer atomically.
    #[inline(always)]
    pub(crate) fn cas_root_ptr_generic(
        &self,
        expected: *mut u8,
        new: *mut u8,
    ) -> Result<(), *mut u8> {
        self.root_ptr
            .compare_exchange(
                expected,
                new,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            )
            .map(|_| ())
    }

    /// Check if the current root is a leaf node.
    ///
    /// # Safety
    /// Reads the version field through a raw pointer. The `root_ptr` must
    /// point to a valid node (guaranteed by construction).
    #[inline(always)]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "root_ptr points to L or L::Internode, both have NodeVersion \
                  as first field with proper alignment"
    )]
    fn root_is_leaf_generic(&self) -> bool {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: `root_ptr` always points to a valid node.
        // `NodeVersion` is the first field of both leaf and internode types.
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();

        unsafe { (*version_ptr).is_leaf() }
    }

    /// Get a mutable reference to the allocator.
    #[inline(always)]
    pub(crate) const fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    /// Get an immutable reference to the allocator.
    #[inline(always)]
    pub(crate) const fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Get a reference to the collector.
    #[inline(always)]
    pub(crate) const fn collector(&self) -> &Collector {
        &self.collector
    }

    /// Increment the entry count.
    #[inline(always)]
    pub(crate) fn inc_count(&self) {
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    // ========================================================================
    //  Generic Tree Traversal
    // ========================================================================

    /// Reach the leaf node that should contain the given key.
    ///
    /// Traverses from root through internodes to find the target leaf.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// Reference to the leaf node that contains or should contain the key.
    #[inline(always)]
    pub(crate) fn reach_leaf_generic(&self, key: &Key<'_>) -> &L {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both L and L::Internode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or L::Internode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf node
            unsafe { &*(root.cast::<L>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &L::Internode = unsafe { &*(root.cast::<L::Internode>()) };
            self.reach_leaf_via_internode_generic(internode, key)
        }
    }

    /// Traverse from an internode down to the target leaf.
    ///
    /// Uses generic internode search to find the correct child at each level.
    #[expect(
        clippy::unused_self,
        reason = "Method signature matches reach_leaf pattern"
    )]
    fn reach_leaf_via_internode_generic(&self, mut inode: &L::Internode, key: &Key<'_>) -> &L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();

        loop {
            // Find child index using generic search
            let child_idx: usize =
                upper_bound_internode_generic::<LeafValue<V>, L::Internode>(target_ikey, inode);
            let child_ptr: *mut u8 = inode.child(child_idx);

            // Prefetch child node
            prefetch_read(child_ptr);

            // Check child type via NodeVersion
            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to L or L::Internode, both properly aligned"
            )]
            let child_version: &NodeVersion = unsafe { &*(child_ptr.cast::<NodeVersion>()) };

            if child_version.is_leaf() {
                // SAFETY: is_leaf() confirms this is a leaf
                return unsafe { &*(child_ptr.cast::<L>()) };
            }

            // Descend to child internode
            // SAFETY: !is_leaf() confirms InternodeNode
            inode = unsafe { &*(child_ptr.cast::<L::Internode>()) };
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    #[inline(always)]
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "Returns &mut L which requires &mut self for lifetime"
    )]
    pub(crate) fn reach_leaf_mut_generic(&mut self, key: &Key<'_>) -> &mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let root: *mut u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or L::Internode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf
            unsafe { &mut *(root.cast::<L>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &L::Internode = unsafe { &*(root.cast::<L::Internode>()) };

            let ikey: u64 = key.ikey();
            let child_idx: usize =
                upper_bound_internode_generic::<LeafValue<V>, L::Internode>(ikey, internode);
            let start_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node
            prefetch_read(start_ptr);

            let children_are_leaves: bool = internode.children_are_leaves();

            if children_are_leaves {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                unsafe { &mut *start_ptr.cast::<L>() }
            } else {
                // Iterative traversal for deeper trees
                // SAFETY: The returned pointer is valid for the tree's lifetime
                unsafe { &mut *Self::reach_leaf_mut_iterative_generic(start_ptr, ikey) }
            }
        }
    }

    /// Iterative leaf reach for deeply nested trees (generic version).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for as long as the tree's allocations remain valid.
    fn reach_leaf_mut_iterative_generic(mut current: *mut u8, ikey: u64) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &L::Internode = unsafe { &*(current.cast::<L::Internode>()) };
            let child_idx: usize =
                upper_bound_internode_generic::<LeafValue<V>, L::Internode>(ikey, internode);
            let child_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node
            prefetch_read(child_ptr);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                return child_ptr.cast::<L>();
            }

            current = child_ptr;
        }
    }

    // ========================================================================
    //  Generic Optimistic Read Path
    // ========================================================================

    /// Get a value by key.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`get_with_guard`](Self::get_with_guard) to amortize guard creation cost.
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<Arc<V>> {
        let guard = self.guard();
        self.get_with_guard(key, &guard)
    }

    /// Get a value by key using an explicit guard.
    ///
    /// Use this when performing multiple operations to amortize guard overhead.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<Arc<V>> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_concurrent_generic(&mut search_key, guard)
    }

    /// Internal concurrent get implementation with layer descent support.
    fn get_concurrent_generic(&self, key: &mut Key<'_>, guard: &LocalGuard<'_>) -> Option<Arc<V>> {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;

        // Start at tree root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);

        loop {
            // Find the actual layer root (handles layer root promotion)
            layer_root = self.maybe_parent_generic(layer_root);

            // Traverse to leaf for current layer
            let leaf_ptr: *mut L = self.reach_leaf_concurrent_generic(layer_root, key, guard);

            // Search in leaf with version validation
            // SAFETY: leaf_ptr protected by guard
            let leaf: &L = unsafe { &*leaf_ptr };
            let version: u32 = leaf.version().stable();

            // Check for deleted node
            if leaf.version().is_deleted() {
                continue; // Retry
            }

            // Load permutation
            let Ok(perm) = leaf.permutation_try() else {
                continue; // Frozen, retry
            };

            let target_ikey: u64 = key.ikey();

            #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
            let search_keylenx: u8 = if key.has_suffix() {
                KSUF_KEYLENX
            } else {
                key.current_len() as u8
            };

            // Search for matching key
            // CRITICAL: Only RECORD the snapshot (keylenx, ptr) here.
            // Do NOT interpret the pointer until AFTER version validation.
            // This prevents use-after-free if the slot was modified concurrently.
            let mut match_snapshot: Option<(u8, *mut u8)> = None;

            for i in 0..perm.size() {
                let slot: usize = perm.get(i);
                let slot_ikey: u64 = leaf.ikey(slot);

                if slot_ikey != target_ikey {
                    continue;
                }

                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                if slot_ptr.is_null() {
                    continue;
                }

                if slot_keylenx == search_keylenx {
                    // Potential exact match - verify suffix if present
                    let suffix_match: bool = if slot_keylenx == KSUF_KEYLENX {
                        // Both search key and slot have suffixes - compare them
                        leaf.ksuf_equals(slot, key.suffix())
                    } else {
                        // Neither has suffix (inline keys with keylenx 0-8)
                        true
                    };

                    if suffix_match {
                        // Record snapshot - do NOT interpret until version validated
                        match_snapshot = Some((slot_keylenx, slot_ptr));
                        break;
                    }
                } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
                    // Layer pointer - record for descent after validation
                    match_snapshot = Some((slot_keylenx, slot_ptr));
                    break;
                }
            }

            // Validate version BEFORE interpreting any pointers
            if leaf.version().has_changed(version) {
                // Version changed - retry from layer root
                continue;
            }

            // ================================================================
            //  VERSION VALIDATED - NOW SAFE TO INTERPRET SNAPSHOT
            // ================================================================
            // At this point, the (keylenx, ptr) pair is consistent.
            // We can safely determine if ptr is an Arc or layer pointer.

            if let Some((keylenx, ptr)) = match_snapshot {
                if keylenx >= LAYER_KEYLENX {
                    // Layer pointer - descend into sublayer
                    key.shift();
                    layer_root = ptr;
                    continue;
                }

                // Value Arc - NOW safe to clone
                // SAFETY: version validated, so keylenx correctly identifies ptr as Arc<V>
                let arc: Arc<V> = unsafe {
                    let value_ptr: *const V = ptr.cast();
                    Arc::increment_strong_count(value_ptr);
                    Arc::from_raw(value_ptr)
                };
                return Some(arc);
            }

            // Not found
            return None;
        }
    }

    /// Follow parent pointers to find the actual layer root.
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    fn maybe_parent_generic(&self, mut node: *const u8) -> *const u8 {
        loop {
            // SAFETY: node is valid, both types have NodeVersion as first field
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            let parent = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &L = unsafe { &*(node.cast::<L>()) };
                leaf.parent()
            } else {
                // SAFETY: !version.is_leaf() confirmed
                let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };
                inode.parent()
            };

            if parent.is_null() {
                return node;
            }

            node = parent;
        }
    }

    /// Traverse from layer root to target leaf with version validation.
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    fn reach_leaf_concurrent_generic(
        &self,
        start: *const u8,
        key: &Key<'_>,
        _guard: &LocalGuard<'_>,
    ) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();
        let mut node: *const u8 = start;

        loop {
            // SAFETY: node is valid
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            let v: u32 = version.stable();

            if version.is_leaf() {
                // Cast const to mut, then to L
                return node.cast_mut().cast::<L>();
            }

            // It's an internode
            // SAFETY: !is_leaf() confirmed
            let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };

            let child_idx: usize =
                upper_bound_internode_generic::<LeafValue<V>, L::Internode>(target_ikey, inode);
            let child: *mut u8 = inode.child(child_idx);

            prefetch_read(child);

            if child.is_null() {
                node = start;
                continue;
            }

            if inode.version().has_changed(v) {
                if inode.version().has_split(v) {
                    node = start;
                    continue;
                }
                continue;
            }

            node = child;
        }
    }

    // ========================================================================
    //  Generic CAS Insert Path
    // ========================================================================

    /// Maximum CAS retry attempts before falling back to locked path.
    const MAX_CAS_RETRIES_GENERIC: usize = 3;

    /// Try CAS-based lock-free insert.
    ///
    /// Attempts to insert a new key-value pair using optimistic concurrency.
    /// Returns result indicating success or reason for fallback.
    #[expect(clippy::too_many_lines, reason = "Complex concurrency logic")]
    pub(crate) fn try_cas_insert_generic(
        &self,
        key: &Key<'_>,
        value: &Arc<V>,
        guard: &LocalGuard<'_>,
    ) -> CasInsertResultGeneric<V> {
        use crate::leaf_trait::TreePermutation;
        use crate::link::{is_marked, unmark_ptr};
        use std::ptr as StdPtr;

        let ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let keylenx: u8 = key.current_len() as u8;

        // Suffix keys require locked path
        if key.has_suffix() {
            return CasInsertResultGeneric::ContentionFallback;
        }

        let mut retries: usize = 0;
        let mut leaf_ptr: *mut L = StdPtr::null_mut();
        let mut use_reach: bool = true;

        loop {
            // 1. Optimistic traversal to find target leaf
            if use_reach {
                let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
                layer_root = self.maybe_parent_generic(layer_root);
                leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, guard);
            } else {
                use_reach = true;
            }

            let leaf: &L = unsafe { &*leaf_ptr };

            // B-link advance if needed
            let advanced: &L = self.advance_to_key_by_bound_generic(leaf, key, guard);
            if !StdPtr::eq(advanced, leaf) {
                leaf_ptr = StdPtr::from_ref(advanced).cast_mut();
                use_reach = false;
                continue;
            }

            // 2. Get version (fail-fast if dirty)
            let version: u32 = leaf.version().value();
            if leaf.version().is_dirty() {
                return CasInsertResultGeneric::ContentionFallback;
            }

            // Check for frozen permutation
            let Ok(perm) = leaf.permutation_try() else {
                return CasInsertResultGeneric::ContentionFallback;
            };

            // 3. Search for key position
            let search_result = self.search_for_insert_generic(leaf, key, &perm);

            match search_result {
                InsertSearchResultGeneric::Found { slot } => {
                    return CasInsertResultGeneric::ExistsNeedLock { slot };
                }

                InsertSearchResultGeneric::Layer { slot, .. }
                | InsertSearchResultGeneric::Conflict { slot } => {
                    return CasInsertResultGeneric::LayerNeedLock { slot };
                }

                InsertSearchResultGeneric::NotFound { logical_pos } => {
                    // 4. Check if leaf has space
                    if perm.size() >= L::WIDTH {
                        return CasInsertResultGeneric::FullNeedLock;
                    }

                    // 5. Check slot-0 rule
                    let next_free: usize = perm.back();
                    if next_free == 0 && !leaf.can_reuse_slot0(ikey) {
                        return CasInsertResultGeneric::Slot0NeedLock;
                    }

                    // 6. Compute new permutation
                    let (new_perm, slot) = perm.insert_from_back_immutable(logical_pos);

                    // 7. Prepare Arc pointer
                    let arc_ptr: *mut u8 = Arc::into_raw(Arc::clone(value)) as *mut u8;

                    // 8. CAS slot value (NULL-claim semantics)
                    if leaf
                        .cas_slot_value(slot, StdPtr::null_mut(), arc_ptr)
                        .is_err()
                    {
                        let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                        retries += 1;

                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }

                        Self::backoff_generic(retries);

                        continue;
                    }

                    // 9. Validate version before writing key data
                    if leaf.version().has_changed_or_locked(version) {
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 10. Store key data
                    unsafe {
                        leaf.store_key_data_for_cas(slot, ikey, keylenx);
                    }

                    // 11. Secondary version check
                    if leaf.version().has_changed_or_locked(version) {
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 12. Verify slot ownership
                    if leaf.load_slot_value(slot) != arc_ptr {
                        let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 13. Final version check
                    if leaf.version().has_changed_or_locked(version) {
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 14. CAS permutation to publish
                    match leaf.cas_permutation_raw(perm, new_perm) {
                        Ok(()) => {
                            // Verify slot wasn't stolen
                            // NOTE: Do NOT increment count here - the slot was stolen by another
                            // thread, so our insert did not complete. The locked fallback path
                            // will handle the insert and increment the count if successful.
                            if leaf.load_slot_value(slot) != arc_ptr {
                                return CasInsertResultGeneric::ContentionFallback;
                            }

                            // Check for concurrent split
                            if leaf.version().is_splitting() {
                                let _ = leaf.version().stable();
                            }

                            let next_raw = leaf.next_raw();
                            if is_marked(next_raw) {
                                leaf.wait_for_split();
                            }

                            // Check if split moved our entry
                            let current_perm = leaf.permutation_wait();
                            let mut slot_in_perm = false;
                            for i in 0..current_perm.size() {
                                if current_perm.get(i) == slot {
                                    slot_in_perm = true;
                                    break;
                                }
                            }

                            if !slot_in_perm {
                                // Split moved our entry - success
                                self.count.fetch_add(1, AtomicOrdering::Relaxed);
                                return CasInsertResultGeneric::Success(None);
                            }

                            // Check for orphan
                            let next_ptr = unmark_ptr(next_raw);
                            if !next_ptr.is_null() {
                                let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
                                if ikey >= next_bound {
                                    return CasInsertResultGeneric::ContentionFallback;
                                }
                            }

                            // Success!
                            self.count.fetch_add(1, AtomicOrdering::Relaxed);
                            return CasInsertResultGeneric::Success(None);
                        }

                        Err(failure) => {
                            match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                                Ok(()) | Err(_) => {
                                    let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                                }
                            }

                            if failure.is_frozen() {
                                return CasInsertResultGeneric::ContentionFallback;
                            }

                            retries += 1;
                            if retries > Self::MAX_CAS_RETRIES_GENERIC {
                                return CasInsertResultGeneric::ContentionFallback;
                            }
                            Self::backoff_generic(retries);
                        }
                    }
                }
            }
        }
    }

    /// Exponential backoff for CAS retries.
    #[inline(always)]
    fn backoff_generic(retries: usize) {
        let spins = 1usize << retries.min(6);
        for _ in 0..spins {
            std::hint::spin_loop();
        }
    }

    /// Search for insert position in a leaf (generic version).
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn search_for_insert_generic(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;

        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = if key.has_suffix() {
            KSUF_KEYLENX
        } else {
            key.current_len() as u8
        };

        for i in 0..perm.size() {
            let slot: usize = perm.get(i);
            let slot_ikey: u64 = leaf.ikey(slot);

            if slot_ikey == target_ikey {
                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                if slot_ptr.is_null() {
                    continue;
                }

                // Layer pointer - always descend into the layer.
                // Even if the key has no suffix, the value should be stored in the sublayer
                // with ikey=0, keylenx=0 (empty key at that layer level).
                if slot_keylenx >= LAYER_KEYLENX {
                    return InsertSearchResultGeneric::Layer {
                        slot,
                        shift_amount: 8,
                    };
                }

                // Exact match check
                if slot_keylenx == search_keylenx {
                    if slot_keylenx == KSUF_KEYLENX {
                        // Both have suffixes - compare them
                        let key_suffix: &[u8] = key.suffix();
                        if let Some(slot_suffix) = leaf.ksuf(slot) {
                            if key_suffix == slot_suffix {
                                // Same suffix = same key
                                return InsertSearchResultGeneric::Found { slot };
                            }
                            // Different suffixes = conflict, need layer
                            return InsertSearchResultGeneric::Conflict { slot };
                        }
                        // No stored suffix (shouldn't happen for KSUF_KEYLENX)
                        // but treat as conflict to be safe
                        return InsertSearchResultGeneric::Conflict { slot };
                    }
                    // Inline keys (no suffix) with matching keylenx = same key
                    return InsertSearchResultGeneric::Found { slot };
                }

                // Same ikey, different keylenx - conflict
                return InsertSearchResultGeneric::Conflict { slot };
            }

            // Sorted order - found insert position
            if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound {
            logical_pos: perm.size(),
        }
    }

    /// Advance to correct leaf via B-link (generic version).
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_by_bound_generic<'a>(
        &'a self,
        mut leaf: &'a L,
        key: &Key<'_>,
        _guard: &LocalGuard<'_>,
    ) -> &'a L {
        use crate::link::{is_marked, unmark_ptr};

        let key_ikey: u64 = key.ikey();

        if leaf.version().is_splitting() {
            let _ = leaf.version().stable();
        }

        loop {
            let next_raw: *mut L = leaf.next_raw();
            if is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut L = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                return leaf;
            }

            // SAFETY: next_ptr is valid
            let next: &L = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                leaf = next;
                continue;
            }

            return leaf;
        }
    }

    // ========================================================================
    //  Generic Locked Insert Path
    // ========================================================================

    /// Insert a key-value pair.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`insert_with_guard`](Self::insert_with_guard) to amortize guard creation cost.
    ///
    /// # Returns
    ///
    /// * `Ok(None)` - New key inserted
    /// * `Ok(Some(old))` - Key existed, old value returned
    /// * `Err(InsertError)` - Insert failed (key too long)
    ///
    /// # Errors
    /// If insert fails.
    #[inline]
    pub fn insert(&self, key: &[u8], value: V) -> Result<Option<Arc<V>>, InsertError> {
        let guard = self.guard();
        self.insert_with_guard(key, value, &guard)
    }

    /// Insert a key-value pair using an explicit guard.
    ///
    /// This is the main public insert API for `MassTreeGeneric`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert (byte slice)
    /// * `value` - The value to insert
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Ok(None)` - New key inserted
    /// * `Ok(Some(old))` - Key existed, old value returned
    /// * `Err(InsertError)` - Insert failed (key too long)
    ///
    /// # Errors
    /// If insert fails.
    pub fn insert_with_guard(
        &self,
        key: &[u8],
        value: V,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<Arc<V>>, InsertError> {
        let mut key = Key::new(key);
        let arc = Arc::new(value);
        self.insert_concurrent_generic(&mut key, arc, guard)
    }

    /// Internal concurrent insert with CAS fast path and locked fallback.
    #[expect(clippy::too_many_lines, reason = "Complex concurrency logic")]
    fn insert_concurrent_generic(
        &self,
        key: &mut Key<'_>,
        value: Arc<V>,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<Arc<V>>, InsertError> {
        // Track current layer root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);

        loop {
            // Follow parent pointers to actual layer root
            layer_root = self.maybe_parent_generic(layer_root);

            // Try CAS fast path first (only for simple cases)
            if !key.has_suffix() {
                match self.try_cas_insert_generic(key, &value, guard) {
                    CasInsertResultGeneric::Success(old) => {
                        return Ok(old);
                    }
                    CasInsertResultGeneric::ExistsNeedLock { .. }
                    | CasInsertResultGeneric::FullNeedLock
                    | CasInsertResultGeneric::LayerNeedLock { .. }
                    | CasInsertResultGeneric::Slot0NeedLock
                    | CasInsertResultGeneric::ContentionFallback => {
                        // Fall through to locked path
                    }
                }
            }

            // Locked path
            let leaf_ptr: *mut L = self.reach_leaf_concurrent_generic(layer_root, key, guard);

            let leaf: &L = unsafe { &*leaf_ptr };

            // B-link advance if needed
            let leaf: &L = self.advance_to_key_by_bound_generic(leaf, key, guard);

            // Lock the leaf
            let mut lock = leaf.version().lock();

            // Get permutation (must not be frozen since we hold lock)
            let perm = leaf.permutation();

            // Search for insert position
            let search_result = self.search_for_insert_generic(leaf, key, &perm);

            match search_result {
                InsertSearchResultGeneric::Found { slot } => {
                    // Key exists - update value
                    let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                    if !old_ptr.is_null() {
                        // Clone old Arc for return value BEFORE we store new pointer.
                        // SAFETY: old_ptr is non-null and came from Arc::into_raw
                        let old_arc: Arc<V> = unsafe {
                            let arc_ptr: *const V = old_ptr.cast();
                            Arc::increment_strong_count(arc_ptr);
                            Arc::from_raw(arc_ptr)
                        };

                        let new_ptr: *mut u8 = Arc::into_raw(value) as *mut u8;

                        // Mark insert, store value, unlock happens on drop
                        lock.mark_insert();
                        leaf.set_leaf_value_ptr(slot, new_ptr);
                        drop(lock);

                        // Defer retirement of the old Arc.
                        // This ensures readers who captured old_ptr before our store
                        // can safely complete their validation and retry.
                        // SAFETY: old_ptr came from Arc::into_raw
                        unsafe {
                            guard.defer_retire(old_ptr.cast::<V>(), |ptr, _| {
                                drop(Arc::from_raw(ptr));
                            });
                        }

                        return Ok(Some(old_arc));
                    }
                    drop(lock);
                }

                InsertSearchResultGeneric::NotFound { logical_pos } => {
                    let ikey: u64 = key.ikey();

                    // New key - check if leaf has space
                    if perm.size() >= L::WIDTH {
                        // Leaf is full - perform split using SPLIT-THEN-RETRY pattern
                        let leaf_ptr_current: *mut L = std::ptr::from_ref(leaf).cast_mut();

                        // Split the leaf (takes lock ownership, releases before returning)
                        self.handle_leaf_split_generic(
                            leaf_ptr_current,
                            lock, // Move lock ownership
                            logical_pos,
                            ikey,
                            guard,
                        )?;

                        // Lock was released by handle_leaf_split_generic.
                        // Retry the insert - next iteration will find correct leaf with space.
                        continue;
                    }

                    // Check slot-0 rule
                    let next_free: usize = perm.back();

                    if next_free == 0 && !leaf.can_reuse_slot0(ikey) {
                        // Slot-0 violation - need swap logic
                        // For now, try to use a different slot via back_at_offset
                        let slot = if perm.size() < L::WIDTH - 1 {
                            perm.back_at_offset(1)
                        } else {
                            // Leaf is full and slot-0 violation - need split
                            let leaf_ptr_current: *mut L = std::ptr::from_ref(leaf).cast_mut();

                            // Split using SPLIT-THEN-RETRY pattern (takes lock ownership)
                            self.handle_leaf_split_generic(
                                leaf_ptr_current,
                                lock, // Move lock ownership
                                logical_pos,
                                ikey,
                                guard,
                            )?;

                            // Lock was released by handle_leaf_split_generic.
                            // Retry the insert.
                            continue;
                        };

                        // Assign to alternative slot
                        self.assign_slot_generic(leaf, &mut lock, slot, key, &value);

                        // Update permutation with swap logic
                        let mut new_perm = perm;
                        new_perm.swap_free_slots(0, 1);
                        let _ = new_perm.insert_from_back(logical_pos);
                        leaf.set_permutation(new_perm);

                        drop(lock);
                        self.count.fetch_add(1, AtomicOrdering::Relaxed);
                        return Ok(None);
                    }

                    // Normal insert
                    let (new_perm, slot) = perm.insert_from_back_immutable(logical_pos);
                    self.assign_slot_generic(leaf, &mut lock, slot, key, &value);

                    // Update permutation
                    leaf.set_permutation(new_perm);
                    drop(lock);

                    self.count.fetch_add(1, AtomicOrdering::Relaxed);
                    return Ok(None);
                }

                InsertSearchResultGeneric::Layer { slot, .. } => {
                    // Descend into sublayer
                    let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                    drop(lock);

                    key.shift();

                    layer_root = layer_ptr;
                }

                InsertSearchResultGeneric::Conflict { slot } => {
                    // =================================================================
                    // Suffix Conflict: Same ikey, different suffix
                    // Create a new layer to distinguish the keys
                    // =================================================================

                    // Mark insert before modifying the node
                    lock.mark_insert();

                    // Create new layer for the conflicting keys
                    //
                    // SAFETY:
                    // - We hold the lock on `leaf`
                    // - `guard` is from this tree's collector
                    let layer_ptr: *mut u8 = unsafe {
                        self.create_layer_concurrent_generic(
                            leaf,
                            slot,
                            key,
                            Arc::clone(&value),
                            guard,
                        )
                    };

                    // CRITICAL: Drop the existing Arc in the conflict slot.
                    //
                    // The create_layer_concurrent_generic function cloned it via try_clone_arc(),
                    // so the slot's reference is now redundant. We must drop it to avoid
                    // leaking memory when we overwrite with the layer pointer.
                    //
                    // SAFETY:
                    // - We hold the lock, so no concurrent access
                    let old_ptr: *mut u8 = leaf.take_leaf_value_ptr(slot);
                    if !old_ptr.is_null() {
                        // SAFETY: old_ptr came from Arc::into_raw during the original insert
                        let _old_arc: Arc<V> = unsafe { Arc::from_raw(old_ptr.cast::<V>()) };
                        // _old_arc is dropped here, decrementing refcount
                    }

                    // Clear any existing suffix for this slot
                    // SAFETY: We hold the lock
                    unsafe { leaf.clear_ksuf(slot, guard) };

                    // Install the layer pointer in the conflict slot
                    //
                    // NOTE: The original ikey remains unchanged (it's the shared prefix).
                    // We only change keylenx to indicate this is now a layer pointer,
                    // and set the pointer to the new layer chain.
                    leaf.set_keylenx(slot, LAYER_KEYLENX);
                    leaf.set_leaf_value_ptr(slot, layer_ptr);

                    // Release lock and increment entry count
                    // (new key was added to the layer, so count increases by 1)
                    drop(lock);
                    self.count.fetch_add(1, AtomicOrdering::Relaxed);

                    return Ok(None);
                }
            }
        }
    }

    /// Assign a value to a slot in a locked leaf.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn assign_slot_generic(
        &self,
        leaf: &L,
        lock: &mut crate::nodeversion::LockGuard<'_>,
        slot: usize,
        key: &Key<'_>,
        value: &Arc<V>,
    ) {
        let ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let keylenx: u8 = key.current_len() as u8;

        let value_ptr: *mut u8 = Arc::into_raw(Arc::clone(value)) as *mut u8;

        // Mark insert dirty
        lock.mark_insert();

        // Store key data and value
        leaf.set_ikey(slot, ikey);
        leaf.set_keylenx(slot, keylenx);
        leaf.set_leaf_value_ptr(slot, value_ptr);
    }

    // ========================================================================
    // Generic Split Methods
    // ========================================================================

    /// Handle a leaf split when the leaf is full.
    ///
    /// This method:
    /// 1. Calculates the split point
    /// 2. Allocates a new leaf
    /// 3. Performs the split (moves entries)
    /// 4. Links the leaves in B-link order
    /// 5. Propagates the split to the parent
    ///
    /// # Arguments
    ///
    /// * `left_leaf_ptr` - Pointer to the leaf being split
    /// * `lock` - Lock guard (takes ownership, will mark split and release)
    /// * `logical_pos` - Insert position in the leaf (for split point calculation)
    /// * `ikey` - The key being inserted (for split point calculation)
    /// * `guard` - Memory reclamation guard
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The caller should retry the insert (SPLIT-THEN-RETRY pattern).
    ///
    /// # Note
    ///
    /// This function takes ownership of the lock and releases it before returning.
    #[inline]
    fn handle_leaf_split_generic(
        &self,
        left_leaf_ptr: *mut L,
        mut lock: crate::nodeversion::LockGuard<'_>,
        logical_pos: usize,
        ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        let left_leaf: &L = unsafe { &*left_leaf_ptr };

        // Calculate split point
        let split_point = left_leaf
            .calculate_split_point(logical_pos, ikey)
            .ok_or(InsertError::SplitFailed)?;

        // Allocate new leaf
        let new_leaf: Box<L> = L::new_boxed();

        // Mark split in progress (must be called before freeze_permutation)
        lock.mark_split();

        // Perform the split (insert_target ignored - we use SPLIT-THEN-RETRY)
        let (new_leaf_box, split_ikey, _insert_target) =
            unsafe { left_leaf.split_into_preallocated(split_point.pos, new_leaf, guard) };

        // Store new leaf in allocator
        let right_leaf_ptr: *mut L = self.allocator.alloc_leaf(new_leaf_box);

        // Link leaves in B-link order
        // NOTE: right_leaf's parent pointer is NULL at this point. It will be set
        // in propagate_split_generic after we insert into the parent internode.
        // If another thread splits right_leaf before we set its parent, that thread
        // will see NULL parent and wait in propagate_split_generic.
        unsafe { left_leaf.link_sibling(right_leaf_ptr) };

        // OPTIMIZATION: Release leaf lock BEFORE parent propagation.
        // Once link succeeds, the split is visible via B-link chain.
        // Readers use advance_to_key() to follow B-links.
        drop(lock);

        // Propagate split to parent (lock already released)
        self.propagate_split_generic(left_leaf_ptr, right_leaf_ptr, split_ikey, guard)?;

        Ok(())
    }

    /// Lock a parent internode from a leaf with validation (generic version).
    ///
    /// Handles the race where the parent pointer may change between reading
    /// and locking. Uses an optimistic lock-then-validate pattern.
    ///
    /// # Returns
    ///
    /// `(parent_ptr, lock_guard)` - Successfully locked parent
    ///
    /// # Panics
    ///
    /// Panics if the leaf has no parent (is a root).
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn locked_parent_leaf_generic(
        &self,
        leaf: &L,
    ) -> (*mut L::Internode, crate::nodeversion::LockGuard<'_>) {
        let mut retries: usize = 0;

        #[cfg(feature = "tracing")]
        eprintln!(
            "[LOCKED_PARENT_LEAF] entering, leaf={:p}",
            std::ptr::from_ref(leaf)
        );

        loop {
            // Step 1: Read parent pointer
            let parent_ptr: *mut u8 = leaf.parent();

            #[cfg(feature = "tracing")]
            eprintln!(
                "[LOCKED_PARENT_LEAF] parent_ptr={:p} is_null={}",
                parent_ptr,
                parent_ptr.is_null()
            );

            debug_assert!(
                !parent_ptr.is_null(),
                "locked_parent_leaf_generic called on root"
            );

            // Step 2: Lock the parent
            // SAFETY: parent_ptr is non-null, seize guard ensures it won't be freed
            let parent: &L::Internode = unsafe { &*parent_ptr.cast::<L::Internode>() };

            #[cfg(feature = "tracing")]
            eprintln!("[LOCKED_PARENT_LEAF] about to lock parent");

            let lock = parent.version().lock_with_yield();

            #[cfg(feature = "tracing")]
            eprintln!("[LOCKED_PARENT_LEAF] locked, revalidating");

            // Step 3: Revalidate - check parent pointer hasn't changed
            let current_parent: *mut u8 = leaf.parent();
            if current_parent == parent_ptr {
                // Success: parent is stable
                #[cfg(feature = "tracing")]
                eprintln!("[LOCKED_PARENT_LEAF] success, returning");
                return (parent_ptr.cast(), lock);
            }

            // Step 4: Parent changed, release lock and retry
            #[cfg(feature = "tracing")]
            eprintln!("[LOCKED_PARENT_LEAF] parent changed, retrying");

            drop(lock);

            retries += 1;
            assert!(
                retries < Self::MAX_PROPAGATION_RETRIES,
                "locked_parent_leaf_generic: exceeded {} retries",
                Self::MAX_PROPAGATION_RETRIES
            );

            std::hint::spin_loop();
        }
    }

    /// Propagate a leaf split to the parent.
    #[expect(clippy::too_many_lines)]
    #[expect(clippy::panic, reason = "FATAL: Cannot continue")]
    fn propagate_split_generic(
        &self,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        #[cfg(feature = "tracing")]
        eprintln!(
            "[SPLIT] propagate_split_generic: split_ikey={split_ikey:016x} left={left_leaf_ptr:p} right={right_leaf_ptr:p}"
        );

        // Check if left leaf was the main tree root
        let left_was_main_root: bool = self.root_is_leaf_generic() && {
            let root_ptr: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            std::ptr::eq(root_ptr, left_leaf_ptr)
        };

        if left_was_main_root {
            #[cfg(feature = "tracing")]
            eprintln!("[SPLIT] left was main tree root, creating root internode");
            return self.create_root_internode_generic(left_leaf_ptr, right_leaf_ptr, split_ikey);
        }

        // Check if this is a layer root (is_root flag + null parent)
        let left_leaf: &L = unsafe { &*left_leaf_ptr };

        // Handle NULL parent case
        if left_leaf.parent().is_null() {
            // Actual layer root - promote
            if left_leaf.version().is_root() {
                #[cfg(feature = "tracing")]
                eprintln!("[SPLIT] left is layer root, promoting");
                return self.promote_layer_root_generic(left_leaf_ptr, right_leaf_ptr, split_ikey);
            }

            // NULL parent but NOT a root: newly split sibling whose parent hasn't
            // been set yet. Wait for the creating thread to set it.
            // Strategy: spin briefly, then yield, then short sleep.
            let mut spins: usize = 0;
            loop {
                // Check if parent is now set
                if !left_leaf.parent().is_null() {
                    break;
                }

                spins += 1;

                // Backoff strategy tuned for parent-set latency (typically < 1µs):
                // Phase 1 (0-64): Spin - handles majority of cases
                // Phase 2 (64-1024): Yield - moderate backoff
                // Phase 3 (1024+): Sleep 10µs - shorter sleep for faster recovery
                if spins <= 64 {
                    std::hint::spin_loop();
                } else if spins <= 1024 {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(std::time::Duration::from_micros(10));
                }

                // Safety limit
                assert!(
                    spins <= 1_000_000,
                    "propagate_split_generic: parent pointer never set after {} iterations. \
                     left_leaf_ptr={left_leaf_ptr:p} is_root={}",
                    spins,
                    left_leaf.version().is_root()
                );
            }
        }

        // Lock parent with validation and find child index.
        // Use a retry loop to handle the case where child is not found in parent,
        // which can happen during concurrent splits (matches WIDTH=15 behavior).
        let mut retry_count: usize = 0;
        let (parent_ptr, mut parent_lock, child_idx) = loop {
            let (parent_ptr, parent_lock) = self.locked_parent_leaf_generic(left_leaf);
            let parent: &L::Internode = unsafe { &*parent_ptr };

            #[cfg(feature = "tracing")]
            eprintln!(
                "[SPLIT] locked parent={:p} nkeys={} is_full={}",
                parent_ptr,
                parent.nkeys(),
                parent.is_full()
            );

            // Find child index using pointer scan
            if let Some(idx) = self.try_find_child_index_generic(parent, left_leaf_ptr.cast::<u8>())
            {
                #[cfg(feature = "tracing")]
                eprintln!("[SPLIT] child_idx={idx}");
                break (parent_ptr, parent_lock, idx);
            }

            // Child not found - this can happen if the parent was split and
            // the child moved to a sibling, but the child's parent pointer
            // was updated AFTER we validated in locked_parent_leaf_generic.
            retry_count += 1;

            #[cfg(feature = "tracing")]
            eprintln!(
                "[SPLIT] child not found in parent, retry #{retry_count} parent={:p} nkeys={}",
                parent_ptr,
                parent.nkeys()
            );

            // Child not found - this happens during concurrent internode splits.
            // The child was moved to a sibling but its parent pointer hasn't been
            // updated yet. Yield and retry - the window is small.
            drop(parent_lock);

            if retry_count < 1000 {
                std::hint::spin_loop();
            } else {
                // After many retries, yield to let the splitting thread complete
                std::thread::yield_now();
            }

            // Safety valve - but with much higher limit since this is a valid race
            if retry_count >= 100_000 {
                let current_parent = left_leaf.parent();
                panic!(
                    "propagate_split_generic: child not found after {retry_count} retries. \
                     left_leaf_ptr={left_leaf_ptr:p} parent_ptr={parent_ptr:p} \
                     current_parent={current_parent:p}"
                );
            }
        };

        let parent: &L::Internode = unsafe { &*parent_ptr };

        if parent.is_full() {
            // Parent is full - split the parent internode
            #[cfg(feature = "tracing")]
            eprintln!(
                "[SPLIT] parent FULL (nkeys={}), triggering internode split",
                parent.nkeys()
            );

            // CRITICAL: Save is_root BEFORE unlocking, because SPLIT_UNLOCK_MASK clears ROOT_BIT
            let parent_was_root: bool = parent.is_root();

            // CRITICAL FIX: Set right_leaf's parent pointer BEFORE releasing the lock.
            // This prevents a race where another thread tries to split right_leaf and
            // finds NULL parent (not a root), causing it to spin forever waiting.
            // The internode split may move right_leaf to a different parent, but:
            // 1. Having a non-NULL parent allows the other thread to make progress
            // 2. The "child not found" retry loop handles parent changes gracefully
            // 3. propagate_internode_split_generic will update the parent if needed
            unsafe {
                (*right_leaf_ptr).set_parent(parent_ptr.cast::<u8>());
            }

            // Release parent lock WITHOUT mark_split - propagate_internode_split_generic
            // will re-acquire the lock and call mark_split when it performs the split.
            // This matches WIDTH=15 behavior (locked.rs:1614-1616).
            drop(parent_lock);

            self.propagate_internode_split_generic(
                parent_ptr,
                split_ikey,
                right_leaf_ptr.cast::<u8>(),
                parent_was_root,
                guard,
            )
        } else {
            // Parent has room - insert directly
            #[cfg(feature = "tracing")]
            eprintln!(
                "[SPLIT] parent has room, inserting at idx={child_idx} ikey={split_ikey:016x}"
            );

            parent_lock.mark_insert();
            parent.insert_key_and_child(child_idx, split_ikey, right_leaf_ptr.cast::<u8>());

            // Update right leaf's parent pointer
            unsafe {
                (*right_leaf_ptr).set_parent(parent_ptr.cast::<u8>());
            }

            #[cfg(feature = "tracing")]
            eprintln!(
                "[SPLIT] insert complete, parent now has nkeys={}",
                parent.nkeys()
            );

            drop(parent_lock);
            Ok(())
        }
    }

    /// Create a new root internode when the root was a leaf.
    #[inline]
    fn create_root_internode_generic(
        &self,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> Result<(), InsertError> {
        #[cfg(feature = "tracing")]
        eprintln!(
            "[CREATE_ROOT_INTERNODE] creating root internode for split_ikey={split_ikey:016x}"
        );

        // Create new root internode (height=0, children are leaves)
        let new_root: Box<L::Internode> = L::Internode::new_root_boxed(0);

        #[cfg(feature = "tracing")]
        eprintln!(
            "[CREATE_ROOT_INTERNODE] new_root created, is_root={}",
            new_root.version().is_root()
        );

        // Set up children: [left_leaf] -split_ikey- [right_leaf]
        new_root.set_child(0, left_leaf_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_leaf_ptr.cast());
        new_root.set_nkeys(1);

        // Allocate internode
        let new_root_ptr: *mut u8 = self
            .allocator
            .alloc_internode_erased(Box::into_raw(new_root).cast());

        #[cfg(feature = "tracing")]
        {
            let root_ref: &L::Internode = unsafe { &*new_root_ptr.cast::<L::Internode>() };
            eprintln!(
                "[CREATE_ROOT_INTERNODE] after alloc new_root_ptr={:p} is_root={}",
                new_root_ptr,
                root_ref.version().is_root()
            );
        }

        // Atomically update root pointer
        // NOTE: We set parent pointers AFTER CAS succeeds to avoid dangling pointers
        // if another thread already installed a new root. This matches WIDTH=15 behavior.
        let expected: *mut u8 = left_leaf_ptr.cast();
        match self.root_ptr.compare_exchange(
            expected,
            new_root_ptr,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => {
                // CAS succeeded - now safe to update parent pointers
                unsafe {
                    // Fence before making the new root reachable via parent pointers.
                    // Ensures the new internode is fully constructed before it becomes visible.
                    // Required for correctness on weakly-ordered architectures (ARM, etc.).
                    atomicFence(AtomicOrdering::Release);

                    (*left_leaf_ptr).set_parent(new_root_ptr);
                    (*right_leaf_ptr).set_parent(new_root_ptr);

                    // Clear root flag on left leaf (it's no longer the root)
                    (*left_leaf_ptr).version().mark_nonroot();
                }
                Ok(())
            }
            Err(_) => {
                // Root changed concurrently - shouldn't happen if we hold lock
                Err(InsertError::SplitFailed)
            }
        }
    }

    /// Promote a layer root to a new layer root internode.
    #[inline]
    #[expect(clippy::unnecessary_wraps, reason = "API consistency")]
    fn promote_layer_root_generic(
        &self,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> Result<(), InsertError> {
        // Create new internode root for this layer (height=0, children are leaves)
        let new_inode: Box<L::Internode> = L::Internode::new_boxed(0);

        // Set up children
        new_inode.set_child(0, left_leaf_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_leaf_ptr.cast());
        new_inode.set_nkeys(1);

        // Mark as root (layer roots have the root flag)
        new_inode.version().mark_root();

        // Allocate internode
        let new_inode_ptr = self
            .allocator
            .alloc_internode_erased(Box::into_raw(new_inode).cast());

        // Update children's parent pointers
        // Now left_leaf.parent() != null, so maybe_parent pattern works
        unsafe {
            // Fence before making the new root reachable via parent pointers.
            // Ensures the new internode is fully constructed before it becomes visible.
            // Required for correctness on weakly-ordered architectures (ARM, etc.).
            atomicFence(AtomicOrdering::Release);

            (*left_leaf_ptr).set_parent(new_inode_ptr);
            (*right_leaf_ptr).set_parent(new_inode_ptr);

            // Clear root flag on both leaves - they're no longer layer roots
            (*left_leaf_ptr).version().mark_nonroot();
            (*right_leaf_ptr).version().mark_nonroot();
        }

        Ok(())
    }

    /// Try to find the child index for a given child pointer in an internode.
    ///
    /// Returns `Some(index)` if found, `None` if not found. Use this in retry loops
    /// where not finding the child is a valid transient state during concurrent splits.
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn try_find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> Option<usize> {
        use crate::leaf_trait::TreeInternode;

        let nkeys = parent.nkeys();
        (0..=nkeys).find(|&i| parent.child(i) == child)
    }

    /// Find the child index for a given child pointer in an internode.
    /// Panics if not found.
    #[expect(clippy::expect_used, reason = "FATAL: Fail Fast")]
    fn find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> usize {
        self.try_find_child_index_generic(parent, child)
            .expect("Child not found in parent internode")
    }

    // ========================================================================
    //  Internode Split Propagation (Generic)
    // ========================================================================

    /// Maximum number of retries for split propagation before panicking.
    ///
    /// This bounds retry loops to prevent livelock. If exceeded, it indicates
    /// a bug in the concurrency logic rather than normal contention.
    const MAX_PROPAGATION_RETRIES: usize = 64;

    /// Lock a parent internode with validation (generic version).
    ///
    /// Handles the race where the parent pointer may change between reading
    /// and locking. Uses an optimistic lock-then-validate pattern.
    ///
    /// # Returns
    ///
    /// - `Some((parent_ptr, lock_guard))` - Successfully locked parent
    /// - `None` - Node has no parent (is a root)
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn locked_parent_internode_generic(
        &self,
        inode: &L::Internode,
    ) -> Option<(*mut L::Internode, crate::nodeversion::LockGuard<'_>)> {
        use crate::leaf_trait::TreeInternode;

        let mut retries: usize = 0;

        loop {
            // Step 1: Optimistic read of parent pointer
            let parent_ptr: *mut u8 = inode.parent();

            // No parent means this is a root node
            if parent_ptr.is_null() {
                return None;
            }

            // Step 2: Lock the parent using yield-based locking
            // SAFETY: parent_ptr is non-null, seize guard ensures it won't be freed
            let parent: &L::Internode = unsafe { &*parent_ptr.cast::<L::Internode>() };
            let lock = parent.version().lock_with_yield();

            // Step 3: Revalidate - check parent pointer hasn't changed
            let current_parent: *mut u8 = inode.parent();
            if current_parent == parent_ptr {
                // Success: parent is stable
                return Some((parent_ptr.cast(), lock));
            }

            // Step 4: Parent changed, release lock and retry
            drop(lock);

            retries += 1;
            assert!(
                retries < Self::MAX_PROPAGATION_RETRIES,
                "locked_parent_internode_generic: exceeded {} retries",
                Self::MAX_PROPAGATION_RETRIES
            );

            // Brief pause to reduce contention
            std::hint::spin_loop();
        }
    }

    /// Propagate an internode split up the tree (generic version).
    ///
    /// When a parent internode becomes full during leaf split propagation,
    /// this function splits the parent and propagates the separator key upward.
    ///
    /// # Algorithm
    ///
    /// 1. Lock the parent internode
    /// 2. If parent has space (another thread split it), insert directly
    /// 3. If parent is full, split it:
    ///    - Create new sibling internode
    ///    - Split keys and children between parent and sibling
    ///    - Update children's parent pointers
    /// 4. If parent is a root, create new root internode
    /// 5. Otherwise, recursively propagate to grandparent
    #[expect(
        clippy::too_many_lines,
        reason = "Complex node splitting logic with conditionally compiled logs"
    )]
    #[expect(
        clippy::only_used_in_recursion,
        reason = "TODO: Consider iterative algorithm (may be too complex)"
    )]
    fn propagate_internode_split_generic(
        &self,
        parent_ptr: *mut L::Internode,
        insert_ikey: u64,
        insert_child: *mut u8,
        parent_was_root: bool,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        use crate::leaf_trait::TreeInternode;

        #[cfg(feature = "tracing")]
        eprintln!(
            "[INTERNODE_SPLIT] propagate_internode_split_generic: parent={parent_ptr:p} insert_ikey={insert_ikey:016x} insert_child={insert_child:p} parent_was_root={parent_was_root}"
        );

        let mut retries: usize = 0;

        'retry: loop {
            retries += 1;
            assert!(
                retries <= Self::MAX_PROPAGATION_RETRIES,
                "propagate_internode_split_generic: exceeded {} retries",
                Self::MAX_PROPAGATION_RETRIES
            );

            // SAFETY: parent_ptr is valid (from locked_parent or caller)
            let parent: &L::Internode = unsafe { &*parent_ptr };

            // Lock the parent using yield-based locking
            let mut parent_lock = parent.version().lock_with_yield();

            // Recompute child index after acquiring lock (may have changed)
            let child_idx: usize = parent.find_insert_position(insert_ikey);

            #[cfg(feature = "tracing")]
            eprintln!(
                "[INTERNODE_SPLIT] parent nkeys={} is_full={} child_idx={}",
                parent.nkeys(),
                parent.is_full(),
                child_idx
            );

            // Check if parent is still full (another thread may have split it)
            if !parent.is_full() {
                // Parent was split by another thread - just insert
                #[cfg(feature = "tracing")]
                eprintln!("[INTERNODE_SPLIT] parent not full, inserting directly");

                parent_lock.mark_insert();
                parent.insert_key_and_child(child_idx, insert_ikey, insert_child);

                // Update child's parent pointer
                unsafe {
                    if parent.children_are_leaves() {
                        (*insert_child.cast::<L>()).set_parent(parent_ptr.cast::<u8>());
                    } else {
                        (*insert_child.cast::<L::Internode>()).set_parent(parent_ptr.cast::<u8>());
                    }
                }
                return Ok(());
            }

            // Parent is full - must split
            #[cfg(feature = "tracing")]
            eprintln!(
                "[INTERNODE_SPLIT] parent FULL, splitting. height={} children_are_leaves={} is_root_BEFORE_mark_split={}",
                parent.height(),
                parent.children_are_leaves(),
                parent.is_root()
            );

            parent_lock.mark_split();

            #[cfg(feature = "tracing")]
            eprintln!(
                "[INTERNODE_SPLIT] after mark_split, is_root={}",
                parent.is_root()
            );

            // Create new sibling internode
            let sibling: Box<L::Internode> = L::Internode::new_boxed(parent.height());
            let sibling_ptr: *mut L::Internode = Box::into_raw(sibling);

            #[cfg(feature = "tracing")]
            eprintln!("[INTERNODE_SPLIT] created sibling={sibling_ptr:p}");

            // Track sibling for cleanup
            self.allocator
                .track_internode_erased(sibling_ptr.cast::<u8>());

            // Split and insert simultaneously
            // NOTE: split_into now updates all children's parent pointers in sibling internally
            // (matching C++ masstree_split.hh:163-165). This is critical for correctness.
            let (popup_key, insert_went_left) = unsafe {
                parent.split_into(
                    &mut *sibling_ptr,
                    sibling_ptr,
                    child_idx,
                    insert_ikey,
                    insert_child,
                )
            };

            #[cfg(feature = "tracing")]
            {
                let sibling_ref: &L::Internode = unsafe { &*sibling_ptr };
                eprintln!(
                    "[INTERNODE_SPLIT] after split: popup_key={:016x} insert_went_left={} parent.nkeys={} sibling.nkeys={}",
                    popup_key,
                    insert_went_left,
                    parent.nkeys(),
                    sibling_ref.nkeys()
                );
            }

            // NOTE: split_into updates internode children's parents internally (height > 0).
            // For leaf children (height == 0), we must update them here because split_into
            // doesn't know the actual leaf type (could be LeafNode<S, WIDTH> or LeafNode24<S>).
            // This must happen while still holding the parent lock to prevent races.
            unsafe {
                let sibling_ref: &L::Internode = &*sibling_ptr;

                if parent.children_are_leaves() {
                    // Update all leaf children's parent pointers in sibling
                    for i in 0..=sibling_ref.nkeys() {
                        let child: *mut u8 = sibling_ref.child(i);
                        if !child.is_null() {
                            (*child.cast::<L>()).set_parent(sibling_ptr.cast::<u8>());
                        }
                    }
                }
                // For internode children, split_into already updated them

                // If insert_child stayed in the LEFT parent, set its parent explicitly
                if insert_went_left {
                    if parent.children_are_leaves() {
                        (*insert_child.cast::<L>()).set_parent(parent_ptr.cast::<u8>());
                    } else {
                        (*insert_child.cast::<L::Internode>()).set_parent(parent_ptr.cast::<u8>());
                    }
                }
            }

            // Check if parent is a root (null parent AND was_root flag passed from caller)
            // NOTE: We use parent_was_root instead of parent.is_root() because
            // SPLIT_UNLOCK_MASK clears ROOT_BIT when we unlocked in propagate_split_generic
            #[cfg(feature = "tracing")]
            eprintln!(
                "[INTERNODE_SPLIT] checking if parent is root: parent.parent()={:p} is_null={} parent_was_root={}",
                parent.parent(),
                parent.parent().is_null(),
                parent_was_root
            );

            if parent.parent().is_null() && parent_was_root {
                #[cfg(feature = "tracing")]
                eprintln!("[INTERNODE_SPLIT] parent IS root, creating new root internode");

                let current_root: *mut u8 = self.root_ptr.load(AtomicOrdering::Acquire);

                if current_root == parent_ptr.cast::<u8>() {
                    // MAIN TREE ROOT INTERNODE SPLIT
                    let result = self.create_root_internode_from_internode_split_generic(
                        parent_ptr,
                        sibling_ptr,
                        popup_key,
                    );
                    drop(parent_lock);
                    return result;
                }

                // LAYER ROOT INTERNODE SPLIT
                let result =
                    self.promote_layer_root_internode_generic(parent_ptr, sibling_ptr, popup_key);
                drop(parent_lock);
                return result;
            }

            // Not a root - propagate to grandparent
            #[expect(
                clippy::manual_let_else,
                reason = "Unnecessary refactor would add complexity"
            )]
            let (grandparent_ptr, mut grandparent_lock) =
                if let Some(result) = self.locked_parent_internode_generic(parent) {
                    result
                } else {
                    // Parent became a root while we were working - retry
                    drop(parent_lock);
                    continue 'retry;
                };

            let grandparent: &L::Internode = unsafe { &*grandparent_ptr };

            // Find parent's position in grandparent
            let parent_idx: Option<usize> = {
                let mut found: Option<usize> = None;
                for i in 0..=grandparent.nkeys() {
                    if grandparent.child(i) == parent_ptr.cast::<u8>() {
                        found = Some(i);
                        break;
                    }
                }
                found
            };

            let Some(_parent_idx) = parent_idx else {
                // Parent not found in grandparent - structure changed, retry
                drop(grandparent_lock);
                drop(parent_lock);
                continue 'retry;
            };

            // Check if grandparent has space
            if !grandparent.is_full() {
                // Grandparent has space - insert separator and sibling
                grandparent_lock.mark_insert();
                let insert_pos = grandparent.find_insert_position(popup_key);
                grandparent.insert_key_and_child(insert_pos, popup_key, sibling_ptr.cast::<u8>());
                unsafe {
                    (*sibling_ptr).set_parent(grandparent_ptr.cast::<u8>());
                }

                // Release locks in order
                drop(grandparent_lock);
                drop(parent_lock);
                return Ok(());
            }

            // Grandparent full - need recursive split
            // CRITICAL: Save is_root BEFORE unlocking, because SPLIT_UNLOCK_MASK clears ROOT_BIT
            let grandparent_was_root: bool = grandparent.is_root();

            // Release grandparent lock without mark_split - recursive call will handle it
            // The recursive call will re-acquire the lock and mark_split when appropriate
            drop(grandparent_lock);

            // Recursive call to split grandparent
            let result = self.propagate_internode_split_generic(
                grandparent_ptr,
                popup_key,
                sibling_ptr.cast::<u8>(),
                grandparent_was_root,
                guard,
            );

            drop(parent_lock);
            return result;
        }
    }

    /// Create a new root internode from an internode split (generic version).
    fn create_root_internode_from_internode_split_generic(
        &self,
        left_ptr: *mut L::Internode,
        right_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> Result<(), InsertError> {
        use crate::leaf_trait::TreeInternode;

        // SAFETY: left_ptr is valid
        let left: &L::Internode = unsafe { &*left_ptr };

        // Create new root with height = left.height + 1
        let new_root: Box<L::Internode> = L::Internode::new_root_boxed(left.height() + 1);

        // Set up children: [left] -split_ikey- [right]
        new_root.set_child(0, left_ptr.cast::<u8>());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_ptr.cast::<u8>());
        new_root.set_nkeys(1);

        // Allocate and track
        let new_root_ptr: *mut u8 = self
            .allocator
            .alloc_internode_erased(Box::into_raw(new_root).cast());

        // Atomically install new root
        let expected: *mut u8 = left_ptr.cast::<u8>();

        match self.root_ptr.compare_exchange(
            expected,
            new_root_ptr,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => {
                // CAS succeeded - update parent pointers
                unsafe {
                    // Fence before making the new root reachable via parent pointers.
                    atomicFence(AtomicOrdering::Release);

                    (*left_ptr).set_parent(new_root_ptr);
                    (*right_ptr).set_parent(new_root_ptr);
                    (*left_ptr).version().mark_nonroot();
                }
                Ok(())
            }
            Err(_) => {
                // CAS failed - another thread already updated root
                Err(InsertError::SplitFailed)
            }
        }
    }

    /// Promote a layer root internode to a new layer root internode (generic version).
    #[expect(clippy::unnecessary_wraps, reason = "API Consistency")]
    fn promote_layer_root_internode_generic(
        &self,
        left_ptr: *mut L::Internode,
        right_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> Result<(), InsertError> {
        use crate::leaf_trait::TreeInternode;

        // SAFETY: left_ptr is valid
        let left: &L::Internode = unsafe { &*left_ptr };

        // Create new root for this layer
        let new_root: Box<L::Internode> = L::Internode::new_boxed(left.height() + 1);

        // Set up children
        new_root.set_child(0, left_ptr.cast::<u8>());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_ptr.cast::<u8>());
        new_root.set_nkeys(1);

        // Mark as layer root
        new_root.version().mark_root();

        // Allocate and track
        let new_root_ptr: *mut u8 = self
            .allocator
            .alloc_internode_erased(Box::into_raw(new_root).cast());

        // Update parent pointers
        unsafe {
            atomicFence(AtomicOrdering::Release);

            (*left_ptr).set_parent(new_root_ptr);
            (*right_ptr).set_parent(new_root_ptr);
            (*left_ptr).version().mark_nonroot();
        }

        Ok(())
    }
}

// =============================================================================
// Generic Layer Creation
// =============================================================================

impl<V, L, A> MassTreeGeneric<V, L, A>
where
    V: Send + Sync + 'static,
    L: LayerCapableLeaf<V>,
    A: NodeAllocatorGeneric<LeafValue<V>, L>,
{
    /// Create a new layer for suffix conflict (generic version).
    ///
    /// Called when two keys share the same 8-byte ikey but have different suffixes.
    /// Creates a twig chain if needed, ending in a leaf with both keys.
    ///
    /// # Algorithm
    ///
    /// 1. Extract existing key's suffix and Arc value from conflict slot
    /// 2. Shift `new_key` past the matching ikey
    /// 3. While both keys have matching ikeys AND both have more bytes:
    ///    - Create intermediate "twig" layer node with just the matching ikey
    ///    - Chain twig nodes together via layer pointers
    /// 4. Create final leaf with both keys (now diverged)
    /// 5. Link twig chain to final leaf
    /// 6. Return head of chain (or final leaf if no chain)
    ///
    /// # Arguments
    ///
    /// * `parent_leaf` - The leaf containing the conflict slot
    /// * `conflict_slot` - Physical slot index with the existing key
    /// * `new_key` - The new key being inserted (will be mutated via shift)
    /// * `new_value` - Arc value for the new key
    /// * `guard` - Seize guard for memory reclamation
    ///
    /// # Returns
    ///
    /// Raw pointer to the head of the layer chain (either a twig or the final leaf).
    /// This pointer should be stored in the conflict slot with `LAYER_KEYLENX`.
    ///
    /// # Safety
    ///
    /// - Caller must hold the lock on `parent_leaf`
    /// - Caller must have called `lock.mark_insert()` before calling this
    /// - `guard` must come from this tree's collector
    unsafe fn create_layer_concurrent_generic(
        &self,
        parent_leaf: &L,
        conflict_slot: usize,
        new_key: &mut Key<'_>,
        new_value: Arc<V>,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        // =====================================================================
        // Step 1: Extract existing key's suffix and Arc value
        // =====================================================================

        // Get existing suffix (empty slice if no suffix stored)
        let existing_suffix: &[u8] = parent_leaf.ksuf(conflict_slot).unwrap_or(&[]);

        // Create a Key iterator from the existing suffix for comparison
        let mut existing_key: Key<'_> = Key::from_suffix(existing_suffix);

        // Clone the existing Arc value from the conflict slot
        // INVARIANT: Conflict case means the slot contains a value, not a layer pointer.
        let existing_arc: Option<Arc<V>> = parent_leaf.try_clone_arc(conflict_slot);
        debug_assert!(
            existing_arc.is_some(),
            "create_layer_concurrent_generic: conflict slot {} should contain a value, \
             not a layer pointer. keylenx={}",
            conflict_slot,
            parent_leaf.keylenx(conflict_slot)
        );

        // =====================================================================
        // Step 2: Shift new_key past the matching ikey
        // =====================================================================

        // The new_key's current ikey matched the conflict slot's ikey.
        // If new_key has more bytes (suffix), shift to the next 8-byte chunk.
        if new_key.has_suffix() {
            new_key.shift();
        }

        // =====================================================================
        // Step 3: Compare keys to determine twig chain depth
        // =====================================================================

        // Compare the next ikeys of both keys
        let mut cmp: Ordering = existing_key.compare(new_key.ikey(), new_key.current_len());

        // =====================================================================
        // Step 4: Create twig chain while ikeys match AND both have more bytes
        // =====================================================================

        // Twig chain head (first twig node, returned to caller)
        let mut twig_head: Option<*mut L> = None;
        // Twig chain tail (last twig node, where we link the next node)
        let mut twig_tail: *mut L = std::ptr::null_mut();

        while cmp == Ordering::Equal && existing_key.has_suffix() && new_key.has_suffix() {
            // Both keys have the same ikey at this level AND both have more bytes.
            // Create an intermediate twig node that just holds this matching ikey.

            // Allocate new twig node configured as layer root
            let twig: Box<L> = L::new_layer_root_boxed();
            let twig_ptr: *mut L = self.allocator.alloc_leaf(twig);

            // Initialize twig with the matching ikey in slot 0
            // SAFETY: twig_ptr is valid, we just allocated it
            unsafe {
                (*twig_ptr).set_ikey(0, existing_key.ikey());
                // Twig has exactly 1 entry (the matching ikey, will point to next layer)
                (*twig_ptr).set_permutation(<L::Perm as TreePermutation>::make_sorted(1));
            }

            // Link to previous twig in chain (if any)
            if twig_head.is_some() {
                // Previous twig's slot 0 now points to this twig as a layer
                // SAFETY: twig_tail is valid from previous iteration
                unsafe {
                    (*twig_tail).set_keylenx(0, LAYER_KEYLENX);
                    (*twig_tail).set_leaf_value_ptr(0, twig_ptr.cast::<u8>());
                }
            } else {
                // First twig becomes the head of the chain
                twig_head = Some(twig_ptr);
            }
            twig_tail = twig_ptr;

            // Shift both keys to compare the next 8-byte chunk
            existing_key.shift();
            new_key.shift();
            cmp = existing_key.compare(new_key.ikey(), new_key.current_len());
        }

        // =====================================================================
        // Step 5: Create final leaf with both keys (now diverged or one is prefix)
        // =====================================================================

        let final_leaf: Box<L> = L::new_layer_root_boxed();
        let final_ptr: *mut L = self.allocator.alloc_leaf(final_leaf);

        // Assign both entries to the final leaf in sorted order
        // SAFETY: final_ptr is valid (just allocated), guard is from caller
        unsafe {
            self.assign_final_layer_entries(
                final_ptr,
                &existing_key,
                existing_arc,
                new_key,
                Some(new_value),
                cmp,
                guard,
            );
        }

        // =====================================================================
        // Step 6: Link twig chain to final leaf
        // =====================================================================

        twig_head.map_or_else(
            || final_ptr.cast::<u8>(),
            |head| {
                // Link last twig to the final leaf
                // SAFETY: twig_tail is valid (we have at least one twig since head is Some)
                unsafe {
                    (*twig_tail).set_keylenx(0, LAYER_KEYLENX);
                    (*twig_tail).set_leaf_value_ptr(0, final_ptr.cast::<u8>());
                }
                // Return head of twig chain
                head.cast::<u8>()
            },
        )
    }

    /// Assign two entries to the final layer leaf in sorted order.
    ///
    /// The entries are ordered by:
    /// 1. ikey comparison (lexicographic via u64 big-endian)
    /// 2. If ikeys equal: shorter key first (prefix before extension)
    ///
    /// # Safety
    ///
    /// - `final_ptr` must be valid and point to an empty leaf
    /// - `guard` must come from this tree's collector
    /// - Caller must ensure no concurrent access to `final_ptr`
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    unsafe fn assign_final_layer_entries(
        &self,
        final_ptr: *mut L,
        existing_key: &Key<'_>,
        existing_arc: Option<Arc<V>>,
        new_key: &Key<'_>,
        new_arc: Option<Arc<V>>,
        cmp: Ordering,
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: final_ptr is valid per caller contract
        let final_leaf: &L = unsafe { &*final_ptr };

        match cmp {
            Ordering::Less => {
                // existing_key.ikey() < new_key.ikey()
                // existing goes in slot 0, new goes in slot 1
                // SAFETY: guard requirement passed through from caller
                unsafe {
                    final_leaf.assign_from_key_arc(0, existing_key, existing_arc, guard);
                    final_leaf.assign_from_key_arc(1, new_key, new_arc, guard);
                }
            }
            Ordering::Greater => {
                // new_key.ikey() < existing_key.ikey()
                // new goes in slot 0, existing goes in slot 1
                // SAFETY: guard requirement passed through from caller
                unsafe {
                    final_leaf.assign_from_key_arc(0, new_key, new_arc, guard);
                    final_leaf.assign_from_key_arc(1, existing_key, existing_arc, guard);
                }
            }
            Ordering::Equal => {
                // Keys have same ikey at this level.
                // This happens when one key is a prefix of the other.
                // Convention: shorter key first (prefix before extension).
                if existing_key.current_len() <= new_key.current_len() {
                    // existing is shorter or equal length -> existing first
                    // SAFETY: guard requirement passed through from caller
                    unsafe {
                        final_leaf.assign_from_key_arc(0, existing_key, existing_arc, guard);
                        final_leaf.assign_from_key_arc(1, new_key, new_arc, guard);
                    }
                } else {
                    // new is shorter -> new first
                    // SAFETY: guard requirement passed through from caller
                    unsafe {
                        final_leaf.assign_from_key_arc(0, new_key, new_arc, guard);
                        final_leaf.assign_from_key_arc(1, existing_key, existing_arc, guard);
                    }
                }
            }
        }

        // Set permutation: final leaf now has exactly 2 entries in slots 0 and 1
        final_leaf.set_permutation(<L::Perm as TreePermutation>::make_sorted(2));
    }
}
