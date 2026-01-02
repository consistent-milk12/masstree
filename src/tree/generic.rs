use std::{
    cmp::Ordering,
    marker::PhantomData,
    ptr as StdPtr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering},
};

use seize::{Collector, Guard, LocalGuard};

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric, TreeInternode, TreePermutation, is_marked,
    key::Key,
    leaf_trait::LayerCapableLeaf,
    leaf24::{KSUF_KEYLENX, LAYER_KEYLENX},
    nodeversion::NodeVersion,
    slot::ValueSlot,
    tree::{
        InsertError, InsertSearchResultGeneric,
        coalesce::{self, CoalesceQueue},
        remove::NodeCleaner,
        split::Propagation,
    },
    unmark_ptr,
};

use super::remove::RemoveError;

mod insert;
mod layer;
mod optimistic_reads;
mod search;
mod split;

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Create a new empty `MassTreeGeneric` with the given allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn with_allocator(allocator: A) -> Self {
        // Create root leaf directly in allocator memory (bypasses Box for pool allocators).
        let root_ptr: *mut L = allocator.alloc_leaf_direct(true, false);

        Self {
            collector: Collector::new(),
            allocator,
            root_ptr: AtomicPtr::new(root_ptr.cast::<u8>()),
            count: AtomicUsize::new(0),
            coalesce_queue: CoalesceQueue::new(),
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
    ///
    /// Returns `true` if the tree contains no keys.
    ///
    /// Note: After deletions, the tree structure (internodes, empty leaves) may
    /// persist even when empty. This method checks the key count, not the structure.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    //  Lazy Coalescing (Empty Leaf Cleanup)
    // ========================================================================

    /// Get the number of empty leaves pending cleanup.
    ///
    /// This is the number of leaves that became empty after key removal
    /// and are waiting to be processed by `process_coalesce()`.
    #[must_use]
    #[inline]
    pub fn pending_coalesce(&self) -> usize {
        self.coalesce_queue.len()
    }

    /// Process all pending empty leaf removals.
    ///
    /// Call this during low-contention periods to clean up empty leaves.
    /// This is safe to call concurrently with other operations.
    ///
    /// # Returns
    ///
    /// The number of entries processed (including skipped/re-queued).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree::new();
    /// let guard = tree.guard();
    ///
    /// // ... perform many removes ...
    ///
    /// // Clean up empty leaves
    /// let processed = tree.process_coalesce(&guard);
    /// println!("Processed {} empty leaves", processed);
    /// ```
    #[inline]
    pub fn process_coalesce(&self, guard: &LocalGuard<'_>) -> usize {
        coalesce::process_all::<S, L, A>(&self.coalesce_queue, &self.allocator, guard)
    }

    /// Process up to `limit` pending empty leaf removals.
    ///
    /// Useful for bounded cleanup during normal operations.
    ///
    /// # Returns
    ///
    /// The number of entries processed.
    #[inline]
    pub fn process_coalesce_batch(&self, guard: &LocalGuard<'_>, limit: usize) -> usize {
        coalesce::process_batch::<S, L, A>(&self.coalesce_queue, &self.allocator, guard, limit)
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Load the root pointer atomically.
    #[inline(always)]
    pub(crate) fn load_root_ptr_generic(&self, _guard: &LocalGuard<'_>) -> *const u8 {
        self.root_ptr.load(AtomicOrdering::Acquire)
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
    #[expect(
        dead_code,
        reason = "Kept for future use when is_empty() needs structural check"
    )]
    fn root_is_leaf_generic(&self) -> bool {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: `root_ptr` always points to a valid node.
        // `NodeVersion` is the first field of both leaf and internode types.
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();

        // SAFETY: version_ptr is valid per above invariant
        unsafe { (*version_ptr).is_leaf() }
    }

    /// Get a mutable reference to the allocator.
    #[inline(always)]
    #[allow(dead_code, reason = "infrastructure for future deletion")]
    pub(crate) const fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    /// Get an immutable reference to the allocator.
    #[inline(always)]
    #[allow(dead_code, reason = "infrastructure for future deletion")]
    pub(crate) const fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Get a reference to the collector.
    #[inline(always)]
    #[allow(dead_code, reason = "infrastructure for future deletion")]
    pub(crate) const fn collector(&self) -> &Collector {
        &self.collector
    }

    /// Increment the entry count.
    #[inline(always)]
    #[allow(dead_code, reason = "infrastructure for entry count tracking")]
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
    #[allow(dead_code, reason = "traversal helper for future features")]
    pub(crate) fn reach_leaf_generic(&self, key: &Key<'_>) -> &L {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both L and L::Internode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or L::Internode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        // SAFETY: version_ptr is valid per comment above
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
    #[allow(dead_code, reason = "traversal helper for future features")]
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
                upper_bound_internode_generic::<S, L::Internode>(target_ikey, inode);
            let child_ptr: *mut u8 = inode.child(child_idx);

            // Prefetch child node
            prefetch_read(child_ptr);

            // Check child type via NodeVersion
            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to L or L::Internode, both properly aligned"
            )]
            // SAFETY: child_ptr points to valid node with NodeVersion as first field
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
    #[allow(dead_code, reason = "traversal helper for future features")]
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
        // SAFETY: version_ptr is valid per comment above
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf
            unsafe { &mut *(root.cast::<L>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &L::Internode = unsafe { &*(root.cast::<L::Internode>()) };

            let ikey: u64 = key.ikey();
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(ikey, internode);
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
    #[allow(dead_code, reason = "traversal helper for future features")]
    fn reach_leaf_mut_iterative_generic(mut current: *mut u8, ikey: u64) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &L::Internode = unsafe { &*(current.cast::<L::Internode>()) };
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(ikey, internode);
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

    /// Follow parent pointers to find the actual layer root.
    #[inline(always)]
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

    /// Single-step parent traversal (C++ `maybe_parent()` equivalent).
    ///
    /// Returns the parent node if it exists, otherwise returns `node` itself.
    /// Used in root fix-up loop during traversal retries.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    fn maybe_parent_one_step(&self, node: *const u8) -> *const u8 {
        // SAFETY: node is valid, both types have NodeVersion as first field
        #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
        let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed
            let leaf: &L = unsafe { &*(node.cast::<L>()) };
            leaf.parent()
        } else {
            // SAFETY: !version.is_leaf() confirmed
            let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };
            inode.parent()
        };

        if parent.is_null() {
            node
        } else {
            parent.cast_const()
        }
    }

    /// Double-buffer traversal from root to leaf.
    ///
    /// This implements the C++ Masstree `reach_leaf()` pattern with:
    /// - Two-pointer technique to avoid stale child pointer validation
    /// - Split detection with key comparison to avoid unnecessary root restarts
    ///
    /// # Algorithm
    ///
    /// The double-buffer technique uses two slots for node/version pairs:
    /// 1. Read child pointer into the OTHER buffer (sense ^ 1)
    /// 2. Get child's stable version
    /// 3. Validate parent version
    /// 4. If valid: swap sense (we're now "at" child)
    /// 5. If invalid due to split: check if key > `last_key`
    /// 6. Only restart from root if key actually escaped to sibling
    #[expect(clippy::indexing_slicing, reason = "sense is always 0 or 1")]
    pub(crate) fn reach_leaf_concurrent_generic(
        &self,
        start: *const u8,
        key: &Key<'_>,
        _is_sublayer: bool,
        _guard: &LocalGuard<'_>,
    ) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::leaf_trait::TreeInternode;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();

        // Double-buffer: two node/version pairs, alternating via sense
        // n[sense] is current node, n[sense ^ 1] is candidate child
        // Note: n[0] is set in root fix-up loop, n[1] is set before being read
        let mut n: [*const u8; 2] = [StdPtr::null(); 2];
        let mut v: [u32; 2] = [0; 2];

        #[expect(unused_assignments, reason = "sense is set at top of retry loop")]
        let mut sense: usize = 0;

        'retry: loop {
            // ==================================================================
            // Root fix-up: walk up until we find the actual layer root
            // ==================================================================
            // C++ comment: "Get a non-stale root. Detect staleness by checking
            // whether n has ever split. The true root has never split."
            //
            // When a node splits, SPLIT_UNLOCK_MASK clears the ROOT_BIT.
            // So we walk up via parent pointers until we find a node with
            // is_root() set, which is the current layer root.
            sense = 0;
            n[sense] = start;
            loop {
                // SAFETY: n[sense] is valid, NodeVersion is first field
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let version: &NodeVersion = unsafe { &*(n[sense].cast::<NodeVersion>()) };
                v[sense] = version.stable();

                if version.is_root() {
                    // Found the current layer root
                    break;
                }

                // Walk up to parent
                n[sense] = self.maybe_parent_one_step(n[sense]);
            }

            // ==================================================================
            // Traverse internodes using double-buffer pattern
            // ==================================================================
            loop {
                // SAFETY: n[sense] is valid, NodeVersion is first field
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let version: &NodeVersion = unsafe { &*(n[sense].cast::<NodeVersion>()) };

                // Get stable version of current node
                v[sense] = version.stable();

                // Check if we've reached a leaf
                if version.is_leaf() {
                    // Return the leaf pointer - caller must check deleted_layer()
                    // and handle by unshifting key and restarting from main root.
                    // We cannot handle deleted_layer here because `start` is fixed
                    // to the sublayer root, causing an infinite loop if we retry.
                    return n[sense].cast_mut().cast::<L>();
                }

                // It's an internode - traverse down
                // SAFETY: !is_leaf() confirmed above
                let inode: &L::Internode = unsafe { &*(n[sense].cast::<L::Internode>()) };

                // Binary/linear search for child pointer
                let child_idx: usize =
                    upper_bound_internode_generic::<S, L::Internode>(target_ikey, inode);

                // Read child into the OTHER buffer (sense ^ 1)
                let other: usize = sense ^ 1;
                let child: *mut u8 = inode.child(child_idx);
                n[other] = child.cast_const();

                // Null child means concurrent split in progress
                if child.is_null() {
                    // Retry - root fix-up loop will find current root
                    continue 'retry;
                }

                // Prefetch child while we validate (hide memory latency)
                prefetch_read(child);

                // Get stable version of child BEFORE validating parent
                // This is the key insight: we have child's version captured,
                // so even if parent changes, we know child state at read time
                // SAFETY: child is non-null and valid
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let child_version: &NodeVersion = unsafe { &*(child.cast::<NodeVersion>()) };
                v[other] = child_version.stable();

                // Now validate parent - did it change during our read?
                if !inode.version().has_changed(v[sense]) {
                    // Success! Parent unchanged, child read is valid
                    // Swap sense to "move" to child
                    sense = other;
                    continue;
                }

                // Parent changed - check if it was a split that moved our key
                let old_v: u32 = v[sense];
                v[sense] = inode.version().stable(); // Re-read parent version

                if inode.version().has_split(old_v) {
                    // Split occurred - check if our key moved to the right sibling
                    // If target_ikey > last_key, key is beyond this internode's range
                    let nkeys: usize = inode.nkeys();
                    if nkeys > 0 {
                        let last_key: u64 = inode.ikey(nkeys - 1);
                        if target_ikey > last_key {
                            // Key escaped to sibling - must restart from root
                            // Retry - root fix-up loop will find current root
                            continue 'retry;
                        }
                    }
                }

                // Minor change (insert) or split that didn't move our key
                // Just retry this level (sense stays the same, loop continues)
            }
        }
    }

    /// Compare a key with the last key in a leaf, retrying until stable.
    ///
    /// Returns:
    /// - `Ordering::Less` if key < last key in leaf
    /// - `Ordering::Equal` if key == last key in leaf
    /// - `Ordering::Greater` if key > last key in leaf, or leaf is empty
    ///
    /// This matches C++ `leaf::stable_last_key_compare()` from `masstree_struct.hh:603-626`.
    ///
    /// # Reference
    /// ```cpp
    /// template <typename P>
    /// inline int leaf<P>::stable_last_key_compare(const key_type& k, nodeversion_type v,
    ///                                              threadinfo& ti) const {
    ///     while (true) {
    ///         typename leaf<P>::permuter_type perm(permutation_);
    ///         int n = perm.size();
    ///         int p = perm[n ? n - 1 : 0];
    ///         int cmp = compare_key(k, p);
    ///         if (likely(!this->has_changed(v))) {
    ///             return cmp;
    ///         }
    ///         v = this->stable_annotated(ti.stable_fence());
    ///     }
    /// }
    /// ```
    #[inline]
    fn stable_last_key_compare(leaf: &L, key: &Key<'_>, mut version: u32) -> Ordering {
        loop {
            let perm: L::Perm = leaf.permutation();
            let n: usize = perm.size();

            // If empty, return Greater (key should look elsewhere)
            // If non-empty, compare with last slot in permutation
            let cmp: Ordering = if n == 0 {
                // Empty leaf - return Greater so caller will check siblings
                // C++ comment: "it is always safe to return 1, because then the
                // caller will check more precisely whether k belongs in this"
                Ordering::Greater
            } else {
                let last_slot: usize = perm.get(n - 1);
                let slot_ikey: u64 = leaf.ikey(last_slot);
                let slot_keylenx: u8 = leaf.keylenx(last_slot);
                key.compare(slot_ikey, slot_keylenx as usize)
            };

            // Check if version is still valid
            if !leaf.version().has_changed(version) {
                return cmp;
            }

            // Version changed, get new stable version and retry
            version = leaf.version().stable();
        }
    }

    /// Advance to correct leaf via B-link after version change detected.
    ///
    /// This is called when `has_changed(old_version)` returns true, indicating
    /// a split may have occurred. It follows B-links to find the correct leaf.
    ///
    /// # Algorithm (from C++ `leaf::advance_to_key`)
    /// 1. Get stable version
    /// 2. If `has_split` AND key > last key in leaf: walk B-link chain
    /// 3. Otherwise: key belongs in current leaf
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_generic<'a>(
        &'a self,
        mut leaf: &'a L,
        key: &Key<'_>,
        old_version: u32,
        _guard: &LocalGuard<'_>,
    ) -> (&'a L, u32) {
        use crate::link::{is_marked, unmark_ptr};

        let key_ikey: u64 = key.ikey();
        let mut version: u32 = leaf.version().stable();

        // Check if we need to walk the B-link chain
        // C++: if (unlikely(v.has_split(oldv)) && n->stable_last_key_compare(ka, v, ti) > 0)
        let needs_walk: bool = if leaf.version().has_split(old_version) {
            // Split occurred - check if key is greater than last key in leaf
            Self::stable_last_key_compare(leaf, key, version) == Ordering::Greater
        } else if leaf.version().is_splitting() {
            // Split in progress - wait and check
            version = leaf.version().stable();
            Self::stable_last_key_compare(leaf, key, version) == Ordering::Greater
        } else {
            false
        };

        if !needs_walk {
            return (leaf, version);
        }

        // Walk B-link chain to find correct leaf
        while !leaf.version().is_deleted() {
            let next_raw: *mut L = leaf.next_raw();

            // Check for marked pointer (split in progress)
            if is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut L = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr protected by guard
            let next: &L = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                // Key belongs in next leaf or further
                leaf = next;
                version = leaf.version().stable();
                continue;
            }

            // Key belongs in current leaf
            break;
        }

        (leaf, version)
    }

    /// Maximum B-link hops before giving up and signaling re-descent.
    ///
    /// This prevents unbounded sibling walks when routing is inconsistent.
    /// After this many hops, we return the current leaf and let the caller
    /// detect the mismatch and retry from the root.
    const MAX_BLINK_HOPS: usize = 128;

    /// Advance to correct leaf via B-link (generic version).
    /// Used by insert path before locking.
    ///
    /// Returns `(leaf, exceeded_hop_limit)`. If `exceeded_hop_limit` is true,
    /// the caller should retry from the root instead of trusting this leaf.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip_all, fields(ikey = %format_args!("{:016x}", key.ikey())))
    )]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_by_bound_generic<'a>(
        &'a self,
        mut leaf: &'a L,
        key: &Key<'_>,
        _guard: &LocalGuard<'_>,
    ) -> (&'a L, bool) {
        use crate::link::{is_marked, unmark_ptr};

        let key_ikey: u64 = key.ikey();
        let mut hops: usize = 0;

        // Wait for any in-progress split to complete
        if leaf.version().is_splitting() {
            let _ = leaf.version().stable();
        }

        loop {
            // Check hop limit to prevent unbounded B-link walks
            if hops >= Self::MAX_BLINK_HOPS {
                return (leaf, true);
            }

            let next_raw: *mut L = leaf.next_raw();
            if is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut L = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr is valid
            let next: &L = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                // Key belongs in next leaf or further
                leaf = next;
                hops += 1;
                continue;
            }

            break;
        }

        (leaf, false)
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
    pub fn insert(&self, key: &[u8], value: S::Value) -> Result<Option<S::Output>, InsertError> {
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
        value: S::Value,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, InsertError> {
        let mut key = Key::new(key);
        let output = S::into_output(value);
        self.insert_concurrent_generic(&mut key, output, guard)
    }

    // ========================================================================
    //  Remove Operations
    // ========================================================================

    /// Remove a key from the tree.
    ///
    /// Returns `Ok(Some(value))` if the key existed and was removed,
    /// `Ok(None)` if the key was not found, or an error on failure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tree = MassTree24::new();
    /// tree.insert(b"key", 42)?;
    ///
    /// let removed = tree.remove(b"key")?;
    /// assert_eq!(removed, Some(Arc::new(42)));
    ///
    /// let not_found = tree.remove(b"key")?;
    /// assert_eq!(not_found, None);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `RemoveError::RetryLimitExceeded` if the operation cannot
    /// complete after the retry limit (extremely rare, indicates contention).
    pub fn remove(&self, key: &[u8]) -> Result<Option<S::Output>, RemoveError> {
        let guard = self.guard();
        self.remove_with_guard(key, &guard)
    }

    /// Remove a key using an existing guard.
    ///
    /// Use this when performing multiple operations under the same guard
    /// to amortize guard creation overhead.
    ///
    /// # Errors
    /// If fails to remove
    pub fn remove_with_guard(
        &self,
        key: &[u8],
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, RemoveError> {
        NodeCleaner::remove_concurrent_generic(self, key, guard)
    }

    /// Decrement the entry count after successful removal.
    ///
    /// Called by `finish_remove_generic` after updating the permutation.
    #[inline(always)]
    pub(crate) fn dec_count(&self) {
        self.count.fetch_sub(1, AtomicOrdering::Relaxed);
    }

    /// Assign a value to a slot in a locked leaf.
    ///
    /// Handles both inline keys (0-8 bytes) and suffix keys (>8 bytes).
    /// For suffix keys, stores `keylenx = KSUF_KEYLENX` and allocates suffix storage.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    #[expect(clippy::too_many_arguments, reason = "Internals")]
    fn assign_slot_generic(
        &self,
        leaf: &L,
        lock: &mut crate::nodeversion::LockGuard<'_>,
        slot: usize,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) {
        let ikey: u64 = key.ikey();
        let value_ptr: *mut u8 = S::output_to_raw(value);

        // Mark insert dirty
        lock.mark_insert();

        // Store key data and value
        leaf.set_ikey(slot, ikey);
        leaf.set_leaf_value_ptr(slot, value_ptr);

        // Handle suffix keys correctly
        if key.has_suffix() {
            // Key has suffix bytes beyond the 8-byte ikey
            leaf.set_keylenx(slot, KSUF_KEYLENX);
            // SAFETY: We hold the lock, guard is from this tree's collector
            unsafe { leaf.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            // Inline key (0-8 bytes total, no suffix)
            #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
            let keylenx: u8 = key.current_len() as u8;
            leaf.set_keylenx(slot, keylenx);
        }
    }
}
