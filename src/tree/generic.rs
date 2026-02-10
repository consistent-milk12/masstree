use std::{
    cmp::Ordering,
    marker::PhantomData,
    ptr as StdPtr,
    sync::atomic::{AtomicPtr, Ordering as AtomicOrdering},
};

use seize::{Collector, Guard, LocalGuard};

use crate::{
    BatchedRetire, Linker, MassTreeGeneric, TreeAllocator, TreePermutation,
    hints::{likely, unlikely},
    internode::InternodeNode,
    key::Key,
    leaf_trait::{LayerCapableLeaf, TreeLeafNode},
    leaf15::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode15},
    nodeversion::{LockGuard, NodeVersion},
    policy::LeafPolicy,
    prefetch::prefetch_read,
    shard_counter::ShardedCounter,
    tree::{
        InsertError, InsertSearchResultGeneric,
        coalesce::{Coalesce, CoalesceQueue},
        generic::entry::Entry,
        remove::NodeCleaner,
        split::Propagation,
    },
};

use super::remove::RemoveError;

mod batch;
mod entry;
mod insert;
mod layer;
mod optimistic_reads;
mod search;
mod split;

// Re-export batch types
pub use batch::{BatchEntry, BatchInsertResult};

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Create a new empty `MassTreeGeneric` with the given allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    /// Uses default batch size for memory reclamation (max(CPU count, 32)).
    #[must_use]
    pub fn with_allocator(allocator: A) -> Self {
        Self::with_allocator_batch_size(allocator, None)
    }

    /// Create a new empty `MassTreeGeneric` with custom batch size.
    ///
    /// # Arguments
    ///
    /// * `allocator` - Node allocator for leaf and internode allocation
    /// * `batch_size` - Number of retirements to batch before triggering reclamation.
    ///   - `None` uses default (max(CPU count, 32))
    ///   - Smaller values (16-32): Lower latency variance, more frequent barriers
    ///   - Larger values (64-128): Better throughput, but latency spikes
    ///
    /// # Performance Tuning
    ///
    /// The batch size controls how many retired pointers accumulate before
    /// seize issues a heavy memory barrier (`sys_membarrier` on Linux).
    /// Each barrier costs ~1-5µs, so larger batches amortize this cost.
    #[must_use]
    pub fn with_allocator_batch_size(allocator: A, batch_size: Option<usize>) -> Self {
        // Create root leaf directly in allocator memory (bypasses Box for pool allocators).
        let root_ptr: *mut LeafNode15<P> = allocator.alloc_leaf_direct(true, false);

        let collector: Collector =
            batch_size.map_or_else(Collector::new, |size| Collector::new().batch_size(size));

        Self {
            collector,
            allocator,
            root_ptr: AtomicPtr::new(root_ptr.cast::<u8>()),
            count: ShardedCounter::new(),
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

    /// Flush pending retirements to accelerate memory reclamation.
    ///
    /// This triggers seize to process any accumulated retirements in the
    /// thread-local batch. Call this at natural boundaries in your workload
    /// to get more predictable reclamation timing.
    ///
    /// # Note
    ///
    /// Actual reclamation won't happen until all guards protecting the
    /// retired pointers are dropped. This just starts the process.
    #[inline(always)]
    pub fn flush(&self, guard: &LocalGuard<'_>) {
        self.verify_guard(guard);
        BatchedRetire::flush(guard);
    }

    /// Get the approximate number of keys in the tree.
    ///
    /// # Consistency
    ///
    /// During concurrent mutations, this returns an approximate count
    /// (sharded counter sums multiple atomics). After all mutating threads
    /// have joined/quiesced, this returns the exact count.
    ///
    /// # Performance
    ///
    /// O(16) - reads 16 cache-line-aligned shards.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load()
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
    /// # Algorithm
    ///
    /// For each queued empty leaf:
    /// 1. Lock the leaf and verify still empty
    /// 2. Mark as deleted
    /// 3. Unlink from B-link chain
    /// 4. Remove from parent internode (lock coupling)
    /// 5. Retire the leaf for epoch-based reclamation
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
    /// // Clean up empty leaves (fully removes and reclaims memory)
    /// let processed = tree.process_coalesce(&guard);
    /// println!("Processed {} empty leaves", processed);
    /// ```
    #[inline]
    pub fn process_coalesce(&self, guard: &LocalGuard<'_>) -> usize {
        self.verify_guard(guard);
        Coalesce::process_all::<P, A>(&self.coalesce_queue, &self.allocator, guard)
    }

    /// Process up to `limit` pending empty leaf removals.
    ///
    /// Useful for bounded cleanup during normal operations.
    ///
    /// See [`process_coalesce`](Self::process_coalesce) for the full algorithm.
    ///
    /// # Returns
    ///
    /// The number of entries processed.
    #[inline]
    pub fn process_coalesce_batch(&self, guard: &LocalGuard<'_>, limit: usize) -> usize {
        self.verify_guard(guard);
        Coalesce::process_batch::<P, A>(&self.coalesce_queue, &self.allocator, guard, limit)
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Load the root pointer atomically.
    #[inline(always)]
    pub(crate) fn load_root_ptr_generic(&self, _guard: &LocalGuard<'_>) -> *const u8 {
        self.root_ptr.load(AtomicOrdering::Acquire)
    }

    /// Returns the root pointer for debugging/testing purposes.
    ///
    /// # Safety
    /// The returned pointer is only valid while the guard is held.
    /// Caller must not dereference after dropping the guard.
    #[doc(hidden)]
    pub fn debug_root_ptr(&self, _guard: &LocalGuard<'_>) -> *mut u8 {
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
        reason = "root_ptr points to L or InternodeNode, both have NodeVersion \
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

    /// Verify that a guard was created from this tree's collector.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the guard's collector does not match this tree's collector.
    /// This indicates the caller passed a guard from a different tree, which
    /// is a bug that could lead to use-after-free.
    #[inline(always)]
    pub(crate) fn verify_guard(&self, guard: &LocalGuard<'_>) {
        debug_assert!(
            *guard.collector() == self.collector,
            "Guard was created from a different collector. \
             Use tree.guard() to create guards for this tree."
        );
    }

    /// Increment the entry count.
    #[inline(always)]
    #[allow(dead_code, reason = "infrastructure for entry count tracking")]
    pub(crate) fn inc_count(&self) {
        self.count.increment();
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
    pub(crate) fn reach_leaf_generic(&self, key: &Key<'_>) -> &LeafNode15<P> {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both L and InternodeNode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or InternodeNode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        // SAFETY: version_ptr is valid per comment above
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf node
            unsafe { &*(root.cast::<LeafNode15<P>>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &InternodeNode = unsafe { &*(root.cast::<InternodeNode>()) };
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
    fn reach_leaf_via_internode_generic(
        &self,
        mut inode: &InternodeNode,
        key: &Key<'_>,
    ) -> &LeafNode15<P> {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();

        loop {
            // Find child index using generic search
            let child_idx: usize =
                upper_bound_internode_generic::<InternodeNode>(target_ikey, inode);
            // SAFETY: Dead-code traversal helper, no concurrent retirement.
            let child_ptr: *mut u8 = unsafe { inode.child_unguarded(child_idx) };

            // Prefetch child node (hides memory latency for next iteration)
            prefetch_read(child_ptr);

            // Speculatively prefetch grandchild if children are internodes
            // This overlaps the grandchild fetch with the current child access
            if !inode.children_are_leaves() {
                // Child is an internode - prefetch its first child (grandchild)
                // SAFETY: children_are_leaves() is false, so child is an internode
                let child_inode: &InternodeNode = unsafe { &*(child_ptr.cast::<InternodeNode>()) };
                // SAFETY: Dead-code traversal helper, no concurrent retirement.
                let grandchild_ptr: *mut u8 = unsafe { child_inode.child_unguarded(0) };
                prefetch_read(grandchild_ptr);
            }

            // Check child type via NodeVersion
            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to L or InternodeNode, both properly aligned"
            )]
            // SAFETY: child_ptr points to valid node with NodeVersion as first field
            let child_version: &NodeVersion = unsafe { &*(child_ptr.cast::<NodeVersion>()) };

            if child_version.is_leaf() {
                // SAFETY: is_leaf() confirms this is a leaf
                return unsafe { &*(child_ptr.cast::<LeafNode15<P>>()) };
            }

            // Descend to child internode
            // SAFETY: !is_leaf() confirms InternodeNode
            inode = unsafe { &*(child_ptr.cast::<InternodeNode>()) };
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    #[inline(always)]
    #[allow(dead_code, reason = "traversal helper for future features")]
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "Returns &mut L which requires &mut self for lifetime"
    )]
    pub(crate) fn reach_leaf_mut_generic(&mut self, key: &Key<'_>) -> &mut LeafNode15<P> {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let root: *mut u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or InternodeNode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        // SAFETY: version_ptr is valid per comment above
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf
            unsafe { &mut *(root.cast::<LeafNode15<P>>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &InternodeNode = unsafe { &*(root.cast::<InternodeNode>()) };

            let ikey: u64 = key.ikey();
            let child_idx: usize = upper_bound_internode_generic::<InternodeNode>(ikey, internode);
            // SAFETY: Dead-code traversal helper, no concurrent retirement.
            let start_ptr: *mut u8 = unsafe { internode.child_unguarded(child_idx) };

            // Prefetch child node
            prefetch_read(start_ptr);

            let children_are_leaves: bool = internode.children_are_leaves();

            if children_are_leaves {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                unsafe { &mut *start_ptr.cast::<LeafNode15<P>>() }
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
    fn reach_leaf_mut_iterative_generic(mut current: *mut u8, ikey: u64) -> *mut LeafNode15<P> {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &InternodeNode = unsafe { &*(current.cast::<InternodeNode>()) };
            let child_idx: usize = upper_bound_internode_generic::<InternodeNode>(ikey, internode);
            // SAFETY: Dead-code traversal helper, no concurrent retirement.
            let child_ptr: *mut u8 = unsafe { internode.child_unguarded(child_idx) };

            // Prefetch child node
            prefetch_read(child_ptr);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                return child_ptr.cast::<LeafNode15<P>>();
            }

            // Child is an internode - prefetch its first child (grandchild)
            // SAFETY: !children_are_leaves() so child is an internode
            let child_inode: &InternodeNode = unsafe { &*(child_ptr.cast::<InternodeNode>()) };
            // SAFETY: Dead-code traversal helper, no concurrent retirement.
            let grandchild_ptr: *mut u8 = unsafe { child_inode.child_unguarded(0) };
            prefetch_read(grandchild_ptr);

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

            // SAFETY: Dead-code traversal helper, no concurrent retirement of parent pointers.
            let parent = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &LeafNode15<P> = unsafe { &*(node.cast::<LeafNode15<P>>()) };
                unsafe { leaf.parent_unguarded() }
            } else {
                // SAFETY: !version.is_leaf() confirmed
                let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };
                unsafe { inode.parent_unguarded() }
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

        // SAFETY: Dead-code traversal helper, no concurrent retirement of parent pointers.
        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed
            let leaf: &LeafNode15<P> = unsafe { &*(node.cast::<LeafNode15<P>>()) };
            unsafe { leaf.parent_unguarded() }
        } else {
            // SAFETY: !version.is_leaf() confirmed
            let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };
            unsafe { inode.parent_unguarded() }
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
    #[expect(
        clippy::too_many_lines,
        reason = "hot-path tree traversal, logically cohesive and performance-critical"
    )]
    pub(crate) fn reach_leaf_concurrent_generic(
        &self,
        start: *const u8,
        key: &Key<'_>,
        _is_sublayer: bool,
        guard: &LocalGuard<'_>,
    ) -> *mut LeafNode15<P> {
        use crate::ksearch::upper_bound_internode_generic;
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

                // Break on root OR deleted node. A deleted sublayer root has
                // ROOT_BIT cleared by SPLIT_UNLOCK_MASK (mark_deleted sets
                // SPLITTING_BIT, and LockGuard::drop clears ROOT_BIT for
                // splitting nodes). Without this check, a reader that enters
                // a deleted sublayer with null parent spins forever since
                // is_root() is false and maybe_parent_one_step returns self.
                // The caller handles deleted nodes downstream via
                // deleted_layer() / handle_deleted_leaf recovery paths.
                if version.is_root() || version.is_deleted() {
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

                // OPTIMIZATION: Use try_stable() to avoid spinning on locked internodes.
                // If the node is locked (being modified), retry from root fix-up loop
                // rather than spinning locally. This reduces OCC contention by letting
                // the writer make progress while we re-traverse.
                //
                // Benchmarks show this reduces CPU time in NodeVersion::stable() from
                // 11% to ~7% under high contention (12 threads, rw3 workload) and
                // improves throughput by ~22% on read-heavy workloads.
                //
                // Trade-off: Write-heavy workloads may see slight regression (~15%)
                // due to more retries, but read-heavy workloads (the common case)
                // get significant speedups.
                //
                v[sense] = match version.try_stable() {
                    Some(v) => v,
                    None => continue 'retry, // Node is locked, restart from root
                };

                // Check if we've reached a leaf
                if version.is_leaf() {
                    // Return the leaf pointer - caller must check deleted_layer()
                    // and handle by unshifting key and restarting from main root.
                    // We cannot handle deleted_layer here because `start` is fixed
                    // to the sublayer root, causing an infinite loop if we retry.
                    return n[sense].cast_mut().cast::<LeafNode15<P>>();
                }

                // It's an internode - traverse down
                // SAFETY: !is_leaf() confirmed above
                let inode: &InternodeNode = unsafe { &*(n[sense].cast::<InternodeNode>()) };

                // ============================================================
                // OPTIMIZATION: Prefetch internode's ikeys BEFORE search
                // ============================================================
                // This matches C++ `in->prefetch()` at masstree_struct.hh:659
                // The prefetch hides memory latency during the upper_bound search
                // that follows. We prefetch cache lines beyond the version field
                // which contains the ikeys array.
                //
                // Internode layout: [version:4][nkeys:1][height:4][ikeys:WIDTH*8][children:(WIDTH+1)*8]
                // First cache line (64 bytes) contains version + first ~7 ikeys
                // Second cache line contains remaining ikeys
                prefetch_read(
                    StdPtr::from_ref::<InternodeNode>(inode)
                        .cast::<u8>()
                        .wrapping_add(64),
                );

                // Binary/linear search for child pointer
                let child_idx: usize =
                    upper_bound_internode_generic::<InternodeNode>(target_ikey, inode);

                // DEBUG: Trace routing path
                #[cfg(feature = "debug-routing")]
                {
                    let nkeys = inode.nkeys();
                    eprintln!(
                        "[ROUTE] inode={:p} height={} nkeys={} target_ikey={:016x} -> child_idx={}",
                        inode,
                        inode.height(),
                        nkeys,
                        target_ikey,
                        child_idx
                    );
                    // Print separator keys around the chosen child
                    if child_idx > 0 && child_idx <= nkeys {
                        let left_sep = inode.ikey(child_idx - 1);
                        eprintln!("        left_sep[{}]={:016x}", child_idx - 1, left_sep);
                    }
                    if child_idx < nkeys {
                        let right_sep = inode.ikey(child_idx);
                        eprintln!("        right_sep[{child_idx}]={right_sep:016x}");
                    }
                }

                // Read child into the OTHER buffer (sense ^ 1)
                let other: usize = sense ^ 1;
                let child: *mut u8 = inode.child(child_idx, guard);
                n[other] = child.cast_const();

                // Null child means concurrent split in progress (rare)
                if unlikely(child.is_null()) {
                    // Retry - root fix-up loop will find current root
                    continue 'retry;
                }

                // Prefetch child while we validate (hide memory latency)
                prefetch_read(child);

                // Speculatively prefetch for internode chains
                // This overlaps the grandchild fetch with validation
                if !inode.children_are_leaves() {
                    // Child is an internode - prefetch its ikeys (second cache line)
                    // SAFETY: children_are_leaves() is false, so child is an internode
                    let child_inode: &InternodeNode = unsafe { &*(child.cast::<InternodeNode>()) };
                    prefetch_read(
                        StdPtr::from_ref::<InternodeNode>(child_inode)
                            .cast::<u8>()
                            .wrapping_add(64),
                    );

                    // OPTIMIZATION: Simple grandchild prefetch (matches C++ approach)
                    // Always prefetch child(0) - the first child is on the same cache line
                    // as the internode header, so this prefetches the likely-needed node.
                    // Complex bidirectional logic showed no measurable benefit in benchmarks
                    // but added 2-3% overhead from the conditional logic and nkeys_relaxed().
                    //
                    // Prefetch is safe even for null/invalid pointers (no-op on x86/ARM)
                    let grandchild_ptr: *mut u8 = child_inode.child(0, guard);
                    prefetch_read(grandchild_ptr);
                }

                // Get stable version of child BEFORE validating parent
                // This is the key insight: we have child's version captured,
                // so even if parent changes, we know child state at read time
                // SAFETY: child is non-null and valid
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let child_version: &NodeVersion = unsafe { &*(child.cast::<NodeVersion>()) };
                v[other] = child_version.stable();

                // Now validate parent - did it change during our read?
                // Common case: unchanged (no concurrent modification)
                if likely(!inode.version().has_changed(v[sense])) {
                    // Success! Parent unchanged, child read is valid
                    // Swap sense to "move" to child
                    sense = other;
                    continue;
                }

                // Parent changed - check if it was a split that moved our key
                let old_v: u32 = v[sense];
                v[sense] = inode.version().stable(); // Re-read parent version

                if unlikely(inode.version().has_split(old_v)) {
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
    fn stable_last_key_compare(leaf: &LeafNode15<P>, key: &Key<'_>, mut version: u32) -> Ordering {
        loop {
            let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();
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

            // Check if version is still valid (common case: unchanged)
            if likely(!leaf.version().has_changed(version)) {
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
        mut leaf: &'a LeafNode15<P>,
        key: &Key<'_>,
        old_version: u32,
        guard: &LocalGuard<'_>,
    ) -> (&'a LeafNode15<P>, u32) {
        let key_ikey: u64 = key.ikey();
        let mut version: u32 = leaf.version().stable();

        // Check if we need to walk the B-link chain
        // C++: if (unlikely(v.has_split(oldv)) && n->stable_last_key_compare(ka, v, ti) > 0)
        let needs_walk: bool = if unlikely(leaf.version().has_split(old_version)) {
            // Split occurred - check if key is greater than last key in leaf
            Self::stable_last_key_compare(leaf, key, version) == Ordering::Greater
        } else if unlikely(leaf.version().is_splitting()) {
            // Split in progress - wait and check
            version = leaf.version().stable();
            Self::stable_last_key_compare(leaf, key, version) == Ordering::Greater
        } else {
            false
        };

        if !needs_walk {
            return (leaf, version);
        }

        // Walk B-link chain to find correct leaf (common case: not deleted)
        while likely(!leaf.version().is_deleted()) {
            let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

            // Check for marked pointer (split in progress OR deleted node)
            if Linker::is_marked(next_raw) {
                leaf.wait_for_split();
                // After wait_for_split returns, re-check is_deleted via the
                // while condition. If deleted, we exit the loop. If not deleted
                // (split completed), we re-read next_raw which should now be
                // unmarked.
                continue;
            }

            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr protected by guard
            let next: &LeafNode15<P> = unsafe { &*next_ptr };
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

        // Note: With proper parent cleanup in coalesce, deleted leaves are now
        // unreachable from the tree. The while loop's is_deleted() check provides
        // defense-in-depth, but the post-loop handling is no longer needed since
        // we won't exit the loop due to deletion (leaf is already removed from tree).

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
    /// Advance along B-links to find the leaf containing the key.
    ///
    /// # Provenance
    ///
    /// This function takes and returns `*mut LeafNode15<P>` to preserve mutable provenance
    /// throughout the B-link walk. This is required for Miri/Stacked Borrows
    /// compliance: pointers that will eventually be freed via `Box::from_raw`
    /// must maintain their mutable provenance.
    ///
    /// # Returns
    ///
    /// - `(*mut LeafNode15<P>, false)` - Found the correct leaf
    /// - `(*mut LeafNode15<P>, true)` - Exceeded hop limit, caller should retry from root
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_by_bound_generic(
        &self,
        mut leaf_ptr: *mut LeafNode15<P>,
        key: &Key<'_>,
        guard: &LocalGuard<'_>,
    ) -> (*mut LeafNode15<P>, bool) {
        let key_ikey: u64 = key.ikey();
        let mut hops: usize = 0;

        // SAFETY: leaf_ptr is valid, protected by guard
        let mut leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        // Wait for any in-progress split to complete
        let version = leaf.version();
        if version.is_splitting() {
            let _ = version.stable();
        }

        loop {
            // Check hop limit to prevent unbounded B-link walks
            if hops >= Self::MAX_BLINK_HOPS {
                return (leaf_ptr, true);
            }

            // Check if current leaf was deleted (e.g., by coalescing) - rare
            // If so, follow B-link to successor and continue from there
            if unlikely(leaf.version().is_deleted()) {
                let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
                let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);
                if next_ptr.is_null() {
                    // Deleted leaf has no successor - return it, caller will
                    // detect the deleted state and retry from root
                    break;
                }
                // SAFETY: next_ptr is valid, protected by guard
                leaf_ptr = next_ptr;
                leaf = unsafe { &*leaf_ptr };
                hops += 1;
                continue;
            }

            let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
            if Linker::is_marked(next_raw) {
                leaf.wait_for_split();
                // After wait_for_split returns, re-check: either the split
                // completed (next unmarked) or the node was deleted (handled
                // at top of loop on next iteration)
                continue;
            }

            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr is valid
            let next: &LeafNode15<P> = unsafe { &*next_ptr };

            // OPTIMIZATION: Prefetch next->next while we compare ikey_bound.
            // This hides memory latency for the next B-link hop (~50-100ns L3 hit).
            // Prefetch is safe even for null/invalid pointers (no-op on x86/ARM).
            //
            // Prefetch 2 cache lines for B-link traversal:
            // - CL0 (0):  header (version, next ptr, ikey_bound) - needed for B-link decisions
            // - CL1 (64): permutation - needed when search starts
            //
            // We don't prefetch ikey0 here because:
            // 1. The B-link loop only checks ikey_bound, not individual keys
            // 2. prefetch_for_search() will fetch ikeys when we land on the final leaf
            let next_next_raw: *mut LeafNode15<P> = next.next_raw(guard);
            let next_next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_next_raw);
            let prefetch_base: *const u8 = next_next_ptr.cast::<u8>();
            prefetch_read(prefetch_base);
            prefetch_read(prefetch_base.wrapping_add(64));

            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                // Key belongs in next leaf or further
                leaf_ptr = next_ptr;
                leaf = next;
                hops += 1;
                continue;
            }

            break;
        }

        (leaf_ptr, false)
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
    /// * `None` - New key inserted
    /// * `Some(old)` - Key existed, old value returned
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption (should not happen in normal operation).
    #[inline]
    pub fn insert(&self, key: &[u8], value: P::Value) -> Option<P::Output> {
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
    /// * `None` - New key inserted
    /// * `Some(old)` - Key existed, old value returned
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption (should not happen in normal operation).
    pub fn insert_with_guard(
        &self,
        key: &[u8],
        value: P::Value,
        guard: &LocalGuard<'_>,
    ) -> Option<P::Output> {
        self.verify_guard(guard);
        let mut key: Key<'_> = Key::new(key);
        let output: <P as LeafPolicy>::Output = P::into_output(value);

        // Internal errors indicate bugs, not recoverable conditions
        match self.insert_concurrent_generic(&mut key, output, guard) {
            Ok(result) => result,
            Err(e) => {
                // SplitFailed indicates tree corruption or internal bug
                panic!(
                    "Insert failed unexpectedly: {e:?}. This indicates a bug in the tree implementation."
                );
            }
        }
    }

    /// Internal: Insert with a pre-created output value
    ///
    /// Used by entry API to avoid double traversal, the output is created
    /// before insertion so it can be cloned and returned without re-fetching.
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption.
    pub(crate) fn insert_output_with_guard(
        &self,
        key: &[u8],
        output: P::Output,
        guard: &LocalGuard<'_>,
    ) -> Option<P::Output> {
        self.verify_guard(guard);

        let mut key: Key<'_> = Key::new(key);
        match self.insert_concurrent_generic(&mut key, output, guard) {
            Ok(result) => result,
            Err(e) => {
                panic!(
                    "Insert failed unexpectedly: {e:?}. This indicates a bug in the tree implementation."
                );
            }
        }
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
    pub fn remove(&self, key: &[u8]) -> Result<Option<P::Output>, RemoveError> {
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
    ) -> Result<Option<P::Output>, RemoveError> {
        self.verify_guard(guard);
        NodeCleaner::remove_concurrent_generic(self, key, guard)
    }

    /// Decrement the entry count after successful removal.
    ///
    /// Called by `finish_remove_generic` after updating the permutation.
    #[inline(always)]
    pub(crate) fn dec_count(&self) {
        self.count.decrement();
    }

    /// Assign a value to a slot in a locked leaf.
    ///
    /// Handles both inline keys (0-8 bytes) and suffix keys (>8 bytes).
    /// For suffix keys, stores `keylenx = KSUF_KEYLENX` and allocates suffix storage.
    ///
    /// Returns a raw pointer to a suffix bag that must be retired after
    /// releasing the lock, or null if no retirement is needed.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    #[expect(clippy::too_many_arguments, reason = "Internals")]
    fn assign_slot_generic(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        key: &Key<'_>,
        value: &P::Output,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> *mut u8 {
        // DEBUG: Verify membership invariant at commit point.
        // If prev exists (non-leftmost leaf), key must be >= ikey_bound.
        debug_assert!(
            leaf.prev(guard).is_null() || key.ikey() >= leaf.ikey_bound(),
            "INSERT INVARIANT VIOLATION: key {:016x} < leaf.ikey_bound {:016x}",
            key.ikey(),
            leaf.ikey_bound()
        );

        let ikey: u64 = key.ikey();

        // Mark insert dirty
        lock.mark_insert();

        // Store key data and value with Relaxed ordering.
        // The permutation update (Release) is the linearization point,
        // so these intermediate stores don't need memory barriers.
        leaf.set_ikey_relaxed(slot, ikey);
        leaf.store_value_relaxed(slot, value);

        // Handle suffix keys correctly
        if key.has_suffix() {
            // Key has suffix bytes beyond the 8-byte ikey
            leaf.set_keylenx_relaxed(slot, KSUF_KEYLENX);
            // SAFETY: We hold the lock, guard is from this tree's collector.
            // Use prealloc path if buffer available, otherwise normal path.
            pre_allocated.map_or_else(
                || unsafe { leaf.assign_ksuf(slot, key.suffix(), guard) },
                |buffer| unsafe { leaf.assign_ksuf_prealloc(slot, key.suffix(), guard, buffer) },
            )
        } else {
            // Inline key (0-8 bytes total, no suffix)
            #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
            let keylenx: u8 = key.current_len() as u8;
            leaf.set_keylenx_relaxed(slot, keylenx);
            StdPtr::null_mut()
        }
    }

    // ========================================================================
    //  Entry API
    // ========================================================================

    /// Gets the given key's corresponding entry for in-place manipulation.
    ///
    /// # Differences from `HashMap::entry`
    ///
    /// - Requires a [`LocalGuard`] for concurrent access
    /// - Returns values by-value, not by mutable reference
    /// - `and_modify` takes a transform function `FnOnce(&Output) -> Value`
    /// - Key is borrowed (zero allocation for entry creation)
    /// - Provides fallible `try_*` variants
    ///
    /// # Why No `entry()` Without Guard
    ///
    /// A guard-less `entry(&self, key)` would need to create a guard internally,
    /// but the Entry must hold a reference to that guard. This creates a
    /// self-referential struct, which Rust does not support without unsafe.
    ///
    /// # Concurrency Warning
    ///
    /// Entry operations are **NOT atomic**. Between the initial lookup and
    /// any modifications, other threads may update the same key. This follows
    /// "last-writer-wins" semantics.
    ///
    /// **Critical:** `or_insert` may OVERWRITE a value inserted by another
    /// thread between `entry_with_guard()` and `or_insert()`. This differs
    /// from `HashMap::or_insert`.
    ///
    /// ```text
    /// Thread A: entry_with_guard(key) -> Vacant
    /// Thread B: insert(key, 100)       // Inserts 100
    /// Thread A: or_insert(0)           // OVERWRITES with 0!
    /// // Result: 0 (Thread B's insert is lost)
    /// ``
    /// For atomic read-modify-write, use external synchronization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use masstree::MassTree;
    ///
    /// let tree: MassTree<u64> = MassTree::new();
    /// let guard = tree.guard();
    ///
    /// // Insert if absent, returning the value
    /// let value = tree.entry_with_guard(b"key", &guard).or_insert(42);
    ///
    /// // Increment counter, defaulting to 0
    /// let value = tree.entry_with_guard(b"counter", &guard)
    ///     .and_modify(|v| *v + 1)  // For Arc<u64>: dereference Arc
    ///     .or_insert(0);
    ///
    /// // Fallible insertion
    /// let result = tree.entry_with_guard(b"key", &guard)
    ///     .or_try_insert(100);
    ///
    /// // Pattern matching
    /// use masstree::Entry;
    /// match tree.entry_with_guard(b"maybe", &guard) {
    ///     Entry::Occupied(o) => {
    ///         // Remove returns the actually removed value, or None if deleted
    ///         if let Some(val) = o.remove() {
    ///             println!("Removed: {:?}", val);
    ///         }
    ///     }
    ///     Entry::Vacant(v) => {
    ///         v.insert(100);
    ///         println!("Inserted default");
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn entry_with_guard<'t, 'e>(
        &'t self,
        key: &'e [u8],
        guard: &'e LocalGuard<'t>,
    ) -> Entry<'t, 'e, P, A>
    where
        P::Output: Clone,
    {
        self.verify_guard(guard);
        Entry::new(self, key, guard)
    }
}

// ============================================================================
//  Standard Collection Traits (FromIterator, Extend)
// ============================================================================

impl<P, A, K, V> FromIterator<(K, V)> for MassTreeGeneric<P, A>
where
    P: LeafPolicy<Value = V>,
    A: TreeAllocator<P> + Default,
    K: AsRef<[u8]>,
{
    /// Creates a tree from an iterator of key-value pairs.
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails during tree construction.
    /// This matches std collection behavior.
    ///
    /// # Performance
    ///
    /// Uses sequential single-threaded insertion. For parallel bulk insertion
    /// of large datasets, use [`insert_batch`](Self::insert_batch) or rayon
    /// with explicit guards.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use masstree::MassTree;
    ///
    /// let data = vec![
    ///     (b"key1".to_vec(), 1u64),
    ///     (b"key2".to_vec(), 2u64),
    ///     (b"key3".to_vec(), 3u64),
    /// ];
    ///
    /// let tree: MassTree<u64> = data.into_iter().collect();
    ///
    /// assert_eq!(tree.len(), 3);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let allocator: A = A::default();
        let tree: Self = Self::with_allocator(allocator);
        let guard: LocalGuard<'_> = tree.guard();

        for (key, value) in iter {
            // Inserts are infallible (abort on OOM like std collections)
            tree.insert_with_guard(key.as_ref(), value, &guard);
        }

        drop(guard);

        tree
    }
}

impl<P, A, K, V> Extend<(K, V)> for MassTreeGeneric<P, A>
where
    P: LeafPolicy<Value = V>,
    A: TreeAllocator<P>,
    K: AsRef<[u8]>,
{
    /// Extends the tree with key-value pairs from an iterator.
    ///
    /// # Allocation
    ///
    /// Allocations are infallible (abort on OOM like std collections).
    ///
    /// # Upsert Semantics
    ///
    /// If a key already exists, it's value is replaced (same as `insert()`).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use masstree::MassTree;
    ///
    /// let mut tree: MassTree<u64> = MassTree::new();
    /// tree.insert(b"existing", 0);
    ///
    /// // Can use byte literals directly, &[u8; N] implements AsRef<[u8]>
    /// tree.extend([
    ///     (b"key1", 1u64);
    ///     (b"key2", 2u64);
    /// ]);
    ///
    /// assert_eq!(tree.len(), 3);
    /// ```
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let guard: LocalGuard<'_> = self.guard();

        for (key, value) in iter {
            // Skip failed inserts, Extend returns () so we can't propagate errors
            let _ = self.insert_with_guard(key.as_ref(), value, &guard);
        }
    }
}
