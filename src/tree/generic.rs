use std::{
    cmp::Ordering,
    marker::PhantomData,
    ptr as StdPtr,
    sync::atomic::{AtomicPtr, Ordering as AtomicOrdering},
};

use seize::{Collector, Guard, LocalGuard};

use crate::{
    BatchedRetire, Linker, MassTreeGeneric, TreeAllocator, TreePermutation, ValueRef,
    hints::{likely, unlikely},
    internode::InternodeNode,
    key::Key,
    leaf_trait::TreeLeafNode,
    leaf15::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode15, MODSTATE_INSERT},
    nodeversion::{LockGuard, NodeVersion},
    policy::{LeafPolicy, PreparedValue},
    prefetch::prefetch_read,
    shard_counter::ShardedCounter,
    tree::{
        InsertError, InsertSearchResultGeneric,
        coalesce::{Coalesce, CoalesceQueue},
        remove::NodeCleaner,
        split::Propagation,
    },
};

use super::remove::RemoveError;

mod batch;
mod entry;
mod insert;
mod insert_strategy;
mod layer;
mod optimistic_reads;
mod search;
mod split;

#[cfg(feature = "debug-print")]
mod print;

// Re-export batch types
pub use batch::{BatchEntry, BatchInsertResult};
pub use entry::{Entry, OccupiedEntry, RemoveEntryResult, VacantEntry};

#[cfg(test)]
std::thread_local! {
    // Counts actual root-to-leaf descents for Entry's performance regression.
    static REACH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// ============================================================================
//  Shared Types
// ============================================================================

/// Result of finding a usable slot for insertion.
pub enum FindSlotResult {
    /// Found a usable slot at the given physical index.
    Found { slot: usize, back_offset: usize },

    /// No usable slot available - the leaf must be split.
    NeedsSplit,
}

/// Errors from membership validation.
pub enum MembershipError {
    /// A split is in progress on this leaf.
    SplitInProgress,

    /// Key's ikey is >= the next sibling's lower bound.
    KeyMovedToSibling,

    /// Key's ikey is < this leaf's lower bound (`ikey_bound()`).
    KeyBelowLowerBound,
}

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Default retirement batch size. Larger values reduce the frequency of
    /// `sys_membarrier` syscalls at the cost of slightly delayed reclamation.
    /// 256 is 8 times the [`seize`] default (32) and amortizes well under write-heavy
    /// workloads while adding negligible peak memory.
    const DEFAULT_RETIRE_BATCH_SIZE: usize = 256;

    /// Create a new empty `MassTreeGeneric` with the given allocator.
    #[must_use]
    pub fn with_allocator(allocator: A) -> Self {
        Self::with_allocator_batch_size(allocator, Some(Self::DEFAULT_RETIRE_BATCH_SIZE))
    }

    /// Create a new empty `MassTreeGeneric` with custom batch size.
    #[must_use]
    pub fn with_allocator_batch_size(allocator: A, batch_size: Option<usize>) -> Self {
        let root_ptr: *mut LeafNode15<P> = allocator.alloc_leaf_direct(true, false);

        let collector: Collector = match batch_size {
            Some(size) => Collector::new().batch_size(size),
            None => Collector::new(),
        };

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
    #[must_use]
    #[inline(always)]
    pub fn guard(&self) -> LocalGuard<'_> {
        self.collector.enter()
    }

    /// Flush pending retirements to accelerate memory reclamation.
    #[inline(always)]
    pub fn flush(&self, guard: &LocalGuard<'_>) {
        self.verify_guard(guard);
        BatchedRetire::flush(guard);
    }

    /// Perform maintenance: reclaim retired memory and clean up empty leaves.
    ///
    /// Call periodically during long-running single-threaded workloads or
    /// between batch operations to reclaim memory from deleted entries.
    /// Requires exclusive access to the guard (no nested guards active).
    ///
    /// Internally runs coalesce processing followed by `guard.flush()` + `guard.refresh()`.
    #[inline]
    pub fn maintenance(&self, guard: &mut LocalGuard<'_>) {
        self.verify_guard(guard);
        Coalesce::process_all::<P, A>(self, guard);
        guard.flush();
        guard.refresh();
    }

    /// Get the approximate number of keys in the tree.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load()
    }

    /// Check if the tree is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    //  Lazy Coalescing (Empty Leaf Cleanup)
    // ========================================================================

    /// Get the number of empty leaves pending cleanup.
    #[must_use]
    #[inline]
    pub fn pending_coalesce(&self) -> usize {
        self.coalesce_queue.len()
    }

    /// Process all pending empty leaf removals.
    #[inline]
    pub fn process_coalesce(&self, guard: &LocalGuard<'_>) -> usize {
        self.verify_guard(guard);
        #[cfg(feature = "tracing")]
        tracing::trace!("process_coalesce");

        Coalesce::process_all::<P, A>(self, guard)
    }

    /// Process up to `limit` pending empty leaf removals.
    #[inline]
    pub fn process_coalesce_batch(&self, guard: &LocalGuard<'_>, limit: usize) -> usize {
        self.verify_guard(guard);
        Coalesce::process_batch::<P, A>(self, guard, limit)
    }

    /// Number of coalesce entries permanently abandoned.
    /// Non-zero values indicate cleanup debt from sustained contention.
    #[must_use]
    #[inline]
    pub fn coalesce_abandoned(&self) -> usize {
        self.coalesce_queue.abandoned()
    }

    /// Number of coalesce entries deferred to the next maintenance call.
    #[must_use]
    #[inline]
    pub fn coalesce_deferred(&self) -> usize {
        self.coalesce_queue.deferred_len()
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Access the coalesce queue.
    #[inline(always)]
    pub(crate) const fn coalesce_queue(&self) -> &CoalesceQueue {
        &self.coalesce_queue
    }

    /// Access the allocator.
    #[inline(always)]
    pub(crate) const fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Load the root pointer atomically.
    ///
    /// Uses `Guard::protect`: the returned node is guaranteed to stay
    /// allocated for the guard's lifetime, matching seeize's reclamation
    /// contract (a plain Acquire load does not participate in the retirement
    /// barrier's total order).
    #[inline(always)]
    pub(crate) fn load_root_ptr_generic(&self, guard: &LocalGuard<'_>) -> *const u8 {
        guard
            .protect(&self.root_ptr, AtomicOrdering::Acquire)
            .cast_const()
    }

    /// Returns the root pointer for debugging/testing purposes.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while `guard` is held and while the
    /// tree is not concurrently mutating the root. The caller must not
    /// dereference it unless it has independently established the node's kind
    /// and validity, and must never use it to mutate or free nodes through
    /// this API.
    #[doc(hidden)]
    pub unsafe fn debug_root_ptr(&self, guard: &LocalGuard<'_>) -> *mut u8 {
        guard.protect(&self.root_ptr, AtomicOrdering::Acquire)
    }

    /// Verify that the guard was created from this tree's collector.
    ///
    /// # Panics
    ///
    /// Panics if the guard's collector does not match this tree's collector.
    /// This indicates the caller passed a guard from a different tree, which
    /// is a bug that could lead to use-after-free.
    #[inline(always)]
    pub(crate) fn verify_guard(&self, guard: &LocalGuard<'_>) {
        if *guard.collector() != self.collector {
            Self::guard_collector_mismatch();
        }
    }

    #[cold]
    #[inline(never)]
    fn guard_collector_mismatch() -> ! {
        panic!(
            "Guard was created from a different collector. \
             Use tree.guard() to create guards for this tree."
        );
    }

    /// Follow parent pointers to find the actual layer root.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    pub(crate) fn maybe_parent_generic(
        &self,
        mut node: *const u8,
        guard: &LocalGuard<'_>,
    ) -> *const u8 {
        loop {
            // SAFETY: node is valid, both types have NodeVersion as first field
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            let parent = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &LeafNode15<P> = unsafe { &*(node.cast::<LeafNode15<P>>()) };
                leaf.parent(guard)
            } else {
                // SAFETY: !version.is_leaf() confirmed
                let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };
                inode.parent(guard)
            };

            if parent.is_null() {
                return node;
            }

            node = parent.cast_const();
        }
    }

    /// Single-step parent traversal
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    fn maybe_parent_one_step(&self, node: *const u8, guard: &LocalGuard<'_>) -> *const u8 {
        // SAFETY: node is valid, both types have NodeVersion as first field
        #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
        let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

        let parent: *mut u8 = if version.is_leaf() {
            let leaf: &LeafNode15<P> = unsafe { &*(node.cast::<LeafNode15<P>>()) };

            // SAFETY: version.is_leaf() confirmed
            leaf.parent(guard)
        } else {
            let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };

            // SAFETY: !version.is_leaf() confirmed
            inode.parent(guard)
        };

        if parent.is_null() {
            node
        } else {
            parent.cast_const()
        }
    }

    /// Double-buffer traversal from root to leaf.
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

        #[cfg(test)]
        REACH_COUNT.with(|count| count.set(count.get() + 1));

        let target_ikey: u64 = key.ikey();

        let mut n: [*const u8; 2] = [StdPtr::null(); 2];
        let mut v: [u32; 2] = [0; 2];

        #[expect(unused_assignments, reason = "sense is set at top of retry loop")]
        let mut sense: usize = 0;

        'retry: loop {
            sense = 0;
            n[sense] = start;

            loop {
                // SAFETY: n[sense] is valid, NodeVersion is first field
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let version: &NodeVersion = unsafe { &*(n[sense].cast::<NodeVersion>()) };
                v[sense] = version.stable();

                if version.is_root() || version.is_deleted() {
                    break;
                }

                // Walk up to parent
                n[sense] = self.maybe_parent_one_step(n[sense], guard);
            }

            loop {
                // SAFETY: n[sense] is valid, NodeVersion is first field
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let version: &NodeVersion = unsafe { &*(n[sense].cast::<NodeVersion>()) };

                v[sense] = match version.try_stable() {
                    Some(v) => v,
                    None => continue 'retry, // Node is locked, restart from root
                };

                // Check if we've reached a leaf
                if version.is_leaf() {
                    return n[sense].cast_mut().cast::<LeafNode15<P>>();
                }

                // It's an internode - traverse down
                // SAFETY: !is_leaf() confirmed above
                let inode: &InternodeNode = unsafe { &*(n[sense].cast::<InternodeNode>()) };

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
                    let nkeys: usize = inode.nkeys();
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
                        let left_sep: u64 = inode.ikey(child_idx - 1);
                        eprintln!("        left_sep[{}]={:016x}", child_idx - 1, left_sep);
                    }

                    if child_idx < nkeys {
                        let right_sep: u64 = inode.ikey(child_idx);
                        eprintln!("        right_sep[{child_idx}]={right_sep:016x}");
                    }
                }

                let other: usize = sense ^ 1;
                let child: *mut u8 = inode.child(child_idx, guard);
                n[other] = child.cast_const();

                if unlikely(child.is_null()) {
                    continue 'retry;
                }

                prefetch_read(child);

                if !inode.children_are_leaves() {
                    // SAFETY: children_are_leaves() is false, so child is an internode
                    let child_inode: &InternodeNode = unsafe { &*(child.cast::<InternodeNode>()) };
                    prefetch_read(
                        StdPtr::from_ref::<InternodeNode>(child_inode)
                            .cast::<u8>()
                            .wrapping_add(64),
                    );

                    // Prefetch is safe even for null/invalid pointers (no-op on x86/ARM)
                    let grandchild_ptr: *mut u8 = child_inode.child(0, guard);
                    prefetch_read(grandchild_ptr);
                }

                // SAFETY: child is non-null and valid
                #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
                let child_version: &NodeVersion = unsafe { &*(child.cast::<NodeVersion>()) };
                v[other] = child_version.stable();

                // A cascade can leave a DELETED internode linked until the
                // parent update completes (or after RetryExhausted). Descending
                // into it would route into a zero-key node with a null child
                // and spin forever. Restart from the root instead.
                if child_version.is_deleted() {
                    continue 'retry;
                }

                if likely(!inode.version().has_changed(v[sense])) {
                    sense = other;
                    continue;
                }

                let old_v: u32 = v[sense];
                v[sense] = inode.version().stable();

                if unlikely(inode.version().has_split(old_v)) {
                    let nkeys: usize = inode.nkeys();

                    if nkeys > 0 {
                        let last_key: u64 = inode.ikey(nkeys - 1);
                        if target_ikey > last_key {
                            continue 'retry;
                        }
                    }
                }
            }
        }
    }

    /// Compare a key with the last key in a leaf, retrying until stable.
    #[inline]
    fn stable_last_key_compare(leaf: &LeafNode15<P>, key: &Key<'_>, mut version: u32) -> Ordering {
        loop {
            let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();
            let n: usize = perm.size();

            // If empty, return Greater (key should look elsewhere)
            // If non-empty, compare with last slot in permutation
            let cmp: Ordering = if n == 0 {
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

        let needs_walk: bool = if unlikely(leaf.version().has_split(old_version)) {
            Self::stable_last_key_compare(leaf, key, version) == Ordering::Greater
        } else if unlikely(leaf.version().is_splitting()) {
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

        (leaf, version)
    }

    /// Maximum B-link hops before giving up and signaling re-descent.
    const MAX_BLINK_HOPS: usize = 128;

    /// Advance to correct leaf via B-link (generic version).
    /// Used by insert path before locking.
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub(crate) fn advance_to_key_by_bound_generic(
        &self,
        mut leaf_ptr: *mut LeafNode15<P>,
        key: &Key<'_>,
        guard: &LocalGuard<'_>,
    ) -> (*mut LeafNode15<P>, bool) {
        let key_ikey: u64 = key.ikey();
        let mut hops: usize = 0;

        // SAFETY: leaf_ptr is valid, protected by guard
        let mut leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        let version: &NodeVersion = leaf.version();

        if version.is_splitting() {
            let _ = version.stable();
        }

        loop {
            if hops >= Self::MAX_BLINK_HOPS {
                return (leaf_ptr, true);
            }

            if unlikely(leaf.version().is_deleted()) {
                let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
                let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

                if next_ptr.is_null() {
                    // A deleted rightmost leaf has no successor. Returning it
                    // again would make the caller re-enter the same leaf
                    // forever; force a restart from the root instead.
                    return (leaf_ptr, true);
                }

                leaf_ptr = next_ptr;

                // SAFETY: next_ptr is valid, protected by guard
                leaf = unsafe { &*leaf_ptr };
                hops += 1;
                continue;
            }

            let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

            if Linker::is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr is valid
            let next: &LeafNode15<P> = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                let next_next_raw: *mut LeafNode15<P> = next.next_raw(guard);
                let next_next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_next_raw);
                let prefetch_base: *const u8 = next_next_ptr.cast::<u8>();

                prefetch_read(prefetch_base);
                prefetch_read(prefetch_base.wrapping_add(64));

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
    /// Returns an owned clone of the old value if the key already existed.
    /// The internal EBR guard is released before returning, so the value
    /// is cloned to ensure independent ownership.
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption (should not happen in normal operation).
    #[inline]
    pub fn insert(&self, key: &[u8], value: P::Value) -> Option<P::Value>
    where
        P::Value: Clone,
    {
        let guard = self.guard();
        self.insert_with_guard(key, value, &guard)
            .map(ValueRef::into_owned)
    }

    /// Insert a key-value pair using an explicit guard.
    ///
    /// Returns the old value as a guard-scoped snapshot. Primitive boxed values
    /// update in place and return an owned copy, with no allocation or retirement
    /// on updates. Other boxed values return a borrow protected by `guard`.
    /// No `Clone` bound is required.
    ///
    /// A borrowed old value cannot escape its guard:
    ///
    /// ```compile_fail
    /// use masstree::MassTree15;
    /// let old = {
    ///     let tree = MassTree15::<String>::new();
    ///     tree.insert(b"key", "old".to_owned());
    ///     let guard = tree.guard();
    ///     tree.insert_with_guard(b"key", "new".to_owned(), &guard).unwrap()
    /// };
    /// println!("{old:?}");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on an invalid key, a foreign guard, or internal tree corruption.
    #[inline]
    pub fn insert_with_guard<'g>(
        &self,
        key: &[u8],
        value: P::Value,
        guard: &'g LocalGuard<'_>,
    ) -> Option<ValueRef<'g, P::Value>> {
        self.verify_guard(guard);
        #[cfg(feature = "tracing")]
        tracing::debug!(key_len = key.len(), "insert");
        let mut key = Key::new(key);
        let result = if P::can_write_through() {
            self.insert_concurrent_value(&mut key, value, guard)
                .map(|old| old.map(ValueRef::Owned))
        } else {
            let output = P::into_output(value);
            self.insert_concurrent_generic(&mut key, output, guard)
                .map(|old| {
                    old.map(|output| {
                        // SAFETY: The insert retired the previous allocation, and
                        // this tree's guard keeps it alive for the returned borrow.
                        // Inline policies return an owned copy instead.
                        unsafe { P::output_to_value_ref(&output) }
                    })
                })
        };

        match result {
            Ok(old) => old,
            Err(e) => panic!(
                "Insert failed unexpectedly: {e:?}. This indicates a bug in the tree implementation."
            ),
        }
    }

    /// Insert a key-value pair using an explicit guard, returning the old value.
    ///
    /// Converts the snapshot from [`insert_with_guard`](Self::insert_with_guard)
    /// into an owned value, cloning only borrowed values. Primitive boxed values
    /// use the same allocation-free write-through update path.
    ///
    /// # Panics
    ///
    /// Panics on internal tree corruption.
    pub fn insert_value_with_guard(
        &self,
        key: &[u8],
        value: P::Value,
        guard: &LocalGuard<'_>,
    ) -> Option<P::Value>
    where
        P::Value: Clone,
    {
        self.insert_with_guard(key, value, guard)
            .map(ValueRef::into_owned)
    }

    // ========================================================================
    //  Remove Operations
    // ========================================================================

    /// Remove a key from the tree.
    ///
    /// Returns an owned clone of the removed value if the key existed.
    /// The internal EBR guard is released before returning, so the value
    /// is cloned to ensure independent ownership.
    ///
    /// # Example
    ///
    /// ```rust
    /// use masstree::MassTree15;
    ///
    /// let tree: MassTree15<u64> = MassTree15::new();
    /// tree.insert(b"key", 42);
    ///
    /// let removed = tree.remove(b"key").unwrap();
    /// assert_eq!(removed, Some(42));
    ///
    /// let not_found = tree.remove(b"key").unwrap();
    /// assert_eq!(not_found, None);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `RemoveError::RetryLimitExceeded` if the operation cannot
    /// complete after the retry limit (extremely rare, indicates contention).
    pub fn remove(&self, key: &[u8]) -> Result<Option<P::Value>, RemoveError>
    where
        P::Value: Clone,
    {
        let guard: LocalGuard<'_> = self.guard();
        let result: Option<ValueRef<'_, P::Value>> = self.remove_with_guard(key, &guard)?;

        Ok(result.map(ValueRef::into_owned))
    }

    /// Remove a key using an existing guard.
    ///
    /// The returned value is scoped to `guard`: a removed write-through value
    /// is an atomic copy, and a boxed value remains alive until the guard is
    /// dropped even though it has already been scheduled for reclamation.
    ///
    /// # Errors
    /// If fails to remove
    pub fn remove_with_guard<'g>(
        &self,
        key: &[u8],
        guard: &'g LocalGuard<'_>,
    ) -> Result<Option<ValueRef<'g, P::Value>>, RemoveError> {
        self.verify_guard(guard);
        #[cfg(feature = "tracing")]
        tracing::debug!(key_len = key.len(), "remove");

        let removed: Option<P::Output> = NodeCleaner::remove_concurrent_generic(self, key, guard)?;

        Ok(removed.map(|output: P::Output| {
            // SAFETY: `output` was removed under `guard`; the allocation is
            // retired but cannot be reclaimed until the guard is dropped.
            unsafe { P::output_to_value_ref(&output) }
        }))
    }

    /// Decrement the entry count after successful removal.
    ///
    /// Called by `finish_remove_generic` after updating the permutation.
    #[inline(always)]
    pub(crate) fn dec_count(&self) {
        self.count.decrement();
    }

    /// Assign a value to a slot in a locked leaf.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    #[expect(clippy::too_many_arguments, reason = "Internals")]
    fn assign_slot_generic(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        key: &Key<'_>,
        prepared: &PreparedValue<P>,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> *mut u8 {
        debug_assert!(
            leaf.prev(guard).is_null() || key.ikey() >= leaf.ikey_bound(),
            "INSERT INVARIANT VIOLATION: key {:016x} < leaf.ikey_bound {:016x}",
            key.ikey(),
            leaf.ikey_bound()
        );

        let ikey: u64 = key.ikey();

        // OCC-01: every new-slot write marks the node dirty unconditionally.
        // The previous modstate-conditional form took the normal (INSERT)
        // state for granted and published a new permutation with no version
        // change, so a reader that snapshotted before the insert could not
        // detect it. `mark_insert` is idempotent.
        lock.mark_insert();

        if leaf.modstate_relaxed() != MODSTATE_INSERT {
            leaf.set_modstate_relaxed(MODSTATE_INSERT);
        }

        leaf.set_ikey_relaxed(slot, ikey);
        leaf.store_prepared_value_relaxed(slot, prepared);

        if key.has_suffix() {
            leaf.set_keylenx_relaxed(slot, KSUF_KEYLENX);

            match pre_allocated {
                Some(buffer) => unsafe {
                    leaf.assign_ksuf_prealloc(slot, key.suffix(), guard, buffer)
                },

                None => unsafe { leaf.assign_ksuf(slot, key.suffix(), guard) },
            }
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
    /// # Example
    ///
    /// ```rust
    /// use masstree::{MassTree, Entry};
    ///
    /// let tree: MassTree<u64> = MassTree::new();
    /// let guard = tree.guard();
    ///
    /// // Insert if absent, returning the value
    /// let _value = tree.entry_with_guard(b"key", &guard).or_insert(42);
    ///
    /// // Increment counter, defaulting to 0
    /// let _value = tree.entry_with_guard(b"counter", &guard)
    ///     .and_modify(|v| *v + 1)
    ///     .or_insert(0);
    ///
    /// // Pattern matching
    /// match tree.entry_with_guard(b"maybe", &guard) {
    ///     Entry::Occupied(o) => {
    ///         let _val = o.remove();
    ///     }
    ///     Entry::Vacant(v) => {
    ///         v.insert(100);
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
    /// Panics on an internal tree invariant violation. A memory-allocation
    /// failure aborts the process, matching std collection behavior.
    ///
    /// # Example
    ///
    /// ```rust
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
        let mut tree = Self::with_allocator(A::default());
        tree.extend(iter);

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
    /// # Example
    ///
    /// ```rust
    /// use masstree::MassTree;
    ///
    /// let mut tree: MassTree<u64> = MassTree::new();
    /// tree.insert(b"existing", 0);
    ///
    /// // Can use byte literals directly, &[u8; N] implements AsRef<[u8]>
    /// tree.extend([
    ///     (b"key1" as &[u8], 1u64),
    ///     (b"key2" as &[u8], 2u64),
    /// ]);
    ///
    /// assert_eq!(tree.len(), 3);
    /// ```
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        const MAX_CHUNK_SIZE: usize = 16_384;
        let guard: LocalGuard<'_> = self.guard();
        let mut iter = iter.into_iter();
        let mut chunk_size = if iter.size_hint().0 >= MAX_CHUNK_SIZE {
            MAX_CHUNK_SIZE
        } else {
            1_024
        };
        let mut chunk = Vec::with_capacity(iter.size_hint().0.min(chunk_size));
        loop {
            for (key, value) in iter.by_ref().take(chunk_size) {
                chunk.push((key, Some(value)));
            }
            if chunk.is_empty() {
                break;
            }
            // Keep keys stationary in the chunk while the entries are sorted.
            // Snapshot AsRef once per input, so even a custom wrapper that
            // selects different slices on later calls cannot change routing.
            let mut batch: Vec<BatchEntry<P, &[u8]>> = chunk
                .iter_mut()
                .map(|(key, value)| {
                    BatchEntry::from_key(
                        K::as_ref(&*key),
                        value.take().expect("chunk value already consumed"),
                    )
                })
                .collect();
            let _ = self.sort_and_insert_batch::<false, _>(&mut batch, &guard);
            drop(batch);
            chunk.clear();
            chunk_size = (chunk_size * 2).min(MAX_CHUNK_SIZE);
        }
    }
}
