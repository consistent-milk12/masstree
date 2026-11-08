//! Guarded traversal primitives shared by point operations, scans and GC.

use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::{
    Linker,
    hints::{likely, unlikely},
    internode::InternodeNode,
    leaf15::LeafNode15,
    nodeversion::NodeVersion,
    policy::LeafPolicy,
    prefetch::prefetch_read,
};

#[cfg(test)]
std::thread_local! {
    pub(super) static ROOT_CLIMB_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Resolve a layer-slot target or cached ancestor to its current root.
///
/// Returns `None` for a null pointer or an excessive parent chain. Callers can
/// retry a transient traversal without accepting a partial climb as the root.
///
/// # Safety
/// `node` must be null or a live leaf/internode protected by this tree's guard.
/// Every parent pointer must belong to the same layer and collector.
#[inline]
pub(super) unsafe fn resolve_layer_root<P: LeafPolicy>(
    mut node: *const u8,
    guard: &LocalGuard<'_>,
) -> Option<*const u8> {
    for _ in 0..64 {
        if node.is_null() {
            return None;
        }
        // SAFETY: The caller protects this node; node_parent protects each
        // subsequently loaded ancestor for the same guard lifetime.
        let parent = unsafe { node_parent::<P>(node, guard) };
        if parent.is_null() {
            return Some(node);
        }
        #[cfg(test)]
        ROOT_CLIMB_COUNT.with(|count| count.set(count.get() + 1));
        node = parent.cast_const();
    }
    None
}

/// Load the guarded parent of either node kind.
///
/// # Safety
/// `node` must be a non-null, aligned node protected by `guard`.
#[inline(always)]
pub(super) unsafe fn node_parent<P: LeafPolicy>(
    node: *const u8,
    guard: &LocalGuard<'_>,
) -> *mut u8 {
    // SAFETY: Both node layouts begin with NodeVersion and the caller guarantees
    // the allocation's kind/alignment/lifetime. The leaf-kind bit is immutable.
    #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
    let version = unsafe { &*node.cast::<NodeVersion>() };
    if version.is_leaf() {
        // SAFETY: The kind was checked above; the parent load is EBR-protected.
        unsafe { &*node.cast::<LeafNode15<P>>() }.parent(guard)
    } else {
        // SAFETY: The kind was checked above; the parent load is EBR-protected.
        unsafe { &*node.cast::<InternodeNode>() }.parent(guard)
    }
}

/// Resolve a validated layer slot and opportunistically refresh its root.
///
/// # Safety
/// `raw` must be a live layer node loaded from `parent` under `guard`. The
/// caller must have validated the slot's OCC snapshot and must hold no node
/// locks. Refresh takes only a nonblocking parent-leaf lock; it never acquires
/// a sublayer lock, preserving the top-down order used by layer GC.
#[inline]
pub(super) unsafe fn resolve_layer_slot<P: LeafPolicy>(
    parent: &LeafNode15<P>,
    slot: usize,
    ikey: u64,
    raw: *mut u8,
    guard: &LocalGuard<'_>,
) -> *mut u8 {
    // SAFETY: The caller supplies a node protected by the same collector guard.
    let Some(root) = (unsafe { resolve_layer_root::<P>(raw, guard) }) else {
        return raw;
    };
    if root == raw.cast_const() {
        return raw;
    }
    if let Some(mut lock) = parent.version().try_lock() {
        let perm = parent.permutation();
        let active = (0..perm.size()).any(|i| perm.get(i) == slot);
        if active
            && !parent.version().is_deleted()
            && !parent.deleted_layer()
            && parent.ikey_relaxed(slot) == ikey
            && parent.keylenx_relaxed(slot) >= crate::leaf15::LAYER_KEYLENX
            && parent.load_layer_raw(slot) == raw
        {
            lock.mark_insert();
            // SAFETY: The parent slot still names this same live sublayer,
            // and `root` is its guarded ancestor. We hold the parent lock and
            // publish the replacement before Release unlock. A concurrent
            // promotion can make root stale again, but keeps it in this layer.
            // Sublayer nodes are retired as a whole only after GC clears this
            // owning slot; chain coalesce operates on the top layer only.
            unsafe { parent.store_layer(slot, root.cast_mut()) };
        }
    }
    root.cast_mut()
}

/// Double-buffer traversal from root to leaf. A null result means the
/// cached layer root was retired; callers must restart from the tree root.
/// The second result is the resolved root for subsequent guarded retries.
///
/// # Safety
/// `start` must be null or a tree node protected by `guard`. All linked
/// nodes must belong to the same layer, leaf policy and collector.
#[expect(clippy::indexing_slicing, reason = "sense is always 0 or 1")]
#[expect(
    clippy::too_many_lines,
    reason = "hot-path tree traversal, logically cohesive and performance-critical"
)]
pub(super) unsafe fn reach_leaf<P: LeafPolicy>(
    start: *const u8,
    target_ikey: u64,
    guard: &LocalGuard<'_>,
) -> (*mut LeafNode15<P>, *const u8) {
    use crate::ksearch::upper_bound_internode_generic;

    if start.is_null() {
        return (StdPtr::null_mut(), start);
    }
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

            if version.is_deleted() {
                return (StdPtr::null_mut(), start);
            }
            if version.is_root() {
                break;
            }

            // SAFETY: This node and its ancestors are protected by guard.
            let Some(root) = (unsafe { resolve_layer_root::<P>(n[sense], guard) }) else {
                return (StdPtr::null_mut(), start);
            };
            if root == n[sense] {
                // A detached non-root cannot supply a valid descent.
                return (StdPtr::null_mut(), start);
            }
            n[sense] = root;
        }

        let resolved_root = n[sense];
        loop {
            // SAFETY: Both node kinds start with NodeVersion; each pointer
            // was loaded under guard and remains allocated for its lifetime.
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(n[sense].cast::<NodeVersion>()) };

            v[sense] = match version.try_stable() {
                Some(v) => v,
                None => continue 'retry, // Node is locked, restart from root
            };

            if version.is_deleted() {
                return (StdPtr::null_mut(), start);
            }

            // Check if we've reached a leaf
            if version.is_leaf() {
                return (n[sense].cast_mut().cast::<LeafNode15<P>>(), resolved_root);
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

/// Advance to correct leaf via B-link (generic version).
/// Returns `true` when callers must re-descend rather than use this leaf.
///
/// # Safety
/// `leaf_ptr` must be null or a leaf protected by `guard`. Its sibling
/// chain must belong to the same layer, leaf policy and collector.
pub(super) unsafe fn advance_by_bound<P: LeafPolicy>(
    mut leaf_ptr: *mut LeafNode15<P>,
    key_ikey: u64,
    guard: &LocalGuard<'_>,
) -> (*mut LeafNode15<P>, bool) {
    if leaf_ptr.is_null() {
        return (leaf_ptr, true);
    }
    let mut hops: usize = 0;

    // SAFETY: leaf_ptr is valid, protected by guard
    let mut leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

    let version: &NodeVersion = leaf.version();

    if version.is_splitting() {
        let _ = version.stable();
    }

    loop {
        if hops >= 128 {
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

#[cfg(test)]
mod unit_tests;
