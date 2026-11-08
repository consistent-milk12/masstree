//! Guarded traversal primitives shared by point operations, scans and GC.

use seize::LocalGuard;

use crate::{
    internode::InternodeNode, leaf15::LeafNode15, nodeversion::NodeVersion, policy::LeafPolicy,
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

#[cfg(test)]
mod unit_tests;
