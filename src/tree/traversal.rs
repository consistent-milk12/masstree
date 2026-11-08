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
