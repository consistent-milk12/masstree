//! Range positioning over the guarded traversal kernel used by point operations.

use seize::LocalGuard;

use crate::tree::traversal::{advance_by_bound, reach_leaf};
use crate::{leaf15::LeafNode15, policy::LeafPolicy};

use super::cursor_key::CursorKey;

/// Descend from a layer root and follow split siblings to the cursor's leaf.
/// Cache the returned root so a subsequent retry avoids parent climbs.
#[inline]
pub fn reach_leaf_for_scan<P: LeafPolicy>(
    start: *const u8,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> (*mut LeafNode15<P>, *const u8) {
    let target = cursor_key.current_ikey();
    let mut root = start;
    // Bound retries for a corrupted or continually changing sibling chain.
    for _ in 0..64 {
        // SAFETY: The scan loaded start from the tree under this guard; each
        // returned root and leaf remains protected for the same guard lifetime.
        let (leaf, resolved) = unsafe { reach_leaf::<P>(root, target, guard) };
        root = resolved;
        if leaf.is_null() {
            return (leaf, root);
        }
        // SAFETY: The shared descent returned a guard-protected leaf.
        let (leaf, retry) = unsafe { advance_by_bound(leaf, target, guard) };
        if retry {
            continue;
        }
        // SAFETY: A successful advance returns a non-null guarded leaf.
        let candidate = unsafe { &*leaf };
        if !candidate.prev(guard).is_null() && target < candidate.ikey_bound() {
            continue;
        }
        return (leaf, root);
    }
    (std::ptr::null_mut(), root)
}
