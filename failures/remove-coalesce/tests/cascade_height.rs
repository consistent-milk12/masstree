//! GC-04: collapsing a non-root internode must preserve the B+tree level.
use masstree::{InternodeNode, MassTree, NodeVersion};

#[test]
fn cascade_preserves_child_height() {
    let tree = MassTree::<u64>::new();
    for key in 0u64..300 {
        tree.insert(&key.to_be_bytes(), key);
    }
    // Sequential splits put 135..149 in the first leaf of the right internode.
    // Leave that leaf live while emptying all its siblings.
    for key in 150u64..300 {
        assert_eq!(tree.remove(&key.to_be_bytes()).unwrap(), Some(key));
    }
    let mut guard = tree.guard();
    tree.maintenance(&mut guard);
    let root = tree.debug_root_ptr(&guard);
    // SAFETY: The guard belongs to this live tree, and inspection is quiescent.
    // NodeVersion is at offset zero in both node layouts.
    assert!(!unsafe { &*root.cast::<NodeVersion>() }.is_leaf());
    // SAFETY: The version above establishes the node type; guard keeps it live.
    let root = unsafe { &*root.cast::<InternodeNode>() };
    assert_eq!(root.height(), 1, "probe requires a height-1 root");
    let child = root.child(1, &guard);
    assert!(!child.is_null());
    // SAFETY: This is a published child protected by the same guard.
    assert!(
        !unsafe { &*child.cast::<NodeVersion>() }.is_leaf(),
        "height-1 root now directly references a leaf after cascade"
    );
}
