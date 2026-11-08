//! GC-05: one contended attempt should defer cleanup to a later call.
use masstree::{BoxPolicy, InternodeNode, LeafNode15, MassTree15, NodeVersion};

#[test]
fn contention_does_not_spend_all_retries_in_one_call() {
    let tree = MassTree15::<u64>::new();
    for key in 0u64..20 {
        tree.insert(&key.to_be_bytes(), key);
    }
    for key in 15u64..20 {
        assert_eq!(tree.remove(&key.to_be_bytes()).unwrap(), Some(key));
    }
    let guard = tree.guard();
    let root = tree.debug_root_ptr(&guard);
    // SAFETY: Quiescent inspection of this tree under its guard; both layouts
    // place NodeVersion at offset zero.
    assert!(!unsafe { &*root.cast::<NodeVersion>() }.is_leaf());
    // SAFETY: Root version identifies an internode, kept live by guard.
    let root = unsafe { &*root.cast::<InternodeNode>() };
    let child = root.child(1, &guard);
    assert!(!child.is_null());
    // SAFETY: Height zero means children are leaves. This tree uses BoxPolicy
    // and the guard protects this child. The lock models ordinary writer contention.
    assert_eq!(root.height(), 0);
    let leaf = unsafe { &*child.cast::<LeafNode15<BoxPolicy<u64>>>() };
    let lock = leaf.version().lock();
    let processed = tree.process_coalesce(&guard);
    eprintln!(
        "processed={processed} deferred={} abandoned={}",
        tree.coalesce_deferred(),
        tree.coalesce_abandoned()
    );
    drop(lock);
    assert_eq!(tree.coalesce_abandoned(), 0, "one brief conflict exhausted the entire retry budget");
    assert_eq!(tree.coalesce_deferred(), 1);
    tree.process_coalesce(&guard);
    assert_eq!(tree.pending_coalesce(), 0);
}
