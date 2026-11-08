//! Probe: does a plain insert bump the leaf's OCC version counter?
//! (Context for remove's post-lock validation.)

use masstree::{MassTree15Inline, NodeVersion};

#[test]
fn insert_version_bump_probe() {
    let tree = MassTree15Inline::<u64>::new();
    let g = tree.guard();
    // SAFETY: Quiescent inspection of this tree under its guard; the root node
    // is live for the guard's lifetime.
    let p: *mut NodeVersion = unsafe { tree.debug_root_ptr(&g) }.cast();
    let v = |p: *mut NodeVersion| unsafe { (*p).value() };
    println!("initial: {:#x}", v(p));
    tree.insert_with_guard(b"aaaa", 1, &g);
    println!("after first insert: {:#x}", v(p));
    tree.insert_with_guard(b"bbbb", 2, &g);
    println!("after second insert: {:#x}", v(p));
    tree.insert_with_guard(b"cccc", 3, &g);
    println!("after third insert: {:#x}", v(p));
    tree.insert_with_guard(b"aaaa", 9, &g);
    println!("after update: {:#x}", v(p));
    let _ = tree.remove_with_guard(b"bbbb", &g);
    println!("after remove: {:#x}", v(p));
    // Second leaf / split check.
    for i in 0..40u64 {
        tree.insert_with_guard(&i.to_be_bytes(), i, &g);
    }
    println!("after 40 inserts (root may be internode): {:#x}", v(p));
}
