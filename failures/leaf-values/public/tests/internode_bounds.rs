//! Control for the SAFE-04 follow-up: ordinary array indexing still checks bounds
//! in release builds. These calls panic; they do not perform unchecked OOB writes.
use masstree::InternodeNode;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

#[test]
fn indexed_accessors_panic() {
    let node = InternodeNode::new(0);
    assert!(catch_unwind(AssertUnwindSafe(|| node.set_ikey(15, 0))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| node.set_child(16, ptr::null_mut()))).is_err());
}

#[test]
fn full_insert_panics() {
    let node = InternodeNode::new(0);
    for index in 0..15 {
        node.insert_key_and_child(index, index as u64, ptr::null_mut());
    }
    assert!(catch_unwind(AssertUnwindSafe(|| {
        node.insert_key_and_child(0, 0, ptr::null_mut());
    }))
    .is_err());
}
