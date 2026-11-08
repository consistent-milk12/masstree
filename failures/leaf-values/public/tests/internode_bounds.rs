//! Control for the SAFE-04 follow-up: ordinary array indexing still checks bounds
//! in release builds. These calls panic; they do not perform unchecked OOB writes.
use masstree::InternodeNode;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

#[test]
fn indexed_accessors_panic() {
    let mut node = Box::<InternodeNode>::new_uninit();
    // SAFETY: Exclusive ownership of correctly aligned uninitialized storage.
    let node = unsafe {
        InternodeNode::init_at(node.as_mut_ptr(), 0);
        node.assume_init()
    };
    assert!(catch_unwind(AssertUnwindSafe(|| node.set_ikey(15, 0))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| node.set_child(16, ptr::null_mut()))).is_err());
}

// insert_key_and_child and shift_from are now unsafe crate-internal helpers.
// Their former public trait forwarders were removed rather than retaining a
// safe entry point that could bypass the pointer-publication preconditions.
