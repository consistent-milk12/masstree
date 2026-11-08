//! ALLOC-02: a panicking user value `Drop` must not abort or corrupt the tree.
//!
//! Updating a key retires the old box, and `MassTree15::insert` clones the old
//! value for the caller. Both copies can panic when dropped. The guarantee
//! under test is that reclamation and tree teardown contain those panics and
//! the allocator/reclaimer state remains usable.
use masstree::MassTree15;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone)]
struct Boom;

impl Drop for Boom {
    fn drop(&mut self) {
        panic!("boom value drop");
    }
}

#[test]
fn panicking_value_drop() {
    let tree: MassTree15<Boom> = MassTree15::new();

    // The returned old value (and the retired box) panic when dropped. Catch
    // those caller-side panics; they are legal Rust, not tree corruption.
    let r1 = catch_unwind(AssertUnwindSafe(|| {
        tree.insert(b"k", Boom);
    }));
    let r2 = catch_unwind(AssertUnwindSafe(|| {
        tree.insert(b"k", Boom);
    }));
    assert!(r1.is_ok(), "first insert must not panic");
    assert!(r2.is_err(), "dropping the returned old Boom panics");

    // Tree teardown must complete despite the panicking value destructor.
    let dropped = catch_unwind(AssertUnwindSafe(|| drop(tree)));
    assert!(dropped.is_ok(), "tree drop must not unwind or abort");
}
