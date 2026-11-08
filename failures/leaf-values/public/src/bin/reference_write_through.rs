//! SAFE-05 follow-up: Copy + no padding is still insufficient for pointer values.
use masstree::{BoxPolicy, LeafPolicy, MassTree15};

static VALUE: u64 = 123;

fn main() {
    assert!(<BoxPolicy<&'static u64> as LeafPolicy>::CAN_WRITE_THROUGH);
    let tree = MassTree15::<&'static u64>::new();
    tree.insert(b"key", &VALUE);
    let guard = tree.guard();
    let value = tree.get_ref(b"key", &guard).unwrap();
    assert_eq!(**value, VALUE);
}
