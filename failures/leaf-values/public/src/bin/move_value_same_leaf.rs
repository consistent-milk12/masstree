// SAFE-04: `move_value_to` is now an unsafe API.
//
// The caller must clear/retire the source slot. This program demonstrates the
// required contract: clear the source after the move so the Box is owned by one
// slot only.
use masstree::{BoxPolicy, LeafNode15, LeafPolicy, Permuter15};

fn main() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new();
    let value = <BoxPolicy<u64> as LeafPolicy>::into_output(7u64);
    // SAFETY: this leaf is exclusively owned and `value` is stored once.
    unsafe { leaf.store_value(0, &value) };
    leaf.set_permutation(Permuter15::make_sorted(1));

    // SAFETY: source slot 0 is cleared immediately after the move, so only
    // slot 1 retains the Box pointer.
    unsafe {
        leaf.move_value_to(&leaf, 0, 1);
    }
    leaf.clear_value(0);

    drop(leaf);
    println!("value moved without double free");
}
