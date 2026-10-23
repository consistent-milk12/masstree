// Miri: safe `move_value_to` copies a Box pointer without clearing the source.
// Expected: double free when the leaf is dropped.
use masstree::{
    BoxPolicy, LeafNode15, LeafPolicy, Permuter15,
};

fn main() {
    let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new();
    let value = <BoxPolicy<u64> as LeafPolicy>::into_output(7u64);
    leaf.store_value(0, &value);
    leaf.set_permutation(Permuter15::make_sorted(1));
    leaf.move_value_to(&leaf, 0, 1);
    drop(leaf);
}
