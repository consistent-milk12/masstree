// Miri: `insert_batch_with_guard` re-boxes old write-through values and never
// retires them. Expected: one leaked allocation per updated entry.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    let _ = tree.insert_batch_with_guard(vec![(b"k".to_vec(), 1u64)], &guard);
    let result = tree.insert_batch_with_guard(vec![(b"k".to_vec(), 2u64)], &guard);
    assert_eq!(result.old_values.len(), 1);
    drop(result);
    drop(guard);
    drop(tree);
}
