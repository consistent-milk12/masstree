//! TOOL-14: the printer must climb from a promoted layer's original left leaf.
use masstree::MassTree15;

#[test]
fn printer_includes_promoted_layer_values() {
    let tree = MassTree15::<String>::new();
    for index in 0u64..40 {
        let mut key = b"prefix00".to_vec();
        key.extend_from_slice(&index.to_be_bytes());
        tree.insert(&key, format!("VALUE_{index:03}_END"));
    }
    let printed = tree.print_tree();
    assert!(printed.contains("VALUE_000_END"));
    assert!(
        printed.contains("VALUE_039_END"),
        "printer omitted values in the promoted layer's right siblings"
    );
}
