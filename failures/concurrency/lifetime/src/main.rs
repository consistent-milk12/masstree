use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    let escaped = {
        let guard = tree.guard();
        tree.insert_with_guard(b"key", "value".to_string(), &guard);
        tree.get_with_guard(b"key", &guard).unwrap()
    };
    drop(tree);
    println!("escaped value: {}", &*escaped);
}
