//! Debug test to find where the hang occurs.

fn main() {
    use masstree::MassTree24;
    
    let tree: MassTree24<u64> = MassTree24::new();
    
    eprintln!("Inserting 50 keys...");
    for i in 0_u64..50 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }
    eprintln!("Inserted 50 keys, len = {}", tree.len());
    
    eprintln!("Removing keys 0-24...");
    for i in 0_u64..25 {
        eprintln!("  Removing key {}", i);
        tree.remove(&i.to_be_bytes()).unwrap();
        eprintln!("  Removed key {}, len = {}", i, tree.len());
    }
    
    eprintln!("Verifying remaining keys...");
    for i in 25_u64..50 {
        eprintln!("  Getting key {}", i);
        let result = tree.get(&i.to_be_bytes());
        eprintln!("  Key {} = {:?}", i, result);
    }
    
    eprintln!("Done!");
}
