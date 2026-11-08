//! GC-04: a public-API remove/coalesce/insert sequence eventually hangs.
//! Build first, then run with timeout. Also usable under Miri.
use masstree::MassTree;

fn main() {
    let tree = MassTree::<u64>::new();
    for key in 0u64..300 {
        tree.insert(&(key * 1000).to_be_bytes(), key);
    }
    for key in 150u64..300 {
        assert_eq!(tree.remove(&(key * 1000).to_be_bytes()).unwrap(), Some(key));
    }
    let mut guard = tree.guard();
    tree.maintenance(&mut guard);
    eprintln!("coalesce finished; reinserting beyond surviving leaf");
    for key in 300u64..1000 {
        tree.insert_with_guard(&(key * 1000).to_be_bytes(), key, &guard);
        if key % 100 == 0 {
            eprintln!("inserted {key}");
        }
    }
    assert_eq!(tree.len(), 850);
    for key in (0u64..150).chain(300..1000) {
        assert_eq!(tree.get_with_guard(&(key * 1000).to_be_bytes(), &guard).map(|v| *v), Some(key));
    }
    eprintln!("all surviving and reinserted keys verified");
    // This full leaf moved during the height-1 internode's split. Its real
    // parent pointer was not updated because it was cast as an internode.
    eprintln!("splitting a moved leaf at 450001");
    tree.insert_with_guard(&450001u64.to_be_bytes(), 450001, &guard);
    eprintln!("moved leaf split completed");
}
