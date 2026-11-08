use masstree::{BoxPolicy, LeafNode15, LeafPolicy, NodeVersion, Permuter};

fn main() {
    println!("== NodeVersion: INSERTING window is invisible to has_changed ==");
    {
        let v = NodeVersion::new(true);
        let old = v.stable();               // reader snapshot, clean
        let mut g = v.lock();               // writer
        g.mark_insert();                    // dirty (INSERTING) while payload is mutated
        let changed = v.has_changed(old);   // reader validation
        println!("has_changed while writer holds INSERTING = {changed} (old={old:#010x}, cur={:#010x})", v.value());
        assert!(changed, "has_changed must observe the INSERTING dirty window");
        drop(g);
        println!("after unlock has_changed = {} (cur={:#010x})", v.has_changed(old), v.value());
    }

    println!("== NodeVersion: snapshot taken while LOCK is set, then INSERTING ==");
    {
        let v = NodeVersion::new(false);
        let mut g = v.lock();
        let old = v.stable(); // clean: LOCK is not part of DIRTY_MASK
        g.mark_insert();
        println!("old={old:#010x} cur={:#010x} has_changed={}", v.value(), v.has_changed(old));
        assert!(v.has_changed(old), "mark_insert must make the snapshot stale");
        drop(g);
    }

    println!("== LeafNode15: ValueSlot becomes empty while version check says Valid ==");
    {
        let leaf: Box<LeafNode15<BoxPolicy<u64>>> = LeafNode15::new();
        leaf.set_ikey(0, 0x4141414141414141);
        let value = <BoxPolicy<u64> as LeafPolicy>::into_output(0xdead_beef_u64);
        leaf.store_value(0, &value);
        leaf.set_keylenx(0, 8);
        leaf.set_permutation(Permuter::make_sorted(1));

        // Reader: OCC snapshot then "search_leaf_multi_layer" would return ValueSlot(0)
        let old = leaf.version().stable();
        let perm = leaf.permutation();
        let slot = perm.get(0);
        assert_eq!(leaf.ikey(slot), 0x4141414141414141);
        assert_eq!(leaf.keylenx(slot), 8);
        assert!(leaf.load_value(slot).is_some(), "ValueSlot precondition");

        // Writer: insert path's in-place mutation before unlock
        let mut g = leaf.version().lock();
        g.mark_insert();
        let _ = leaf.take_value_for_layer(slot); // same call handle_suffix_conflict makes

        // Reader: load_value(slot) then validate
        let out = leaf.load_value(slot);
        let changed = leaf.version().has_changed(old);
        println!("reader output = {out:?}, has_changed = {changed}");
        assert!(out.is_none());
        assert!(changed, "get_guarded must reject a snapshot whose value was cleared");
        drop(g);
    }

}
