// Append these modules at the end of `src/policy/tests.rs` in a temporary
// checkout. The first three also have public-API equivalents in
// `failures/leaf-values/public/src/bin/`.
//
//   cargo test --lib audit_ -- --nocapture
//   cargo +nightly miri test --lib audit_padded_write_through
//   cargo +nightly miri test --lib audit_valueptr_lifetime
//   cargo +nightly miri test --lib audit_valueptr_write_through_race
//   cargo +nightly miri test --lib audit_batch_write_through_leak

#[cfg(test)]
mod audit_padded_write_through {
    use crate::policy::{BoxPolicy, LeafPolicy};

    #[repr(align(8))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Padded {
        a: u32,
    }

    #[test]
    fn padded_type_is_not_write_through() {
        assert!(!BoxPolicy::<Padded>::can_write_through());
        let p: crate::policy::ValuePtr<Padded> = BoxPolicy::<Padded>::into_output(Padded { a: 7 });
        // Non-write-through: clone through the pointer rather than an atomic.
        let copy: Padded = BoxPolicy::<Padded>::clone_value_from_output(&p);
        assert_eq!(copy, Padded { a: 7 });
    }
}

#[cfg(test)]
mod audit_valueptr_lifetime {
    use crate::tree::MassTree15;

    #[test]
    fn valueptr_outlives_guard_and_tree() {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(b"k", 42u64);
        let leaked = {
            let guard = tree.guard();
            tree.get_with_guard(b"k", &guard).unwrap()
        };
        drop(tree);
        assert_eq!(*leaked, 42u64);
    }
}

#[cfg(test)]
mod audit_valueptr_write_through_race {
    use crate::tree::MassTree15;

    #[test]
    fn valueptr_deref_races_with_write_through_insert() {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert(b"k", 1u64);
        let ptr = {
            let guard = tree.guard();
            tree.get_with_guard(b"k", &guard).unwrap()
        };

        let writer = std::thread::spawn(move || {
            for v in 0..64u64 {
                let _ = tree.insert(b"k", v);
            }
        });

        let mut sum = 0u64;
        for _ in 0..64 {
            sum = sum.wrapping_add(*ptr);
        }
        writer.join().unwrap();
        assert!(sum > 0);
    }
}

#[cfg(test)]
mod audit_batch_write_through_leak {
    use crate::tree::MassTree15;

    #[test]
    fn batch_with_guard_leaks_old_value_boxes() {
        let tree: MassTree15<u64> = MassTree15::new();
        let guard = tree.guard();
        let _ = tree.insert_batch_with_guard(vec![(b"k".to_vec(), 1u64)], &guard);
        let result = tree.insert_batch_with_guard(vec![(b"k".to_vec(), 2u64)], &guard);
        assert_eq!(result.old_values.len(), 1);
        drop(result);
        drop(guard);
        drop(tree);
    }
}
