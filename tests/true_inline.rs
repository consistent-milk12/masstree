//! Memory leak test.

use masstree::MassTree15Inline;
use seize::LocalGuard;

#[test]
#[expect(clippy::indexing_slicing, clippy::unwrap_used)]
fn masstree15_inline_no_leak_with_long_keys() {
    fn make_key(i: u64) -> [u8; 64] {
        let mut key = [0u8; 64];
        key[0..8].copy_from_slice(&i.to_be_bytes());

        // Fill rest with pattern to ensure suffix is used
        (8..64).for_each(|j| {
            key[j] = ((i + (j as u64)) & 0xFF) as u8;
        });

        key
    }

    // Create tree, insert enough keys to trigger external suffix bags
    // Inline capacity is 256 bytes, ~56 bytes per suffix, so ~5 keys fill it
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard: LocalGuard<'_> = tree.guard();

    for i in 0..20 {
        let key = make_key(i);
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    drop(guard);
    drop(tree); // Should free external suffix bags via `Drop`

    // If `Drop` is missing memory tools (miri, asan etc) will detect the leak.
}
