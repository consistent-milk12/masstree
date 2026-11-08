use super::*;
use crate::policy::InlinePolicy;
use crate::{BoxPolicy, MassTreeGeneric, SeizeAllocator};

/// Force slot reuse at the boundary between OCC validation and callback
/// delivery. This models a writer at that exact point without timing sleeps.
struct ReuseBeforeVisit<M, V> {
    mutate: M,
    inner: V,
}

impl<P: LeafPolicy, M: FnMut(), V: SlotVisitor<P>> SlotVisitor<P> for ReuseBeforeVisit<M, V> {
    type Emitter = V::Emitter;

    fn visit(&mut self, snapshot: <Self::Emitter as ScanEmitter<P>>::Snapshot, key: &[u8]) -> bool {
        (self.mutate)();
        self.inner.visit(snapshot, key)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "One deterministic capture-to-delivery interleaving"
)]
fn check_slot_reuse<P: LeafPolicy<Value = u64>>(reverse: bool, reference: bool, layer: bool) {
    let tree = MassTreeGeneric::<P, SeizeAllocator<P>>::with_allocator(SeizeAllocator::new());
    let guard = tree.guard();
    let make_key = |value: u64| {
        let mut key = value.to_be_bytes().to_vec();
        if layer {
            key.push(b'x');
        }
        key
    };
    for key in 1u64..=15 {
        tree.insert_with_guard(&make_key(key), key, &guard);
    }
    let leaf_ptr = tree
        .load_root_ptr_generic(&guard)
        .cast::<LeafNode15<P>>()
        .cast_mut();
    // SAFETY: Fifteen keys fit the initial leaf; this tree's guard protects it.
    let leaf = unsafe { &*leaf_ptr };
    let version = leaf.version().stable();
    let perm = leaf.permutation();
    let mut cursor = CursorKey::empty();
    let mut layers = LayerStack::new();
    let mut ctx = BatchCtx {
        leaf,
        perm,
        perm_size: 15,
        cached_version: version,
        ki: 5,
        root: leaf_ptr.cast(),
        leaf_ptr,
        cursor_key: &mut cursor,
        layer_stack: &mut layers,
        down_follow_suffix: false,
        down_fresh_subtree: false,
    };
    let mutate = || {
        if layer {
            let mut new_key = 6u64.to_be_bytes().to_vec();
            new_key.push(b'y');
            tree.insert_with_guard(&new_key, 16, &guard);
            assert!(leaf.is_layer(perm.get(5)), "must convert the observed slot");
        } else {
            assert_eq!(
                *tree
                    .remove_with_guard(&make_key(6), &guard)
                    .unwrap()
                    .unwrap(),
                6
            );
            tree.insert_with_guard(&make_key(16), 16, &guard);
            assert_eq!(
                leaf.ikey_relaxed(perm.get(5)),
                16,
                "must reuse the observed slot"
            );
        }
    };
    let mut seen = None;
    let mut record = |key: &[u8], value: u64| {
        assert_eq!(key, make_key(6));
        seen = Some((
            u64::from_be_bytes(key.get(..8).unwrap().try_into().unwrap()),
            value,
        ));
        false
    };
    let mut count = 0;
    if reference {
        let mut visitor = ReuseBeforeVisit {
            mutate,
            inner: RefSlotVisitor(|key: &[u8], value: &u64| record(key, *value)),
        };
        run_batch(reverse, &mut ctx, &mut visitor, &mut count);
    } else {
        let mut visitor = ReuseBeforeVisit {
            mutate,
            inner: CopySlotVisitor(|key: &[u8], output: P::Output| {
                // SAFETY: The test holds this tree's guard over the complete scan.
                record(key, unsafe { P::clone_value_from_output(&output) })
            }),
        };
        run_batch(reverse, &mut ctx, &mut visitor, &mut count);
    }
    assert_eq!(count, 1);
    assert_eq!(
        seen,
        Some((6, 6)),
        "key and value must come from the same validated slot"
    );
}

fn run_batch<P: LeafPolicy, V: SlotVisitor<P>>(
    reverse: bool,
    ctx: &mut BatchCtx<'_, P>,
    visitor: &mut V,
    count: &mut usize,
) {
    if reverse {
        let result = process_batch_keyed::<Backward, _, _>(
            ctx,
            &RangeBound::Unbounded,
            CursorBound::Inside,
            visitor,
            count,
            &mut ReverseFlags::new(),
        );
        assert!(matches!(result, LeafBatchResultBack::Stopped));
    } else {
        let result = process_batch_keyed::<Forward, _, _>(
            ctx,
            &RangeBound::Unbounded,
            CursorBound::Inside,
            visitor,
            count,
            &mut (),
        );
        assert!(matches!(result, LeafBatchResult::Stopped));
    }
}

#[test]
fn keyed_batch_snapshot_survives_slot_reuse() {
    for reverse in [false, true] {
        check_slot_reuse::<InlinePolicy<u64>>(reverse, false, false);
        check_slot_reuse::<BoxPolicy<u64>>(reverse, false, false);
        check_slot_reuse::<BoxPolicy<u64>>(reverse, true, false);
    }
}

#[test]
fn keyed_batch_snapshot_survives_layer_conversion() {
    for reverse in [false, true] {
        check_slot_reuse::<InlinePolicy<u64>>(reverse, false, true);
        check_slot_reuse::<BoxPolicy<u64>>(reverse, false, true);
        check_slot_reuse::<BoxPolicy<u64>>(reverse, true, true);
    }
}
