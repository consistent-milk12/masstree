# Leaf/values reproducers

## Public Miri binaries

These compile against the repository and demonstrate defects that do not need
crate-private access:

- `valueptr_escape` — `ValuePtr` outlives its guard/tree (use-after-free).
- `valueptr_race` — non-atomic `ValuePtr::deref` races a write-through insert.
- `padded_write_through` — atomic write-through reads uninitialized padding.
- `move_value_same_leaf` — safe `move_value_to` duplicates ownership.
- `batch_write_through_leak` — old batch values are re-boxed and never retired.

Run with the nightly Miri component:

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_escape
cargo +nightly miri run --bin padded_write_through
cargo +nightly miri run --bin move_value_same_leaf
```

The expected failure is a Miri `Undefined Behavior` or `memory leaked` report.

## Private unit tests

The files in `private-tests/` are exact snippets to append to a temporary
checkout of this repository. They need `pub(crate)` access and therefore are
not integration tests:

- `leaf15_unit_tests.add.rs`
  - `audit_new_insert_does_not_bump_version` (OCC-01)
  - `audit_value_cleared_under_inserting_is_not_detected_by_has_changed` (OCC-02)
  - `audit_move_value_to_same_leaf_double_owns` (SAFE-04)
- `policy_tests.add.rs`
  - padded write-through UB, `ValuePtr` lifetime/race, batch leak (Miri)
- `remove_unit_tests.add.rs`
  - staged same-ikey wrong-key removal (REM-01)

Suggested procedure:

```bash
cp -a . /tmp/masstree-private-audit
cd /tmp/masstree-private-audit
cat failures/leaf-values/private-tests/leaf15_unit_tests.add.rs >> src/leaf15/unit_tests.rs
cat failures/leaf-values/private-tests/policy_tests.add.rs >> src/policy/tests.rs
cat failures/leaf-values/private-tests/remove_unit_tests.add.rs >> src/tree/remove/unit_tests.rs
cargo test --lib audit_ -- --nocapture
```

Run the Miri-labelled ones individually:

```bash
cargo +nightly miri test --lib audit_padded_write_through
cargo +nightly miri test --lib audit_move_value_to_same_leaf_double_owns
```

The snippets are additive; none of them modify production code in the
repository.
