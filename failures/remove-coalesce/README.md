# Remove/coalesce reproducers

`nonempty_alloc` is a passing PERF-17 regression. A real counting allocator
checks eight nonempty removals for boxed and inline values with 32- and
256-byte keys: the former Vec routes allocated [8, 8, 32, 32] times, and stack
routes allocate [0, 0, 0, 0]. Reclamation bookkeeping is primed before counting. It also
compares surviving keys/values with BTreeMap and empties the remaining deep
layers to exercise owned queued routes. Native and Miri both pass.

```bash
CARGO_TARGET_DIR=/tmp/masstree-target cargo run --bin nonempty_alloc
CARGO_TARGET_DIR=/tmp/masstree-target MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri run --bin nonempty_alloc
```

The table below records historical audit behavior.

| Target | Finding | Expected |
|---|---|---|
| `tests/wrong_key.rs` | REM-01 | `remove(short)` returns a longer same-ikey key's value; 100k+ mismatches in a few seconds |
| `tests/lost_remove.rs` | REM-02 | `remove(key)` returns `None` while the key is still present after a concurrent split |
| `tests/double_unlink_deterministic.rs` | GC-01 | `pending_coalesce()==2`; `process_coalesce` never returns, watchdog exits 42 |
| `tests/double_unlink.rs` | GC-01 | stress variant; maintenance wedges or UAFs |
| `tests/sublayer_leak.rs` | GC-02 | live leaf/internode counts do not return to the empty-tree baseline |
| `tests/multi_coalesce.rs` | GC-01/GC-02 | repeated/duplicate maintenance entries |
| `tests/version_probe.rs` | OCC-01 | new-key inserts leave the version word unchanged |

```bash
cargo test --release --test wrong_key -- --nocapture
cargo test --release --test lost_remove -- --nocapture
timeout 30s cargo test --release --test double_unlink_deterministic -- --nocapture
cargo test --release --test sublayer_leak -- --nocapture
```

The intentionally failing tests are the reproduction. Run hang-prone tests
under `timeout`.
