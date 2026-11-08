# Batch/API reproducers

Proposal 4 update: `leak3` is now a passing regression. It asserts that dropping
entries frees their allocations, invalid-key batches leave the tree empty, and
15 preconstructed long-key write-through updates allocate at most three times
(for the old-value result). The prior runners allocated six times. The matching
integration drop-counter regression catches the residual invalid-key output leak
(2 of 3 values dropped before, all 3 afterward), and Miri covers unwinding after
a prior suffix replacement. The table below retains historical failure behavior.


| Binary | Finding | Expected |
|---|---|---|
| `alias` | SAFE-03 | two keys share one `ValuePtr`; updating one changes the other |
| `doublefree` | SAFE-03 | one Box is retired twice; glibc double-free abort |
| `live_view` | SAFE-01 | a held `ValuePtr` changes after `insert_value_with_guard` |
| `mut` | BATCH-01 | mutated `BatchEntry.key` with stale cached ikey loses the key (`get(MAX)=None`, `len/scan` disagree) |
| `leak`/`leak2`/`leak3` | BATCH-02/03 | measured allocation growth; `BatchEntry` drop leak; partial-panic leak |
| `layer_updates` | BATCH-02 | stale/duplicate values around layer updates |
| `teardown_safe` | SAFE-04 | safe `debug_root_ptr` + `teardown_tree` frees a live tree; UAF/double-free |
| `uaf_compile` | SAFE-01 | compiles a safe escaped `ValuePtr` |
| `main` | BATCH-04 | duplicate-key batch order/last-wins check |
| `mimalloc-repro/` | BATCH-05 | downstream binary with its own global allocator fails to compile when `masstree/mimalloc` is enabled |

```bash
cargo run --bin alias
cargo run --bin doublefree       # abort expected
cargo run --bin mut
cargo run --bin leak2
cargo run --bin leak3
cargo run --bin teardown_safe    # abort expected
(cd mimalloc-repro && cargo check)  # expected global-allocator conflict
```

`probes/` contains two standalone `rustc` programs that show the unstable-sort
reordering behind BATCH-04.

Do not run all binaries in one process; several intentionally abort.
