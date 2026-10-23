# Concurrency cross-cut reproducers

| Crate | Finding | Expected |
|---|---|---|
| `vercheck/` | OCC-01 | version word unchanged across three new-key inserts, then changes on update/remove |
| `layer_race/` | OCC-02 | `get_ref` transient misses while another key with the same ikey creates a layer |
| `typeconf/` | OCC-02 | `MassTree15<String>` returns a `String` whose contents/pointer are not the expected value (layer pointer aliases a value) |
| `remove_race/` | REM-01 | `remove(short)` returns the long key's value |
| `repro/` | GC-01 | empty-leaf reuse queues two raw chain entries; `process_coalesce` wedges |
| `lifetime/` | SAFE-01 | Miri use-after-free from a guard-scoped `ValuePtr` |

```bash
(cd vercheck && cargo run --quiet)
(cd layer_race && cargo run --release)
(cd typeconf && cargo run --release)
(cd remove_race && cargo run --release)
(cd lifetime && cargo +nightly miri run)
timeout 30s bash -c 'cd repro && cargo run --release'
```

`layer_race` and `typeconf` are stochastic; they may need several hundred
thousand operations. `repro` is expected to hang.
