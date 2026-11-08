# Range scan reproducers

`promoted` is a passing regression: 40- and 300-key promoted layers and
256-byte keys are checked against a BTreeMap across forward/reverse ranges,
excluded and later-prefix bounds, keyed/reference/batch/value callbacks,
full-key prefixes, and mixed next/back.
The corresponding unit regression verifies that both scan directions cache
the true promoted root and retries require zero parent hops.

`race` now asserts ordering and key/value identity during concurrent writes
for both policies, using barriers and a bounded watchdog. Run it as
`cargo test --bin race` or `cargo +nightly miri test --bin race`; Miri uses fewer
rounds while retaining all scan modes and key shapes. The table below retains
historical failure observations.

| Binary | Finding | Expected observation |
|---|---|---|
| `nul` | RNG-01, RNG-03 | `start inc [0]` includes `[]`; excluded bound is emitted; reverse `end inc [0]` drops `[0]` |
| `nulpfx` | RNG-01 | `scan_prefix([0])` returns `[]` |
| `layerend` | RNG-02 | reverse end at `abcdefgh` returns keys `abcdefghX..Z` (expected none), same for `Excluded`; `scan_rev_batch` also returns them |
| `promoted` | RNG-04 | start indexes ≥ 15 skip the bound key; full-key `scan_prefix` returns 0 |
| `vals` | RNG-05 | values-only scans emit values for empty/inverted ranges |
| `valfar` | RNG-05 | values-only scan descends a layer past the end bound |
| `maxprefix` | RNG-05 | `scan_prefix_values([a;256])` gets two values, including the adjacent `[a;255]+b` key |
| `apis` | RNG-01/02/03 | keyed/ref/batch variants disagree with the `BTreeMap` bounds; reverse batch can be empty |
| `race` | RNG-07 | remove+reinsert stress eventually emits the same key twice; debug ordering assertion fires |
| `doubleended` | RNG-06 | draining `next()` to `None`, then `next_back()`, re-emits `c,b,a` |
| `empty_right` | RNG-08 | removing the rightmost five keys makes `last()` `None` and reverse iteration empty while forward returns 15 |
| `uaf` | SAFE-01 | `ScanEntry` outlives guard/tree and dereferences a freed value |
| `main`, `main_all` | RNG-01/03/04/05 | broad differential checks and exploratory API matrices |

Run a single repro with:

```bash
cargo run --quiet --bin <name>
```

Some stress binaries print rather than assert. Use `timeout 30s`.
