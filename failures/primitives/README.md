# Primitives reproducers

## `nodeversion/`

| Binary | Finding | Expected |
|---|---|---|
| `cargo run` (`src/main.rs`) | OCC-01/02 | `has_changed` stays false while the writer holds `INSERTING`; a `ValueSlot` is cleared under a valid snapshot |
| `cargo run --release --bin layer_race` | OCC-02 | transient `get_ref` misses while a layer is created for the same ikey |
| `cargo run --release --bin layer_race_update` | OCC-02 | variant with value-update control |
| `cargo run --quiet --bin perm_raw` | PRIM-02 | `Permuter<7>::remove_to_back` leaves bits above the encoded width |

`layer_race` accepts an optional iteration count and `--skip-b` control mode.

## `shard/`

`cargo run --quiet` hammers `ShardedCounter::increment/decrement/load`.
`load()`'s debug assertion can observe a negative sum even though the logical
count never goes negative. Build without `--release` to keep debug assertions.

## `scripts/`

`perm_model.py`/`perm_model2.py` are the exhaustive/random Permuter models used
to validate the float-free permutation arithmetic.
