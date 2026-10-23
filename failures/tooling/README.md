# Tooling/workflow failure checks

Each script exits 0 when the documented defect is reproduced.

| Script | Finding |
|---|---|
| `01-no-default-features.sh` | TOOL-01: `--all-targets --no-default-features` fails on `print_tree` |
| `02-shuttle-noop.sh` | TOOL-02: `test-shuttle` runs zero tests |
| `03-miri-leaks-target.sh` | TOOL-03: `miri-leaks` uses `--lib`, where no leak tests exist |
| `04-clippy-strict.sh` | TOOL-04: strict Clippy fails on four warnings |
| `05-cpp-bench.sh` | TOOL-05: missing C++ binary, path disagreement, wrong redirect |
| `06-test-logs.sh` | TOOL-06: tests write untracked `logs/` |
| `07-vacuous-tests.sh` | TOOL-09/10: vacuous parity tests and one-sided scan proptests |

```bash
./run-all.sh
```

The checks that compile use `CARGO_TARGET_DIR=/tmp/masstree-failure-tooling-target`
by default. Set it explicitly to reuse a build cache.
