#!/usr/bin/env bash
# TOOL-01 regression: `--no-default-features` must build every target.
# The print_tree example is now gated by `debug-print` and the examples and
# benchmarks use the guard-scoped value API.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
cargo check --all-targets --no-default-features >"$OUT" 2>&1
status=$?
rm -f "$OUT"
if [ "$status" -eq 0 ]; then
  echo "FIXED: --no-default-features builds all targets"
  exit 0
fi
echo "REGRESSION: --no-default-features failed for all targets (exit $status)"
tail -20 "$OUT" 2>/dev/null || true
exit 1
