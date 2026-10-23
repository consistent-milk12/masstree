#!/usr/bin/env bash
# BUG-TOOL-01: `--no-default-features` cannot build all targets because
# examples/print_tree.rs is not gated by `debug-print`.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
cargo check --all-targets --no-default-features >"$OUT" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  echo "UNEXPECTED PASS: --no-default-features now builds all targets"
  rm -f "$OUT"
  exit 1
fi
if grep -q "doesn't implement .*Display" "$OUT"; then
  echo "REPRODUCED BUG-TOOL-01 (exit $status): print_tree requires debug-print"
  grep -m1 "doesn't implement .*Display" "$OUT" || true
  rm -f "$OUT"
  exit 0
fi
echo "FAILED TO REPRODUCE: build failed for a different reason"
tail -20 "$OUT"
rm -f "$OUT"
exit 1
