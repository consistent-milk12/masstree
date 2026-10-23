#!/usr/bin/env bash
# BUG-TOOL-02: `just test-shuttle` selects a module that does not exist.
# It exits 0 after running zero tests, and `test-all` depends on it.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
cargo test --lib tree::shuttle_tests >"$OUT" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  echo "UNEXPECTED FAILURE: cargo test exited $status"
  tail -20 "$OUT"
  rm -f "$OUT"
  exit 1
fi
if grep -Eq "running 0 tests|0 passed" "$OUT"; then
  echo "REPRODUCED BUG-TOOL-02: test-shuttle is a zero-test no-op"
  grep -E "running 0 tests|test result:" "$OUT" | head -3
  rm -f "$OUT"
  exit 0
fi
echo "FAILED TO REPRODUCE: shuttle filter actually selected tests"
grep -E "running [0-9]+ tests|test result:" "$OUT" | head -3
rm -f "$OUT"
exit 1
