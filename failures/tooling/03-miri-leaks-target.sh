#!/usr/bin/env bash
# BUG-TOOL-03: `miri-leaks` runs `--lib`, where none of the 20 tests in
# tests/leak_tests.rs exist.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
LIB_LEAKS="$(cargo test --lib -- --list 2>/dev/null | grep -ci leak || true)"
INT_LEAKS="$(cargo test --test leak_tests -- --list 2>/dev/null | grep -c ': test' || true)"
echo "lib leak tests: $LIB_LEAKS"
echo "integration leak tests: $INT_LEAKS"
if [ "$LIB_LEAKS" -eq 0 ] && [ "$INT_LEAKS" -gt 0 ]; then
  echo "REPRODUCED BUG-TOOL-03: miri-leaks --lib executes no leak tests"
  exit 0
fi
echo "FAILED TO REPRODUCE: the lib target now contains leak tests"
exit 1
