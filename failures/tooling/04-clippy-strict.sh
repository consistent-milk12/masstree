#!/usr/bin/env bash
# BUG-TOOL-04: `just lint-strict` fails on current stable Clippy, so
# `check-all` cannot be green.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
cargo clippy --all-targets --all-features -- -D warnings >"$OUT" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  echo "UNEXPECTED PASS: strict Clippy is clean"
  rm -f "$OUT"
  exit 1
fi
echo "REPRODUCED BUG-TOOL-04: strict Clippy exits $status"
grep -E "^error: |^error\[" "$OUT" | head -10 || true
rm -f "$OUT"
exit 0
