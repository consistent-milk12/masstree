#!/usr/bin/env bash
# TOOL-04 regression: strict Clippy must stay clean.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/masstree-failure-tooling-target}"
cd "$ROOT"
cargo clippy --all-targets --all-features -- -D warnings >"$OUT" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  echo "FIXED: strict Clippy is clean"
  rm -f "$OUT"
  exit 0
fi
echo "REGRESSION: strict Clippy exits $status"
grep -E "^error: |^error\[" "$OUT" | head -10 || true
rm -f "$OUT"
exit 1
