#!/usr/bin/env bash
# TOOL-02 regression: the unused shuttle dependency, cfg, and zero-test
# recipe have been removed.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0

if grep -q 'shuttle' Cargo.toml; then
  echo "REGRESSION: Cargo.toml still mentions shuttle"
  status=1
fi

if grep -q 'test-shuttle' justfile; then
  echo "REGRESSION: justfile still has a test-shuttle recipe"
  status=1
fi

if grep -q 'cfg(shuttle)' Cargo.toml; then
  echo "REGRESSION: cfg(shuttle) is still declared"
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "FIXED: shuttle dependency/cfg/recipe are gone"
fi
exit "$status"
