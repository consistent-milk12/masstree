#!/usr/bin/env bash
# BUG-TOOL-06: test tracing writes into the repository and `logs/` is not
# ignored; many tests initialize it by default.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0
if grep -q 'logs/masstree.jsonl' tests/common/mod.rs; then
  echo "REPRODUCED: tests/common writes logs/masstree.jsonl"
else
  echo "NOT REPRODUCED: tracing path changed"
  status=1
fi
if grep -qx 'logs/' .gitignore 2>/dev/null || grep -qx '/logs/' .gitignore 2>/dev/null; then
  echo "NOT REPRODUCED: logs/ is now ignored"
  status=1
else
  echo "REPRODUCED: logs/ is absent from .gitignore"
fi
if [ -d logs ]; then
  echo "REPRODUCED: untracked logs/ directory is present"
fi
exit "$status"
