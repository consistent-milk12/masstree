#!/usr/bin/env bash
# TOOL-06 regression: test tracing must not write into the repository and
# logs/ is ignored as a backstop.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0

if grep -q 'PathBuf::from("logs")' tests/common/mod.rs; then
  echo "REGRESSION: tests/common still defaults to the repository logs/ dir"
  status=1
fi

if ! grep -q 'temp_dir' tests/common/mod.rs; then
  echo "REGRESSION: tests/common does not default to a temp directory"
  status=1
fi

if ! grep -qx 'logs/' .gitignore 2>/dev/null; then
  echo "REGRESSION: logs/ is not ignored"
  status=1
fi

if [ -d logs ] && [ -n "$(ls -A logs 2>/dev/null)" ]; then
  echo "REGRESSION: an untracked logs/ directory with content is present"
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "FIXED: test logs go to a temp directory and logs/ is ignored"
fi
exit "$status"
