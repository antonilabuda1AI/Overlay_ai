#!/usr/bin/env bash
set -euo pipefail
if ! command -v kernprof >/dev/null 2>&1; then
  echo "kernprof (line_profiler) not found. pip install line_profiler"
  exit 1
fi
if [ $# -lt 1 ]; then
  echo "Usage: $0 <python_file> [args...]"
  exit 1
fi
kernprof -l "$@"
python -m line_profiler $(basename "$1").lprof

