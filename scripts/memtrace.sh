#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <python_entry> [args...]"
  exit 1
fi
PYTHONTRACEMALLOC=25 python "$@"

