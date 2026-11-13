#!/usr/bin/env bash
set -euo pipefail

# Only idempotent dependency installation uses this helper, never tests/builds.
for attempt in 1 2 3; do
  if timeout 900 "$@"; then
    exit 0
  fi
  echo "Dependency install attempt $attempt/3 failed: $*" >&2
  if [[ "$attempt" == 3 ]]; then
    echo 'Dependency installation exhausted its retry budget' >&2
    exit 1
  fi
  sleep 2
done
