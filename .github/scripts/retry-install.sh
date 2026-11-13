#!/usr/bin/env bash
set -euo pipefail

# Only idempotent dependency installation uses this helper, never tests/uploads.
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
for attempt in 1 2 3; do
  mirror=primary
  [[ "$attempt" != 2 ]] || mirror=alternate
  export ORBITKV_INSTALL_MIRROR="$mirror"
  echo "Dependency install attempt $attempt/3, mirror=$mirror: $*" >&2
  if [[ "$1" == apt-get ]]; then
    python3 "$scripts/apt-mirrors.py" /etc/apt "$workspace/$attempt" "$mirror"
    options=(-o "Dir::Etc::sourcelist=$workspace/$attempt/sources.list"
      -o "Dir::Etc::sourceparts=$workspace/$attempt/sources.list.d"
      -o APT::Update::Error-Mode=any -o Acquire::Retries=0)
    # Refresh the selected mirror's indexes before install; do not use stale lists.
    if timeout 900 sudo apt-get "${options[@]}" update &&
        { [[ "${2:-}" == update ]] || timeout 900 sudo apt-get "${options[@]}" "${@:2}"; }; then
      exit 0
    fi
  elif timeout 900 "$@"; then
    exit 0
  fi
  if [[ "$attempt" == 3 ]]; then
    echo 'Dependency installation exhausted its retry budget' >&2
    exit 1
  fi
  sleep 2
done
