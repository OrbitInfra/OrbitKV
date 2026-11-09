#!/usr/bin/env bash
set -euo pipefail

# Keep the build input pinned across CI and wheel/release jobs.
version=27.5
case "$(uname -sm)" in
  'Linux x86_64') platform=linux-x86_64 ;;
  'Linux aarch64') platform=linux-aarch_64 ;;
  *) echo 'protoc installer requires Linux x86_64 or aarch64' >&2; exit 1 ;;
esac
workspace=$(mktemp -d "${RUNNER_TEMP:?}/orbitkv-protoc.XXXXXX")
trap 'rm -rf "$workspace"' EXIT
archive="$workspace/protoc.zip"
url="https://github.com/protocolbuffers/protobuf/releases/download/v${version}/protoc-${version}-${platform}.zip"
for attempt in 1 2 3; do
  if curl --fail --location --silent --show-error --connect-timeout 15 --max-time 120 \
    --output "$archive" "$url"; then
    break
  fi
  if [[ "$attempt" == 3 ]]; then
    echo 'protoc download failed after 3 attempts' >&2
    exit 1
  fi
  sleep 2
done
destination="$RUNNER_TEMP/protoc-$version"
mkdir -p "$destination"
unzip -q -o "$archive" -d "$destination"
if [[ "$("$destination/bin/protoc" --version)" != "libprotoc $version" ]]; then
  echo 'downloaded protoc does not match the pinned version' >&2
  exit 1
fi
printf '%s\n' "$destination/bin" >> "${GITHUB_PATH:?}"
