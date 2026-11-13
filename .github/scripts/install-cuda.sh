#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == keyring ]]; then
  # APT verifies signed repository metadata on either mirror; authenticate bootstrap too.
  . /etc/os-release
  case "$(uname -m)" in
    x86_64) arch=x86_64 ;;
    aarch64) arch=sbsa ;;
    *) echo 'CUDA installer requires x86_64 or aarch64' >&2; exit 1 ;;
  esac
  case "$VERSION_ID" in
    22.04|24.04) distro="ubuntu${VERSION_ID//./}" ;;
    *) echo 'CUDA installer requires Ubuntu 22.04 or 24.04' >&2; exit 1 ;;
  esac
  case "$distro/$arch" in
    ubuntu2204/x86_64) sha256=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df ;;
    ubuntu2404/x86_64) sha256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba ;;
    ubuntu2404/sbsa) sha256=6ea7d2737648936820e85677177957a0f6521b840d98eb0bbae0a4f003fa7249 ;;
    *) echo 'CUDA keyring is not pinned for this runner' >&2; exit 1 ;;
  esac
  host=developer.download.nvidia.com
  [[ "${ORBITKV_INSTALL_MIRROR:-primary}" != alternate ]] || host=developer.download.nvidia.cn
  workspace=$(mktemp -d)
  trap 'rm -rf "$workspace"' EXIT
  curl --fail --location --silent --show-error --connect-timeout 15 --max-time 120 \
    "https://$host/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb" -o "$workspace/keyring.deb"
  printf '%s  %s\n' "$sha256" "$workspace/keyring.deb" | sha256sum --check --strict
  sudo dpkg -i "$workspace/keyring.deb"
else
  bash "$scripts/retry-install.sh" bash "$scripts/install-cuda.sh" keyring
  # The same CUDA 13.0.2 component versions on x86_64 and aarch64.
  bash "$scripts/retry-install.sh" apt-get install -y \
    cuda-nvcc-13-0=13.0.88-1 cuda-cudart-dev-13-0=13.0.96-1 cuda-nvrtc-dev-13-0=13.0.88-1
  /usr/local/cuda-13.0/bin/nvcc --version | grep -F 'release 13.0, V13.0.88'
  echo '/usr/local/cuda-13.0/bin' >> "${GITHUB_PATH:?}"
  echo 'CUDA_PATH=/usr/local/cuda-13.0' >> "${GITHUB_ENV:?}"
fi
