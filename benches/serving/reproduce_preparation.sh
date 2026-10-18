#!/usr/bin/env bash
# Run from the repository root with matching ordinary native binaries installed.
set -euo pipefail

export ORBITKV_CACHE_MANAGER_BINARY="${ORBITKV_CACHE_MANAGER_BINARY:-$PWD/python/orbitkv/orbitkv-cache-manager-py}"
trial_root="${1:?provide an external output root, e.g. /var/tmp/orbitkv-bench/preparation}"
trial_root="$(realpath -m -- "$trial_root")"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
case "$trial_root/" in
  "$repo_dir/"*) echo "output root must be outside the source checkout" >&2; exit 2 ;;
esac

for engine in vllm sglang; do
  export TORCHINDUCTOR_CACHE_DIR="${trial_root}-compiler-cache/$engine/inductor"
  export TRITON_CACHE_DIR="${trial_root}-compiler-cache/$engine/triton"
  export VLLM_CACHE_ROOT="${trial_root}-compiler-cache/$engine/vllm"
  command=(".venv/$engine-release/bin/python" -m benches.serving.single_node
    --engine "$engine" --backend orbitkv --model /workspace/models/qwen3-8b
    --workload sustained --lengths 1024 4096 --concurrencies 4
    --gpu-tokens 16384 --host-gib 4 --ssd-gib 16 --query-budget-gib 3
    --duration-seconds 60 --max-requests 64 --working-set 12
    --queue-warmup off --trace-transfers --seed 20260922)

  # Same fixed request cap/seed and reversed middle-pair order.
  for repetition in 1 2 3; do
    order=(off on)
    if [[ "$repetition" == 2 ]]; then order=(on off); fi
    for preparation in "${order[@]}"; do
      "${command[@]}" --read-batch-mib 32 --prepare-requests "$preparation" \
        --output "$trial_root-$engine-pair-$repetition-$preparation"
    done
  done

  # Isolate batching and conservative stopping from the ownership comparison.
  "${command[@]}" --output "$trial_root-$engine-unbounded"
  "${command[@]}" --read-batch-mib 32 --prepare-requests on --read-timeout-ms 100 \
    --output "$trial_root-$engine-deadline"
  "${command[@]}" --read-batch-mib 32 --prepare-requests on --read-max-batches 1 \
    --output "$trial_root-$engine-best-effort"
done

# Dedicated DRAM-only tier controls; the larger host pool is recorded explicitly.
for engine in vllm sglang; do
  export TORCHINDUCTOR_CACHE_DIR="${trial_root}-compiler-cache/$engine/inductor"
  export TRITON_CACHE_DIR="${trial_root}-compiler-cache/$engine/triton"
  export VLLM_CACHE_ROOT="${trial_root}-compiler-cache/$engine/vllm"
  ".venv/$engine-release/bin/python" -m benches.serving.single_node \
    --engine "$engine" --backend orbitkv --model /workspace/models/qwen3-8b \
    --workload concurrent --lengths 1024 4096 --concurrencies 1 4 --repeats 3 \
    --gpu-tokens 16384 --host-gib 8 --ssd-gib 0 --query-budget-gib 3 \
    --trace-transfers --seed 20260922 --output "$trial_root-$engine-dram"
done
