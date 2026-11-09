#!/usr/bin/env bash
# Official vLLM 0.33.0 NIXL + MultiConnector with independent OrbitKV cache.
# Experimental composition: see docs/pd.md for qualification limits.
set -euo pipefail

MODEL="${1:?Usage: scripts/run_pd_local.sh /path/to/model}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/vllm-release/bin/python}"
ROUTER="${VLLM_ROUTER:-vllm-router}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8101}"
PREFILL_PORT="${PREFILL_PORT:-8102}"
export VLLM_USE_V2_MODEL_RUNNER=0
PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-8998}"
DECODE_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT:-8999}"
PREFILL_CACHE_SOCKET="${PREFILL_CACHE_SOCKET:?Set PREFILL_CACHE_SOCKET to the local Manager UDS}"
DECODE_CACHE_SOCKET="${DECODE_CACHE_SOCKET:?Set DECODE_CACHE_SOCKET to the local Manager UDS}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/orbitkv-native-pd.XXXXXX)}"
[[ -x "$PYTHON" ]] || { echo "Missing official vLLM environment: $PYTHON" >&2; exit 1; }
[[ "$PREFILL_GPU" != "$DECODE_GPU" ]] || { echo "PREFILL_GPU and DECODE_GPU must differ" >&2; exit 1; }
[[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "STARTUP_TIMEOUT must be a positive number of seconds" >&2; exit 1; }
for command in setsid curl "$ROUTER"; do command -v "$command" >/dev/null; done
mkdir -p "$LOG_DIR"

kv_config() {
    "$PYTHON" - "$1" "$2" <<'CONFIG'
import json
import sys

role, socket = sys.argv[1:]
native = {
    "kv_connector": "NixlConnector",
    "kv_role": f"kv_{role}",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {"backends": ["UCX"]},
}
cache = {
    "kv_connector": "OrbitKVConnector",
    "kv_connector_module_path": "orbitkv.vllm",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "orbitkv.bootstrap_socket": socket,
        "orbitkv.mode": "read_write" if role == "producer" else "save_only",
    },
}
print(json.dumps({
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "connectors": [cache, native] if role == "producer" else [native, cache],
    },
}))
CONFIG
}

DECODE_CONFIG="$(kv_config consumer "$DECODE_CACHE_SOCKET")"
PREFILL_CONFIG="$(kv_config producer "$PREFILL_CACHE_SOCKET")"
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "Logs: $LOG_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" CUDA_VISIBLE_DEVICES="$DECODE_GPU" "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 --port "$DECODE_PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
    --kv-transfer-config "$DECODE_CONFIG" > "$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")
setsid env VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" CUDA_VISIBLE_DEVICES="$PREFILL_GPU" "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 --port "$PREFILL_PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
    --kv-transfer-config "$PREFILL_CONFIG" > "$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")

for index in 0 1; do
    if [[ "$index" == 0 ]]; then name=decode; port="$DECODE_PORT"; else name=prefill; port="$PREFILL_PORT"; fi
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    until curl --max-time 2 -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "$name failed to become ready; see $LOG_DIR/$name.log" >&2
            exit 1
        fi
        sleep 1
    done
    echo "$name ready: http://127.0.0.1:$port"
done

setsid "$ROUTER" --vllm-pd-disaggregation --kv-connector nixl \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --host 127.0.0.1 --port "$ROUTER_PORT" > "$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
echo "Native P/D router starting: http://127.0.0.1:$ROUTER_PORT; logs: $LOG_DIR"
wait -n "${pids[@]}"
