#!/usr/bin/env bash
# CPU callback controls against installed official classes and our built wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(realpath -m "${1:?external result directory required}")"
WHEEL="$(realpath "${2:?built Python 3.12 wheel required}")"
if [[ -e "$RUN_DIR" ]]; then
  echo "Choose a new result directory; preserve prior failures and controls." >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp -R "$ROOT/python/tests" "$RUN_DIR/"
cp "$ROOT/python/pyproject.toml" "$RUN_DIR/"
for engine in vllm sglang; do
  version="$(python3 - "$ROOT" "$engine" <<'PY'
import runpy
import sys
pins = runpy.run_path(sys.argv[1] + "/python/tests/support/installed_artifacts.py")
print(pins["ENGINE_VERSIONS"][sys.argv[2]])
PY
)"
  engine_env="$ROOT/.venv/engine-api-$engine"
  uv venv --python 3.12 "$engine_env"
  uv pip install --python "$engine_env/bin/python" --torch-backend cpu \
    "$engine==$version" pytest requests
  uv pip install --python "$engine_env/bin/python" --no-deps "$WHEEL"
  uv pip freeze --python "$engine_env/bin/python" > "$RUN_DIR/$engine-freeze.txt"
  (
    cd "$RUN_DIR"
    # Exercise the GPU adapter's Python contracts, not SGLang's CPU serving backend.
    env -u PYTHONPATH VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS='' SGLANG_USE_CPU_ENGINE=0 \
      ORBITKV_REQUIRE_ENGINE_TESTS=1 \
      "$engine_env/bin/python" -m pytest --import-mode=importlib -m '' \
      --engine="$engine" --engine-evidence=installed-cpu \
      --engine-report="$RUN_DIR/$engine-evidence.json" --junitxml="$RUN_DIR/$engine.xml"
  )
done
