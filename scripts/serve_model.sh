#!/usr/bin/env bash
set -euo pipefail
if [ -d /metadisk/may/muse-glimmer ]; then
  MUSE_ROOT="${MUSE_ROOT:-/metadisk/may/muse-glimmer}"
else
  MUSE_ROOT="${MUSE_ROOT:-$HOME/.local/share/muse-glimmer}"
fi
split_args=()
if [ -n "${GPU_SPLIT:-}" ]; then
  split_args=(--tensor-split "$GPU_SPLIT")
fi
exec "$MUSE_ROOT/llama.cpp/build-cuda128/bin/llama-server" \
  -m "$MUSE_ROOT/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" \
  --mmproj "${MUSE_MMPROJ:-$MUSE_ROOT/models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf}" \
  --alias muse-glimmer --host 127.0.0.1 --port 8081 \
  --n-gpu-layers 99 --split-mode layer "${split_args[@]}" \
  --ctx-size 131072 --parallel 1 --jinja \
  --chat-template-kwargs '{"reasoning_strength":"medium"}' "$@"
