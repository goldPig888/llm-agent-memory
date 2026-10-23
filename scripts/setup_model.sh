#!/usr/bin/env bash
# Build and download the tested local multimodal runtime. No cloud inference.
set -euo pipefail
if [ -d /metadisk/may/muse-glimmer ]; then
  MUSE_ROOT="${MUSE_ROOT:-/metadisk/may/muse-glimmer}"
else
  MUSE_ROOT="${MUSE_ROOT:-$HOME/.local/share/muse-glimmer}"
fi
MUSE_PYTHON="${MUSE_PYTHON:-python}"
if [ -z "${CUDA_ARCHITECTURES:-}" ]; then
  # CMake <3.24 does not support the special "native" architecture value.
  # Query numeric compute capabilities instead (e.g. 8.6 -> 86).
  CUDA_ARCHITECTURES="$($MUSE_PYTHON - <<'PY'
import subprocess
try:
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True)
    values = sorted(set(line.strip().replace('.', '') for line in output.splitlines()))
    if not values or not all(value.isdigit() for value in values):
        raise ValueError('invalid compute capability')
    print(';'.join(values))
except (OSError, ValueError, subprocess.CalledProcessError) as exc:
    raise SystemExit('Cannot detect GPU architecture; set CUDA_ARCHITECTURES explicitly (e.g. 86 for A5000).') from exc
PY
  )"
fi
if [ -d /usr/local/cuda-12.8 ]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
else
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
mkdir -p "$MUSE_ROOT"
if [ ! -d "$MUSE_ROOT/llama.cpp/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$MUSE_ROOT/llama.cpp"
fi
git -C "$MUSE_ROOT/llama.cpp" checkout 0ae0e1976670fac25f4e0ed864df30021accc764
cmake -S "$MUSE_ROOT/llama.cpp" -B "$MUSE_ROOT/llama.cpp/build-cuda128" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -DLLAMA_BUILD_TESTS=OFF
cmake --build "$MUSE_ROOT/llama.cpp/build-cuda128" --target llama-server -j "${BUILD_JOBS:-8}"
"$MUSE_PYTHON" - "$MUSE_ROOT/models" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('meta-models/Muse-Glimmer-30B-GGUF',
    revision='b44f1072cf75e918a1b82a3a17b45c4389b076e7', local_dir=sys.argv[1],
    allow_patterns=['Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf',
                    'mmproj-Muse-Glimmer-30B-Q4_K_M.gguf', 'README.md'])
PY
