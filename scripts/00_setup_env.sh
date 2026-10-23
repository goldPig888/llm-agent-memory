#!/usr/bin/env bash
# 00: one-time setup on a new machine.
#
#   bash scripts/00_setup_env.sh            # conda env "llm-agent-memory"
#   ENV_NAME=myenv bash scripts/00_setup_env.sh
#
# Clones the two pinned sources in vendor.lock into vendor/, then installs,
# into one Python 3.12 conda env: LIBERO (hf-libero), the pinned
# Inspect Robots core + agent plugin, the pinned LLM_LIBERO plugin (same tasks,
# embodiment and prompt as the paper), and this package. Then initialises
# LIBERO's config so the simulator never blocks on its first-run question.
set -euo pipefail

ENV_NAME="${ENV_NAME:-llm-agent-memory}"
cd "$(dirname "$0")/.."

echo "[00] fetching pinned sources (vendor.lock)"
mkdir -p vendor
grep -v '^#' vendor.lock | while read -r name url commit; do
  [ -z "$name" ] && continue
  if [ ! -d "vendor/$name/.git" ]; then
    git clone --quiet "$url" "vendor/$name"
  fi
  git -C "vendor/$name" fetch --quiet origin "$commit" 2>/dev/null || git -C "vendor/$name" fetch --quiet origin
  git -C "vendor/$name" checkout --quiet "$commit"
  echo "  vendor/$name @ $(git -C "vendor/$name" rev-parse --short HEAD)"
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[00] creating conda env $ENV_NAME (python 3.12)"
  conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

echo "[00] installing LIBERO and the pinned framework"
pip install hf-libero==0.1.4 imageio imageio-ffmpeg h5py
pip install -r vendor/LLM_LIBERO/requirements.lock.txt \
  ./vendor/inspect-robots \
  ./vendor/inspect-robots/plugins/inspect-robots-agent \
  ./vendor/LLM_LIBERO
pip install -e ".[dev,setup,drive]"

if [ ! -f "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" ]; then
  echo "[00] initialising LIBERO (answering N to the dataset-path question)"
  echo N | python -c "import libero.libero" || true
fi

echo "LIBERO environment ready. Next: python scripts/01_check_install.py"
