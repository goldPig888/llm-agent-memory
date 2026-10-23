# Muse Glimmer on LIBERO-10, locally

Evaluate Meta Muse Glimmer with RGB observations and structured robot tool calls.
Inference runs on this machine; no Codex sessions or hosted inference credentials.
The pinned LLM-LIBERO tasks, prompts, motion adapter, and simulator success checker
are retained. Default evaluation: tasks 0–9, initial states 0–9, 80 model calls,
900 simulator steps, full text history and images from the latest two observations.
Each episode starts fresh without demonstrations or earlier-episode notes.

## Environment

On another Linux/NVIDIA PC, install Miniconda, Git, CMake, a C++ compiler,
and a CUDA toolkit compatible with its driver (CUDA 12.8 tested here). Then:

```bash
git clone https://github.com/goldPig888/llm-agent-memory.git
cd llm-agent-memory
bash scripts/00_setup_env.sh
conda activate llm-agent-memory
python scripts/01_check_install.py --fake

# Choose a disk with room for ~19 GB of weights, builds, and rollout data.
export MUSE_ROOT="$HOME/.local/share/muse-glimmer"
export CUDA_HOME=/usr/local/cuda  # adjust to your installed toolkit
bash scripts/setup_model.sh
bash scripts/serve_model.sh     # keep this terminal running
```

Keep `MUSE_ROOT` set to the same location for setup and serving. The build
detects the local GPU architecture; override with `CUDA_ARCHITECTURES` if needed.
`CUDA_VISIBLE_DEVICES` selects GPUs. `GPU_SPLIT=0.4,0.6` optionally sets a split
for two GPUs; otherwise llama.cpp distributes weights automatically. GPU memory
requirements include the vision encoder and context cache, not just the weights.

The existing Python environment name is retained. `scripts/00_setup_env.sh`
can install the pinned simulator dependencies on a new machine.

This machine has two RTX A5000 24 GB GPUs. The official 17 GB Q4 GGUF and vision
projector are under `/metadisk/may/muse-glimmer/models`. CUDA 12.8 builds the
llama.cpp server at `/metadisk/may/muse-glimmer/llama.cpp/build-cuda128/bin/llama-server`.
Full BF16 requires more memory; this evaluation uses quantization explicitly.

```bash
bash scripts/setup_model.sh  # reproducible download and CUDA build on this machine
bash scripts/serve_model.sh
python scripts/smoke_model.py  # requires two-camera image understanding + a tool call
python scripts/run_local.py --tasks 2 --inits 0
# Full paper-sized evaluation, only after the smoke test:
python scripts/run_local.py --tasks 0 1 2 3 4 5 6 7 8 9 --inits 0 1 2 3 4 5 6 7 8 9
```

### Medium and high on separate PCs

In a second terminal on the **medium** PC:

```bash
conda activate llm-agent-memory
python scripts/smoke_model.py
python -u scripts/run_local.py --effort medium --output outputs/muse
```

On the **high** PC use `--effort high` instead. Each command defaults to all
ten tasks and initial states 0–9 (100 episodes). The runner sets reasoning
strength on every request, overriding the server default. Run inside `tmux`
or `screen` if disconnecting SSH; keep both server and runner alive.
Use the same pinned weights and 80-call / 900-step / two-observation defaults
across machines. This runner does not automatically resume interrupted runs.

Point the viewer at the same `--output` directory used by the runner. Its
`campaign.json` appears when the first run starts.

The model endpoint is `http://127.0.0.1:8081/v1`. The runner rejects non-loopback
endpoints and does not inherit cloud keys. The local endpoint uses an HTTP
chat interface; it does not send requests to a paid provider.
Use `--effort low|medium|high`, `--image-horizon`, `--max-output-tokens`, and
`--output` to configure runs. The server has one 131,072-token slot; requests
that exceed context or truncate output must be investigated rather than
silently counted as valid policy comparisons. Protocol metadata records limits.
`--plan-only` writes pending runs without model or simulator calls.

## Viewer

```bash
python scripts/serve_experiment_dashboard.py --output outputs/muse --port 8765
```

Open http://localhost:8765. Adapted from LLM_LIBERO's experiment dashboard:
per-task/state results, official simulator success, videos, trajectories,
conversation, model-call token counts, and latency. Refresh to update.
Raw requests/responses, framework logs, frames and videos live beside each run.
On this machine `outputs/muse` points to `/metadisk/may/muse-glimmer/results`
to leave the main disk free. A separate `outputs/muse_smoke` stores the one-call
setup check; it is excluded from the benchmark viewer and exports.
A fake/smoke run is not a benchmark result.

## Google Drive export and upload

Reused from IRVLUTD/LLM_LIBERO. Default folder:
`1D6vn9HiG9V81ROdV57m7zy8tOsDLL84M` (same as upstream).
Existing OAuth files can be used by path; never commit credentials.

```bash
# Local export only:
python scripts/upload_experiment_bundle.py --source outputs/muse --visualization --export-only
# Export and merge into the shared Drive results (explicit upload):
python scripts/upload_experiment_bundle.py --source outputs/muse --visualization \
  --token /home/may/Desktop/Github/LLM_LIBERO/outputs/drive_auth/token.json
# Fetch selected local rollout videos, then upload them to the same folder:
python scripts/fetch_rollout_videos.py --help
python scripts/upload_rollout_videos.py --help
```

Supply `--file-id` to merge into a specific shared results file. The uploader
preserves existing episode records and writes a receipt. Uploading is separate
from evaluation; no upload happens automatically.

Sources: [Meta deployment guide](https://dev.meta.ai/docs/muse-glimmer/deploy),
[official model](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF),
[LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO).
