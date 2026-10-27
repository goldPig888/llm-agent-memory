# Native detection-format xhigh pilot

This tests a localization-format hypothesis, not an established claim about
Meta training data. The HF gist uses x_min/y_min/x_max/y_max but explicitly
calls its 0–1000 coordinate scale an assumption. The clicking adapter card
reports a different GUI domain and notes that simple rescaling did not rescue
its base-model outputs; those results are not LIBERO evidence.

Sources:
- https://gist.github.com/ariG23498/4f3587eb7753c0ff77c269e2c1efe2c0
- https://huggingface.co/blog/muse-glimmer
- https://huggingface.co/merve/muse-glimmer-ft-clicking-v2

Protocol: saved Task 0 states 0–4, agentview only, raw 512x512 RGB, local base
Glimmer GGUF and existing quantized projector. Each image gets two fresh-context
requests: the existing normalized 0..1 point prompt and a native detection
prompt, both xhigh and 16,384 output tokens. Seed is 42 + state in both formats.
Format order alternates across states. This is ten requests, zero robot actions,
no training, no adapter loading, and no paid API.

The native prompt asks to detect alphabet soup in the model's native
object-detection format, with no explanation. It does not name box keys or
coordinate range. JSON and Python-style literal dictionaries are parsed safely,
using the last dictionary with the four example keys. The primary score uses
box center divided by 1000, with no clipping, reordering, adaptive scaling, or
selection based on ground-truth performance. Invalid/truncated outputs count as
failures and are retained. All box values and raw outputs are saved; values
within 0..1000 alone do not validate that coordinate convention. A scale warning
is emitted if all box coordinates are <=1. Masks and labels are scoring-only;
they are never passed to the model.

The original 44% result used high effort and fifty agentview states; this five-
state xhigh pilot needs its paired xhigh point control. Do not compare percentages
as though the datasets and settings were identical, or generalize five examples
to a benchmark success rate. A box-center hit is not a successful grasp.

The pilot waits for the current xhigh robot campaign to complete, then acquires
the same model lock. If that campaign is incomplete or another experiment owns
the lock, it stops rather than overlapping. Outputs are a fresh directory;
there are no silent reruns or API retries. The raw dataset and old results remain
unchanged.

```bash
python scripts/run_native_pointing.py \
  --dataset /metadisk/may/muse-glimmer/pointing_50states_20261026 \
  --output /metadisk/may/muse-glimmer/native_pointing_xhigh_states0to4_20261027 \
  --states 0 1 2 3 4 \
  --wait-campaign /metadisk/may/muse-glimmer/harness_xhigh_results/visual-ablation-xhigh-v1.json
```

Viewer: http://127.0.0.1:8770/ (separate from the original 50-state pointing viewer).
