# General harness ablations

Task 0/state 0 is a diagnostic fixture. No task-specific prompt, object lookup,
target coordinates, or scripted task solution is added. The existing user task
instruction is unchanged. The original paper baseline is unchanged.

The four variants form a 2 × 2 screen:

| Variant | Remember rejected programs and give recovery feedback | Clean RGB instead of overlays |
| --- | --- | --- |
| control | no | no |
| recovery | yes | no |
| clean-images | no | yes |
| combined | yes | yes |

Recovery canonicalizes program syntax, remembers failed programs at the current
measured robot pose, and refuses identical retries. Zooming does not erase the
failure memory. A changed measured pose clears it. Feedback reminds the model
that a rejected segment did not move the robot, suggests smaller displacements,
and distinguishes IK from collision recovery. It never chooses a robot action
or bypasses the checker. This is a rejection-feedback experiment, not an IK fix.

Clean RGB uses the exact same calibrated crops, camera poses, resolution, image
history, and deprojection. It removes only annotations. A prompt override
explains that grid lines and reticles are absent. The original general prompt
otherwise remains intact; there is no shorter or task-focused replacement.

All trials: local Glimmer, high effort, 16,384 response tokens, 80 decisions,
900 physics steps, Task 0/state 0, two image pairs, context 131,072, existing
v3 motion/checking settings. Requests use generation seed 42 + decision number
in every variant. This controls sampling seeds; GPU inference is not guaranteed
bitwise deterministic. One state per variant cannot establish a success rate.
If a variant improves first-grasp and rejection metrics, validate on held-out
states/tasks before adopting it. No paid API is used.

```bash
python scripts/queue_harness_ablations.py \
  --output /metadisk/may/muse-glimmer/harness_xhigh_results \
  --campaign-id general-ablation-v1 --effort high
```

The queue runs sequentially and each child holds the existing global model and
trial locks. It preserves all historic trials in the harness viewer. Its JSON
state updates every ten seconds, with PID, last progress timestamp, per-status
counts, first grasp proposal, first completed grasp primitive, physics steps,
and environment success. Proposed/completed grasps are not task success.
There are no automatic experiment retries. A process error, interrupted trial,
or twenty minutes without API/turn progress pauses the queue for inspection.
Restart with identical arguments skips completed variants and refuses to rerun
an interrupted variant silently.

Live comparison: http://127.0.0.1:8767/ablations.html. Historic trials,
conversations, observations, and videos remain at /report.html on that viewer.

## Visual-only follow-up

`visual-ablation-v1` compares `clean-images` (fresh control), `clean-bright`
(fixed gamma 0.8), and `clean-contrast` (Pillow autocontrast, 1% cutoff,
preserve_tone=True). These variants all disable recovery, remove overlays,
and use the identical general prompt and motion/checking settings. Only pixel
intensities differ. Crops, resolution, camera calibration, deprojection, and
image history remain identical. Normalization cannot recover hidden labels,
missing image detail, or objects outside the camera. No generative enhancement,
scene-specific tuning, or task-specific prompt is used.

```bash
python scripts/queue_harness_ablations.py \
  --output /metadisk/may/muse-glimmer/harness_xhigh_results \
  --campaign-id visual-ablation-v1 --effort high \
  --variants clean-images clean-bright clean-contrast
```

As before, each trial uses Task 0/state 0, high effort, 80 decisions, 900 physics
steps, 16,384 output tokens, and generation seed 42 + decision number. All run
sequentially and appear in the existing comparison viewer. The crop does not
cause the initially clipped basket; the raw fixed-camera frame is already
clipped there. A full-frame toggle would be a no-op at the initial state, so it
is not part of this experiment.

## DeepSeek max-effort comparison

The Haimaker route requests `deepseek/deepseek-v4.1-flash`, thinking enabled,
`reasoning_effort=max`. This is the highest documented native level (ultra maps
to max in DeepSeek docs). The API's internal effort is not independently
observable; exact requests/responses are retained. No local template kwargs or
unsupported sampling seed are sent. Authentication is never included in captures.

`queue_deepseek_harness.py` waits for all three visual trials to finish, then
runs Task 0 states 0,1,2 sequentially with clean RGB, the existing general
prompt, 80 calls/900 physics steps per episode, and a 16,384-token output cap.
The prompt substitutes only the model identity. There is no task-specific
solution prompt, demonstration, or cross-episode memory.

All three episodes share a $5 peak-list-price accounting cap (not a promotional
invoice). Each request reserves full-context input plus maximum output at
$0.30/M input and $1.20/M output; known usage reconciles that reservation.
Unknown usage retains its reservation and stops the run; errors are not retried.
An insufficient cap can stop the batch before all three episodes complete.
The model's advertised context is 1M; we conservatively reserve 1,048,576 input
tokens. This accounting ignores cache discounts rather than assuming free use.

```bash
python scripts/queue_deepseek_harness.py \
  --output /metadisk/may/muse-glimmer/harness_xhigh_results \
  --env-file /path/to/your/.env
```

The existing harness viewer records model, provider, requested effort, raw
commands, feedback, tokens, estimated cost, and task outcome for these trials.
Credentials and raw outputs remain local. No manipulation result is implied by
passing provider-transport unit tests.

## Xhigh visual rerun

The initial visual experiments used high effort. The requested rerun is
`visual-ablation-xhigh-v1`, with the same three variants on Task 0/state 0,
80 decisions, 900 steps, 16,384 output tokens, and generation seed 42 + decision
number. Only reasoning effort changes to xhigh. Older high results are retained.
The local harness runner and ablation queue now default to xhigh; older high
experiment reproduction commands above explicitly specify --effort high.

```bash
python scripts/queue_harness_ablations.py \
  --output /metadisk/may/muse-glimmer/harness_xhigh_results \
  --campaign-id visual-ablation-xhigh-v1 --effort xhigh \
  --variants clean-images clean-bright clean-contrast
```
