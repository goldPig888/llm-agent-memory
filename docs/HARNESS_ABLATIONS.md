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
  --campaign-id general-ablation-v1
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
  --campaign-id visual-ablation-v1 \
  --variants clean-images clean-bright clean-contrast
```

As before, each trial uses Task 0/state 0, high effort, 80 decisions, 900 physics
steps, 16,384 output tokens, and generation seed 42 + decision number. All run
sequentially and appear in the existing comparison viewer. The crop does not
cause the initially clipped basket; the raw fixed-camera frame is already
clipped there. A full-frame toggle would be a no-op at the initial state, so it
is not part of this experiment.
