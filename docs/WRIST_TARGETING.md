# Wrist targeting and perception diagnostics

This is a new harness profile. The completed original-policy baseline and
previous harness output directories remain unchanged.

## Changes

`run_harness.py --profile wrist-v3` defaults to high reasoning, with 16,384
response tokens, 80 decisions and 900 physics steps. It describes a two-stage
procedure: coarse agentview depth targeting and a 13 cm hover, followed by
selection in the measured wrist view and wrist ray/plane targeting. The model
must confirm the wrist view is appropriate; wrist angle is reported each turn
and is not assumed to stay overhead after arm rotation.

`deproject` and `deproject_depth` accept `camera="agentview"` or `camera="wrist"`.
Agentview uses its current crop; wrist coordinates refer to the full image.
The restricted parser accepts only those camera-name literals, not general
strings, arbitrary Python or shell execution. Existing calls retain agentview
as their default.

The v3 profile crops the first agentview to the projected tabletop extent,
trims repeated observation fields, and rejects a third consecutive zoom-only
program with action-oriented feedback. Observation-only requests also receive
the limit feedback. This does not force an arbitrary motion if the model keeps
asking to observe; those turns still count against the decision allowance.

Grasp approach defaults to 5 cm. Descent can run at up to 0.08 m/s until the
last 2 cm, which retain the 0.04 m/s contact speed. Existing measured settling
and step accounting remain. Contact previews still precede each segment.

`grasp(..., support_z=...)` records the measured pinch height minus the
specified support height. `place_relative(..., support_z=...)` releases at the
destination support height plus that offset and the selected clearance. It
requires a retained grasp, a recorded support height and approximately
unchanged downward orientation. The model must supply the support height;
this does not infer basket rims or basket-floor geometry.

The checker infers a carried-object attachment only from bilateral finger
contact. During preview it transforms that object's free pose along with the
gripper, checks object/world contacts, and restores the simulator state. A
requested carry without verifiable bilateral contact returns CARRY_UNVERIFIED.
Collision feedback includes waypoint and generic contact categories; it does
not reveal object identities or poses to the model. This is privileged
checker infrastructure and remains separately labeled from the baseline.

## Validated geometry and precision

At Task 0/state 0, measured optical-axis angles below horizontal were 31.922°
for agentview and 86.906° for wrist. The corresponding horizontal error per
unit height error is 1.605 and 0.054. This confirms the proposed sensitivity
mechanism; it does not establish that it caused every recorded grasp failure.

The loaded `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` contains 200 Q4_K and 100 Q6_K
tensors, alongside 506 F32 and 3 BF16 tensors. It is quantized. The server
supports `MUSE_MMPROJ=/path/to/a/different/projector.gguf` for a future precision
ablation. No projector was replaced or dequantized. Converting quantized
weights to FP16 does not restore information lost during quantization.

## Pointing diagnostic

`collect_pointing.py` renders Task 0 states 0–49 at the initial pose after the
standard ten neutral settling steps. There are zero scored action steps. For
each state it records clean 512×512 agentview and wrist images. The target is
the alphabet soup can. Body-center projections and visible-object masks are
stored separately under `scoring/` and never included in model requests.

`run_pointing.py` sends only each sample's raw image and instruction. It records
normalized point predictions or abstentions, visible-mask hit rate, projected
body-center pixel error, parse errors, token usage and latency. Truncations are
recorded as parse errors, not silently retried. The Glimmer-only run uses high
reasoning and a 4,096-token allowance; this is a recognition diagnostic, not a
task-success benchmark. Camera views are tested independently at the initial
pose, not at a post-hover pose. A surface can be geometrically visible while
its identifying label is hidden; interpret abstentions with that limitation.

The optional Haimaker comparison requires an explicit execution flag and a
key, and is capped at $5. The current experiment uses Glimmer only.

```bash
python scripts/collect_pointing.py --output "$MUSE_ROOT/pointing_dataset"
python scripts/run_pointing.py --dataset "$MUSE_ROOT/pointing_dataset" \
  --output "$MUSE_ROOT/pointing_glimmer_high" --models glimmer --effort high
python scripts/serve_pointing.py --dataset "$MUSE_ROOT/pointing_dataset" \
  --results "$MUSE_ROOT/pointing_glimmer_high" --port 8769

# After the pointing test exits; a new, separate trial:
python scripts/run_harness.py --profile wrist-v3 --effort high --inits 0 \
  --output outputs/muse_harness_xhigh --append --trial-label wrist-v3-high-state0
```

`check_targeting.py --output /tmp/targeting_check.json` runs model-free geometry
and synthetic attachment/obstacle fixtures. It verifies projection, collision
classification and exact simulator-state restoration. Teleported objects in
those fixtures are not manipulation successes or policy inputs. Real grasp
retention and placement still need a new full-context experiment.
