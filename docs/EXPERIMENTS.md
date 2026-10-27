# Muse Glimmer experiment review

This repository contains two separate evaluation paths. Their raw outputs are
kept in separate campaigns, and an attempt's files are never overwritten.

| Setting | Original-policy baseline | Adapted robot harness |
|---|---|---|
| Model | Muse Glimmer 30B, official Q4_K_M | Same model |
| Reasoning | xhigh | Early xhigh; v3 and ablations high |
| Task coverage | Tasks 0–9, states 0–9: 100 episodes | Task 0 exploratory trials |
| Robot interface | Original `move_to`, `done`, `give_up` | Parsed programs of motion primitives |
| Images | Raw agentview + wrist, 256×256 | 512×512; overlays or clean RGB by variant |
| Image history | Latest two observations | Latest two observations |
| Text history | Full episode; fresh policy each episode | Full episode; fresh context each episode |
| Limits | 80 model calls, 900 physics steps, 20 Hz | Same call/physics limits |
| Response allowance | 16,384 tokens including reasoning | 16,384 tokens |
| Success | LIBERO predicate | LIBERO predicate |
| Demonstrations / cross-episode memory | None | None |

The baseline preserves the original policy, prompt structure, embodiment and
motion adapter. It is **not an identical hosted-API reproduction**: it uses a
local Chat API, quantized model weights, and the user-selected 16,384-token
allowance rather than the paper draft's 3,072-token allowance. Sampling defaults
and actual context size are archived in each new run's protocol.

## Read the code in this order

1. [Baseline runner](../scripts/run_local.py): local-only inference transport,
   exact saved requests/responses, upstream agent policy, scoring and seeds.
2. [Harness runner](../scripts/run_harness.py): prompt adaptation, observation
   history, parsed program execution and per-turn feedback.
3. [Session](../src/llm_agent_memory/harness/session.py): primitive expansion,
   measured pose tracking, grasp/holding proxy, step accounting and settling.
4. [Geometry](../src/llm_agent_memory/harness/geometry.py): calibrated overlays,
   crop-aware projection and ray/plane conversion.
5. [Simulator checks](../src/llm_agent_memory/harness/worker_support.py): camera
   calibration, depth rendering, Panda IK and sampled contact rejection.
6. [Program parser](../src/llm_agent_memory/harness/plan_program.py) and
   [upstream interface](../src/llm_agent_memory/harness/upstream.py): restricted
   action language; model text is never executed as shell or arbitrary Python.
7. [Sequential queue](../scripts/queue_baseline.py) and
   [watchdog](../scripts/watch_baseline.py): recovery, single-worker lock,
   archived interruptions, and bounded retries.

[Upstream provenance and differences](../src/llm_agent_memory/harness/UPSTREAM.md)
identify the source commit and the single-Panda adaptation. The blog's recorded
successes included human motion review; our recorded trials use automatic
checks. Their tasks, camera geometry and model also differ.

## Results available for review

See [latest results and interpretation](LATEST_RESULTS.md),
[ablation protocols](HARNESS_ABLATIONS.md), [ablation results](harness_ablation_results.json),
and [pointing results](pointing_results_snapshot.json).

- [Snapshot metadata and aggregate results](results_snapshot.json)
- [Per-state baseline snapshot](baseline_results_snapshot.csv)
- [Recorded harness programs and step costs](harness_turns_snapshot.csv)

These are a dated snapshot, not a continuously updated benchmark. Pending
episodes and infrastructure errors remain separate from completed failures.
Videos, images, weights, full model traffic and credentials are excluded from
Git. They remain in the local campaign output directories. No new manipulation
success is implied by a completed primitive or a passing wiring test.

The original harness state-0 trial reached 900 steps in 26 model calls without
task success. Its feedback includes five failed grasps, one drop and five
collision rejections. State 1 was interrupted after observation-only turns.
The efficiency-v2 state-0 trial stopped when its 33,091-token request exceeded
the then-configured 32,768 context, after 48 physics steps. It is an incomplete
experiment. The baseline subsequently uses a 131,072-token context.

## Execution-efficiency change

`measured-settling-v2` skips already-open and already-reached operations, ends
settling on measured convergence, and caps retraction at 0.08 m/s instead of
the previous fixed 0.04 m/s. Contact descent remains 0.04 m/s. Every actual
physics step still counts against 900; the simulator rate is unchanged.

A model-free free-space check measured 24 steps and 2.15 mm endpoint error for
an 8 cm lift, versus the previous 45-step schedule. This does not verify grasp
retention at the higher retraction speed or improved full-task success.

## Review questions

- Does the oblique camera/grid make height-plane mistakes likely? Verify real
  rendered projection/crop geometry, beyond synthetic round-trip tests.
- Should target selection and plane deprojection be an explicit stage before
  hovering, descending, closing and lift verification?
- Does the model identify the requested object correctly? It repeatedly
  confused distractors in the recorded harness trace.
- Does the checker reject valid carrying paths because the preview moves
  the robot while leaving a held object's pose unchanged?
- Is the fingertip contact exception too permissive, and does tracking lag
  reliably indicate guarded contact?
- Is the finger-gap holding estimate stable after closing and lifting?
- Can a complete pickup/place fit within 900 steps without removing checks?

Change one component at a time and compare the same task/state pairs. Use
stage-level outcomes (correct object selection, reached hover, retained grasp,
placement) alongside the final predicate. Keep RGB-only and depth-assisted
variants separately labeled.

## Reproduce a small run

Install the pinned dependencies and model as described in the main README.
Set `MUSE_ROOT` to the directory containing `models/` and `llama.cpp/`.

```bash
bash scripts/serve_model.sh --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0

# In another terminal; run one evaluation at a time:
python scripts/run_local.py --output outputs/baseline_review --tasks 0 --inits 0 \
  --effort xhigh --max-calls 80 --max-steps 900 --image-horizon 2 \
  --max-output-tokens 16384

# Separate experiment, after the baseline episode exits:
python scripts/run_harness.py --output outputs/harness_review --inits 0 \
  --effort xhigh --max-calls 80 --max-steps 900 --max-output-tokens 16384

python scripts/serve_experiment_dashboard.py --output outputs/harness_review --port 8767
python -m pytest -q
```

Use `--append --trial-label <label>` to retain earlier harness trials in the
same viewer. Queue/watchdog tools are Linux-specific; their storage root honors
`MUSE_ROOT`. [Service templates](../deploy/systemd/README.md) document the local
monitoring deployment. Do not restart or modify an active evaluation just to
review this repository.
