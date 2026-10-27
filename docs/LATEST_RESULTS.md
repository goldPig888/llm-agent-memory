# Latest harness results

Snapshot captured at 2026-10-27T22:28:26.571816+00:00.
Glimmer ablations are single-state screens; DeepSeek covers three states. Neither establishes a success rate.
All variants preserve the general prompt; no task-specific solution prompt was added.

| Campaign / variant | Effort | Status | Decisions | Physics steps | First grasp proposed | Task outcome |
|---|---|---|---:|---:|---:|---|
| general-ablation-v1 / control | high | completed | 80 | 872 | — | failure |
| general-ablation-v1 / recovery | high | completed | 73 | 900 | 62 | failure |
| general-ablation-v1 / clean-images | high | completed | 41 | 900 | 3 | failure |
| general-ablation-v1 / combined | high | completed | 33 | 894 | 18 | failure |
| visual-ablation-v1 / clean-images | high | completed | 33 | 894 | 18 | failure |
| visual-ablation-v1 / clean-bright | high | completed | 31 | 900 | 3 | failure |
| visual-ablation-v1 / clean-contrast | high | completed | 80 | 788 | 3 | failure |
| visual-ablation-xhigh-v1 / clean-images | xhigh | completed | 36 | 900 | 5 | failure |
| visual-ablation-xhigh-v1 / clean-bright | xhigh | completed | 41 | 889 | 5 | failure |
| visual-ablation-xhigh-v1 / clean-contrast | xhigh | running | 24 | 567 | 3 | pending |

The completed four-way screen had no full grasp primitive completion or task success.
Clean images reached a grasp proposal at decision 3. Combined preprocessing/recovery
produced seven dropped outcomes during grasp/lift attempts; this does not establish
correct-object retention. Recovery blocked four identical rejected retries.
The visual follow-up compares raw clean RGB, gamma 0.8, and contrast normalization.
Rows still running/queued at capture have no final outcome.

The completed original-policy xhigh baseline remains 0/100 successes (99 failures, 1 error).
The completed pointing diagnostic hit the target mask in 22/50 agentview cases and
9/50 initial-wrist cases, with 8 and 41 abstentions respectively.

## DeepSeek max-effort harness comparison

Three Task 0 states (0,1,2), clean images, 80 calls/900 steps per episode,
16,384 output tokens, Haimaker with max requested. Internal provider effort
is not independently observable. See deepseek_harness_results.json.

| Trial / state | Status | Calls | Steps | Completed grasps | Completed placements |
|---|---|---:|---:|---:|---:|
| deepseek-max-clean-states012 / 0 | failure | 23 | 900 | 1 | 0 |
| deepseek-max-clean-states012 / 1 | failure | 31 | 900 | 1 | 0 |
| deepseek-max-clean-states012 / 2 | failure | 28 | 900 | 1 | 1 |

Estimated peak-list-price bound: $0.5401; not a promotional invoice.

Completed grasp/place primitives are intermediate milestones; the LIBERO
predicate determines full-task success. They do not verify target identity.

## Native-format pointing pilot

Snapshot phase: **waiting_for_xhigh_robot_trials**, 0/10 requests recorded.
States 0–4, agent camera, xhigh. Paired normalized-point vs native-box prompts.
Primary box center uses the predeclared 0–1000 assumption; raw outputs are retained.
See NATIVE_POINTING.md and native_pointing_results.json. No result is claimed while waiting.

## Overlay audit and next experiments

The grid/reticle design follows the upstream report, but the LIBERO renderer is our
implementation. Our dense lines overlap labels; coordinate text is small; the prompt
describes yellow x/cyan y while the renderer currently draws both in yellow.
Proposed next comparisons are reticle-only and a sparse faint grid with a consistent
legend. These overlay redesigns have **not** been implemented or run yet.

## Review on another PC

Pull this repository and read this file, HARNESS_ABLATIONS.md, harness_ablation_results.json,
pointing_results_snapshot.json, deepseek_harness_results.json, native_pointing_results.json,
and harness_turns_snapshot.csv. No local model or simulator
is needed to read the summaries and programs. These committed files are a dated snapshot,
not a live copy of the experiment queue. Videos, observations, weights, and raw API requests
remain on the experiment machine and are not included in Git.
