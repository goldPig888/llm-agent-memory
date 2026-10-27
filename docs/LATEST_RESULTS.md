# Latest harness results

Snapshot captured at 2026-10-27T17:09:20.751164+00:00.
These are single-state diagnostic trials, not success-rate estimates.
All variants preserve the general prompt; no task-specific solution prompt was added.

| Campaign / variant | Status | Decisions | Physics steps | First grasp proposed | Task outcome |
|---|---|---:|---:|---:|---|
| general-ablation-v1 / control | completed | 80 | 872 | — | failure |
| general-ablation-v1 / recovery | completed | 73 | 900 | 62 | failure |
| general-ablation-v1 / clean-images | completed | 41 | 900 | 3 | failure |
| general-ablation-v1 / combined | completed | 33 | 894 | 18 | failure |
| visual-ablation-v1 / clean-images | running | 23 | 497 | 18 | pending |
| visual-ablation-v1 / clean-bright | queued | — | — | — | pending |
| visual-ablation-v1 / clean-contrast | queued | — | — | — | pending |

The completed four-way screen had no full grasp primitive completion or task success.
Clean images reached a grasp proposal at decision 3. Combined preprocessing/recovery
produced seven dropped outcomes during grasp/lift attempts; this does not establish
correct-object retention. Recovery blocked four identical rejected retries.
The visual follow-up compares raw clean RGB, gamma 0.8, and contrast normalization.
Rows still running/queued at capture have no final outcome.

The completed original-policy xhigh baseline remains 0/100 successes (99 failures, 1 error).
The completed pointing diagnostic hit the target mask in 22/50 agentview cases and
9/50 initial-wrist cases, with 8 and 41 abstentions respectively.

## Overlay audit and next experiments

The grid/reticle design follows the upstream report, but the LIBERO renderer is our
implementation. Our dense lines overlap labels; coordinate text is small; the prompt
describes yellow x/cyan y while the renderer currently draws both in yellow.
Proposed next comparisons are reticle-only and a sparse faint grid with a consistent
legend. These overlay redesigns have **not** been implemented or run yet.

## Review on another PC

Pull this repository and read this file, HARNESS_ABLATIONS.md, harness_ablation_results.json,
pointing_results_snapshot.json, and harness_turns_snapshot.csv. No local model or simulator
is needed to read the summaries and programs. These committed files are a dated snapshot,
not a live copy of the experiment queue. Videos, observations, weights, and raw API requests
remain on the experiment machine and are not included in Git.
