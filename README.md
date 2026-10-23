# llm-agent-memory

Codex as a closed-loop policy on LIBERO-10, with cross-episode memory.

This repo runs the [LLM-LIBERO](https://irvlutd.github.io/LLM-LIBERO/) protocol
with one change: instead of a stateless API call per step, the policy is a
persistent Codex session (your ChatGPT plan, the same Codex you use in the
terminal). The tasks, initial states, embodiment, prompt, tools, 80-call budget
and 900-step limit are the paper's, imported from pinned sources rather than
copied, so Codex results compare directly with the paper's zero-shot baselines.

On top of that, every episode's `hindsight` note (the model's own "what I wish
I had known") can be fed forward to later episodes. In the paper those notes
were logged and never used. Here they are the variable under study.

## Setup on a new machine

Linux with an NVIDIA GPU (headless EGL), `git`, `conda`, and Node.js for Codex.

```bash
git clone https://github.com/goldPig888/llm-agent-memory.git
cd llm-agent-memory
bash scripts/00_setup_env.sh          # pinned sources + conda env + LIBERO + this package
conda activate llm-agent-memory
python scripts/01_check_install.py    # unit tests + one real LIBERO episode, no model calls

npm install -g @openai/codex          # if you do not have it yet
codex login                           # sign in with your ChatGPT plan
python scripts/02_install_codex_config.py
```

## The scripts, in order

| Script | What it does |
|---|---|
| `00_setup_env.sh` | One-time install: clones `vendor.lock` into `vendor/`, then a conda env |
| `01_check_install.py` | Tests plus a scripted LIBERO episode (lift 3 cm, close, open). `--fake` skips LIBERO |
| `02_install_codex_config.py` | Registers the `libero` MCP server in `~/.codex/config.toml`. Safe to rerun |
| `03_open_codex_episode.py` | Opens one episode as a normal interactive Codex chat you can watch |
| `04_run_codex_batch.py` | Runs many episodes headless with `codex exec`, resumable |
| `05_summarize.py` | Success per batch and per task, next to the paper's rates |

### First run: check Codex can see the cameras

```bash
python scripts/03_open_codex_episode.py --task 2 --init 0 --fake
```

The fake simulator draws a red square where the gripper is. Once Codex has
called `start_episode`, ask it what it sees in the camera images. If it cannot
describe a red square, it is not receiving the MCP image blocks and will fall
back to opening the saved PNGs (the AGENTS.md rules allow exactly that). Fix
this before running real episodes, since a blind policy is not the experiment.

### One real episode, interactively

```bash
python scripts/03_open_codex_episode.py --task 2 --init 0 --effort high
```

### A pilot batch

Task 2 (33% pooled in the paper) leaves room to move in both directions:

```bash
python scripts/04_run_codex_batch.py --label pilot_zs --tasks 2 --inits 0-9 --effort high
python scripts/05_summarize.py --label pilot_zs
```

### Memory experiments

```bash
# zero-shot baseline, then the same episodes with notes from it
python scripts/04_run_codex_batch.py --label zs  --tasks 2 --inits 0-9
python scripts/04_run_codex_batch.py --label mem --tasks 2 --inits 0-9 \
    --memory same_task --memory-from zs
```

| `--memory` | What the episode sees |
|---|---|
| `off` | Nothing extra. Zero-shot, identical to the paper (default) |
| `same_task` | Up to 5 most recent notes from earlier episodes of the same task |
| `all_tasks` | Up to 5 most recent notes from earlier episodes of any task |

Each batch writes its own notes to `outputs/memory/<label>.jsonl`.
`--memory-from` adds another batch's file as a read-only source. An episode
only sees notes written before it started. The header text is the upstream
agent's `prior_learnings` wording.

## How it works

```text
codex (your plan) ── MCP stdio ──> llm_agent_memory.mcp_server
                                      │  start_episode, move_to, done, give_up, episode_status
                                      v
                                   Episode (Algorithm 1 of the paper)
                                      │  upstream agent Toolset: same validation and interpolation
                                      v
                                   LiberoEmbodiment (pinned LLM_LIBERO) ──> LIBERO / MuJoCo
```

Each episode gets a workspace, `workspaces/<time>_<label>_taskXX_initYY/`,
whose `AGENTS.md` holds the paper's system prompt verbatim plus the session
rules. Codex runs there with a read-only sandbox. The MCP server allows one
`start_episode` per session, on the task and initial state the launch script
assigned, so an episode cannot be restarted or swapped for an easier one.

Outputs per episode, in `outputs/codex_runs/<same folder name>/`:

```text
protocol.json    task, prompt, limits, memory notes shown (if any)
episode.jsonl    every tool call, its arguments, and what the model was told
frames/          the PNG of every observation, per camera
result.json      outcome, success, calls, steps, final hindsight note
<time>_taskXX_initYY/rollout.mp4   agentview | wrist video, 20 fps
```

## Differences from the paper's API runs

| | Paper (API) | This repo (Codex) |
|---|---|---|
| Model access | Provider API, metered | Codex on a ChatGPT plan |
| Context | Full text history, images from the last 2 observations | Whatever Codex keeps (persistent session, its own compaction) |
| Budget | 80 LLM calls | 80 robot tool calls; Codex's own reasoning turns are not counted |
| Tokens and cost | Measured per call | Not reported on a plan |
| Prompt, tools, limits, scoring | | Identical (imported from the same pinned code) |

The context row is the point of the experiment. The budget and cost rows mean
Codex numbers are comparable on success rate, not on efficiency. For cost
numbers, the same loop has to run against the API.

## Keeping episodes honest

Codex can run shell commands, and a read-only sandbox still lets it read
files, including LIBERO's task files with object positions. Three guards:

1. AGENTS.md forbids shell, file, web and code use (except opening the frame PNGs).
2. Codex runs with `--sandbox read-only`.
3. `04` saves Codex's event stream and counts any shell, file or web action
   as `off_protocol`. `05` lists those episodes. Exclude them from results.

Interactive runs (`03`) have no event log, so the audit only covers batches.
Use `03` for watching and debugging, `04` for numbers.

## Pinned sources

`vendor.lock` names two repos and the exact commits to use. `00_setup_env.sh`
clones them into `vendor/` (which git ignores) and checks those commits out.

| Repo | Commit | Why |
|---|---|---|
| `inspect-robots` | `4d35fe4` | The framework commit the paper's runs used (`upstream_commit.txt` in LLM_LIBERO) |
| `LLM_LIBERO` | `7db834a` | LIBERO-10 embodiment, tasks and prompt from the paper |

To move to newer commits, edit `vendor.lock`, rerun `00_setup_env.sh`, and
rerun `python -m pytest`. The prompt test fails if the paper's prompt changed.

## Layout

```text
scripts/                 00 to 05, run in order
src/llm_agent_memory/
  protocol.py            prompt, limits, task list (imported, not retyped)
  episode.py             one episode: tool calls in, observations out
  mcp_server.py          the Codex-facing MCP server
  memory.py              hindsight store and the memory modes
  workspace.py           per-episode AGENTS.md
  codex.py               Codex command lines, config block, audit
  fake.py                LIBERO stand-in for wiring checks
tests/                   no simulator or model needed
vendor.lock              pinned upstream commits (cloned into vendor/)
```

## Citation

Built on

```bibtex
@misc{espinola2026reasoningaction,
  title  = {From Reasoning to Action: Benchmarking Frontier AI Models on LIBERO-10 Robot Manipulation Tasks},
  author = {Espinola, May and Zhang, Qifan and Xiang, Yu},
  year   = {2026},
  url    = {https://irvlutd.github.io/LLM-LIBERO/}
}
```
