"""03: open one episode as an interactive Codex chat (the way you normally use Codex).

    python scripts/03_open_codex_episode.py --task 2 --init 0
    python scripts/03_open_codex_episode.py --task 2 --init 0 --effort high --memory same_task
    python scripts/03_open_codex_episode.py --task 2 --init 0 --fake   # wiring test, no LIBERO

Creates workspaces/<time>_taskXX_initYY/ with an AGENTS.md holding the paper's
prompt and the episode rules, then starts `codex` there with the first message
already sent. You watch the chat; the robot's outputs (frames, episode.jsonl,
result.json, rollout.mp4) land in outputs/codex_runs/ under the same folder name.

Interactive runs are for prototyping and watching. For numbers, use 04.
"""

from __future__ import annotations

import argparse
import os

from llm_agent_memory.codex import REPO, find_codex, interactive_command, session_env
from llm_agent_memory.memory import MODES
from llm_agent_memory.workspace import episode_tag, first_message, make_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, required=True, help="LIBERO-10 task id, 0-9")
    parser.add_argument("--init", type=int, default=0, help="initial state, 0-49")
    parser.add_argument("--model", help="Codex model id (default: your Codex default)")
    parser.add_argument("--effort", help="reasoning effort: minimal, low, medium, high")
    parser.add_argument("--memory", default="off", choices=MODES)
    parser.add_argument("--fake", action="store_true", help="fake simulator (no LIBERO)")
    parser.add_argument("--label", default="", help="tag added to the folder name")
    parser.add_argument("--no-approvals", action="store_true", help="pass --ask-for-approval never")
    args = parser.parse_args()

    codex = find_codex()
    tag = episode_tag(args.label)
    workspace = make_workspace(REPO / "workspaces", args.task, args.init, tag)
    env = session_env(args.task, args.init, tag, memory=args.memory, fake=args.fake)
    cmd = interactive_command(
        codex, workspace, first_message(args.task, args.init), args.model, args.effort, args.no_approvals
    )
    print(f"[03] workspace: {workspace}")
    print(f"[03] outputs:   {REPO / 'outputs' / 'codex_runs' / workspace.name}")
    print(f"[03] memory={args.memory} fake={args.fake}")
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
