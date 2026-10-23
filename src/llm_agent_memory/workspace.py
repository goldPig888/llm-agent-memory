"""Per-episode Codex workspaces: a folder holding the AGENTS.md Codex reads on start.

Codex runs with its working directory set to the workspace, so the AGENTS.md
there acts as the system prompt. It carries the paper's prompt verbatim plus
the rules that keep a Codex episode comparable to an API episode (act only
through the robot tools, no shell, no reading the simulator's files).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_agent_memory.protocol import MAX_CALLS, instruction, system_prompt

CODEX_RULES = """\
## Rules for this session

You are being evaluated. Act on the robot only through the `libero` MCP tools.

1. First call `start_episode` with task_id={task_id} and init_state={init_state}.
2. Then make exactly one robot tool call (`move_to`, `done`, or `give_up`) at a time, and
   read its result (text and camera images) before deciding the next one.
3. Do not run shell commands, read or write files, search the web, or write code. All
   you know about the scene must come from the tool results. The simulator's source,
   task files and logs are off limits.
4. If a tool result lists saved frame paths but no images are visible to you, open
   exactly those PNG paths with your image viewer. Nothing else on disk.
5. The episode ends when a tool result says it is over. Then stop and reply with one
   line: the outcome reported by the tool.
"""


def episode_tag(label: str = "") -> str:
    """Timestamp prefix shared by the workspace and the MCP server's output folder."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"_{label}" if label else ""
    return f"{stamp}{suffix}"


def agents_md(task_id: int, init_state: int, max_calls: int = MAX_CALLS) -> str:
    """AGENTS.md text for one episode."""
    return (
        f"# LIBERO-10 episode: task {task_id}, initial state {init_state}\n\n"
        f"Goal: {instruction(task_id)}\n\n"
        "## Robot instructions\n\n"
        "```text\n" + system_prompt(task_id, max_calls) + "\n```\n\n"
        + CODEX_RULES.format(task_id=task_id, init_state=init_state)
    )


def first_message(task_id: int, init_state: int) -> str:
    """The opening user message for a Codex episode."""
    return (
        f"Start the LIBERO-10 episode now: call start_episode with task_id={task_id} "
        f"and init_state={init_state}, then control the robot following AGENTS.md."
    )


def make_workspace(
    root: Path, task_id: int, init_state: int, tag: str, max_calls: int = MAX_CALLS
) -> Path:
    """Create ``root/<tag>_taskXX_initYY`` with AGENTS.md and a small manifest."""
    path = Path(root) / f"{tag}_task{task_id:02d}_init{init_state:02d}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "AGENTS.md").write_text(agents_md(task_id, init_state, max_calls))
    (path / "workspace.json").write_text(
        json.dumps({"task_id": task_id, "init_state": init_state, "tag": tag}, indent=2) + "\n"
    )
    return path
