"""Launching Codex for an episode: command lines, environment, config block, audit.

Codex runs on your ChatGPT plan (interactive ``codex`` or headless
``codex exec``), not on an API key. The ``libero`` MCP server is registered
once in ``~/.codex/config.toml`` by ``scripts/02_install_codex_config.py``;
per-episode settings travel as ``LAM_*`` environment variables, which that
config forwards to the server.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FORWARDED_ENV = (
    "LAM_TASK_ID",
    "LAM_INIT_STATE",
    "LAM_RUN_TAG",
    "LAM_OUTPUT_ROOT",
    "LAM_MEMORY",
    "LAM_MEMORY_FILE",
    "LAM_MEMORY_FROM",
    "LAM_MAX_CALLS",
    "LAM_FAKE",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "MUJOCO_EGL_DEVICE_ID",
    "CUDA_VISIBLE_DEVICES",
    "LIBERO_CONFIG_PATH",
    "LIBERO_PYTHON",
)

BLOCK_START = "# >>> llm-agent-memory >>>"
BLOCK_END = "# <<< llm-agent-memory <<<"

EFFORTS = ("minimal", "low", "medium", "high")

# Event types in `codex exec --json` output that mean Codex acted outside the
# robot tools. Matching is on the raw line, so it survives schema changes.
OFF_PROTOCOL = re.compile(r'"(command_execution|exec_command\w*|local_shell\w*|web_search\w*|file_change|patch_apply\w*)"')


def find_codex() -> str:
    """Path to the ``codex`` binary, or exit with install instructions."""
    path = shutil.which("codex")
    if path is None:
        sys.exit("codex not found on PATH. Install it with: npm install -g @openai/codex")
    return path


def config_block(python: str = sys.executable, repo: Path = REPO) -> str:
    """The ``[mcp_servers.libero]`` table, wrapped in markers so it can be replaced."""
    env_list = ", ".join(f'"{name}"' for name in FORWARDED_ENV)
    return (
        f"{BLOCK_START}\n"
        "[mcp_servers.libero]\n"
        f"command = {json.dumps(python)}\n"
        'args = ["-m", "llm_agent_memory.mcp_server"]\n'
        f"cwd = {json.dumps(str(repo))}\n"
        "startup_timeout_sec = 120.0\n"
        "# LIBERO downloads ~400 MB of assets on first use and a long move can take a while.\n"
        "tool_timeout_sec = 900.0\n"
        f"env_vars = [{env_list}]\n"
        f"{BLOCK_END}\n"
    )


def install_block(config_path: Path, block: str) -> str:
    """Insert or replace the marked block in a Codex config file. Returns what happened."""
    text = config_path.read_text() if config_path.exists() else ""
    pattern = re.compile(re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n?", re.S)
    if pattern.search(text):
        new_text, action = pattern.sub(block, text), "replaced"
    else:
        if "[mcp_servers.libero]" in text:
            raise SystemExit(
                f"{config_path} already has a [mcp_servers.libero] table that this script did "
                "not write. Remove it, then run this script again."
            )
        new_text, action = text + ("\n" if text and not text.endswith("\n") else "") + "\n" + block, "added"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        backup = config_path.with_suffix(".toml.bak")
        if not backup.exists():
            shutil.copy2(config_path, backup)
    config_path.write_text(new_text)
    return action


def codex_home() -> Path:
    """``$CODEX_HOME`` or ``~/.codex``."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def session_env(
    task_id: int,
    init_state: int,
    tag: str,
    memory: str = "off",
    fake: bool = False,
    output_root: Path | None = None,
    memory_file: Path | None = None,
    memory_from: Path | None = None,
) -> dict[str, str]:
    """Environment for one Codex episode (the MCP server reads these)."""
    env = dict(os.environ)
    env.update(
        LAM_TASK_ID=str(task_id),
        LAM_INIT_STATE=str(init_state),
        LAM_RUN_TAG=tag,
        LAM_OUTPUT_ROOT=str(output_root or REPO / "outputs" / "codex_runs"),
        LAM_MEMORY=memory,
        LAM_MEMORY_FILE=str(memory_file or memory_path("interactive", fake)),
        LAM_MEMORY_FROM=str(memory_from) if memory_from else "",
        LAM_FAKE="1" if fake else "0",
    )
    return env


def memory_path(name: str, fake: bool = False) -> Path:
    """Hindsight file for a batch (or for interactive runs). Fake runs never share a real file."""
    return REPO / "outputs" / "memory" / f"{name}{'_fake' if fake else ''}.jsonl"


def model_args(model: str | None, effort: str | None) -> list[str]:
    """``-m`` and reasoning-effort overrides, when given."""
    args: list[str] = []
    if model:
        args += ["-m", model]
    if effort:
        if effort not in EFFORTS:
            raise SystemExit(f"--effort must be one of {EFFORTS}")
        args += ["-c", f'model_reasoning_effort="{effort}"']
    return args


def interactive_command(
    codex: str, workspace: Path, message: str, model: str | None, effort: str | None, no_approvals: bool
) -> list[str]:
    """The ``codex`` TUI in the episode workspace, opened with the first message."""
    cmd = [codex, "--cd", str(workspace), "--sandbox", "read-only", *model_args(model, effort)]
    if no_approvals:
        cmd += ["--ask-for-approval", "never"]
    return [*cmd, message]


def exec_command(
    codex: str, workspace: Path, message: str, model: str | None, effort: str | None
) -> list[str]:
    """Headless ``codex exec`` with JSONL events on stdout."""
    return [
        codex,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(workspace),
        "--sandbox",
        "read-only",
        *model_args(model, effort),
        message,
    ]


def count_off_protocol(events_path: Path) -> int:
    """How many event lines in a ``codex exec --json`` log show non-robot actions."""
    if not events_path.exists():
        return 0
    return sum(1 for line in events_path.read_text().splitlines() if OFF_PROTOCOL.search(line))


def parse_ids(spec: str, upper: int) -> list[int]:
    """Parse ``"0-9"``, ``"2"`` or ``"0,3,5-7"`` into a sorted list within ``0..upper-1``."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            ids.update(range(low, high + 1))
        else:
            ids.add(int(part))
    bad = [i for i in ids if not 0 <= i < upper]
    if bad:
        raise SystemExit(f"ids out of range 0..{upper - 1}: {sorted(bad)}")
    return sorted(ids)
