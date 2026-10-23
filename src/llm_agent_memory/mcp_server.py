"""MCP server that lets a Codex session drive one LIBERO-10 episode.

Codex launches this over stdio (see ``scripts/02_install_codex_config.py``).
The robot tools are the paper's three tools with the upstream schemas,
descriptions and validation unchanged. Two extra tools manage the episode:
``start_episode`` (allowed once per session) and ``episode_status`` (free, does
not use a call).

Settings come from environment variables, which the launch scripts set and
the Codex config forwards:

=================  ============================================================
LAM_TASK_ID        LIBERO-10 task this session must run (0-9)
LAM_INIT_STATE     initial state this session must run (0-49)
LAM_RUN_TAG        name of the output folder for this episode
LAM_OUTPUT_ROOT    where episode folders go (default ``outputs/codex_runs``)
LAM_MEMORY         ``off`` (zero-shot, default), ``same_task`` or ``all_tasks``
LAM_MEMORY_FILE    hindsight store this session writes (one file per batch)
LAM_MEMORY_FROM    another batch's hindsight file, read but never written
LAM_MAX_CALLS      robot tool-call budget (default 80, as in the paper)
LAM_FAKE           ``1`` to use the fake simulator instead of LIBERO
=================  ============================================================
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from llm_agent_memory.episode import ROBOT_TOOLS, Episode, Reply, libero_factory
from llm_agent_memory.memory import MODES, HindsightStore
from llm_agent_memory.protocol import MAX_CALLS

SERVER_NAME = "libero"

INSTRUCTIONS = (
    "Controls one simulated LIBERO-10 robot episode. Call start_episode once, then "
    "control the robot with move_to, done and give_up, one call at a time, looking at "
    "each result before the next call."
)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


class Session:
    """State for one MCP session: at most one episode, started once."""

    def __init__(self) -> None:
        self.task_id = _env_int("LAM_TASK_ID")
        self.init_state = _env_int("LAM_INIT_STATE")
        self.run_tag = os.environ.get("LAM_RUN_TAG") or time.strftime("%Y%m%d-%H%M%S")
        self.output_root = Path(os.environ.get("LAM_OUTPUT_ROOT") or "outputs/codex_runs")
        self.memory_mode = os.environ.get("LAM_MEMORY", "off").strip() or "off"
        if self.memory_mode not in MODES:
            raise SystemExit(f"LAM_MEMORY must be one of {MODES}, got {self.memory_mode!r}")
        extra = os.environ.get("LAM_MEMORY_FROM", "").strip()
        self.memory = HindsightStore(
            Path(os.environ.get("LAM_MEMORY_FILE") or "outputs/memory/unspecified.jsonl"),
            read_also=[Path(extra)] if extra else None,
        )
        self.max_calls = _env_int("LAM_MAX_CALLS") or MAX_CALLS
        if os.environ.get("LAM_FAKE", "") == "1":
            from llm_agent_memory.fake import fake_factory

            self.factory: Any = fake_factory
        else:
            self.factory = libero_factory
        self.episode: Episode | None = None
        self.stored = False
        # Schemas only need the action/observation spaces, which do not depend on the task.
        probe = Episode(0, 0, self.output_root / "_probe", factory=self._schema_factory)
        self.schemas = {s["function"]["name"]: s["function"] for s in probe.tool_schemas()}

    def _schema_factory(self, task_id: int, run_dir: Path) -> Any:
        # Building the real embodiment does not start the simulator, so this is cheap.
        return self.factory(task_id, run_dir)

    def tools(self) -> list[types.Tool]:
        """start_episode, the three robot tools, and episode_status."""
        start = types.Tool(
            name="start_episode",
            description=(
                "Reset the simulator and start the episode. Returns the goal and the "
                "initial observation. Allowed once per session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "LIBERO-10 task, 0-9"},
                    "init_state": {"type": "integer", "description": "Initial state, 0-49"},
                },
            },
        )
        robot = [
            types.Tool(
                name=name,
                description=self.schemas[name]["description"],
                inputSchema=self.schemas[name]["parameters"],
            )
            for name in ROBOT_TOOLS
        ]
        status = types.Tool(
            name="episode_status",
            description="Calls used, control steps used, and whether the episode is over.",
            inputSchema={"type": "object", "properties": {}},
        )
        return [start, *robot, status]

    def start(self, arguments: dict[str, Any]) -> Reply:
        """Start the session's single episode, enforcing the launch script's task and state."""
        if self.episode is not None:
            return Reply("error: this session already started its episode; it cannot be restarted")
        task_id = arguments.get("task_id", self.task_id)
        init_state = arguments.get("init_state", self.init_state)
        if task_id is None or init_state is None:
            return Reply("error: pass task_id and init_state")
        if self.task_id is not None and task_id != self.task_id:
            return Reply(f"error: this session is set up for task_id={self.task_id}")
        if self.init_state is not None and init_state != self.init_state:
            return Reply(f"error: this session is set up for init_state={self.init_state}")
        started = time.time()
        notes = self.memory.render(self.memory_mode, int(task_id), before=started)
        run_dir = self.output_root / f"{self.run_tag}_task{int(task_id):02d}_init{int(init_state):02d}"
        self.episode = Episode(
            int(task_id),
            int(init_state),
            run_dir,
            factory=self.factory,
            max_calls=self.max_calls,
            prior_notes=notes,
            meta={"policy": "codex", "memory_mode": self.memory_mode, "run_tag": self.run_tag},
        )
        return self.episode.start()

    def call(self, name: str, arguments: dict[str, Any]) -> Reply:
        """Dispatch one tool call."""
        if name == "start_episode":
            return self.start(arguments)
        if self.episode is None:
            return Reply("error: call start_episode first")
        if name == "episode_status":
            status = self.episode.status()
            return Reply("\n".join(f"{key}: {value}" for key, value in status.items()))
        if name not in ROBOT_TOOLS:
            return Reply(f"error: unknown tool {name!r}")
        reply = self.episode.call(name, arguments)
        if self.episode.over and not self.stored:
            self.stored = True
            episode = self.episode
            self.memory.add(
                {
                    "task_id": episode.task_id,
                    "init_state": episode.init_state,
                    "outcome": episode.outcome,
                    "success": episode.success,
                    "memory_mode": self.memory_mode,
                    "run_tag": self.run_tag,
                    "hindsight": episode.hindsight,
                }
            )
            episode.close()  # flushes rollout.mp4 now rather than when Codex exits
        return reply

    def close(self) -> None:
        """Release the simulator if an episode was started."""
        if self.episode is not None:
            self.episode.close()


def to_content(reply: Reply) -> list[types.TextContent | types.ImageContent]:
    """Text first, then one image block per camera, plus the PNG paths on disk."""
    content: list[types.TextContent | types.ImageContent] = [
        types.TextContent(type="text", text=reply.text)
    ]
    for frame in reply.frames:
        content.append(types.TextContent(type="text", text=f"{frame.camera} camera:"))
        content.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(frame.png).decode("ascii"),
                mimeType="image/png",
            )
        )
    if reply.frames:
        paths = ", ".join(f"{f.camera}={f.path.resolve()}" for f in reply.frames)
        content.append(types.TextContent(type="text", text=f"Frames saved: {paths}"))
    return content


def build_server(session: Session) -> Server:
    """Wire a ``Session`` into an MCP low-level server."""
    server: Server = Server(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return session.tools()

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent | types.ImageContent]:
        reply = await anyio.to_thread.run_sync(session.call, name, dict(arguments or {}))
        return to_content(reply)

    return server


async def _serve() -> None:
    session = Session()
    server = build_server(session)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        session.close()


def main() -> None:
    """Entry point: ``python -m llm_agent_memory.mcp_server``."""
    anyio.run(_serve)


if __name__ == "__main__":
    main()
