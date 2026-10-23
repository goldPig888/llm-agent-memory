"""One closed-loop LIBERO-10 episode, driven one tool call at a time.

This is Algorithm 1 of the LLM-LIBERO paper with the LLM taken out: whoever
holds an ``Episode`` (the MCP server, a test, a script) supplies the tool
calls. Motion handling is the upstream agent's ``Toolset``, so ``move_to``
validates, bounds and interpolates exactly as it did in the paper's runs.

Episode rules (unchanged from the paper):

* every tool call counts against the ``max_calls`` budget, including calls
  that come back as an error;
* LIBERO's success predicate is checked after every control step, and success
  ends the episode at once, even in the middle of a motion;
* ``done`` and ``give_up`` end the episode as a failure (the task was not
  complete when the model stopped, or the checker would already have ended it);
* the episode also fails at ``max_steps`` control steps, or when the last call
  of the budget is used without success.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots.embodiment import EmbodimentBase
from inspect_robots.types import Observation
from inspect_robots_agent._tools import Toolset, build_toolset
from inspect_robots_agent.policy import _state_lines
from inspect_robots_libero.tasks import libero_10_task
from PIL import Image

from llm_agent_memory.protocol import (
    MAX_CALLS,
    MAX_STEPS,
    check_init_state,
    instruction,
    system_prompt,
)

EmbodimentFactory = Callable[[int, Path], EmbodimentBase]

# The paper's agent policy default (``max_speed_frac``).
MAX_SPEED_FRAC = 0.1

ROBOT_TOOLS = ("move_to", "done", "give_up")


@dataclass(frozen=True)
class _Call:
    """The shape ``Toolset.execute`` expects: a name and JSON-encoded arguments."""

    name: str
    arguments: str


@dataclass(frozen=True)
class Frame:
    """One camera image from the latest observation, saved as a PNG on disk."""

    camera: str
    path: Path
    png: bytes


@dataclass(frozen=True)
class Reply:
    """What the caller gets back from one tool call: text plus camera frames."""

    text: str
    frames: tuple[Frame, ...] = ()
    over: bool = False


def libero_factory(task_id: int, run_dir: Path) -> EmbodimentBase:
    """Build the real LIBERO embodiment; videos and worker logs go to ``run_dir``."""
    from inspect_robots_libero.embodiment import LiberoEmbodiment

    return LiberoEmbodiment(task_id=task_id, output_dir=run_dir)


class Episode:
    """A single LIBERO-10 episode. Create, then ``start()``, then call tools."""

    def __init__(
        self,
        task_id: int,
        init_state: int,
        run_dir: Path,
        factory: EmbodimentFactory = libero_factory,
        max_calls: int = MAX_CALLS,
        max_steps: int = MAX_STEPS,
        prior_notes: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.init_state = check_init_state(init_state)
        self.run_dir = Path(run_dir)
        self.max_calls = max_calls
        self.max_steps = max_steps
        self.prior_notes = prior_notes.strip()
        self.meta = dict(meta or {})
        self.embodiment = factory(task_id, self.run_dir)
        info = self.embodiment.info
        self.toolset: Toolset = build_toolset(
            info.action_space, info.observation_space, info.control_hz, MAX_SPEED_FRAC
        )
        self.calls = 0
        self.steps = 0
        self.outcome: str | None = None  # "success" | "done" | "give_up" | "call_budget" | "step_limit" | "error"
        self.final_text = ""
        self.hindsight = ""
        self.observation: Observation | None = None
        self.started_at = 0.0
        self.ended_at: float | None = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def over(self) -> bool:
        """True once the episode has ended for any reason."""
        return self.outcome is not None

    @property
    def success(self) -> bool:
        """True only if LIBERO's success predicate fired."""
        return self.outcome == "success"

    def tool_schemas(self) -> list[dict[str, Any]]:
        """The upstream OpenAI-format schemas for move_to, done and give_up."""
        return self.toolset.schemas()

    def start(self) -> Reply:
        """Reset the simulator to the initial state and return the goal and first observation."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "frames").mkdir(exist_ok=True)
        scene = libero_10_task(self.task_id, num_scenes=1, start=self.init_state).scenes[0]
        self.started_at = time.time()
        self.observation = self.embodiment.reset(scene)
        self._write_json(
            "protocol.json",
            {
                "task_id": self.task_id,
                "init_state": self.init_state,
                "instruction": instruction(self.task_id),
                "max_calls": self.max_calls,
                "max_steps": self.max_steps,
                "zero_shot": not self.prior_notes,
                "prior_notes": self.prior_notes,
                "system_prompt": system_prompt(self.task_id, self.max_calls),
                **self.meta,
            },
        )
        lines = [f"Goal: {instruction(self.task_id)}"]
        if self.prior_notes:
            lines += ["", self.prior_notes]
        lines += ["", "Initial observation:", *self._state_text()]
        reply = Reply("\n".join(lines), self._frames("call00"))
        self._log({"call": 0, "tool": "start", "result": reply.text})
        return reply

    def close(self) -> None:
        """Release the simulator. Safe to call more than once."""
        self.embodiment.close()

    # -------------------------------------------------------------- tool calls

    def call(self, name: str, arguments: dict[str, Any]) -> Reply:
        """Run one robot tool call (move_to, done, give_up) and report what happened."""
        if self.observation is None:
            return Reply("error: the episode has not started")
        if self.over:
            return Reply(f"error: the episode is over ({self.outcome})", over=True)

        self.calls += 1
        tag = f"call{self.calls:02d}"
        result = self.toolset.execute(_Call(name, json.dumps(arguments)), self.observation)

        if result.error is not None:
            text = f"error: {result.error}"
        elif name in ("done", "give_up"):
            self.final_text = str(arguments.get("summary") or arguments.get("reason") or "")
            self.hindsight = str(arguments.get("hindsight") or "").strip()
            self._end(name)
            text = f"{name} recorded. The episode is over."
        else:
            assert result.chunk is not None
            text = self._play(result.chunk.actions, result.target)

        if not self.over and self.calls >= self.max_calls:
            self._end("call_budget")
            text += f"\nThe budget of {self.max_calls} calls is used up. The episode is over."

        frames: tuple[Frame, ...] = ()
        if not self.over or self.outcome in ("success", "step_limit", "call_budget"):
            frames = self._frames(tag)
            text += "\n" + "\n".join(self._state_text())
        text += f"\n[calls used: {self.calls}/{self.max_calls}, control steps: {self.steps}/{self.max_steps}]"
        self._log(
            {
                "call": self.calls,
                "tool": name,
                "arguments": arguments,
                "result": text,
                "steps": self.steps,
                "outcome": self.outcome,
            }
        )
        return Reply(text, frames, over=self.over)

    def status(self) -> dict[str, Any]:
        """Counters and outcome; reading it does not use a call."""
        return {
            "task_id": self.task_id,
            "init_state": self.init_state,
            "calls": self.calls,
            "max_calls": self.max_calls,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "over": self.over,
            "outcome": self.outcome,
        }

    # --------------------------------------------------------------- internals

    def _play(self, actions: Any, target: Any) -> str:
        planned = len(actions)
        played = 0
        for action in actions:
            step = self.embodiment.step(action)
            self.steps += 1
            played += 1
            self.observation = step.observation
            if step.terminated and step.termination_reason == "success":
                self._end("success")
                return f"The task succeeded during this motion ({played} of {planned} steps)."
            if self.steps >= self.max_steps:
                self._end("step_limit")
                return (
                    f"The episode reached its {self.max_steps}-step limit "
                    f"({played} of {planned} steps played). The episode is over."
                )
        text = f"The motion finished playing ({planned} of {planned} steps)."
        if target is not None and self.observation is not None:
            residual = self.toolset.residual(target, self.observation)
            if residual is not None:
                label, magnitude = residual
                text += f" Largest remaining offset from the requested target is {magnitude:.4g} on {label}."
        return text

    def _end(self, outcome: str) -> None:
        self.outcome = outcome
        self.ended_at = time.time()
        self._write_json(
            "result.json",
            {
                **self.status(),
                "success": self.success,
                "final_text": self.final_text,
                "hindsight": self.hindsight,
                "seconds": round(self.ended_at - self.started_at, 2),
                **self.meta,
            },
        )

    def _state_text(self) -> list[str]:
        assert self.observation is not None
        return _state_lines(self.observation, self.toolset.state_labels())

    def _frames(self, tag: str) -> tuple[Frame, ...]:
        assert self.observation is not None
        frames = []
        for camera, image in self.observation.images.items():
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
            path = self.run_dir / "frames" / f"{tag}_{camera}.png"
            path.write_bytes(buffer.getvalue())
            frames.append(Frame(camera, path, buffer.getvalue()))
        return tuple(frames)

    def _log(self, record: dict[str, Any]) -> None:
        record = {"t": round(time.time(), 3), **record}
        with (self.run_dir / "episode.jsonl").open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")
