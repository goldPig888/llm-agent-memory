"""A stand-in for the LIBERO simulator, for wiring checks without MuJoCo or a GPU.

It declares exactly the same action and observation spaces as the real
``libero`` embodiment (it subclasses it), so the agent toolset, the MCP tool
schemas and the policy wiring are all exercised for real. Only the physics is
fake: the end effector jumps to each commanded pose, and the "task" succeeds
when the end effector is lifted above ``success_z`` with the gripper closed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_libero.embodiment import LiberoEmbodiment

from llm_agent_memory.protocol import CONTROL_HZ, instruction

START_POSE = np.array([-0.2, 0.0, 1.18, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0])


class FakeLibero(LiberoEmbodiment):
    """Same spaces as LIBERO, trivial kinematics, no worker process."""

    def __init__(self, task_id: int, output_dir: Path, success_z: float = 1.30) -> None:
        super().__init__(task_id=task_id, output_dir=output_dir)
        self.success_z = success_z
        self.pose = START_POSE.copy()
        self.step_count = 0

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return to the start pose; ignores the scene's init state."""
        self.instruction = scene.instruction or instruction(self.task_id or 0)
        self.pose = START_POSE.copy()
        self.step_count = 0
        return self._fake_observation()

    def step(self, action: Action) -> StepResult:
        """Move straight to the commanded pose; succeed when lifted with gripper closed."""
        self.pose = np.asarray(action.data, dtype=float).copy()
        self.step_count += 1
        success = bool(self.pose[2] >= self.success_z and self.pose[9] > 0.5)
        return StepResult(
            self._fake_observation(),
            reward=float(success),
            terminated=success,
            termination_reason="success" if success else None,
            info={"success": success},
        )

    def close(self) -> None:
        """Nothing to release."""

    def _fake_observation(self) -> Observation:
        images = {}
        for camera, shade in (("agentview", 90), ("wrist", 60)):
            image = np.full((256, 256, 3), shade, dtype=np.uint8)
            u = int(np.clip(128 + self.pose[1] * 200, 4, 251))
            v = int(np.clip(128 - (self.pose[2] - 1.0) * 200, 4, 251))
            image[v - 4 : v + 4, u - 4 : u + 4] = (230, 60, 60)
            images[camera] = image
        closed = self.pose[9] > 0.5
        return Observation(
            images=images,
            state={
                "eef_state": self.pose.copy(),
                "gripper_qpos": np.array([0.0, 0.0] if closed else [0.04, -0.04]),
            },
            instruction=self.instruction,
            state_time=self.step_count / CONTROL_HZ,
        )


def fake_factory(task_id: int, run_dir: Path) -> FakeLibero:
    """Factory with the same signature as ``episode.libero_factory``."""
    return FakeLibero(task_id, run_dir)
