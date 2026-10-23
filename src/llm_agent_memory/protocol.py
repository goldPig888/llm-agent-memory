"""The LLM-LIBERO evaluation protocol, imported from the pinned sources.

Nothing here is retyped: the system prompt comes from the upstream agent
policy's template and the embodiment notes come from the LLM_LIBERO plugin, so
a Codex episode sees the same text as the episodes in the paper.
"""

from __future__ import annotations

from inspect_robots_agent.policy import _SYSTEM_TEMPLATE
from inspect_robots_libero.embodiment import BASE_DOCS
from inspect_robots_libero.tasks import LIBERO_10_TASKS, NUM_INIT_STATES, check_task_id

# Per-episode limits from the paper (Sec. 2.3, Algorithm 1).
MAX_CALLS = 80
MAX_STEPS = 900
CONTROL_HZ = 20

TASK_IDS = tuple(range(len(LIBERO_10_TASKS)))


def embodiment_name(task_id: int) -> str:
    """Name the model is told it controls, e.g. ``libero_10_task02``."""
    return f"libero_10_task{check_task_id(task_id):02d}"


def instruction(task_id: int) -> str:
    """LIBERO's own language instruction for ``task_id``."""
    return LIBERO_10_TASKS[check_task_id(task_id)][1]


def system_prompt(task_id: int, max_calls: int = MAX_CALLS) -> str:
    """The exact system prompt used in the paper (Figure 2) for ``task_id``."""
    head = _SYSTEM_TEMPLATE.format(name=embodiment_name(task_id), budget=max_calls)
    return head + "\n\nEmbodiment notes:\n" + BASE_DOCS.strip()


def check_init_state(init_state: int) -> int:
    """Return ``init_state`` if LIBERO has it, else raise ``ValueError``."""
    if isinstance(init_state, bool) or not isinstance(init_state, int):
        raise ValueError(f"init_state must be an integer, got {init_state!r}")
    if not 0 <= init_state < NUM_INIT_STATES:
        raise ValueError(f"init_state must be in 0..{NUM_INIT_STATES - 1}, got {init_state}")
    return init_state
