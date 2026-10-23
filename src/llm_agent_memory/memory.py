"""Cross-episode memory: the research variable this repo exists to study.

Every episode that ends with ``done`` or ``give_up`` leaves a ``hindsight``
note (the model's own "what I wish I had known"). In the LLM-LIBERO paper
those notes were logged and never fed back, so every episode stayed zero-shot.
Here they can be fed forward, under one of these modes:

* ``off``: zero-shot, identical to the paper (the default).
* ``same_task``: notes from earlier episodes of the same LIBERO-10 task.
* ``all_tasks``: notes from earlier episodes of any task.

Only notes written before the current episode started are used, and the
header wording is the upstream agent's ``prior_learnings`` wording.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MODES = ("off", "same_task", "all_tasks")

HEADER = (
    "Notes from a previous attempt at tasks like this one. They may be wrong or "
    "stale; the current observation always wins:"
)


class HindsightStore:
    """An append-only JSONL file of hindsight notes, one record per episode."""

    def __init__(self, path: Path, read_also: list[Path] | None = None) -> None:
        self.path = Path(path)
        # Extra files that are read but never written (another batch's notes).
        self.read_also = [Path(p) for p in read_also or [] if Path(p) != self.path]

    def add(self, record: dict[str, Any]) -> bool:
        """Store a finished episode's note. Empty and 'none' notes are skipped."""
        note = str(record.get("hindsight") or "").strip()
        if not note or note.lower() == "none":
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.time(), **record, "hindsight": note}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return True

    def records(self) -> list[dict[str, Any]]:
        """Records from this file and any read-only sources, oldest first."""
        rows: list[dict[str, Any]] = []
        for path in [*self.read_also, self.path]:
            if path.exists():
                rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return sorted(rows, key=lambda row: row.get("t", 0))

    def render(self, mode: str, task_id: int, limit: int = 5, before: float | None = None) -> str:
        """Prompt block for ``mode`` and ``task_id``; empty string when there is nothing to add."""
        if mode not in MODES:
            raise ValueError(f"memory mode must be one of {MODES}, got {mode!r}")
        if mode == "off":
            return ""
        cutoff = time.time() if before is None else before
        rows = [r for r in self.records() if r.get("t", 0) < cutoff]
        if mode == "same_task":
            rows = [r for r in rows if r.get("task_id") == task_id]
        rows = rows[-limit:]
        if not rows:
            return ""
        lines = [HEADER]
        for row in rows:
            ending = "succeeded" if row.get("success") else f"ended with {row.get('outcome')}"
            lines.append(f"- (task {row.get('task_id')}, {ending}) {row['hindsight']}")
        return "\n".join(lines)
