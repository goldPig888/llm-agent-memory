"""04: run many episodes headless with `codex exec`, one after another.

    python scripts/04_run_codex_batch.py --label pilot --tasks 2 --inits 0-9 --effort high
    python scripts/04_run_codex_batch.py --label mem_same --tasks 2 --inits 0-9 --memory same_task
    python scripts/04_run_codex_batch.py --label full --tasks 0-9 --inits 0-9      # the paper's 100 episodes

Episodes run in order (task, then initial state), which matters for the memory
modes: each episode only sees notes from episodes that finished before it.
Each batch keeps its own hindsight file, outputs/memory/<label>.jsonl, so
batches never leak notes into each other. To start a memory batch from the
notes another batch wrote (for example a zero-shot one), pass
--memory-from <that label>; those notes are read but never written to.
Progress goes to outputs/batches/<label>.jsonl. Re-running the same command
skips episodes that already finished, so an interrupted batch resumes.

Every episode's Codex event stream is saved next to its AGENTS.md, and any
shell, file or web action Codex took is counted as `off_protocol` (05 reports
these; such episodes are not comparable with the paper's API runs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from llm_agent_memory.codex import (
    REPO,
    count_off_protocol,
    exec_command,
    find_codex,
    memory_path,
    parse_ids,
    session_env,
)
from llm_agent_memory.memory import MODES
from llm_agent_memory.protocol import TASK_IDS
from llm_agent_memory.workspace import episode_tag, first_message, make_workspace


def finished(manifest: Path) -> set[tuple[int, int]]:
    if not manifest.exists():
        return set()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return {(r["task_id"], r["init_state"]) for r in rows if r.get("outcome")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="batch name; also the resume key")
    parser.add_argument("--tasks", default="0-9", help='e.g. "0-9", "2", "0,3,5-7"')
    parser.add_argument("--inits", default="0-9", help="initial states, same syntax (paper: 0-9)")
    parser.add_argument("--model")
    parser.add_argument("--effort", help="minimal, low, medium, high")
    parser.add_argument("--memory", default="off", choices=MODES)
    parser.add_argument(
        "--memory-from",
        help="also read the hindsight notes of this earlier batch (read only)",
    )
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per episode")
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = parser.parse_args()

    codex = "codex" if args.dry_run else find_codex()
    pairs = [(t, i) for t in parse_ids(args.tasks, len(TASK_IDS)) for i in parse_ids(args.inits, 50)]
    manifest = REPO / "outputs" / "batches" / f"{args.label}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = finished(manifest)
    todo = [p for p in pairs if p not in done]
    print(f"[04] {args.label}: {len(pairs)} episodes, {len(pairs) - len(todo)} already done, {len(todo)} to run")

    for n, (task_id, init_state) in enumerate(todo, 1):
        tag = episode_tag(args.label)
        if args.dry_run:
            planned = REPO / "workspaces" / f"{tag}_task{task_id:02d}_init{init_state:02d}"
            print(" ".join(exec_command(codex, planned, first_message(task_id, init_state), args.model, args.effort)))
            continue
        workspace = make_workspace(REPO / "workspaces", task_id, init_state, tag)
        env = session_env(
            task_id,
            init_state,
            tag,
            memory=args.memory,
            fake=args.fake,
            memory_file=memory_path(args.label, args.fake),
            memory_from=memory_path(args.memory_from, args.fake) if args.memory_from else None,
        )
        cmd = exec_command(codex, workspace, first_message(task_id, init_state), args.model, args.effort)
        print(f"[04] {n}/{len(todo)} task {task_id} init {init_state} -> {workspace.name}", flush=True)
        started = time.time()
        events = workspace / "codex_events.jsonl"
        with events.open("w") as out, (workspace / "codex_stderr.log").open("w") as err:
            try:
                code: int | str = subprocess.run(cmd, stdout=out, stderr=err, env=env, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        result_path = REPO / "outputs" / "codex_runs" / workspace.name / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = {
            "label": args.label,
            "task_id": task_id,
            "init_state": init_state,
            "run": workspace.name,
            "model": args.model,
            "effort": args.effort,
            "memory": args.memory,
            "fake": args.fake,
            "codex_exit": code,
            "outcome": result.get("outcome") or ("no_result" if code != 0 else "unfinished"),
            "success": bool(result.get("success")),
            "calls": result.get("calls"),
            "steps": result.get("steps"),
            "off_protocol": count_off_protocol(events),
            "seconds": round(time.time() - started, 1),
        }
        with manifest.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"     {row['outcome']}  calls={row['calls']} steps={row['steps']} off_protocol={row['off_protocol']}")
    print("[04] done. Summarize with: python scripts/05_summarize.py --label " + args.label)


if __name__ == "__main__":
    sys.exit(main())
