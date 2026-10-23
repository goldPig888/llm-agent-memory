"""05: summarize batches in the paper's format.

    python scripts/05_summarize.py                 # every batch in outputs/batches/
    python scripts/05_summarize.py --label pilot   # one batch

Prints success rate per batch and per task (with the paper's zero-shot API
rates for the same task alongside), the termination breakdown, and any
episodes where Codex acted outside the robot tools. Writes
outputs/summary.csv with one row per episode.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from llm_agent_memory.codex import REPO
from llm_agent_memory.protocol import LIBERO_10_TASKS

# Table 2 of the paper: per-task success pooled over 18 zero-shot API configurations.
PAPER_TASK_RATE = {1: 58, 0: 55, 7: 50, 2: 33, 8: 12, 3: 7, 5: 7, 9: 1, 6: 0, 4: 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label")
    args = parser.parse_args()

    folder = REPO / "outputs" / "batches"
    files = [folder / f"{args.label}.jsonl"] if args.label else sorted(folder.glob("*.jsonl"))
    rows = [json.loads(line) for f in files if f.exists() for line in f.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no batch results in {folder}. Run scripts/04_run_codex_batch.py first.")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    for label, items in by_label.items():
        ok = sum(r["success"] for r in items)
        first = items[0]
        print(f"\n== {label}: {ok}/{len(items)} = {100 * ok / len(items):.1f}%  "
              f"(model={first.get('model') or 'codex default'}, effort={first.get('effort') or 'default'}, "
              f"memory={first.get('memory')}, fake={first.get('fake')})")
        print("   outcomes: " + ", ".join(f"{k}={v}" for k, v in Counter(r["outcome"] for r in items).most_common()))
        per_task: dict[int, list[dict]] = defaultdict(list)
        for r in items:
            per_task[r["task_id"]].append(r)
        print(f"   {'task':<5}{'success':>9}{'rate':>7}{'paper':>7}  instruction")
        for task_id in sorted(per_task):
            group = per_task[task_id]
            wins = sum(r["success"] for r in group)
            print(f"   {task_id:<5}{f'{wins}/{len(group)}':>9}{100 * wins / len(group):>6.0f}%"
                  f"{PAPER_TASK_RATE[task_id]:>6}%  {LIBERO_10_TASKS[task_id][1]}")
        flagged = [r for r in items if r.get("off_protocol")]
        if flagged:
            print(f"   WARNING: {len(flagged)} episode(s) used shell/file/web actions (not comparable):")
            for r in flagged:
                print(f"     {r['run']}  off_protocol={r['off_protocol']}")

    out = REPO / "outputs" / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
