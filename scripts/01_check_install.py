"""01: check the install without calling any model.

    python scripts/01_check_install.py          # unit tests + one real LIBERO episode
    python scripts/01_check_install.py --fake   # unit tests + fake simulator only

The LIBERO check starts task 2 at initial state 0, lifts the arm 3 cm, closes
and opens the gripper, and confirms the frames, protocol.json and video were
written. It prints one JSON line ending in "passed": true.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
from llm_agent_memory.episode import Episode, libero_factory
from llm_agent_memory.fake import fake_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true", help="skip LIBERO, use the fake simulator")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if not args.skip_tests:
        print("[01] unit tests", flush=True)
        code = subprocess.call([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")])
        if code:
            sys.exit(code)

    print(f"[01] {'fake' if args.fake else 'LIBERO'} episode check", flush=True)
    out = REPO / "outputs" / "checks" / time.strftime("%Y%m%d-%H%M%S")
    started = time.time()
    episode = Episode(2, 0, out, factory=fake_factory if args.fake else libero_factory)
    try:
        first = episode.start()
        z0 = float(episode.observation.state["eef_state"][2])  # type: ignore[union-attr]
        lift = episode.call("move_to", {"targets": {"z": round(z0 + 0.03, 4)}, "note": "install check: lift 3 cm"})
        close = episode.call("move_to", {"targets": {"gripper": 1.0}, "note": "install check: close"})
        opened = episode.call("move_to", {"targets": {"gripper": 0.0}, "note": "install check: open"})
        z1 = float(episode.observation.state["eef_state"][2])  # type: ignore[union-attr]
    finally:
        episode.close()
    videos = sorted(out.rglob("rollout.mp4"))
    checks = {
        "goal_shown": first.text.startswith("Goal: turn on the stove"),
        "two_cameras": len(first.frames) == 2,
        "moves_played": all("finished playing" in r.text for r in (lift, close, opened)),
        "lifted_cm": round((z1 - z0) * 100, 2),
        "lift_ok": z1 - z0 > 0.015,
        "video": str(videos[0]) if videos else None,
    }
    passed = checks["goal_shown"] and checks["two_cameras"] and checks["moves_played"] and checks["lift_ok"]
    if not args.fake:
        passed = passed and bool(videos)
    print(json.dumps({"output": str(out), "seconds": round(time.time() - started, 1), **checks, "passed": passed}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
