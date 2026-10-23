"""Episode rules, checked against the real upstream toolset with the fake simulator."""

from __future__ import annotations

import json
from pathlib import Path

from llm_agent_memory.episode import Episode
from llm_agent_memory.fake import fake_factory


def make(tmp_path: Path, **kwargs: object) -> Episode:
    episode = Episode(2, 0, tmp_path / "ep", factory=fake_factory, **kwargs)  # type: ignore[arg-type]
    episode.start()
    return episode


def move(episode: Episode, **targets: float) -> str:
    return episode.call("move_to", {"targets": targets, "note": "test move"}).text


def test_start_reports_goal_state_and_two_cameras(tmp_path: Path) -> None:
    episode = Episode(2, 0, tmp_path / "ep", factory=fake_factory)
    reply = episode.start()
    assert reply.text.startswith("Goal: turn on the stove and put the moka pot on it")
    assert "state[eef_state]: x=" in reply.text
    assert [f.camera for f in reply.frames] == ["agentview", "wrist"]
    assert all(f.path.exists() for f in reply.frames)
    protocol = json.loads((tmp_path / "ep" / "protocol.json").read_text())
    assert protocol["zero_shot"] is True and protocol["max_calls"] == 80


def test_move_interpolates_like_upstream(tmp_path: Path) -> None:
    episode = make(tmp_path)
    text = move(episode, z=1.08)  # 10 cm at 1 cm/step plus upstream headroom
    assert "The motion finished playing" in text
    assert episode.steps > 10 and episode.calls == 1 and not episode.over


def test_success_ends_mid_motion(tmp_path: Path) -> None:
    episode = make(tmp_path)
    move(episode, gripper=1.0)
    text = move(episode, z=1.40)
    assert episode.outcome == "success" and episode.success
    assert "succeeded during this motion" in text
    result = json.loads((tmp_path / "ep" / "result.json").read_text())
    assert result["success"] is True


def test_done_is_a_failure_and_keeps_hindsight(tmp_path: Path) -> None:
    episode = make(tmp_path)
    reply = episode.call("done", {"summary": "finished", "hindsight": "arm undershoots"})
    assert reply.over and episode.outcome == "done" and not episode.success
    assert episode.hindsight == "arm undershoots"
    assert "over" in episode.call("move_to", {"targets": {"z": 1.0}, "note": "x"}).text


def test_errors_count_against_the_budget(tmp_path: Path) -> None:
    episode = make(tmp_path, max_calls=3)
    assert move(episode, z=99.0).startswith("error: target for z is outside")
    assert episode.call("move_to", {"targets": {"z": 1.1}}).text.startswith("error: note is required")
    move(episode, z=1.1)
    assert episode.outcome == "call_budget" and episode.calls == 3


def test_step_limit(tmp_path: Path) -> None:
    episode = make(tmp_path, max_steps=5)
    text = move(episode, z=1.0)
    assert episode.outcome == "step_limit" and episode.steps == 5
    assert "5-step limit" in text
