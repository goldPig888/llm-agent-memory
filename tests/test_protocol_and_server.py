"""The prompt matches the paper, and the MCP server exposes the paper's tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from inspect_robots_libero.embodiment import BASE_DOCS

from llm_agent_memory import mcp_server
from llm_agent_memory.memory import HindsightStore
from llm_agent_memory.protocol import system_prompt
from llm_agent_memory.workspace import agents_md, make_workspace

PAPER_OPENING = (
    "You are controlling a real robot embodiment named 'libero_10_task00' through tool calls."
)


def test_system_prompt_is_the_papers() -> None:
    prompt = system_prompt(0)
    assert prompt.startswith(PAPER_OPENING)
    assert "You have a budget of 80 LLM calls for the whole trial." in prompt
    assert prompt.endswith("Embodiment notes:\n" + BASE_DOCS.strip())


def test_agents_md_embeds_prompt_and_rules(tmp_path: Path) -> None:
    text = agents_md(4, 7)
    assert system_prompt(4) in text
    assert "task_id=4 and init_state=7" in text
    path = make_workspace(tmp_path, 4, 7, "tag")
    assert (path / "AGENTS.md").read_text() == text


@pytest.fixture()
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mcp_server.Session:
    monkeypatch.setenv("LAM_FAKE", "1")
    monkeypatch.setenv("LAM_TASK_ID", "1")
    monkeypatch.setenv("LAM_INIT_STATE", "3")
    monkeypatch.setenv("LAM_RUN_TAG", "test")
    monkeypatch.setenv("LAM_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("LAM_MEMORY_FILE", str(tmp_path / "memory.jsonl"))
    return mcp_server.Session()


def test_tools_are_the_upstream_ones(session: mcp_server.Session) -> None:
    tools = {t.name: t for t in session.tools()}
    assert set(tools) == {"start_episode", "move_to", "done", "give_up", "episode_status"}
    assert tools["move_to"].description.startswith("Move to absolute Cartesian end-effector targets")
    assert tools["move_to"].inputSchema["required"] == ["targets", "note"]
    assert tools["give_up"].inputSchema["required"] == ["reason", "hindsight"]


def test_session_enforces_one_episode_on_the_assigned_task(session: mcp_server.Session) -> None:
    assert session.call("move_to", {}).text == "error: call start_episode first"
    assert "set up for task_id=1" in session.call("start_episode", {"task_id": 4, "init_state": 3}).text
    assert session.call("start_episode", {}).text.startswith("Goal: put both the cream cheese")
    assert "cannot be restarted" in session.call("start_episode", {}).text
    content = mcp_server.to_content(
        session.call("move_to", {"targets": {"z": 1.1}, "note": "lift a little"})
    )
    assert [c.type for c in content] == ["text", "text", "image", "text", "image", "text"]


def test_hindsight_is_stored_and_fed_forward(session: mcp_server.Session, tmp_path: Path) -> None:
    session.call("start_episode", {})
    session.call("give_up", {"reason": "stuck", "hindsight": "the basket is to the left"})
    store = HindsightStore(tmp_path / "memory.jsonl")
    assert store.records()[0]["hindsight"] == "the basket is to the left"
    assert "the basket is to the left" in store.render("same_task", 1)
    assert store.render("same_task", 2) == ""
    assert store.render("off", 1) == ""
    assert "the basket" in store.render("all_tasks", 2)


def test_memory_from_is_read_only(tmp_path: Path) -> None:
    source = HindsightStore(tmp_path / "zs.jsonl")
    source.add({"task_id": 1, "outcome": "done", "hindsight": "from zero-shot"})
    store = HindsightStore(tmp_path / "mem.jsonl", read_also=[tmp_path / "zs.jsonl"])
    store.add({"task_id": 1, "outcome": "give_up", "hindsight": "from memory batch"})
    text = store.render("same_task", 1)
    assert "from zero-shot" in text and "from memory batch" in text
    assert len(source.records()) == 1
