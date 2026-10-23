"""Codex launch helpers: config block, command lines, audit, id parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_agent_memory.codex import (
    BLOCK_START,
    config_block,
    count_off_protocol,
    exec_command,
    install_block,
    interactive_command,
    parse_ids,
)


def test_config_block_is_replaced_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "y"\n')
    assert install_block(path, config_block("/env/bin/python", Path("/repo"))) == "added"
    assert install_block(path, config_block("/new/bin/python", Path("/repo"))) == "replaced"
    text = path.read_text()
    assert text.count(BLOCK_START) == 1 and '"/new/bin/python"' in text
    assert '[mcp_servers.other]' in text and (tmp_path / "config.toml.bak").exists()


def test_foreign_libero_table_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[mcp_servers.libero]\ncommand = 'mine'\n")
    with pytest.raises(SystemExit):
        install_block(path, config_block())


def test_commands_are_read_only_and_carry_effort(tmp_path: Path) -> None:
    run = exec_command("codex", tmp_path, "go", "gpt-x", "high")
    assert run[:3] == ["codex", "exec", "--json"] and run[-1] == "go"
    assert "read-only" in run and 'model_reasoning_effort="high"' in run
    chat = interactive_command("codex", tmp_path, "go", None, None, no_approvals=True)
    assert chat[-3:] == ["--ask-for-approval", "never", "go"]


def test_off_protocol_audit(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"type":"item.completed","item":{"type":"mcp_tool_call"}}\n'
        '{"type":"item.completed","item":{"type":"command_execution","command":"ls"}}\n'
    )
    assert count_off_protocol(events) == 1


def test_parse_ids() -> None:
    assert parse_ids("0,3,5-7", 10) == [0, 3, 5, 6, 7]
    with pytest.raises(SystemExit):
        parse_ids("8-11", 10)
