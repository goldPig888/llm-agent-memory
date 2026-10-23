"""02: register the `libero` MCP server with Codex (run once per machine).

    python scripts/02_install_codex_config.py           # writes ~/.codex/config.toml
    python scripts/02_install_codex_config.py --print   # only show the block

Adds a [mcp_servers.libero] table that launches this env's Python with
`-m llm_agent_memory.mcp_server` from the repo root. The block sits between
marker comments, so running this again (for example after moving the repo or
recreating the env) replaces it in place. The first edit keeps a backup at
config.toml.bak.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from llm_agent_memory.codex import codex_home, config_block, find_codex, install_block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print", action="store_true", help="print the block, change nothing")
    args = parser.parse_args()

    block = config_block()
    if args.print:
        print(block)
        return

    codex = find_codex()
    version = subprocess.run([codex, "--version"], capture_output=True, text=True).stdout.strip()
    path = codex_home() / "config.toml"
    action = install_block(path, block)
    print(f"[02] {codex} ({version or 'unknown version'})")
    print(f"[02] {action} the libero MCP server in {path}:\n")
    print(block)
    listed = subprocess.run([codex, "mcp", "list"], capture_output=True, text=True)
    if listed.returncode == 0:
        print("[02] codex mcp list:\n" + listed.stdout)
    print("Next: python scripts/03_open_codex_episode.py --task 2 --init 0 --fake")


if __name__ == "__main__":
    sys.exit(main())
