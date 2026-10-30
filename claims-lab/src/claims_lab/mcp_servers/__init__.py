"""Reference MCP servers for the Lakeside systems, plus helpers to connect them."""
from __future__ import annotations

import json
import os
import sys

SERVERS = {
    "claims_systems": "claims_lab.mcp_servers.claims_systems",
    "policy_docs": "claims_lab.mcp_servers.policy_docs",
    "reference_data": "claims_lab.mcp_servers.reference_data",
    "comms": "claims_lab.mcp_servers.comms",
}

# Servers for experiments only (not part of Lakeside's systems, not in the Claude Desktop configuration)
EXPERIMENTS = {
    "garage_ratings": "claims_lab.mcp_servers.garage_ratings",   # notebook 03, B3: a description aimed at the model
}


READ_ONLY = {"readOnlyHint": True}


def _usable_errlog_for_stdio() -> None:
    """In a notebook kernel sys.stderr is not a real file (no file descriptor), and the MCP stdio client, which
    passes it to the server subprocess, fails with 'fileno'. Point the client's default at the real stderr instead."""
    import mcp.client.stdio as stdio
    fn = stdio.stdio_client.__wrapped__
    try:
        fn.__defaults__[0].fileno()
    except Exception:
        fn.__defaults__ = (sys.__stderr__,)


_usable_errlog_for_stdio()


def _server_env() -> dict:
    # stdio subprocesses only inherit a minimal environment; pass on what the servers need (embeddings, workdir)
    keep = ("PATH", "HOME", "PYTHONPATH", "CLAIMS_LAB_", "AZURE_OPENAI_", "OPENAI_")
    env = {k: v for k, v in os.environ.items() if k.startswith(keep)}
    env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
    return env


def stdio_connections(names: list[str] | None = None) -> dict:
    """Connection config for langchain_mcp_adapters.MultiServerMCPClient (each server runs as a subprocess)."""
    modules = {**SERVERS, **EXPERIMENTS}
    return {n: {"transport": "stdio", "command": sys.executable, "args": ["-m", modules[n]], "env": _server_env()}
            for n in (names or SERVERS)}


def script_connection(path: str) -> dict:
    """Connection config for a server script of your own, e.g. the lakeside_server.py written in notebook 03."""
    from pathlib import Path
    return {"transport": "stdio", "command": sys.executable, "args": [str(Path(path).resolve())], "env": _server_env()}


def claude_desktop_config(names: list[str] | None = None, workdir: str | None = None) -> str:
    """JSON snippet for Claude Desktop's claude_desktop_config.json (instructor demo).

    Claude Desktop starts each server in a folder of its own choosing, so the snippet sets an absolute working
    folder (default: ~/claims_lab_work) where the server keeps its copy of the database."""
    from pathlib import Path
    env = {"CLAIMS_LAB_WORKDIR": str(Path(workdir or Path.home() / "claims_lab_work").expanduser().resolve()),
           "FASTMCP_LOG_LEVEL": "WARNING"}
    cfg = {"mcpServers": {f"lakeside-{n.replace('_', '-')}": {"command": sys.executable, "args": ["-m", SERVERS[n]], "env": env}
                          for n in (names or SERVERS)}}
    text = json.dumps(cfg, indent=2)
    print(text)
    return text
