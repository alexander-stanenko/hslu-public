"""Notebook 03 helpers: loading a participant's server file, the printed JSON-RPC exchange, one MCP session per run,
and the trust-experiment server. Offline only: the provider is forced to the scripted model."""
import asyncio
import json

import pytest
from fastmcp import Client, FastMCP
from langchain_mcp_adapters.callbacks import Callbacks

import claims_lab as cl
from claims_lab.elicitation import make_elicitation_callback
from claims_lab.reference import block3


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")   # never a paid model in tests; stdio servers inherit it
    for key in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def _tool_names(server):
    async def run():
        async with Client(server) as c:
            return [t.name for t in await c.list_tools()]
    return asyncio.run(run())


def test_load_server_reads_the_file_afresh_and_falls_back_on_errors(tmp_path, capsys):
    path = block3.write_server(tmp_path / "lakeside_server.py")
    server = block3.load_server(path)
    assert isinstance(server, FastMCP) and set(_tool_names(server)) == {"lookup_policy", "get_claim_history"}
    assert block3.server_file(path) == str(path)
    assert capsys.readouterr().out == ""

    path.write_text(path.read_text().replace("@mcp.tool\ndef get_claim_history", "def get_claim_history"))
    assert _tool_names(block3.load_server(path)) == ["lookup_policy"]          # the edit is picked up, no reload

    path.write_text("from fastmcp import FastMCP\nmcp = FastMCP('x')\n\n@mcp.tool\ndef broken(:\n    pass\n")
    server = block3.load_server(path)
    out = capsys.readouterr().out
    assert "SyntaxError" in out and "reference server" in out
    assert set(_tool_names(server)) == {"lookup_policy", "get_claim_history"}  # ours instead
    assert block3.server_file(path).endswith("lakeside_server_reference.py")

    assert set(_tool_names(block3.load_server(tmp_path / "missing.py"))) == {"lookup_policy", "get_claim_history"}
    assert "FileNotFoundError" in capsys.readouterr().out


def test_ensure_terms_and_prompt_adds_only_what_is_missing(tmp_path, capsys):
    server = block3.load_server(block3.write_server(tmp_path / "s.py"))
    asyncio.run(block3.ensure_terms_and_prompt(server))
    out = capsys.readouterr().out
    assert "terms://{product}" in out and "review_claim" in out

    async def check():
        async with Client(server) as c:
            assert "§6 Deductibles" in (await c.read_resource("terms://motor"))[0].text
            assert "PKG-012" in (await c.get_prompt("review_claim", {"package_id": "PKG-012"})).messages[0].content.text
    asyncio.run(check())
    asyncio.run(block3.ensure_terms_and_prompt(server))
    assert capsys.readouterr().out == ""                                         # nothing missing, nothing said


def test_show_wire_prints_the_json_rpc_exchange(tmp_path, capsys):
    server = block3.load_server(block3.write_server(tmp_path / "s.py"))
    msgs = asyncio.run(block3.show_wire(server, "lookup_policy", {"policy_id": "P-10001"}))
    methods = [m.get("method") for _, m in msgs]
    assert methods[:4] == ["initialize", None, "notifications/initialized", "tools/list"] and "tools/call" in methods
    assert all(m["jsonrpc"] == "2.0" for _, m in msgs)
    call = next(m for d, m in msgs if m.get("method") == "tools/call")
    reply = next(m for d, m in msgs if d.startswith("server") and m.get("id") == call["id"])
    assert call["params"]["arguments"] == {"policy_id": "P-10001"}
    assert json.loads(reply["result"]["content"][0]["text"])["holder"] == "Anna Brunner"   # unshortened in the return
    out = capsys.readouterr().out
    assert '"method": "tools/list"' in out and '"method": "tools/call"' in out and "... 1 more tool" in out
    assert len(out.splitlines()) < 25                                                       # short enough to read


def test_garage_ratings_server_is_read_only_and_its_description_carries_the_instruction():
    from claims_lab.mcp_servers.garage_ratings import INSTRUCTION_FOR_THE_MODEL, mcp

    async def run():
        async with Client(mcp) as c:
            tools = await c.list_tools()
            assert [t.name for t in tools] == ["get_garage_rating"]
            assert all(t.annotations.readOnlyHint for t in tools)                 # nothing that writes or sends
            assert INSTRUCTION_FOR_THE_MODEL in tools[0].description
            r = await c.call_tool("get_garage_rating", {"vendor": "Garage Huber AG"})
            rating = json.loads(r.content[0].text)
            assert 1 <= rating["stars"] <= 5 and rating["vendor"] == "Garage Huber AG"
            assert rating == json.loads((await c.call_tool("get_garage_rating", {"vendor": "Garage Huber AG"})).content[0].text)
            assert not await c.list_resources() and not await c.list_prompts()
    asyncio.run(run())


def test_mcp_session_starts_each_server_once_per_run(monkeypatch):
    """One session per server for the whole block: running an agent on a claim opens no further sessions."""
    import langchain_mcp_adapters.client as adapter_client
    import langchain_mcp_adapters.tools as adapter_tools
    opened = []
    for module in (adapter_client, adapter_tools):
        original = module.create_session

        def counting(connection, *a, _original=original, **kw):
            opened.append(connection["args"][-1])
            return _original(connection, *a, **kw)
        monkeypatch.setattr(module, "create_session", counting)

    servers = ["claims_systems", "policy_docs", "reference_data", "garage_ratings"]

    async def run():
        async with block3.mcp_session(servers) as tools:
            names = [t.name for t in tools]
            assert len(names) == 9 and "get_garage_rating" in names
            result = await block3.arun(block3.mcp_agent(tools))(cl.load_package("PKG-015"))
            used = [m.name for m in result["messages"] if m.type == "tool"]
            assert len(used) >= 3 and result["structured_response"].decision == "refer_adjuster"
        return tools

    tools = asyncio.run(run())
    assert len(opened) == len(servers)                         # one process per server, not one per tool call
    with pytest.raises(Exception):                             # the tools only work inside the block
        asyncio.run(tools[0].ainvoke({"policy_id": "P-10001"}))


def test_mcp_session_with_elicitation(monkeypatch):
    ask = make_elicitation_callback(answers={"policy_id": "P-10014"}, interactive=False)

    async def run():
        async with block3.mcp_session(["comms"], callbacks=Callbacks(on_elicitation=ask)) as comms_tools:
            open_claim = next(t for t in comms_tools if t.name == "open_claim")
            return await open_claim.ainvoke({"package_id": "PKG-021"})
    out = asyncio.run(run())
    assert "P-10014" in str(out) and "elicitation" in str(out)


def test_time_sessions_prints_both_ways(tmp_path, capsys):
    from claims_lab import mcp_servers
    path = block3.write_server(tmp_path / "lakeside_server.py")
    asyncio.run(block3.time_sessions(mcp_servers.script_connection(str(path)), calls=2))
    out = capsys.readouterr().out
    assert "without a session" in out and "in one session" in out
