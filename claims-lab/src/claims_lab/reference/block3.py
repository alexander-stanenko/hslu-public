"""Reference for notebook 03: a server of your own, the checklist agent over MCP, elicitation, and the bridge (stretch).

    from claims_lab.reference import block3
    block3.write_server()                                   # catch-up for Part A: writes lakeside_server.py
    server = block3.load_server("lakeside_server.py")       # (re)import the file, fall back to ours if it is broken
    await block3.show_wire(server)                          # the JSON-RPC messages of tools/list and one tools/call
    async with block3.mcp_session() as tools:               # Lakeside's two read-only servers, one session each
        await cl.aspot_check(block3.mcp_agent(tools), "PKG-001", "PKG-002", "PKG-003")
    agent = await block3.intake_agent(answers={"policy_id": "P-1004"})    # B3
"""
from __future__ import annotations

import contextlib
import importlib.util
import itertools
import json
import textwrap
import time
from pathlib import Path

import anyio
from fastmcp import Client, Context, FastMCP
from fastmcp.client.transports import FastMCPTransport
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from .. import mcp_servers
from ..elicitation import make_elicitation_callback
from ..models import get_model, run_config
from ..schemas import ClaimDecision
from . import block2

# ------------------------------------------------------------------ Part A: your first MCP server
SERVER_SOURCE = '''"""A small MCP server for Lakeside's policy system (notebook 03, Part A)."""
from fastmcp import FastMCP

from claims_lab import tools

mcp = FastMCP("lakeside-policies", instructions="Policy look-ups for Lakeside Insurance AG.")


@mcp.tool
def lookup_policy(policy_id: str) -> dict:
    """Look up a policy by its number (format 'P-1001'). Returns the holder, the car and its plate, the kinds of
    damage the policy covers ('covers'), the deductible in CHF and the dates between which the policy is valid."""
    return tools.lookup_policy(policy_id)


@mcp.tool
def get_claim_history(policy_id: str) -> list[dict]:
    """List the earlier claims on a policy (policy number 'P-1001'): date, kind of damage, amount in CHF, invoice
    number, garage and status. Use it to see whether this damage or this invoice was claimed before."""
    return tools.get_claim_history(policy_id)


if __name__ == "__main__":
    mcp.run(show_banner=False)
'''


def write_server(path: str | Path = "lakeside_server.py") -> Path:
    """Catch-up for Part A: write the reference server file."""
    path = Path(path)
    path.write_text(SERVER_SOURCE, encoding="utf-8")
    return path


_module_ids = itertools.count(1)


def _import_server(path: Path) -> FastMCP:
    if not path.exists():
        raise FileNotFoundError(f"there is no file {path.name} in this folder (run the %%writefile cell first)")
    spec = importlib.util.spec_from_file_location(f"_lakeside_server_{next(_module_ids)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # a fresh import every time, so edits to the file are picked up
    server = getattr(module, "mcp", None)
    if not isinstance(server, FastMCP):
        raise TypeError(f"{path.name} defines no FastMCP object called `mcp`")
    return server


def _reference_file() -> Path:
    from ..data import workdir
    return write_server(workdir() / "lakeside_server_reference.py")


def _check(path: str | Path) -> tuple[Path, FastMCP]:
    path = Path(path)
    try:
        return path, _import_server(path)
    except Exception as e:  # show the problem, continue with ours
        print(f"Your {path.name} could not be loaded: {type(e).__name__}: {e}\n"
              f"This cell continues with our reference server. Fix the file, run its cell again, then run this cell again.\n")
        ref = _reference_file()
        return ref, _import_server(ref)


def load_server(path: str | Path = "lakeside_server.py") -> FastMCP:
    """Import a server file afresh and return its FastMCP object (`mcp`). If the file is missing or has an error,
    print what is wrong and return the reference server instead, so that the next cells still run."""
    return _check(path)[1]


def server_file(path: str | Path = "lakeside_server.py") -> str:
    """The server file to start as a separate process: yours if it loads, otherwise a copy of the reference server."""
    return str(_check(path)[0])


def _add_terms(server) -> None:
    from .. import data

    @server.resource("terms://car")
    def terms() -> str:
        """Lakeside's general terms for car insurance in full (§1 to §4)."""
        return data.terms()


def _add_prompt(server) -> None:
    @server.prompt
    def review_claim(package_id: str) -> str:
        """Instruction for reviewing one claim package with the clerk's cheat sheet."""
        return (f"Review claim package {package_id}. Work through the four questions of the clerk's cheat sheet in order "
                f"and decide: fast_track, request_info, refer_adjuster or decline. Give the reason.")


def add_terms_and_prompt(server) -> None:
    """Exercise A3 solution: a resource and a prompt on an existing FastMCP server."""
    _add_terms(server)
    _add_prompt(server)


async def ensure_terms_and_prompt(server) -> None:
    """After Exercise A3: add our version of whatever is missing (the resource, the prompt), with a message."""
    async with Client(server) as c:
        has_terms = "terms://car" in [str(r.uri) for r in await c.list_resources()]
        has_prompt = "review_claim" in [p.name for p in await c.list_prompts()]
    if not has_terms:
        print("Your server has no resource terms://car yet; we add ours so that you can continue.")
        _add_terms(server)
    if not has_prompt:
        print("Your server has no prompt review_claim yet; we add ours so that you can continue.")
        _add_prompt(server)


# ------------------------------------------------------------------ Part A2: the messages on the wire
class WireTap(FastMCPTransport):
    """FastMCP's in-memory transport with a relay in the middle that keeps a copy of every JSON-RPC message.

    client  <->  relay (records)  <->  server, all in this Python process; nothing is sent over a network."""

    def __init__(self, server):
        super().__init__(server)
        self.messages: list[tuple[str, dict]] = []

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs):
        from fastmcp.client.transports.memory import _enter_server_lifespan

        async def relay(source, target, direction):
            try:
                async for item in source:
                    if hasattr(item, "message"):
                        self.messages.append((direction, item.message.model_dump(by_alias=True, mode="json",
                                                                                 exclude_none=True)))
                    await target.send(item)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
                pass

        async with create_client_server_memory_streams() as (client_side, relay_a), \
                create_client_server_memory_streams() as (relay_b, server_side):
            client_read, client_write = client_side
            from_client, to_client = relay_a
            from_server, to_server = relay_b
            server_read, server_write = server_side
            async with _enter_server_lifespan(server=self.server):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(lambda: self.server._mcp_server.run(
                        server_read, server_write, self.server._mcp_server.create_initialization_options()))
                    tg.start_soon(relay, from_client, to_server, "client -> server")
                    tg.start_soon(relay, from_server, to_client, "server -> client")
                    try:
                        async with ClientSession(read_stream=client_read, write_stream=client_write,
                                                 **session_kwargs) as session:
                            yield session
                    finally:
                        tg.cancel_scope.cancel()


def _shorten(obj, limit: int = 90):
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + " ..."
    if isinstance(obj, dict):
        return {k: _shorten(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shorten(v, limit) for v in obj]
    return obj


def _ordered(m: dict) -> dict:
    """jsonrpc, id and method first, as in the specification's examples."""
    return {k: m[k] for k in ("jsonrpc", "id", "method", "params", "result", "error") if k in m}


async def show_wire(server, tool: str = "lookup_policy", arguments: dict | None = None) -> list[tuple[str, dict]]:
    """Connect to `server` in this process, list its tools and call one, and print the JSON-RPC messages of the
    tools/list and tools/call exchanges. To keep it short, long texts are cut, only the first tool of the list is
    shown, and two repeated or technical parts are replaced by a note (FastMCP's `_meta`, `outputSchema`, and
    `structuredContent`, which repeats the text result as JSON). Returns all messages, unshortened."""
    if isinstance(server, (str, Path)):   # a server file, e.g. "lakeside_server.py": import it, do not start it
        server = _import_server(Path(server))
    tap = WireTap(server)
    async with Client(tap) as c:
        await c.list_tools()
        await c.call_tool(tool, arguments if arguments is not None else {"policy_id": "P-1001"})
    msgs = tap.messages
    print("All messages, in order:", ", ".join(m.get("method", "result") + (" ->" if d.startswith("client") else " <-")
                                                for d, m in msgs), "\n")
    for d, m in msgs:
        if m.get("method") not in ("tools/list", "tools/call"):
            continue
        reply = json.loads(json.dumps(next((r for _, r in msgs if "method" not in r and r.get("id") == m["id"]), {})))
        result = reply.get("result", {})
        if m["method"] == "tools/list":
            tools = [{k: t[k] for k in ("name", "description", "inputSchema") if k in t} for t in result.get("tools", [])]
            more = len(tools) - 1
            result["tools"] = tools[:1] + ([f"... {more} more tool{'s' if more > 1 else ''}"] if more > 0 else [])
        elif "structuredContent" in result:
            result["structuredContent"] = "{... the same result as a JSON object ...}"
        for direction, message in (("client -> server", _ordered(m)), ("server -> client", _shorten(_ordered(reply)))):
            print(textwrap.fill(f"{direction}: {json.dumps(message, ensure_ascii=False)}", width=115,
                                subsequent_indent="    ", break_on_hyphens=False))
        print()
    return msgs


# ------------------------------------------------------------------ Part B: the checklist agent over MCP
READ_SERVERS = ["claims_systems", "policy_docs"]


async def mcp_tools(names: list[str] | None = None) -> list:
    """The tools of Lakeside's servers as LangChain tools, without a session: the adapter then starts a new server
    process for every tool call (about a second each). Prefer `mcp_session`."""
    client = MultiServerMCPClient(mcp_servers.stdio_connections(names or READ_SERVERS))
    return await client.get_tools()


@contextlib.asynccontextmanager
async def mcp_session(names: list[str] | None = None, callbacks: Callbacks | None = None,
                      connections: dict | None = None):
    """Lakeside's servers as LangChain tools, with one MCP session per server for the whole `async with` block:
    each server process starts once, and every tool call inside the block reuses it.

        async with block3.mcp_session(["claims_systems", "policy_docs"]) as tools:
            agent = block3.mcp_agent(tools)
            ...   # the tools only work inside the block

    The documented pattern of langchain-mcp-adapters, for several servers at once:
    `async with client.session(name) as session: tools = await load_mcp_tools(session)`."""
    client = MultiServerMCPClient(connections or mcp_servers.stdio_connections(names or READ_SERVERS),
                                  callbacks=callbacks)
    async with contextlib.AsyncExitStack() as stack:
        tools = []
        for name in client.connections:
            session = await stack.enter_async_context(client.session(name))
            tools += await load_mcp_tools(session, callbacks=client.callbacks, server_name=name)
        yield tools


async def time_sessions(connection: dict, tool: str = "lookup_policy", arguments: dict | None = None,
                        calls: int = 3) -> None:
    """Time `calls` tool calls twice, and print both times: without a session (a new server process per call) and
    in one session."""
    arguments = arguments if arguments is not None else {"policy_id": "P-1001"}
    client = MultiServerMCPClient({"server": connection})
    start = time.perf_counter()
    t = next(t for t in await client.get_tools() if t.name == tool)
    for _ in range(calls):
        await t.ainvoke(arguments)
    without = time.perf_counter() - start
    start = time.perf_counter()
    async with client.session("server") as session:
        t = next(t for t in await load_mcp_tools(session) if t.name == tool)
        for _ in range(calls):
            await t.ainvoke(arguments)
    within = time.perf_counter() - start
    print(f"list the tools, then {calls} calls of {tool}:\n"
          f"  without a session (a new server process for each call): {without:.1f} s\n"
          f"  in one session (one server process):                    {within:.1f} s")


def mcp_agent(tools: list, llm=None):
    """The checklist agent with its tools from MCP servers: same prompt, same tool text, new wiring."""
    return create_agent(llm or get_model(), tools, system_prompt=block2.PLAN_PROMPT, response_format=ClaimDecision)


def arun(agent, max_steps: int = 12):
    """Async version of block2.run: MCP tools can only be called asynchronously."""
    async def system(pkg):
        return await agent.ainvoke({"messages": [("user", pkg.text())]}, config=run_config(max_steps))
    return system


# ------------------------------------------------------------------ B3: elicitation
INTAKE_PROMPT = ("Before the checklist, open a claim file with open_claim. If the documents give no policy number, call "
                 "open_claim without policy_id: the claims system then asks the user. Continue with the policy number "
                 "from its answer. "
                 "If the file is not opened, the policy is not identified: decide request_info (policy_id), and do not "
                 "pick a policy yourself.\n\n" + block2.PLAN_PROMPT)   # without the last sentence, a decline was paid (B3)


async def intake_agent(llm=None, answers: dict | None = None, read_tools: list | None = None):
    """An agent that opens the claim file first. `open_claim` asks the user for a missing policy number.
    `answers` is only used in automated runs (CLAIMS_LAB_NONINTERACTIVE=1); in Colab an input box appears.
    Its tools have no session (a server process per tool call); notebook 03 builds the same agent in `mcp_session`."""
    client = MultiServerMCPClient(mcp_servers.stdio_connections(["comms"]),
                                  callbacks=Callbacks(on_elicitation=make_elicitation_callback(answers=answers)))
    open_claim = [t for t in await client.get_tools() if t.name == "open_claim"]  # offer only this one of the comms tools
    tools = open_claim + (read_tools if read_tools is not None else await mcp_tools())
    return create_agent(llm or get_model(), tools, system_prompt=INTAKE_PROMPT, response_format=ClaimDecision)


# ------------------------------------------------------------------ Stretch (b): the bridge without the adapter library
def mcp_to_langchain(client, mcp_tool_list) -> list[StructuredTool]:
    """What every MCP host does for its model: turn the server's tool list into tool definitions, and forward each
    tool call to the server. `client` is an open fastmcp.Client."""
    def make(t):
        async def call(**kwargs):
            r = await client.call_tool(t.name, kwargs, raise_on_error=False)
            texts = " ".join(c.text for c in r.content if hasattr(c, "text"))
            if r.is_error:
                return "ERROR: " + texts
            return json.dumps(r.structured_content, ensure_ascii=False) if r.structured_content else texts
        return StructuredTool.from_function(coroutine=call, name=t.name, description=t.description or "",
                                            args_schema=t.inputSchema)
    return [make(t) for t in mcp_tool_list]


# ------------------------------------------------------------------ Extra task: test a bridge written in the notebook
async def check_bridge(bridge_fn, server):
    """Notebook 03, extra task: test your bridge on the server from A2 against mcp_to_langchain, with one tool call and
    no model call. Returns the bridge to use: yours if it passes, otherwise ours (mcp_to_langchain), with a message."""
    import inspect
    problem = None
    async with Client(server) as c:
        listing = await c.list_tools()
        ours = {t.name: t for t in mcp_to_langchain(c, listing)}
        try:
            mine = bridge_fn(c, listing)
            if inspect.isawaitable(mine):
                mine = await mine
            names = sorted(str(getattr(t, "name", t)) for t in mine) if isinstance(mine, list) else None
            if names is None:
                problem = f"it returned {type(mine).__name__}, not a list of tools"
            elif not all(isinstance(t, StructuredTool) for t in mine):
                problem = "make(t) does not return a StructuredTool yet (TODO 2)"
            elif names != sorted(ours):
                problem = f"it made the tools {names}, the server offers {sorted(ours)}"
            else:
                by_name = {t.name: t for t in mine}
                differ = [n for n, t in ours.items() if (by_name[n].description or "") != (t.description or "")]
                if differ:
                    problem = f"the description of {differ[0]} is not the server's"
                else:
                    got = str(await by_name["lookup_policy"].ainvoke({"policy_id": "P-1001"}))
                    if "P-1001" not in got:
                        problem = f"lookup_policy('P-1001') returned {got[:80]!r}, not the policy"
        except Exception as e:  # noqa: BLE001
            problem = f"{type(e).__name__}: {e}"
    if problem:
        print(f"Your bridge is not finished yet ({problem}).\nThe cells below use ours, block3.mcp_to_langchain.")
        return mcp_to_langchain
    print(f"Your bridge passes: the same {len(ours)} tools and descriptions as the server's, and lookup_policy('P-1001') "
          f"returns the policy:\n  {got[:160]}...")
    return bridge_fn


# Notebook 03, Extra 2: our text for the prompt the server offers, used when the participant's REVIEW_PROMPT is empty.
REVIEW_PROMPT = """Review claim package {package_id} for Lakeside Insurance AG. Answer four questions in order, each with a tool; the first "no" decides.
1. Is the policy known? lookup_policy, or find_policies by name and plate. No single match: request_info.
2. Is the damage covered on that day? The date lies between valid_from and valid_to, and the kind of damage is in covers. No: decline.
3. Are the documents there? An invoice for every repair; for a theft also a police report. No: request_info.
4. Does everything look normal? search_claims_by_invoice for an invoice used before, search_terms for what the terms exclude. No: refer_adjuster.
If all four pass: above CHF 5,000 refer_adjuster, otherwise fast_track with the payout from calc_payout. Give the reason."""

# Notebook 03, Extra 2: the assistant without our instructions, as in a host like Claude Desktop
GENERIC_PROMPT = "You are a helpful assistant. You can use the tools of the systems you are connected to."


# ------------------------------------------------------------------ Notebook 03, Extra 1: the server asks for approval
LIMIT_CHF = 5000


def _add_approve_payout(server) -> None:
    @server.tool
    async def approve_payout(package_id: str, invoice_chf: float, ctx: Context) -> str:
        """Get the approval for paying a claim. Up to CHF 5,000 the claim is approved at once; above, a person decides."""
        if invoice_chf <= LIMIT_CHF:
            return f"approved: the invoice of CHF {invoice_chf:,.0f} on {package_id} is within the fast-track limit"
        answer = await ctx.elicit(f"{package_id}: the invoice of CHF {invoice_chf:,.0f} is above the CHF 5,000 limit. "
                                  "Approve the payment?", response_type=None)
        if answer.action == "accept":
            return f"approved by a person: the invoice of CHF {invoice_chf:,.0f} on {package_id}"
        return f"NOT approved: the invoice of CHF {invoice_chf:,.0f} on {package_id} goes to an adjuster"


APPROVAL_SOURCE = '''"""Approval before a payment (notebook 03, Extra 1): above CHF 5,000 the server asks a person."""
from fastmcp import Context, FastMCP

mcp = FastMCP("lakeside-approvals", instructions="Approval of payments.")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def approve_payout(package_id: str, invoice_chf: float, ctx: Context) -> str:
    """Get the approval for paying a claim. Up to CHF 5,000 the claim is approved at once; above, a person decides."""
    if invoice_chf <= 5000:
        return f"approved: the invoice of CHF {invoice_chf:,.0f} on {package_id} is within the fast-track limit"
    answer = await ctx.elicit(f"{package_id}: the invoice of CHF {invoice_chf:,.0f} is above the CHF 5,000 limit. "
                              "Approve the payment?", response_type=None)
    if answer.action == "accept":
        return f"approved by a person: the invoice of CHF {invoice_chf:,.0f} on {package_id}"
    return f"NOT approved: the invoice of CHF {invoice_chf:,.0f} on {package_id} goes to an adjuster"


if __name__ == "__main__":
    mcp.run(show_banner=False)
'''


async def check_approval_server(path: str | Path = "approval_server.py") -> str:
    """Notebook 03, Extra 1: the participant's server file if its approve_payout asks above CHF 5,000 and only
    there; otherwise our version, written next to it as approval_server_ours.py. Returns the file to start."""
    path = Path(path)
    try:
        server = _import_server(path)
    except Exception as e:  # noqa: BLE001
        problem = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    else:
        problem = await _approval_problem(server)
    if not problem:
        return str(path)
    ours = path.with_name("approval_server_ours.py")
    ours.write_text(APPROVAL_SOURCE, encoding="utf-8")
    print(f"{path.name} is not finished yet ({problem}), so this cell uses ours, {ours.name}.\n")
    return str(ours)


async def _approval_problem(server) -> str | None:
    """Why the server's approve_payout is not finished, or None if it asks above the limit and only there."""
    from fastmcp.client.elicitation import ElicitResult
    asked = []

    async def decline(message, response_type, params, context):
        asked.append(message)
        return ElicitResult(action="decline")
    try:
        async with Client(server, elicitation_handler=decline) as c:
            if "approve_payout" not in [t.name for t in await c.list_tools()]:
                return "the server has no tool approve_payout"
            low = await c.call_tool("approve_payout", {"package_id": "PKG-001", "invoice_chf": 1150})
            await c.call_tool("approve_payout", {"package_id": "PKG-006", "invoice_chf": 8600})
    except Exception as e:  # noqa: BLE001
        return ("it returns nothing above CHF 5,000" if "None is not of type" in str(e)
                else f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
    if len(asked) != 1:
        return "it should ask above CHF 5,000 and only there"
    if not low.data:
        return "it returns nothing up to CHF 5,000"
    return None


async def ensure_approve_payout(server) -> None:
    """Keep the participant's approve_payout if it asks above the limit and answers below it; otherwise use ours."""
    from fastmcp.client.elicitation import ElicitResult
    asked = []

    async def decline(message, response_type, params, context):
        asked.append(message)
        return ElicitResult(action="decline")

    problem = None
    try:
        async with Client(server, elicitation_handler=decline) as c:
            if "approve_payout" not in [t.name for t in await c.list_tools()]:
                problem = "your server has no tool approve_payout"
            else:
                low = await c.call_tool("approve_payout", {"package_id": "PKG-001", "invoice_chf": 950})
                await c.call_tool("approve_payout", {"package_id": "PKG-006", "invoice_chf": 8600})
                if len(asked) != 1:   # once: for the invoice above the limit only
                    problem = "it should ask above CHF 5,000 and only there"
                elif not low.data:
                    problem = "it returns nothing up to CHF 5,000"
    except Exception as e:  # noqa: BLE001 - any error in the participant's tool means: use ours
        problem = ("it returns nothing above CHF 5,000" if "None is not of type" in str(e)
                   else f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
    if problem:
        print(f"approve_payout is not finished yet ({problem}), so this cell uses ours.\n")
        if "no tool" not in problem:
            server.local_provider.remove_tool("approve_payout")
        _add_approve_payout(server)
