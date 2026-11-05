"""Readable step-by-step view of what an agent did (messages, tool calls, tool results, tokens)."""
from __future__ import annotations

import json

from .costs import CACHED_FLAG, price_for


def _short(x, n: int) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def show_trace(result, max_chars: int = 220, chart: bool = False) -> None:
    """Print an agent run step by step.

    Args:
        result: the dict returned by agent.invoke(...) (with a "messages" key), or a list of messages.
        max_chars: truncate long contents to this many characters.
        chart: also draw the input and output tokens of each model call.
    """
    messages = result["messages"] if isinstance(result, dict) else result
    calls = tin = tout = replayed = 0
    cost = 0.0
    step = 0
    per_call = []
    call_names = {tc["id"]: tc["name"] for m in messages if m.type == "ai" for tc in m.tool_calls}
    for m in messages:
        kind = m.type
        if kind == "human":
            print(f"USER     {_short(m.text, max_chars)}")
        elif kind == "system":
            print(f"SYSTEM   {_short(m.text, max_chars)}")
        elif kind == "ai":
            step += 1
            meta = m.response_metadata or {}
            u = m.usage_metadata or {}
            per_call.append((u.get("input_tokens", 0), u.get("output_tokens", 0)))
            if u or meta:  # a message without metadata was written by code (e.g. middleware)
                calls += 1
                replayed += bool(meta.get(CACHED_FLAG))  # replayed: free now, shown with what it cost when recorded
                tin += u.get("input_tokens", 0)
                tout += u.get("output_tokens", 0)
                pin, pout = price_for(meta.get("model_name"))
                cost += u.get("input_tokens", 0) * pin / 1e6 + u.get("output_tokens", 0) * pout / 1e6
            if m.text.strip() and m.tool_calls:
                print(f"AI  #{step:<3} (thinks) {_short(m.text, max_chars)}")
            for tc in m.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"AI  #{step:<3} calls {tc['name']}({_short(args, max_chars)})")
            if not m.tool_calls:
                print(f"AI  #{step:<3} ANSWER {_short(m.text, max_chars * 2)}")
        elif kind == "tool":
            status = " (ERROR)" if getattr(m, "status", "") == "error" else ""
            name = m.name or call_names.get(getattr(m, "tool_call_id", None), "?")
            print(f"TOOL     {name}{status}: {_short(m.content if isinstance(m.content, str) else m.text, max_chars)}")
    print(f"Total: {calls} LLM calls, {tin:,} in / {tout:,} out tokens, ${cost:.4f}"
          + (f" ({'all' if replayed == calls else replayed} replayed from the recording: free now, cost as recorded)"
             if replayed else ""))
    if chart and any(i for i, _ in per_call):
        from .style import plot_tokens
        plot_tokens(per_call)


def _tokens(text: str) -> int:
    return max(0, len(text) // 4)  # rough estimate: ~4 characters per token


def context_breakdown(result, tools=None, system_prompt: str | None = None, plot: bool = False) -> dict:
    """What filled the model's context window in the LAST request of an agent run (estimated tokens).

    Args:
        result: the dict returned by agent.invoke(...), or a list of messages.
        tools: the tools the agent had (their definitions are sent with every request).
        system_prompt: the agent's system prompt (create_agent does not keep it in the messages).
        plot: also draw the parts as one stacked bar.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    messages = result["messages"] if isinstance(result, dict) else result
    parts = {"system prompt": 0, "tool definitions": 0, "claim package (user)": 0, "tool results": 0, "model's own steps": 0}
    if system_prompt:
        parts["system prompt"] = _tokens(system_prompt)
    if tools:
        parts["tool definitions"] = _tokens(json.dumps([convert_to_openai_tool(t) for t in tools]))
    last_ai = max((i for i, m in enumerate(messages) if m.type == "ai"), default=len(messages))
    for m in messages[:last_ai]:  # everything that was sent with the last request
        text = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
        if m.type == "system":
            parts["system prompt"] += _tokens(text)
        elif m.type == "human":
            parts["claim package (user)"] += _tokens(text)
        elif m.type == "tool":
            parts["tool results"] += _tokens(text)
        elif m.type == "ai":
            parts["model's own steps"] += _tokens(text) + _tokens(json.dumps(m.tool_calls, default=str))
    total = sum(parts.values()) or 1
    print(f"Context of the last request: about {total:,} tokens (estimated)")
    for name, n in parts.items():
        print(f"  {name:<22} {n:>7,}  {n / total:>4.0%}  {'#' * round(30 * n / total)}")
    ais = [m for m in messages if m.type == "ai"]
    sent = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ais)
    if sent:
        print(f"Over the whole run the model received {sent:,} input tokens in {len(ais)} calls: "
              f"the tool definitions and the growing history are re-sent on every call.")
    if plot:
        from .style import plot_context
        plot_context(parts)
    return parts


GRAPH_STYLES = {"model": "fill:#d6eef4,stroke:#000", "code": "fill:#ffffff,stroke:#000", "person": "fill:#f3d4e4,stroke:#000",
                "visited": "stroke-width:4px"}


def show_graph(app, highlight: list[str] | None = None, kinds: dict | None = None) -> None:
    """Draw a compiled LangGraph graph. kinds marks nodes as model, code or person (by colour); highlight draws the path
    one claim took with a thick border. Without the drawing service (offline, automated runs), print the Mermaid text."""
    import os

    from langchain_core.runnables.graph_mermaid import draw_mermaid_png

    from .models import provider
    text = app.get_graph().draw_mermaid()
    nodes = set(app.get_graph().nodes)
    extra = [f"\tclassDef {k} {v}" for k, v in GRAPH_STYLES.items()]
    for kind in ("model", "code", "person"):
        members = [n for n, k in (kinds or {}).items() if k == kind and n in nodes]
        if members:
            extra.append(f"\tclass {','.join(members)} {kind}")
    visited = [n for n in dict.fromkeys(highlight or []) if n in nodes]
    if visited:
        extra.append(f"\tclass {','.join(visited)} visited")
    text = text.rstrip() + "\n" + "\n".join(extra) + "\n"
    legend = ("Blue: a model call; white: code; pink: a person" if kinds else "") + ("; thick border: the path of this claim" if visited else "")
    if provider() != "offline" and not os.environ.get("CLAIMS_LAB_NONINTERACTIVE"):
        try:
            from IPython.display import Image, display
            display(Image(draw_mermaid_png(text, max_retries=2)))  # rendered by the mermaid.ink web service
            if legend:
                print(legend.strip(" ;"))
            return
        except Exception:  # no network, or the service is busy: draw it locally
            _draw_locally(app, kinds, visited)
            if legend:
                print(legend.strip(" ;"))
            return
    print(text)


def _draw_locally(app, kinds, visited) -> None:
    from .style import plot_graph
    g = app.get_graph()
    fills = {n: GRAPH_STYLES[k].split(",")[0].removeprefix("fill:") for n, k in (kinds or {}).items() if k in GRAPH_STYLES}
    plot_graph(list(g.nodes), [(e.source, e.target, e.conditional) for e in g.edges], fills, set(visited))
