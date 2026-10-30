"""Answering MCP elicitation requests (a server asking the *user* for missing information).

In Colab the default handler shows an input box. In automated runs (CLAIMS_LAB_NONINTERACTIVE=1) it
answers from a dict, or declines.
"""
from __future__ import annotations

import os

from mcp.types import ElicitResult


def _interactive() -> bool:
    return os.environ.get("CLAIMS_LAB_NONINTERACTIVE") != "1"


def make_elicitation_callback(answers: dict | None = None, interactive: bool | None = None):
    """Create an `on_elicitation` callback for langchain_mcp_adapters `Callbacks`.

    Args:
        answers: pre-defined answers by field name, e.g. {"policy_id": "P-10014"} (used when not interactive).
        interactive: ask the person at the keyboard via input(). Default: True unless CLAIMS_LAB_NONINTERACTIVE=1.
    """
    answers = answers or {}
    ask = _interactive() if interactive is None else interactive

    async def on_elicitation(mcp_context, params, context) -> ElicitResult:
        props = (getattr(params, "requestedSchema", None) or {}).get("properties", {})
        print(f"\nThe MCP server asks: {params.message}")
        content = {}
        for fname, spec in props.items():
            if ask:
                value = input(f"   {spec.get('title', fname)} (leave empty to decline): ").strip()
            else:
                value = str(answers.get(fname, "")).strip()
                print(f"   {fname}: {value or '(no answer, so decline)'}")
            if not value:
                return ElicitResult(action="decline")
            content[fname] = int(value) if spec.get("type") == "integer" else float(value) if spec.get("type") == "number" else value
        return ElicitResult(action="accept", content=content)

    return on_elicitation
