"""MCP server 'lakeside-policy-docs': Lakeside's general terms for car insurance.

Run as a separate process (stdio):  python -m claims_lab.mcp_servers.policy_docs
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from claims_lab import data
from claims_lab import tools as T
from claims_lab.mcp_servers import READ_ONLY
from claims_lab.reference import block2 as B

mcp = FastMCP("lakeside-policy-docs", instructions="Lakeside's general terms for car insurance.")


@mcp.tool(annotations=READ_ONLY, description=B.search_terms.description)
def search_terms(query: str) -> str:
    return json.dumps(T.search_terms(query), ensure_ascii=False)


@mcp.resource("terms://car")
def terms() -> str:
    """The general terms for car insurance in full (§1 to §4)."""
    return data.terms()


if __name__ == "__main__":
    mcp.run(show_banner=False)
