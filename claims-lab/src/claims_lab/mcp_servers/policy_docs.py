"""MCP server 'lakeside-policy-docs': terms and conditions and the internal claims guideline.

Run as a separate process (stdio):  python -m claims_lab.mcp_servers.policy_docs
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from claims_lab import data
from claims_lab import tools as T
from claims_lab.mcp_servers import READ_ONLY

mcp = FastMCP("lakeside-policy-docs", instructions="Terms and conditions and the internal claims handling guideline.")


@mcp.tool(annotations=READ_ONLY)
def search_policy_terms(query: str) -> str:
    """Search Lakeside's policy terms (motor, household, building) and the internal claims guideline.
    Use it to check what a module covers, exclusions, deductibles, required documents and red flags.
    Returns the 3 most relevant sections with their reference, e.g. 'motor §3'."""
    return json.dumps(T.search_policy_terms(query), ensure_ascii=False)



@mcp.resource("terms://{product}")
def terms(product: str) -> str:
    """Full terms and conditions of a product: motor, household or building."""
    return data.terms(product)


@mcp.resource("terms://guideline")
def guideline() -> str:
    """The internal claims handling guideline (decision rules, required documents, red flags)."""
    return data.terms("claims_guideline")


if __name__ == "__main__":
    mcp.run(show_banner=False)
