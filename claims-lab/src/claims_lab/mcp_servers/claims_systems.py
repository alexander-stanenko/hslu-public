"""MCP server 'lakeside-claims-systems': Lakeside's policies and earlier claims (read only), and the payout calculation.

The tool descriptions are the ones of notebook 02's tools, word for word, so the agent over MCP reads the same text.
Run as a separate process (stdio):  python -m claims_lab.mcp_servers.claims_systems
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from claims_lab import tools as T
from claims_lab.mcp_servers import READ_ONLY
from claims_lab.reference import block2 as B

mcp = FastMCP("lakeside-claims-systems", instructions="Policies and earlier claims of Lakeside Insurance AG.")


@mcp.tool(annotations=READ_ONLY, description=B.lookup_policy.description)
def lookup_policy(policy_id: str) -> str:
    return json.dumps(T.lookup_policy(policy_id), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY, description=B.find_policies.description)
def find_policies(name: str | None = None, plate: str | None = None) -> str:
    return json.dumps(T.find_policies(name, plate), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY, description=B.get_claim_history.description)
def get_claim_history(policy_id: str) -> str:
    return json.dumps(T.get_claim_history(policy_id), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY, description=B.search_claims_by_invoice.description)
def search_claims_by_invoice(invoice_no: str) -> str:
    return json.dumps(T.search_claims_by_invoice(invoice_no), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY, description=B.calc_payout.description)
def calc_payout(amount_chf: float, deductible_chf: float) -> str:
    return json.dumps(T.calc_payout(amount_chf, deductible_chf), ensure_ascii=False)


@mcp.resource("policy://{policy_id}")
def policy_resource(policy_id: str) -> dict:
    """The full policy record as a resource (for applications that attach it as context)."""
    return T.lookup_policy(policy_id)


@mcp.prompt
def review_claim(package_id: str) -> str:
    """Standard instruction for reviewing one claim package with the clerk's cheat sheet."""
    return (f"Review claim package {package_id}. Work through the four questions of the clerk's cheat sheet in order "
            f"and decide: fast_track, request_info, refer_adjuster or decline. Give the reason, and the payout for fast_track.")


if __name__ == "__main__":
    mcp.run(show_banner=False)
