"""MCP server 'lakeside-claims-systems': policy administration and claims history (read only).

Run as a separate process (stdio):  python -m claims_lab.mcp_servers.claims_systems
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from claims_lab import tools as T
from claims_lab.mcp_servers import READ_ONLY

mcp = FastMCP("lakeside-claims-systems", instructions="Policy administration and claims history of Lakeside Insurance AG.")


@mcp.tool(annotations=READ_ONLY)
def lookup_policy(policy_id: str) -> str:
    """Look up an insurance policy by its policy number (format 'P-10001'). Returns the holder, product
    (motor/household/building), status, validity dates, the included modules (what is covered), deductibles in CHF
    and the sum insured in CHF. Use it first for every claim."""
    return json.dumps(T.lookup_policy(policy_id), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY)
def find_policies(name: str | None = None, plate: str | None = None, postcode: str | None = None) -> str:
    """Find policies when the policy number is unknown. Search by customer name (exact or partial), vehicle plate
    (e.g. 'LU 248 113') and/or postcode. Returns all matching policies; more than one match means the policy is not
    uniquely identified."""
    return json.dumps(T.find_policies(name, plate, postcode), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY)
def get_claim_history(policy_id: str) -> str:
    """List all earlier claims on a policy (policy number 'P-10001'): date, claim type, amount in CHF, status,
    invoice number and vendor. Use it to spot repeated claims and reused invoice numbers."""
    return json.dumps(T.get_claim_history(policy_id), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY)
def search_claims_by_invoice(invoice_no: str) -> str:
    """Check whether an invoice number was already used in any earlier claim (across all policies).
    An empty list means the invoice number is new."""
    return json.dumps(T.search_claims_by_invoice(invoice_no), ensure_ascii=False)



@mcp.resource("policy://{policy_id}")
def policy_resource(policy_id: str) -> dict:
    """The full policy record as a resource (for applications that attach it as context)."""
    return T.lookup_policy(policy_id)


@mcp.prompt
def review_claim(package_id: str) -> str:
    """Standard instruction for reviewing one claim package."""
    return (f"Review claim package {package_id}. Identify the policy, check that it was active and that the damage type is "
            f"covered, check the required documents and the red flags of the claims guideline, then decide: fast_track, "
            f"request_info, refer_adjuster or decline. Give the payout for fast_track.")


if __name__ == "__main__":
    mcp.run(show_banner=False)
