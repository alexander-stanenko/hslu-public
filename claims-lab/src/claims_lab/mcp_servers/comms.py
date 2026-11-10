"""MCP server 'lakeside-comms': the claims inbox and actions with side effects (opening a claim file, customer
e-mail, recording decisions). `open_claim` asks the user for a missing policy number (elicitation).

Run as a separate process (stdio):  python -m claims_lab.mcp_servers.comms
"""
from __future__ import annotations

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from claims_lab import data
from claims_lab import tools as T

mcp = FastMCP("lakeside-comms", instructions="Claims inbox and outgoing actions. Write tools have side effects.")

mcp.tool(T.read_package, annotations={"readOnlyHint": True})
mcp.tool(T.request_documents, annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
mcp.tool(T.record_decision, annotations={"readOnlyHint": False, "destructiveHint": True})


@mcp.tool(annotations={"readOnlyHint": True})
def list_inbox(split: str = "dev") -> list[str]:
    """List the package IDs waiting in the claims inbox."""
    return data.list_packages(split)


class PolicyNumber(BaseModel):
    policy_id: str = Field(description="Policy number, e.g. P-1001")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def open_claim(package_id: str, ctx: Context, policy_id: str | None = None) -> str:
    """Open a claim file for a package in the claims system. Call it without policy_id if the documents do not state
    a policy number: the claims system then asks the user for it."""
    source = "stated in the documents"
    if not policy_id:
        answer = await ctx.elicit(f"Package {package_id}: which policy number does this claim belong to?", response_type=PolicyNumber)
        if answer.action != "accept":
            return f"Claim file NOT opened for {package_id}: the user did not provide a policy number."
        policy_id, source = answer.data.policy_id, "given by the user when asked (elicitation)"
    policy = T.lookup_policy(policy_id)
    if "error" in policy:
        return f"Claim file NOT opened: {policy['error']}"
    return f"Claim file opened for {package_id} under policy {policy_id}. Policy number: {source}. Holder: {policy['holder']}."


@mcp.resource("package://{package_id}")
def package(package_id: str) -> str:
    """All documents of a claim package."""
    return data.load_package(package_id).text()


if __name__ == "__main__":
    mcp.run(show_banner=False)
