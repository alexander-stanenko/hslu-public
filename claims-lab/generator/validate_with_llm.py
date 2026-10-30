"""Calibration run (not part of the package): how hard is the dataset for the course model?

Three designs of increasing quality on the 28 dev packages, plus an MCP (stdio) smoke test.
Usage: ANTHROPIC_API_KEY=... python generator/validate_with_llm.py

⚠ LIVE AND PAID: about 4 designs x 28 packages x ~5 calls (≈ 600 model calls). Never run it without the
instructor's explicit OK for that run, and prefer n=8 while developing.
"""
import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

import claims_lab as cl
from claims_lab import tools as T
from claims_lab.mcp_servers import stdio_connections

os.environ.setdefault("CLAIMS_LAB_NONINTERACTIVE", "1")
llm = cl.get_model()

# 1. chatbot: one call, no tools
def chatbot(pkg):
    return cl.structured(llm, cl.ClaimDecision).invoke(
        "You are a claims handler at Lakeside Insurance AG. Decide this claim package.\n\n" + pkg.text())

# 2. agent with good tools, short prompt (must discover the rules through search_policy_terms)
SHORT = ("You are a claims handler at Lakeside Insurance AG. Use the tools to verify the claim against the policy, the claim history, "
         "reference data and the terms/guideline (search_policy_terms) before deciding. Never compute in your head.")
agent_short = create_agent(llm, T.REFERENCE_TOOLS, system_prompt=SHORT, response_format=cl.ClaimDecision)

# 3. agent with the guideline in the system prompt
agent_guided = create_agent(llm, T.REFERENCE_TOOLS, system_prompt=SHORT + "\n\n" + cl.terms("claims_guideline"),
                            response_format=cl.ClaimDecision)

def run(agent):  # always with a step cap: without it create_agent allows 9,999 steps (see planning/09 §10)
    return lambda pkg: agent.invoke({"messages": [("user", pkg.text())]}, config=cl.run_config())

# 4. bad tools (Block 2 starting point)
agent_bad = create_agent(llm, T.BAD_TOOLS, system_prompt="You are a claims handler. Decide the claim.", response_format=cl.ClaimDecision)

FAILS = {}
for sysfn, lab in [(chatbot, "1 chatbot, no tools"), (run(agent_bad), "2 agent, bad tools"),
                   (run(agent_short), "3 agent, good tools"), (run(agent_guided), "4 agent, good tools + guideline")]:
    df = cl.scoreboard(sysfn, n=None, label=lab, show=False)
    FAILS[lab] = df[~df["correct"] | (df["payout_ok"] == False)][["package", "category", "expected", "predicted", "expected_payout", "predicted_payout", "error"]]


async def mcp_smoke():
    client = MultiServerMCPClient(stdio_connections(["claims_systems", "policy_docs", "reference_data"]))
    tools = await client.get_tools()
    agent = create_agent(llm, tools, system_prompt=SHORT + "\n\n" + cl.terms("claims_guideline"), response_format=cl.ClaimDecision)

    async def system(pkg):
        return await agent.ainvoke({"messages": [("user", pkg.text())]})
    await cl.ascoreboard(system, n=8, label="5 same, tools via MCP (stdio)", show=False)

asyncio.run(mcp_smoke())
h = cl.scoreboard_history(plot=False)
print(h[["label", "accuracy", "payout_accuracy", "unsafe_approvals", "issue_recall", "errors", "llm_calls_per_pkg", "cost_usd_per_pkg", "seconds_per_pkg"]].to_string())
print("session:", cl.costs.SESSION_METER)
labs = {l["package_id"]: l["case_key"] for l in cl.labels()}
for lab, f in FAILS.items():
    print(f"\n--- failures: {lab}")
    f = f.assign(case=f["package"].map(labs))
    print(f.to_string(index=False))
