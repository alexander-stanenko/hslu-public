import asyncio
import json

from fastmcp import Client
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import claims_lab as cl
from claims_lab.cache import SQLiteReplayCache
from claims_lab.costs import CostMeter


def oracle_system(pkg):
    e = cl.label(pkg.package_id)["expected"]
    return cl.ClaimDecision(decision=e["decision"], payout_chf=e["payout_chf"], missing_items=e["missing_items"],
                            issues=e["issues"], reasoning="from labels")


def test_perfect_system_scores_perfectly():
    df = cl.scoreboard(oracle_system, split="dev", show=False)
    s = cl.scoreboard_history(plot=False).iloc[-1]
    assert len(df) == 16 and s["accuracy"] == 1.0 and s["unsafe_approvals"] == 0 and s["issue_recall"] == 1.0
    assert s["payout_accuracy"] == 1.0


def test_always_approve_is_unsafe():
    cl.scoreboard(lambda p: {"decision": "fast_track", "payout_chf": 0, "reasoning": "yolo"}, split="dev", show=False, label="yolo")
    s = cl.scoreboard_history(plot=False).iloc[-1]
    assert s["unsafe_approvals"] == 16 - 6 and s["accuracy"] == round(6 / 16, 3) and s["payout_accuracy"] == 0.0


def test_errors_are_reported_not_raised():
    def broken(pkg):
        raise RuntimeError("boom")
    df = cl.scoreboard(broken, n=3, show=False)
    assert df["error"].notna().all()


def test_async_scoreboard():
    async def system(pkg):
        await asyncio.sleep(0)
        return oracle_system(pkg)
    df = asyncio.run(cl.ascoreboard(system, n=5, show=False))
    assert df["correct"].all()


def test_stratified_sample_covers_categories():
    from claims_lab.scoreboard import _sample
    picked = _sample("dev", 6, None)
    assert len(picked) == 6
    assert {l["category"] for l in picked} == {"routine", "incomplete", "not_covered", "judgement", "red_flag", "above_limit"}
    assert [l["package_id"] for l in _sample("quick", None, None)] == cl.list_packages("quick")


def test_replay_cache_marks_hits_and_meter_skips_them(tmp_path):
    cache = SQLiteReplayCache(tmp_path / "c.db")
    msg = AIMessage(content="hello", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
                    response_metadata={"model_name": "claude-haiku-5-5"})
    cache.update("p", "llm", [ChatGeneration(message=msg)])
    hit = cache.lookup("p", "llm")
    assert hit[0].message.text == "hello" and hit[0].message.response_metadata["claims_lab_cached"]
    meter = CostMeter()
    meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    meter.on_llm_end(LLMResult(generations=[hit]))
    assert meter.calls == 1 and meter.cache_hits == 1 and abs(meter.cost_usd - (1000 * 0.10 + 100 * 0.50) / 1e6) < 1e-12


def test_mcp_servers_in_memory_with_elicitation():
    from claims_lab.mcp_servers.comms import mcp as comms
    from claims_lab.mcp_servers.claims_systems import mcp as claims
    from claims_lab.mcp_servers.policy_docs import mcp as docs
    from claims_lab.reference.block2 import FINAL_TOOLS

    async def handler(message, response_type, params, context):
        return response_type(policy_id="P-1004")

    async def run():
        async with Client(claims) as c:
            listed = {t.name: t for t in await c.list_tools()}
            assert set(listed) == {"lookup_policy", "find_policies", "get_claim_history", "search_claims_by_invoice", "calc_payout"}
            block2 = {t.name: t.description for t in FINAL_TOOLS}
            assert all(" ".join(t.description.split()) == " ".join(block2[n].split()) for n, t in listed.items())   # the same words as in Block 2
            r = await c.call_tool("lookup_policy", {"policy_id": "P-1001"})
            assert json.loads(r.content[0].text)["holder"] == "Anna Brunner"
            r = await c.call_tool("search_claims_by_invoice", {"invoice_no": "XX-000"})
            assert r.content[0].text == "[]"                                    # an empty list still arrives as text
        async with Client(comms, elicitation_handler=handler) as c:
            r = await c.call_tool("open_claim", {"package_id": "PKG-004"})   # PKG-004 states no policy number
            assert "P-1004" in r.data and "Thomas Meier" in r.data and "elicitation" in r.data
        async with Client(docs) as c:
            text = (await c.read_resource("terms://car"))[0].text
            assert "§2 Deductible and payout" in text and "§4 Documents a claim needs" in text
            r = await c.call_tool("search_terms", {"query": "documents for a theft"})
            assert "§4" in r.content[0].text
    asyncio.run(run())


def test_block3_agent_over_stdio_offline(monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")   # never a paid model in tests; the servers inherit it
    from claims_lab.reference import block3

    async def run():
        tools = await block3.mcp_tools()
        assert len(tools) == 6
        result = await block3.arun(block3.mcp_agent(tools))(cl.load_package("PKG-002"))   # the reused invoice
        assert result["structured_response"].decision == "refer_adjuster"
        agent = await block3.intake_agent(answers={"policy_id": "P-1004"}, read_tools=tools)
        result = await agent.ainvoke({"messages": [("user", cl.load_package("PKG-004").text())]}, config=cl.run_config())
        opened = [m for m in result["messages"] if getattr(m, "name", None) == "open_claim"]
        assert opened and "P-1004" in str(opened[0].content)
    asyncio.run(run())


def test_scoreboard_with_model_level_meter_and_run_level_meter():
    """Regression: model-level SESSION_METER + scoreboard's per-run meter must not crash (handlers are hashed)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from claims_lab.costs import SESSION_METER
    fake = FakeListChatModel(responses=['{"decision": "decline", "reasoning": "x"}'] * 3, callbacks=[SESSION_METER])
    df = cl.scoreboard(lambda pkg: fake.invoke(pkg.text()).content, n=3, show=False)
    assert df["error"].isna().all()


def test_a_replayed_agent_run_replays_every_step(tmp_path, monkeypatch, capsys):
    """Demo cells replay whole agent runs: a later step's history holds the replayed earlier answers, which carry the
    replay mark and a cost of 0, and must still match the recording (notebooks 02, 04 and 05)."""
    from claims_lab import costs
    from claims_lab.reference import block2
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))
    cl.setup(provider="offline", ping=False)
    agent = block2.final_agent()
    calls = []
    for _ in range(2):
        before = costs.SESSION_METER.calls
        with cl.replay(tmp_path / "replay.db"):
            result = block2.run(agent)(cl.load_package("PKG-002"))
        calls.append(costs.SESSION_METER.calls - before)
    assert calls[0] > 1 and calls[1] == 0
    capsys.readouterr()
    cl.show_trace(result)   # the trace still counts the replayed calls, with their recorded cost
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert f"{calls[0]} LLM calls" in last and "all replayed from the recording" in last


def test_scoreboard_counts_claims_sent_to_a_person(capsys):
    cl.scoreboard(oracle_system, label="oracle")
    out = capsys.readouterr().out
    h = cl.scoreboard_history(plot=False).iloc[-1]
    assert h["to_person"] == 3   # PKG-002, PKG-006 and PKG-007
    assert "sent to a person (refer_adjuster): 3 of 8 claims, 3 expected" in out
