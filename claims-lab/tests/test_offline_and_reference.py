"""Offline provider, Block 1-2 reference implementations, helpers and the cheat sheet (no API calls)."""
import pytest
from langchain.agents import create_agent

import claims_lab as cl
from claims_lab.offline import OfflineChatModel
from claims_lab.reference import block1


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")


def test_get_model_is_offline_and_free():
    llm = cl.get_model()
    assert isinstance(llm, OfflineChatModel)
    d = llm.with_structured_output(cl.ClaimDecision).invoke("Decide this claim: " + cl.load_package("PKG-012").text())
    assert isinstance(d, cl.ClaimDecision)
    from claims_lab.costs import price_for
    assert price_for("offline") == (0.0, 0.0)


def test_offline_model_calls_course_tools_with_args_from_the_package():
    bound = cl.get_model().bind_tools(block1.AGENT_TOOLS)
    ai = bound.invoke(cl.load_package("PKG-012").text())
    assert ai.tool_calls[0]["name"] == "lookup_policy" and ai.tool_calls[0]["args"] == {"policy_id": "P-10001"}


def test_offline_model_works_with_create_agent_and_a_response_format():
    agent = create_agent(cl.get_model(), block1.AGENT_TOOLS, response_format=cl.ClaimDecision)
    result = agent.invoke({"messages": [("user", cl.load_package("PKG-015").text())]}, config=cl.run_config())
    assert isinstance(result["structured_response"], cl.ClaimDecision)


def test_reference_workflow():
    assert block1.workflow(cl.load_package("PKG-012")).payout_chf == 950
    assert block1.workflow(cl.load_package("PKG-019")).decision == "decline"
    assert block1.naive_workflow(cl.load_package("PKG-015")).decision == "fast_track"     # the teaching moment
    assert block1.workflow(cl.load_package("PKG-015")).decision == "refer_adjuster"       # after Exercise B


def test_reference_agent_loop_stops_and_answers():
    messages = block1.run_loop(cl.load_package("PKG-012"), max_steps=8)
    assert messages[-1].type == "ai" and not messages[-1].tool_calls
    assert len([m for m in messages if m.type == "ai"]) <= 8
    assert isinstance(block1.agent(cl.load_package("PKG-012")), cl.ClaimDecision)


def test_reference_systems_run_on_the_scoreboard():
    df = cl.scoreboard(block1.workflow, n=8, show=False, label="offline workflow")
    assert df["error"].isna().all() or (df["error"] == "").all()


def test_offline_replay_cache_stays_out_of_the_package(tmp_path):
    from claims_lab.cache import _default_path
    assert str(_default_path()).startswith(str(cl.data.workdir()))


def test_cheat_sheet():
    text = cl.cheat_sheet(show=False)
    assert "Four possible outcomes" in text and "Five questions" in text
    assert cl.cheat_sheet() is None   # shown, not returned (no double output in notebooks)


# ------------------------------------------------------------------ Block 2
from claims_lab.reference import block2  # noqa: E402


def test_course_tools_always_return_json_text():
    """Every course tool returns text, so an empty list reaches the model as '[]', not as an empty message."""
    from claims_lab.reference import block1
    t = block2.search_claims_by_invoice
    msg = t.invoke({"type": "tool_call", "name": t.name, "args": {"invoice_no": "NEW-0001"}, "id": "1"})
    assert msg.content == "[]"
    for t in block1.AGENT_TOOLS + block2.FINAL_TOOLS + block2.BAD_TOOLS:
        assert t.func.__annotations__.get("return") in (str, "str"), t.name


def test_block2_agents_run_and_return_decisions():
    for agent in (block2.block1_agent(), block2.bad_agent(), block2.fixed_tools_agent(), block2.final_agent(),
                  block2.summarizing_agent()):
        assert isinstance(block2.run(agent)(cl.load_package("PKG-025"))["structured_response"], cl.ClaimDecision)
    assert isinstance(block2.plan_then_execute(cl.load_package("PKG-012"))["structured_response"], cl.ClaimDecision)


def test_fixed_tools_keep_the_bad_tools_capabilities():
    assert len(block2.FIXED_TOOLS) == len(block2.BAD_TOOLS) == 6
    assert {t.name for t in block2.FINAL_TOOLS} >= {"find_policies", "search_claims_by_invoice"}


def test_run_helper_keeps_the_scoreboard_guard():
    """run() passes its own config (step cap); the scoreboard's per-run call limit must still apply."""
    from test_spending_guards import LoopingFakeModel, search
    fake = LoopingFakeModel()
    agent = create_agent(fake, [search])
    df = cl.scoreboard(block2.run(agent, max_steps=100), n=1, show=False)
    assert "BudgetExceeded" in df["error"].iloc[0] and fake.calls <= 25


def test_spot_check_and_context_breakdown(capsys):
    cl.spot_check(block2.run(block2.final_agent()), "PKG-012", "PKG-015")
    out = capsys.readouterr().out
    assert "PKG-012" in out and "expected" in out
    result = block2.run(block2.final_agent())(cl.load_package("PKG-012"))
    parts = cl.context_breakdown(result, tools=block2.FINAL_TOOLS, system_prompt=block2.PLAN_PROMPT)
    assert parts["tool definitions"] > 0 and parts["system prompt"] > 0 and parts["tool results"] > 0


def test_a_motor_windscreen_gets_the_glass_deductible_whatever_the_model_calls_it():
    """The model sometimes labels a windscreen 'glass_breakage' (the household term); cover and deductible agree."""
    from claims_lab.reference import block1
    policy = cl.tools.lookup_policy("P-10031")
    assert block1.is_covered(policy, "glass_breakage") == block1.is_covered(policy, "glass")
    assert block1.deductible(policy, "glass_breakage") == block1.deductible(policy, "glass") == policy["deductibles_chf"]["glass"]
