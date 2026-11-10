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
    d = llm.with_structured_output(cl.ClaimDecision).invoke("Decide this claim: " + cl.load_package("PKG-001").text())
    assert isinstance(d, cl.ClaimDecision)
    from claims_lab.costs import price_for
    assert price_for("offline") == (0.0, 0.0)


def test_offline_model_calls_course_tools_with_args_from_the_package():
    bound = cl.get_model().bind_tools(block1.AGENT_TOOLS)   # plain functions, as in notebook 01
    ai = bound.invoke(cl.load_package("PKG-001").text())
    assert ai.tool_calls[0]["name"] == "lookup_policy" and ai.tool_calls[0]["args"] == {"policy_id": "P-1001"}


def test_offline_model_works_with_create_agent_and_a_response_format():
    agent = create_agent(cl.get_model(), block1.AGENT_TOOLS, response_format=cl.ClaimDecision)
    result = agent.invoke({"messages": [("user", cl.load_package("PKG-002").text())]}, config=cl.run_config())
    assert isinstance(result["structured_response"], cl.ClaimDecision)


def test_reference_workflow():
    assert block1.workflow(cl.load_package("PKG-001")).payout_chf == 650
    assert block1.workflow(cl.load_package("PKG-003")).decision == "decline"
    assert block1.workflow(cl.load_package("PKG-004")).missing_items == ["policy_id"]
    assert block1.workflow(cl.load_package("PKG-006")).decision == "refer_adjuster"      # above CHF 5,000
    assert block1.naive_workflow(cl.load_package("PKG-002")).decision == "fast_track"     # the teaching moment
    assert block1.workflow(cl.load_package("PKG-002")).decision == "refer_adjuster"       # after Exercise B


def test_reference_agent_loop_stops_and_answers():
    messages = block1.run_loop(cl.load_package("PKG-002"), max_steps=8)
    assert messages[-1].role == "assistant" and not messages[-1].tool_calls      # the reply object of OpenAI's client
    replies = [m for m in messages if not isinstance(m, dict)]
    assert len(replies) <= 8
    called = {c.id: c.function.name for m in replies for c in m.tool_calls or []}
    assert any(isinstance(m, dict) and m["role"] == "tool" and called[m["tool_call_id"]] == "search_claims_by_invoice"
               and "P-1002" in m["content"] for m in messages)
    assert block1.outcome(messages) == "refer_adjuster"
    assert isinstance(block1.agent(cl.load_package("PKG-002")), cl.ClaimDecision)


def test_reference_agent_loop_cut_short_has_no_answer():
    messages = block1.run_loop(cl.load_package("PKG-002"), max_steps=1)
    assert block1.outcome(messages).startswith("no answer")


def test_chatbot_and_workflow_use_openais_client():
    from openai.types.chat import ParsedChatCompletion
    client = cl.get_client()
    response = client.chat.completions.parse(model=block1.MODEL, messages=[{"role": "user", "content": "Decide: "
                                             + cl.load_package("PKG-001").text()}], response_format=cl.ClaimDecision)
    assert isinstance(response, ParsedChatCompletion) and isinstance(response.choices[0].message.parsed, cl.ClaimDecision)
    assert isinstance(block1.chatbot(cl.load_package("PKG-001")), cl.ClaimDecision)


def test_client_answers_are_replayed(tmp_path):
    pkg = cl.load_package("PKG-002")
    with cl.replay(path=tmp_path / "replay.db"):
        first = block1.run_loop(pkg)
    with cl.replay(path=tmp_path / "replay.db"):
        again = block1.run_loop(pkg)
    assert [m.model_dump() if not isinstance(m, dict) else m for m in first] == \
           [m.model_dump() if not isinstance(m, dict) else m for m in again]
    from claims_lab.client import info_of
    assert info_of(again[-1])[3] is True        # replayed


def test_scoreboard_counts_each_client_call_once():
    df = cl.scoreboard(block1.agent, ids=["PKG-002"], show=False, label="offline agent")
    assert df["llm_calls"].iloc[0] == 5        # 4 steps of the loop, 1 call that fills in the ClaimDecision


def test_reference_systems_run_on_the_scoreboard():
    df = cl.scoreboard(block1.workflow, n=8, show=False, label="offline workflow")
    assert df["error"].isna().all() or (df["error"] == "").all()


def test_offline_replay_cache_stays_out_of_the_package(tmp_path):
    from claims_lab.cache import _default_path
    assert str(_default_path()).startswith(str(cl.data.workdir()))


def test_cheat_sheet():
    text = cl.cheat_sheet(show=False)
    assert "Four possible outcomes" in text and "Four questions" in text and "CHF 5,000" in text
    assert cl.cheat_sheet() is None   # shown, not returned (no double output in notebooks)


# ------------------------------------------------------------------ Block 2
from claims_lab.reference import block2  # noqa: E402


def test_course_tools_always_return_json_text():
    """Every course tool returns text, so an empty list reaches the model as '[]', not as an empty message."""
    from claims_lab.reference import block1
    t = block2.search_claims_by_invoice
    msg = t.invoke({"type": "tool_call", "name": t.name, "args": {"invoice_no": "NEW-0001"}, "id": "1"})
    assert msg.content == "[]"
    assert block1.search_claims_by_invoice("NEW-0001") == "[]"   # notebook 01's plain functions too
    for t in block1.AGENT_TOOLS + block2.FINAL_TOOLS + block2.BAD_TOOLS:
        fn = getattr(t, "func", t)
        assert fn.__annotations__.get("return") in (str, "str"), fn.__name__


def test_block2_agents_run_and_return_decisions():
    for agent in (block2.block1_agent(), block2.bad_agent(), block2.fixed_tools_agent(), block2.final_agent(),
                  block2.summarizing_agent()):
        assert isinstance(block2.run(agent)(cl.load_package("PKG-007"))["structured_response"], cl.ClaimDecision)
    assert isinstance(block2.plan_then_execute(cl.load_package("PKG-001"))["structured_response"], cl.ClaimDecision)


def test_fixed_tools_keep_the_bad_tools_capabilities():
    assert len(block2.FIXED_TOOLS) == len(block2.BAD_TOOLS) == 5
    assert [t.name for t in block2.FINAL_TOOLS] == [t.name for t in block2.FIXED_TOOLS] + ["find_policies"]


def test_run_helper_keeps_the_scoreboard_guard():
    """run() passes its own config (step cap); the scoreboard's per-run call limit must still apply."""
    from test_spending_guards import LoopingFakeModel, search
    fake = LoopingFakeModel()
    agent = create_agent(fake, [search])
    df = cl.scoreboard(block2.run(agent, max_steps=100), n=1, show=False)
    assert "BudgetExceeded" in df["error"].iloc[0] and fake.calls <= 25


def test_spot_check_and_context_breakdown(capsys):
    cl.spot_check(block2.final_agent(), "PKG-001", "PKG-002")   # an agent is accepted directly (scoreboard.as_system)
    out = capsys.readouterr().out
    assert "PKG-001 (the routine windscreen)" in out and "PKG-002 (the reused invoice)" in out and "expected" in out
    result = block2.run(block2.final_agent())(cl.load_package("PKG-001"))
    parts = cl.context_breakdown(result, tools=block2.FINAL_TOOLS, system_prompt=block2.PLAN_PROMPT)
    assert parts["tool definitions"] > 0 and parts["system prompt"] > 0 and parts["tool results"] > 0



def test_replay_key_ignores_docstring_indentation():
    """Python 3.13+ strips docstring indentation, 3.12 keeps it: a recording must replay on both."""
    from claims_lab.cache import _llm_key
    py312 = "[('tools', [{'description': \"Look up a policy. Returns the holder,\\n    the car and its plate.\"}])]"
    py313 = "[('tools', [{'description': \"Look up a policy. Returns the holder,\\nthe car and its plate.\"}])]"
    assert _llm_key(py312) == _llm_key(py313)
