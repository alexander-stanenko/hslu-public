"""Spending guards: a runaway agent must be stopped long before it gets expensive (no API calls; fake model)."""
import itertools

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

import claims_lab as cl
from claims_lab.costs import BudgetExceeded, CostMeter

_ids = itertools.count()


class LoopingFakeModel(BaseChatModel):
    """Always asks for another tool call, like the stuck agent during course preparation."""
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "looping-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        n = sum(len(str(m.content)) for m in messages) // 4
        msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "terms"}, "id": f"c{next(_ids)}"}],
                        usage_metadata={"input_tokens": n, "output_tokens": 50, "total_tokens": n + 50},
                        response_metadata={"model_name": "claude-haiku-5-5"})
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def search(q: str) -> str:
    """Search the terms."""
    return "§ " + "policy text " * 200  # ~600 tokens per result, like the real search tool


def test_scoreboard_stops_a_runaway_agent():
    fake = LoopingFakeModel()
    agent = create_agent(fake, [search])
    df = cl.scoreboard(lambda pkg: agent.invoke({"messages": [("user", pkg.text())]}), n=1, show=False)
    assert "BudgetExceeded" in df["error"].iloc[0]
    assert fake.calls <= 25          # without the guard: up to ~5,000 calls (recursion_limit 9,999)


def test_request_size_guard_stops_growing_conversations():
    meter = CostMeter(max_input_tokens=5_000)
    fake = LoopingFakeModel(callbacks=[meter])
    with pytest.raises(BudgetExceeded):
        fake.invoke([HumanMessage("x" * 40_000)])   # ~10k tokens > 5k limit, stopped before sending
    assert fake.calls == 0


def test_spending_limit():
    meter = CostMeter(max_usd=0.0001)
    fake = LoopingFakeModel(callbacks=[meter])
    fake.invoke([HumanMessage("x" * 4_000)])            # first call spends ~$0.0001
    with pytest.raises(BudgetExceeded):
        fake.invoke([HumanMessage("x" * 4_000)])


def test_session_meter_has_limits_by_default():
    from claims_lab.costs import SESSION_METER
    assert SESSION_METER.max_usd is not None and SESSION_METER.max_input_tokens is not None
    assert cl.get_model.__module__  # get_model attaches SESSION_METER (see models.py)


def test_shared_ledger_caps_all_calls(tmp_path, monkeypatch):
    """With CLAIMS_LAB_LEDGER set, every live call is recorded and nothing is sent once the shared cap is reached."""
    from claims_lab import costs
    monkeypatch.setenv("CLAIMS_LAB_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("CLAIMS_LAB_LEDGER_CAP_USD", "0.0001")
    meter = CostMeter(ledger=True)
    fake = LoopingFakeModel(callbacks=[meter])
    fake.invoke([HumanMessage("x" * 4_000)])                   # ~$0.0001 recorded
    assert costs.ledger_total() > 0
    with pytest.raises(BudgetExceeded):
        fake.invoke([HumanMessage("x")])
    assert fake.calls == 1                                     # the second call was never sent
