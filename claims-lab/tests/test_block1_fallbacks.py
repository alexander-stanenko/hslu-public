"""Notebook 01 fallbacks: an unfinished exercise is detected without a model call, and our version is used instead."""
import textwrap

import pytest
from langchain_core.messages import AIMessage

import claims_lab as cl
from claims_lab.reference import block1


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")


class _NoModel:
    """The notebook's `bound`; the checks must never call it."""

    def invoke(self, *a, **k):
        raise AssertionError("the check called the real model")


def _notebook(body: str):
    """Define run_loop the way notebook 01 does: in a namespace with `bound`, `agent_tools` and `SYSTEM_PROMPT`."""
    ns = {"bound": _NoModel(), "agent_tools": block1.AGENT_TOOLS, "SYSTEM_PROMPT": block1.SYSTEM_PROMPT}
    exec("from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage\n" + textwrap.dedent(body), ns)
    return ns


LOOP_HEAD = """
def run_loop(pkg, max_steps=8):
    tools_by_name = {t.name: t for t in agent_tools}
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(pkg.text())]
    for step in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
"""
FINISHED = LOOP_HEAD + """
        if not ai.tool_calls:
            return messages
        for call in ai.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return messages
"""
EMPTY = LOOP_HEAD + """
        for call in ai.tool_calls:
            pass
    return messages
"""
NO_STOP = LOOP_HEAD + """
        for call in ai.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return messages
"""


def test_finished_loop_is_kept_and_the_model_is_restored(capsys):
    ns = _notebook(FINISHED)
    model = ns["bound"]
    assert block1.check_my_loop(ns["run_loop"]) is ns["run_loop"]
    assert ns["bound"] is model
    assert "passes our test" in capsys.readouterr().out


def test_empty_loop_falls_back_to_ours_with_a_clear_message(capsys):
    ns = _notebook(EMPTY)
    assert block1.check_my_loop(ns["run_loop"]) is block1.run_loop
    out = capsys.readouterr().out
    assert "not finished yet" in out and "TODO 2 and 3" in out and "TODO 1" in out and "reference.block1" in out


def test_loop_without_stop_is_reported():
    assert block1._loop_problems(_notebook(NO_STOP)["run_loop"]) == ["it does not stop when the model has answered (TODO 1)"]


def test_loop_that_raises_falls_back():
    ns = _notebook(LOOP_HEAD + "        raise KeyError('oops')\n")
    assert block1.check_my_loop(ns["run_loop"]) is block1.run_loop


def test_our_loop_and_untestable_loops_are_used_as_they_are():
    assert block1.check_my_loop(block1.run_loop) is block1.run_loop

    def other(pkg):
        return []
    assert block1.check_my_loop(other) is other


def test_invoice_check_finished_empty_and_wrong():
    assert block1.check_my_invoice_check(block1.check_duplicate_invoice) is block1.check_duplicate_invoice
    assert block1.check_my_invoice_check(lambda facts, policy: None) is block1.check_duplicate_invoice
    declines = lambda facts, policy: cl.ClaimDecision(decision="decline", reasoning="x")  # noqa: E731
    problems = block1._invoice_check_problems(declines)
    assert any("refer_adjuster" in p for p in problems) and any("PKG-012" in p for p in problems)
    assert block1._invoice_check_problems(lambda facts, policy: 1 / 0)[0].startswith("it raised ZeroDivisionError")


@pytest.mark.parametrize("text,expected", [
    ("Decision: refer_adjuster. The invoice was used before.", "refer_adjuster"),
    ("**Decision:** decline — not covered", "decline"),
    ("I recommend we fast-track this claim; payout CHF 950.", "fast_track"),
    ("This claim should be referred to an adjuster.", "refer_adjuster"),
    ("Stopped after 8 steps without a decision.", "no decision (step limit)"),
    ("I am not sure.", "unclear"),
])
def test_outcome_is_read_from_the_final_answer(text, expected):
    assert block1.outcome([AIMessage(content=text)]) == expected
