"""Block 5: watch shows a tool that is not a specialist agent (customer comms), with its whole result and its model calls."""
import pytest
from langchain_core.messages import AIMessage, ToolMessage

import claims_lab as cl
from claims_lab.reference import block5 as B


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("CLAIMS_LAB_NONINTERACTIVE", "1")


def _writer(pkg, d):
    return cl.get_model().invoke(f"Write a letter about {pkg.package_id}: missing {d.missing_items}.").text


def test_the_recorder_counts_the_model_call_inside_customer_comms():
    rec = B._Recorder()
    comms = B.comms_tool(writer=_writer)
    comms.invoke({"package_id": "PKG-005", "missing_items": ["police_report"], "reason": "A theft needs a police report."},
                 config={"callbacks": [rec]})
    assert rec.tool_calls["ask_customer_comms"][0] == 1 and not rec.wrappers


def test_the_recorder_knows_which_tools_run_a_specialist():
    rec = B._Recorder()
    ask = B.ask_tool(B.coverage_analyst(), "ask_coverage_analyst", B.COVERAGE_DESCRIPTION)
    ask.invoke({"question": "Policy P-1003: does it cover glass?"}, config={"callbacks": [rec]})
    assert rec.wrappers == {"ask_coverage_analyst"}


def test_plain_tool_calls_leave_out_the_specialists_and_the_answer_form():
    messages = [AIMessage("", tool_calls=[{"name": "ask_coverage_analyst", "args": {"question": "q"}, "id": "1"},
                                          {"name": "ask_customer_comms", "args": {"package_id": "PKG-005"}, "id": "2"}]),
                ToolMessage("covered", tool_call_id="1"), ToolMessage("Dear Laura Frei, ...", tool_call_id="2"),
                AIMessage("", tool_calls=[{"name": "ClaimDecision", "args": {}, "id": "3"}]),
                ToolMessage("Returning structured response: decision='request_info'", tool_call_id="3")]
    plain = B._plain_tool_calls(messages, {"ask_coverage_analyst"})
    assert [(tc["name"], text) for tc, text in plain] == [("ask_customer_comms", "Dear Laura Frei, ...")]


def _unfinished(state):
    note = ...
    return ...


def _no_return(state):
    note = "x"


def _finished(state):
    note = "\n\nNote from the pipeline: " + "; ".join(state["judgement"].signs)
    return {"decision": note}


@pytest.mark.parametrize("candidate,reason", [(_unfinished, "still ..."), (_no_return, "returns nothing"),
                                              (None, "not a function")])
def test_an_unfinished_team_node_is_replaced_by_ours(candidate, reason, capsys):
    node = B.check_team_node(candidate, B.build_team())
    out = capsys.readouterr().out
    assert reason in out and node.course_reference
    assert B.check_team_node(node, None) is node and "This is our" in capsys.readouterr().out


def test_a_finished_team_node_is_used(capsys):
    assert B.check_team_node(_finished, None) is _finished and "Yes" in capsys.readouterr().out
