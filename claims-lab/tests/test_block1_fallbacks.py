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
    """The notebook's `client`; the checks must never call it."""

    def __getattr__(self, name):
        raise AssertionError("the check called the real model")


def _notebook(body: str):
    """Define run_loop the way notebook 01 does: in a namespace with `client`, `TOOLS`, `tools_by_name`, `MODEL`
    and `SYSTEM_PROMPT`."""
    ns = {"client": _NoModel(), "TOOLS": block1.TOOLS, "tools_by_name": block1.TOOLS_BY_NAME, "MODEL": block1.MODEL,
          "SYSTEM_PROMPT": block1.SYSTEM_PROMPT, "cl": cl}
    exec("import json\n" + textwrap.dedent(body), ns)
    return ns


LOOP_HEAD = """
def run_loop(pkg, max_steps=8):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pkg.text()}]
    for step in range(max_steps):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        reply = response.choices[0].message
        messages.append(reply)
"""
FINISHED = LOOP_HEAD + """
        if not reply.tool_calls:
            return messages
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = tools_by_name[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return messages
"""
EMPTY = LOOP_HEAD + """
        for call in reply.tool_calls:
            pass
    return messages
"""
NO_STOP = LOOP_HEAD + """
        for call in reply.tool_calls or []:
            args = json.loads(call.function.arguments)
            result = tools_by_name[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return messages
"""
GUARDED = """
def run_loop_guarded(pkg, max_steps=8, max_usd=0.01):
    meter = cl.CostMeter(max_usd=max_usd)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pkg.text()}]
    try:
        for step in range(max_steps):
            meter.check()
            response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
            %s
            reply = response.choices[0].message
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                args = json.loads(call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": tools_by_name[call.function.name](**args)})
    except cl.BudgetExceeded as e:
        print("stopped:", e)
    return messages
"""


def test_finished_loop_is_kept_and_the_model_is_restored(capsys):
    ns = _notebook(FINISHED)
    model = ns["client"]
    assert block1.check_my_loop(ns["run_loop"]) is ns["run_loop"]
    assert ns["client"] is model
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
    assert any("refer_adjuster" in p for p in problems) and any("PKG-001" in p for p in problems)
    assert block1._invoice_check_problems(lambda facts, policy: None) == ["it lets the reused invoice GH-3318 of PKG-002 pass"]
    assert block1._invoice_check_problems(lambda facts, policy: 1 / 0)[0].startswith("it raised ZeroDivisionError")


@pytest.mark.parametrize("text,expected", [
    ("Decision: refer_adjuster. The invoice was used before.", "refer_adjuster"),
    ("**Decision:** decline — not covered", "decline"),
    ("I recommend we fast-track this claim; payout CHF 650.", "fast_track"),
    ("This claim should be referred to an adjuster.", "refer_adjuster"),
    ("I am not sure.", "unclear"),
])
def test_outcome_is_read_from_the_final_answer(text, expected):
    assert block1.outcome([{"role": "assistant", "content": text}]) == expected
    assert block1.outcome([AIMessage(content=text)]) == expected       # LangChain's messages too


def test_outcome_of_a_run_stopped_before_the_answer():
    call = {"id": "c1", "type": "function", "function": {"name": "lookup_policy", "arguments": "{}"}}
    assert block1.outcome([{"role": "assistant", "content": None, "tool_calls": [call]}]).startswith("no answer")
    assert block1.outcome([{"role": "tool", "tool_call_id": "c1", "content": "{}"}]).startswith("no answer")


def test_guarded_loop_finished_is_kept(capsys):
    ns = _notebook(GUARDED % "meter.add(response)")
    assert block1.check_guarded_loop(ns["run_loop_guarded"]) is ns["run_loop_guarded"]
    assert "passes our test" in capsys.readouterr().out
    assert isinstance(ns["client"], _NoModel)


def test_guarded_loop_without_add_falls_back(capsys):
    ns = _notebook(GUARDED % "pass")
    assert block1.check_guarded_loop(ns["run_loop_guarded"]) is block1.run_loop_guarded
    assert "second call" in capsys.readouterr().out


def test_guarded_loop_without_except_falls_back(capsys):
    ns = _notebook((GUARDED % "meter.add(response)").replace("cl.BudgetExceeded", "KeyError"))
    assert block1.check_guarded_loop(ns["run_loop_guarded"]) is block1.run_loop_guarded
    assert "does not catch cl.BudgetExceeded" in capsys.readouterr().out


def test_a_check_run_again_on_our_version_says_it_is_ours(capsys):
    ours = block1.check_my_invoice_check(lambda facts, policy: None)   # unfinished: switches to ours
    capsys.readouterr()
    assert block1.check_my_invoice_check(ours) is ours                 # the cell run a second time
    out = capsys.readouterr().out
    assert "This is our check_duplicate_invoice" in out and "Using yours" not in out


def test_rules_tool_extra_runs_a_fifth_tool_without_arguments():
    """Notebook 01, Extra 1: run_loop takes the tool descriptions and the functions; get_rules has no arguments."""
    from types import SimpleNamespace
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_rules", arguments=""))
    functions = {**block1.TOOLS_BY_NAME, "get_rules": block1.get_rules}
    assert "cheat sheet" in block1._run_tool(call, functions)
    assert "Unknown tool" in block1._run_tool(call)                     # without it, the four tools of Part C
    assert block1.RULES_TOOL["function"]["parameters"]["properties"] == {}
    messages = block1.run_loop(cl.load_package("PKG-002"), tools=block1.TOOLS + [block1.RULES_TOOL], functions=functions)
    assert block1.outcome(messages) == "refer_adjuster"
