"""Reference for notebook 01: the same claim handled by a chatbot, a workflow and an agent loop.

All three return a ClaimDecision, so they can be compared on the scoreboard:

    from claims_lab.reference import block1
    cl.scoreboard(block1.chatbot, label="chatbot")
    cl.scoreboard(block1.naive_workflow, label="workflow")       # without the invoice check
    cl.scoreboard(block1.workflow, label="workflow + invoice check")
    cl.scoreboard(block1.agent, label="agent loop")

Notebook 01 works without a framework: every model call goes through OpenAI's own Python client (cl.get_client()),
and the messages are plain dicts. Notebook 02 introduces LangChain.

Notebook 01 also uses helpers that need no model call: check_my_invoice_check, check_my_loop and check_guarded_loop
test a participant's exercise and return ours if it is not finished, and outcome reads the outcome from an agent's answer.
"""
from __future__ import annotations

import json
import os

from .. import tools as T
from ..client import get_client
from ..data import Package
from ..schemas import ClaimDecision, ClaimFacts
from . import blank_line_after

FAST_TRACK_LIMIT_CHF = 5000
MODEL = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini")

# ------------------------------------------------------------------ Part A: the chatbot
# The form exists only in the prompt: the keys and the allowed values. Notebook 02 hands the form to the provider instead.
DECISION_FORMAT = """Answer only with a JSON object with these keys:
- "decision": "fast_track" (pay now), "request_info" (something is missing), "decline" (not insured) or "refer_adjuster" (an adjuster must decide)
- "payout_chf": the amount to pay in CHF, only for fast_track; otherwise null
- "missing_items": a list of what the customer must still send, such as "police_report"
- "issues": a list of red flags or reasons, such as "duplicate_invoice"
- "reasoning": two to four sentences"""


def chatbot(pkg: Package, client=None) -> ClaimDecision:
    """One model call: the package goes in, a decision comes out. No database, no terms, no history."""
    client = client or get_client()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Decide this insurance claim.\n" + DECISION_FORMAT + "\n\n" + pkg.text()}])
    answer = response.choices[0].message.content
    return ClaimDecision(**json.loads(answer))


# ------------------------------------------------------------------ Part B: the workflow (a fixed chain)
FACTS_FORMAT = """Answer only with a JSON object with these keys; use null for anything the documents do not give:
- "policy_id": the policy number, such as "P-1001"
- "customer_name" and "plate": the customer's name and the car's plate, such as "LU 248 113"
- "damage": "glass", "hail", "theft", "collision" or "other"
- "damage_date": the day of the damage, as YYYY-MM-DD
- "amount_chf": the invoice total in CHF, as a number
- "invoice_no" and "garage": the invoice's number and the garage
- "documents_present": a list of the kinds of document in the package ("email", "invoice", "police_report")"""


def extract_facts(pkg: Package, client=None) -> ClaimFacts:
    """Step 1 of the workflow: the model only reads; it decides nothing."""
    client = client or get_client()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Extract the key facts of this claim package.\n" + FACTS_FORMAT + "\n\n" + pkg.text()}])
    answer = response.choices[0].message.content
    return ClaimFacts(**json.loads(answer))


def check_duplicate_invoice(facts: ClaimFacts, policy: dict) -> ClaimDecision | None:
    """Exercise B: an extra check. Returns a decision if the check fails, None if the claim passes."""
    if facts.invoice_no and T.search_claims_by_invoice(facts.invoice_no):
        return ClaimDecision(decision="refer_adjuster", issues=["duplicate_invoice"],
                             reasoning=f"Invoice {facts.invoice_no} was already used in an earlier claim.")
    return None


def workflow(pkg: Package, client=None, extra_checks=(check_duplicate_invoice,)) -> ClaimDecision:
    """The clerk's cheat sheet written as code: extract, policy known?, covered that day?, documents there?, extra
    checks, the limit, pay.

    extra_checks: functions (facts, policy) -> ClaimDecision | None, run after the documents check (cheat-sheet question 4).
    """
    facts = extract_facts(pkg, client)
    if not facts.policy_id:                                                         # 1 policy known?
        return ClaimDecision(decision="request_info", missing_items=["policy_id"], reasoning="No policy number in the package.")
    policy = T.lookup_policy(facts.policy_id)
    if "error" in policy:
        return ClaimDecision(decision="request_info", missing_items=["policy_id"], reasoning=policy["error"])
    if facts.damage_date and not policy["valid_from"] <= facts.damage_date <= policy["valid_to"]:   # 2 covered that day?
        return ClaimDecision(decision="decline", issues=["policy_inactive"],
                             reasoning=f"Policy valid {policy['valid_from']} to {policy['valid_to']}, damage on {facts.damage_date}.")
    if facts.damage not in policy["covers"]:
        return ClaimDecision(decision="decline", issues=["not_covered"],
                             reasoning=f"The policy covers {', '.join(policy['covers'])}, not {facts.damage}.")
    missing = [d for d in ("invoice", "police_report")                              # 3 documents there?
               if d not in facts.documents_present and (d == "invoice" or facts.damage == "theft")]
    if missing:
        return ClaimDecision(decision="request_info", missing_items=missing, reasoning=f"Missing: {', '.join(missing)}.")
    for check in extra_checks:                                                      # 4 anything suspicious?
        if (stop := check(facts, policy)) is not None:
            return stop
    amount = facts.amount_chf or 0.0
    if amount > FAST_TRACK_LIMIT_CHF:                                               # the limit
        return ClaimDecision(decision="refer_adjuster", issues=["above_fast_track_limit"],
                             reasoning=f"CHF {amount:,.2f} is above the fast-track limit of CHF {FAST_TRACK_LIMIT_CHF:,}.")
    payout = T.calc_payout(amount, policy["deductible_chf"])
    return ClaimDecision(decision="fast_track", payout_chf=payout,
                         reasoning=f"{facts.damage} is covered; CHF {amount:,.2f} minus the deductible: CHF {payout:,.2f}.")


def naive_workflow(pkg: Package, client=None) -> ClaimDecision:
    """The workflow as first written in the notebook: no extra checks."""
    return workflow(pkg, client, extra_checks=())


# ------------------------------------------------------------------ Part C: the agent loop, from scratch
# Four of Lakeside's systems as plain Python functions. The name, the arguments and the one-line docstring are all the
# model learns about a tool; each returns its result as JSON text.
def lookup_policy(policy_id: str) -> str:
    """Look up a policy by its number, e.g. 'P-1001'."""
    return json.dumps(T.lookup_policy(policy_id))


def get_claim_history(policy_id: str) -> str:
    """List the earlier claims on a policy."""
    return json.dumps(T.get_claim_history(policy_id))


def search_claims_by_invoice(invoice_no: str) -> str:
    """Find earlier claims, on any policy, that used this invoice number."""
    return json.dumps(T.search_claims_by_invoice(invoice_no))


def calc_payout(amount_chf: float, deductible_chf: float) -> str:
    """Payout in CHF: the invoice amount minus the deductible."""
    return json.dumps(T.calc_payout(amount_chf, deductible_chf))


AGENT_TOOLS = [lookup_policy, get_claim_history, search_claims_by_invoice, calc_payout]

# What the model is told about each tool, in the format of OpenAI's API: a name, a description and the arguments as a
# JSON schema. Notebook 01 writes these out by hand; from notebook 02 on, LangChain's @tool builds them from the function.
TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_policy",
        "description": "Look up a policy by its number, e.g. 'P-1001'.",
        "parameters": {"type": "object", "properties": {"policy_id": {"type": "string"}}, "required": ["policy_id"]}}},
    {"type": "function", "function": {
        "name": "get_claim_history",
        "description": "List the earlier claims on a policy.",
        "parameters": {"type": "object", "properties": {"policy_id": {"type": "string"}}, "required": ["policy_id"]}}},
    {"type": "function", "function": {
        "name": "search_claims_by_invoice",
        "description": "Find earlier claims, on any policy, that used this invoice number.",
        "parameters": {"type": "object", "properties": {"invoice_no": {"type": "string"}}, "required": ["invoice_no"]}}},
    {"type": "function", "function": {
        "name": "calc_payout",
        "description": "Payout in CHF: the invoice amount minus the deductible.",
        "parameters": {"type": "object", "properties": {"amount_chf": {"type": "number"}, "deductible_chf": {"type": "number"}},
                       "required": ["amount_chf", "deductible_chf"]}}},
]
TOOLS_BY_NAME = {f.__name__: f for f in AGENT_TOOLS}

SYSTEM_PROMPT = """You are a claims handler at Lakeside Insurance AG.
Decide the claim package: fast_track (pay now), request_info (something is missing), decline (not insured)
or refer_adjuster (a human must decide). Use the tools to check the policy and the claim history before you decide.
Finish with the decision, the payout in CHF if any, and a short reason."""


def _run_tool(call) -> str:
    """Run one tool the model asked for; an unknown tool or bad arguments come back as an error the model can read."""
    tool = TOOLS_BY_NAME.get(call.function.name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool {call.function.name}"})
    try:
        return tool(**json.loads(call.function.arguments))
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"Bad arguments for {call.function.name}: {e}"})


def run_loop(pkg: Package, max_steps: int = 8, client=None) -> list:
    """The agent loop: ask the model, run the tools it asks for, send the results back, repeat until it answers.

    max_steps stops a confused agent (an agent without a limit can loop, and every step re-sends the whole history).
    Returns the full list of messages (cl.show_trace prints it).
    """
    client = client or get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pkg.text()}]
    for step in range(max_steps):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        reply = response.choices[0].message
        messages.append(reply)
        if not reply.tool_calls:                                # no tool requested: this is the answer
            return messages
        for call in reply.tool_calls:                           # run each requested tool, send the result back
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _run_tool(call)})
    return messages


def agent(pkg: Package, max_steps: int = 8, client=None) -> ClaimDecision:
    """The agent loop plus one small call that turns its final answer into a ClaimDecision (for the scoreboard)."""
    client = client or get_client()
    messages = run_loop(pkg, max_steps, client)
    final = answer_text(messages) or "No decision: the run stopped before the model answered."
    response = client.chat.completions.parse(
        model=MODEL, messages=[{"role": "user", "content": "Put this claim decision into the required format:\n\n" + final}],
        response_format=ClaimDecision)
    return response.choices[0].message.parsed


def _plain(m) -> dict:
    from ..client import plain
    return plain(m) if not hasattr(m, "type") else {"role": {"ai": "assistant", "human": "user"}.get(m.type, m.type),
                                                    "content": m.text, "tool_calls": getattr(m, "tool_calls", None)}


def answer_text(messages: list) -> str | None:
    """The model's final answer, or None if the run stopped before the model answered (the last message is a tool
    result, or a reply that still asks for tools)."""
    if not messages:
        return None
    last = _plain(messages[-1])
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return None
    return last.get("content") or ""


# ------------------------------------------------------------------ Notebook 01: test an exercise, or continue with ours
def _invoice_check_problems(check) -> list[str]:
    """What is wrong with a participant's check_duplicate_invoice, tested on two claims without a model call."""
    reused = ClaimFacts(policy_id="P-1002", damage="glass", damage_date="2026-10-18", amount_chf=980.0,
                        invoice_no="GH-3318")          # PKG-002: this invoice was already paid
    fresh = ClaimFacts(policy_id="P-1001", damage="glass", damage_date="2026-10-20", amount_chf=1150.0,
                       invoice_no="GH-4471-NEW")       # a new invoice
    problems = []
    try:
        d = check(reused, T.lookup_policy("P-1002"))
        if d is None:
            problems.append("it lets the reused invoice GH-3318 of PKG-002 pass")
        elif not isinstance(d, ClaimDecision):
            problems.append(f"it returns a {type(d).__name__}, not a cl.ClaimDecision or None")
        elif d.decision != "refer_adjuster":
            problems.append(f"it returns {d.decision} for a reused invoice; the cheat sheet says refer_adjuster")
        if check(fresh, T.lookup_policy("P-1001")) is not None:
            problems.append("it also stops PKG-001, whose invoice is new")
    except Exception as e:  # noqa: BLE001 - any error in the participant's code means: use ours
        problems.append(f"it raised {type(e).__name__}: {e}")
    return problems


@blank_line_after
def check_my_invoice_check(check):
    """Notebook 01, Exercise B: test your check on PKG-002 and PKG-001 (no model call). Returns yours if it passes,
    otherwise our check_duplicate_invoice, with a message."""
    if check is check_duplicate_invoice:          # the cell was run again after it switched to ours
        print("This is our check_duplicate_invoice from claims_lab.reference.block1.")
        return check
    problems = _invoice_check_problems(check)
    if not problems:
        print("Your check_duplicate_invoice passes our test (reused invoice referred, new invoice passes). Using yours.")
        return check
    print(f"Your check_duplicate_invoice is not finished yet ({'; '.join(problems)}). "
          "Using our version from claims_lab.reference.block1.")
    return check_duplicate_invoice


class _ScriptedClient:
    """A stand-in for OpenAI's client, used to test a loop offline: it asks for lookup_policy once, then answers.
    Its answers are real response objects of the openai package, so the loop reads them as it reads the model's."""

    def __init__(self, policy_id: str, model: str = "offline"):
        self.policy_id, self.model, self.requests = policy_id, model, []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion
        self.requests.append(list(kwargs.get("messages", [])))
        if len(self.requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "test_call_1", "type": "function",
                 "function": {"name": "lookup_policy", "arguments": json.dumps({"policy_id": self.policy_id})}}]}
        else:
            message = {"role": "assistant", "content": "Decision: fast_track (a scripted test answer, not a model)."}
        return ChatCompletion.model_validate({
            "id": "test", "object": "chat.completion", "created": 0, "model": self.model,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if "tool_calls" in message else "stop", "message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}})


def _swap_client(loop, fake, **kwargs):
    """Run the participant's function with the notebook's `client` replaced by a stand-in; returns (output, error)."""
    from ..data import load_package
    namespace = loop.__globals__
    real = namespace.get("client")
    namespace["client"] = fake
    try:
        return loop(load_package("PKG-001"), **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, e
    finally:
        namespace["client"] = real


def _uses_client(loop) -> bool:
    code, namespace = getattr(loop, "__code__", None), getattr(loop, "__globals__", None)
    return code is not None and namespace is not None and "client" in code.co_names


def _tool_result_sent(request: list, call_id: str) -> str | None:
    """The content of the tool message for call_id in a request, or None if the request has none."""
    for m in request:
        d = _plain(m)
        if d.get("role") == "tool" and d.get("tool_call_id") == call_id:
            return str(d.get("content"))
    return None


def _loop_problems(loop) -> list[str] | None:
    """What is wrong with a participant's run_loop, tested with a scripted client (free, instant).
    Returns None if the loop cannot be tested this way (it does not use the notebook's `client`)."""
    if not _uses_client(loop):
        return None
    fake = _ScriptedClient("P-1001")
    out, error = _swap_client(loop, fake)
    # the model's answer has tool_calls=None: a loop without TODO 1 goes on to `for call in None` and fails there
    no_stop = isinstance(error, TypeError) and "NoneType" in str(error) and len(fake.requests) == 2
    if error is not None and not no_stop:
        return [f"it raised {type(error).__name__}: {error}"]
    problems = []
    if len(fake.requests) < 2:
        problems.append("it stopped before the tool the model asked for was run")
    else:
        result = _tool_result_sent(fake.requests[1], "test_call_1")
        if result is None:
            problems.append("the tool result does not go back to the model (TODO 2 and 3)")
        elif "P-1001" not in result:
            problems.append("the message sent back does not contain the tool's result (TODO 2)")
    if no_stop or len(fake.requests) > 2:
        problems.append("it does not stop when the model has answered (TODO 1)")
    elif not isinstance(out, list) or answer_text(out) is None:
        problems.append("it does not return the list of messages, ending with the model's answer")
    return problems


@blank_line_after
def check_my_loop(loop):
    """Notebook 01, Exercise C1: test your run_loop with a scripted stand-in for the model (no model call, no cost).
    Returns yours if it passes, otherwise our run_loop, with a message."""
    if loop is run_loop:
        print("This is our run_loop from claims_lab.reference.block1.")
        return loop
    problems = _loop_problems(loop)
    if problems is None:
        print("Your run_loop does not use `client`, so it cannot be tested offline. Using yours.")
        return loop
    if not problems:
        print("Your run_loop passes our test (it runs the tool, sends the result back and stops at the answer). Using yours.")
        return loop
    print(f"Your run_loop is not finished yet ({'; '.join(problems)}). Using our version from claims_lab.reference.block1.")
    return run_loop


_OUTCOME_WORDS = {"fast_track": ["fast_track", "fast-track", "fast track"],
                  "request_info": ["request_info", "request info"],
                  "decline": ["decline"],
                  "refer_adjuster": ["refer_adjuster", "refer to an adjuster", "refer the claim", "referred to an adjuster"]}


def outcome(messages: list) -> str:
    """The outcome named in the agent's final answer, read from its text (no model call).
    The first outcome mentioned wins; check the printed answer if it matters."""
    import re
    text = answer_text(messages)
    if text is None:
        return "no answer: the run stopped before the model answered"
    text = text.lower()
    if m := re.search(r"decision\W{0,5}(fast_track|request_info|decline|refer_adjuster)", text):
        return m.group(1)
    found = [(text.find(w), name) for name, words in _OUTCOME_WORDS.items() for w in words if w in text]
    return min(found)[1] if found else "unclear"


# ------------------------------------------------------------------ Extra task: a spending limit per run
def run_loop_guarded(pkg: Package, max_steps: int = 8, max_usd: float = 0.01, client=None) -> list:
    """The agent loop with a spending limit: meter.check() before each model call stops the run once it has spent
    max_usd (it raises BudgetExceeded, which the loop catches), and meter.add(response) counts what each call cost."""
    from ..costs import BudgetExceeded, CostMeter
    client = client or get_client()
    meter = CostMeter(max_usd=max_usd)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pkg.text()}]
    try:
        for step in range(max_steps):
            meter.check()
            response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
            meter.add(response)
            reply = response.choices[0].message
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _run_tool(call)})
    except BudgetExceeded as e:
        print("stopped:", e)
    print("this run:", meter)
    return messages


@blank_line_after
def check_guarded_loop(loop):
    """Notebook 01, extra task: test your run_loop_guarded with a scripted stand-in for the model (no model call, no
    cost). Returns yours if it passes, otherwise our run_loop_guarded, with a message."""
    import contextlib
    import io

    if loop is run_loop_guarded:
        return loop
    if not _uses_client(loop):
        print("Your run_loop_guarded is not finished yet (it makes no model call with `client`). "
              "Using our version from claims_lab.reference.block1.")
        return run_loop_guarded
    # each scripted answer costs about $0.0012 (1,000 tokens in, 100 out at the course model's price), so a limit of
    # $0.0005 must stop the run after its first call
    fake = _ScriptedClient("P-1001", model=MODEL)
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        out, error = _swap_client(loop, fake, max_usd=0.0005)
    problems = []
    if error is not None:
        from ..costs import BudgetExceeded
        problems.append("it does not catch cl.BudgetExceeded" if isinstance(error, BudgetExceeded)
                        else f"it raised {type(error).__name__}: {error}")
    elif len(fake.requests) > 1:
        problems.append("it made a second call after the limit was reached: call meter.add(response) after each call "
                        "and meter.check() before it")
    elif "stopped" not in printed.getvalue():
        problems.append('it does not print "stopped: ..." when the limit ends the run')
    if error is None and (not isinstance(out, list) or not out):
        problems.append("it does not return the list of messages")
    if problems:
        print(f"Your run_loop_guarded is not finished yet ({'; '.join(problems)}). "
              "Using our version from claims_lab.reference.block1.")
        return run_loop_guarded
    print("Your run_loop_guarded passes our test (it counts every call, and the limit ends the run without an error). "
          "Using yours.")
    return loop
