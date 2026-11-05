"""Reference for notebook 01: the same claim handled by a chatbot, a workflow and an agent loop.

All three return a ClaimDecision, so they can be compared on the scoreboard:

    from claims_lab.reference import block1
    cl.scoreboard(block1.chatbot, label="chatbot")
    cl.scoreboard(block1.naive_workflow, label="workflow")       # without the invoice check
    cl.scoreboard(block1.workflow, label="workflow + invoice check")
    cl.scoreboard(block1.agent, label="agent loop")

Notebook 01 also uses three helpers that need no model call: check_my_invoice_check and check_my_loop test a
participant's exercise and return ours if it is not finished, and outcome reads the outcome from an agent's answer.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .. import tools as T
from ..data import Package
from ..models import get_model
from ..schemas import ClaimDecision, ClaimFacts

FAST_TRACK_LIMIT_CHF = 5000

# Which claim types each module covers. The workflow hard-codes this; an agent would look it up in the terms.
COVERS = {
    "motor": {"liability": set(), "partial": {"glass", "hail", "storm", "theft", "fire", "animal_collision", "vandalism"},
              "collision": {"collision"}},
    "household": {"basic": {"fire", "hail", "storm", "water_damage", "burglary"}, "theft_outside": {"theft_outside"},
                  "glass": {"glass_breakage"}},
    "building": {"basic": {"fire", "hail", "storm", "water_damage"}, "glass": {"glass_breakage"}},
}


# ------------------------------------------------------------------ Part A: the chatbot
def chatbot(pkg: Package, llm=None) -> ClaimDecision:
    """One model call: the package goes in, a decision comes out. No database, no terms, no history."""
    llm = llm or get_model()
    return llm.with_structured_output(ClaimDecision).invoke("Decide this insurance claim.\n\n" + pkg.text())


# ------------------------------------------------------------------ Part B: the workflow (a fixed chain)
def extract_facts(pkg: Package, llm=None) -> ClaimFacts:
    """Step 1 of the workflow: the model only reads; it decides nothing."""
    llm = llm or get_model()
    return llm.with_structured_output(ClaimFacts).invoke("Extract the key facts of this claim package.\n\n" + pkg.text())


def is_covered(policy: dict, claim_type: str) -> bool:
    if policy["product"] == "motor" and claim_type == "glass_breakage":  # a windscreen is "glass" in the motor terms
        claim_type = "glass"
    modules = COVERS.get(policy["product"], {})
    return any(claim_type in modules.get(m, set()) for m, included in policy["modules"].items() if included)


def deductible(policy: dict, claim_type: str) -> float:
    d = policy["deductibles_chf"]
    if policy["product"] == "motor" and claim_type == "glass_breakage":  # as in is_covered: a windscreen is "glass"
        claim_type = "glass"
    if policy["product"] == "motor":
        return d["glass"] if claim_type == "glass" else d["collision"] if claim_type == "collision" else d["partial"]
    if policy["product"] == "building" and claim_type in ("hail", "storm"):
        return d["natural_hazards"]
    return d["default"]


def check_duplicate_invoice(facts: ClaimFacts, policy: dict) -> ClaimDecision | None:
    """Exercise B: an extra check. Returns a decision if the check fails, None if the claim passes."""
    if facts.invoice_no and T.search_claims_by_invoice(facts.invoice_no):
        return ClaimDecision(decision="refer_adjuster", issues=["duplicate_invoice"],
                             reasoning=f"Invoice {facts.invoice_no} was already used in an earlier claim.")
    return None


def workflow(pkg: Package, llm=None, extra_checks=(check_duplicate_invoice,)) -> ClaimDecision:
    """Part of the clerk's cheat sheet written as code: extract, policy, active?, covered?, extra checks, pay.

    extra_checks: functions (facts, policy) -> ClaimDecision | None, run after the coverage check (cheat-sheet question 5).
    """
    facts = extract_facts(pkg, llm)
    if not facts.policy_id:
        return ClaimDecision(decision="request_info", missing_items=["policy_id"], reasoning="No policy number in the package.")
    policy = T.lookup_policy(facts.policy_id)
    if "error" in policy:
        return ClaimDecision(decision="request_info", missing_items=["policy_id"], reasoning=policy["error"])
    if facts.incident_date and not (policy["valid_from"] <= facts.incident_date <= policy["valid_to"]):
        return ClaimDecision(decision="decline", issues=["policy_inactive"],
                             reasoning=f"Policy valid {policy['valid_from']} to {policy['valid_to']}, incident on {facts.incident_date}.")
    if not is_covered(policy, facts.claim_type):
        included = [m for m, on in policy["modules"].items() if on]
        return ClaimDecision(decision="decline", issues=["not_covered"],
                             reasoning=f"{facts.claim_type} is not covered by the modules {included}.")
    for check in extra_checks:
        if (stop := check(facts, policy)) is not None:
            return stop
    amount = facts.amount_chf or 0.0
    if amount > FAST_TRACK_LIMIT_CHF:
        return ClaimDecision(decision="refer_adjuster", issues=["above_fast_track_limit"],
                             reasoning=f"CHF {amount:,.2f} is above the fast-track limit of CHF {FAST_TRACK_LIMIT_CHF:,}.")
    payout = T.calc_payout(amount, deductible(policy, facts.claim_type), policy["sum_insured_chf"])
    return ClaimDecision(decision="fast_track", payout_chf=payout,
                         reasoning=f"Covered {facts.claim_type} claim of CHF {amount:,.2f}; payout after deductible CHF {payout:,.2f}.")


def naive_workflow(pkg: Package, llm=None) -> ClaimDecision:
    """The workflow as first written in the notebook: no extra checks."""
    return workflow(pkg, llm, extra_checks=())


# ------------------------------------------------------------------ Part C: the agent loop, from scratch
# Four of Lakeside's systems as LangChain tools: each returns its result as JSON text.
@tool
def lookup_policy(policy_id: str) -> str:
    """Look up a policy by its number (e.g. 'P-10001'). Returns holder, product (motor/household/building), status,
    validity dates, included modules (coverage), deductibles in CHF, sum insured in CHF and the insured object."""
    return json.dumps(T.lookup_policy(policy_id), ensure_ascii=False)


@tool
def get_claim_history(policy_id: str) -> str:
    """List all earlier claims on a policy (date, type, amount in CHF, status, invoice number, vendor).
    Use it to check claim frequency and repeated invoices."""
    return json.dumps(T.get_claim_history(policy_id), ensure_ascii=False)


@tool
def search_claims_by_invoice(invoice_no: str) -> str:
    """Check whether an invoice number was already used in any earlier claim (across all policies).
    An empty list means the invoice number is new."""
    return json.dumps(T.search_claims_by_invoice(invoice_no), ensure_ascii=False)


@tool
def calc_payout(amount_chf: float, deductible_chf: float, sum_insured_chf: float | None = None) -> str:
    """Payout in CHF = claimed amount (capped at the sum insured, if any) minus the deductible, never below 0."""
    return json.dumps(T.calc_payout(amount_chf, deductible_chf, sum_insured_chf), ensure_ascii=False)


AGENT_TOOLS = [lookup_policy, get_claim_history, search_claims_by_invoice, calc_payout]

SYSTEM_PROMPT = """You are a claims handler at Lakeside Insurance AG.
Decide the claim package: fast_track (pay now), request_info (something is missing), decline (not insured)
or refer_adjuster (a human must decide). Use the tools to check the policy and the claim history before you decide.
Finish with the decision, the payout in CHF if any, and a short reason."""


def run_loop(pkg: Package, llm=None, tools=None, max_steps: int = 8, system_prompt: str = SYSTEM_PROMPT) -> list:
    """The agent loop: ask the model, run the tools it asks for, send the results back, repeat until it answers.

    max_steps stops a confused agent (an agent without a limit can loop, and every step re-sends the whole history).
    Returns the full message list (use cl.show_trace to read it).
    """
    llm = llm or get_model()
    tools = tools or AGENT_TOOLS
    tools_by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    messages = [SystemMessage(system_prompt), HumanMessage(pkg.text())]
    for _ in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:                                   # no tool requested: this is the answer
            return messages
        for call in ai.tool_calls:                              # run each requested tool, send the result back
            tool = tools_by_name.get(call["name"])
            result = tool.invoke(call["args"]) if tool else json.dumps({"error": f"Unknown tool {call['name']}"})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"]))
    messages.append(AIMessage(content=f"Stopped after {max_steps} steps without a decision."))
    return messages


def agent(pkg: Package, llm=None, max_steps: int = 8) -> ClaimDecision:
    """The agent loop plus one small call that turns its final answer into a ClaimDecision (for the scoreboard)."""
    llm = llm or get_model()
    final = run_loop(pkg, llm, max_steps=max_steps)[-1].text
    return llm.with_structured_output(ClaimDecision).invoke("Put this claim decision into the required format:\n\n" + final)


# ------------------------------------------------------------------ Notebook 01: test an exercise, or continue with ours
def _invoice_check_problems(check) -> list[str]:
    """What is wrong with a participant's check_duplicate_invoice, tested on two claims without a model call."""
    reused = ClaimFacts(policy_id="P-10025", claim_type="glass", incident_date="2026-10-19", amount_chf=1150.0,
                        invoice_no="GH-3318")          # PKG-015: this invoice was already paid
    fresh = ClaimFacts(policy_id="P-10001", claim_type="glass", incident_date="2026-10-20", amount_chf=1150.0,
                       invoice_no="GH-4471")           # PKG-012: a new invoice
    problems = []
    try:
        d = check(reused, T.lookup_policy("P-10025"))
        if d is None:
            problems.append("it lets the reused invoice GH-3318 of PKG-015 pass")
        elif not isinstance(d, ClaimDecision):
            problems.append(f"it returns a {type(d).__name__}, not a cl.ClaimDecision or None")
        elif d.decision != "refer_adjuster":
            problems.append(f"it returns {d.decision} for a reused invoice; the cheat sheet says refer_adjuster")
        if check(fresh, T.lookup_policy("P-10001")) is not None:
            problems.append("it also stops PKG-012, whose invoice is new")
    except Exception as e:  # noqa: BLE001 - any error in the participant's code means: use ours
        problems.append(f"it raised {type(e).__name__}: {e}")
    return problems


def check_my_invoice_check(check):
    """Notebook 01, Exercise B: test your check on PKG-015 and PKG-012 (no model call). Returns yours if it passes,
    otherwise our check_duplicate_invoice, with a message."""
    problems = _invoice_check_problems(check)
    if not problems:
        print("Your check_duplicate_invoice passes our test (reused invoice referred, new invoice passes). Using yours.")
        return check
    print(f"Your check_duplicate_invoice is not finished yet ({'; '.join(problems)}). "
          "Using our version from claims_lab.reference.block1.")
    return check_duplicate_invoice


class _ScriptedModel:
    """A stand-in for the model with tools bound, used to test a loop offline: it asks for one tool, then answers."""

    def __init__(self, policy_id: str):
        self.policy_id, self.inputs = policy_id, []

    def invoke(self, messages, config=None, **kwargs):
        self.inputs.append(list(messages))
        if len(self.inputs) == 1:
            return AIMessage(content="", tool_calls=[{"name": "lookup_policy", "args": {"policy_id": self.policy_id},
                                                      "id": "test_call_1", "type": "tool_call"}])
        return AIMessage(content="Decision: fast_track (a scripted test answer, not a model).")


def _loop_problems(loop) -> list[str] | None:
    """What is wrong with a participant's run_loop, tested with a scripted model (free, instant).
    Returns None if the loop cannot be tested this way (it does not use the notebook's `bound` model)."""
    from ..data import load_package
    code = getattr(loop, "__code__", None)
    namespace = getattr(loop, "__globals__", None)
    if code is None or namespace is None or "bound" not in code.co_names:
        return None
    fake, real = _ScriptedModel("P-10001"), namespace.get("bound")
    namespace["bound"] = fake
    problems = []
    try:
        out = loop(load_package("PKG-012"))
    except Exception as e:  # noqa: BLE001
        return [f"it raised {type(e).__name__}: {e}"]
    finally:
        namespace["bound"] = real
    if len(fake.inputs) < 2:
        problems.append("it stopped before the tool the model asked for was run")
    else:
        results = [m for m in fake.inputs[1] if isinstance(m, ToolMessage) and m.tool_call_id == "test_call_1"]
        if not results:
            problems.append("the tool result does not go back to the model (TODO 2 and 3)")
        elif "P-10001" not in str(results[0].content):
            problems.append("the message sent back does not contain the tool's result (TODO 2)")
    if len(fake.inputs) > 2:
        problems.append("it does not stop when the model has answered (TODO 1)")
    if not isinstance(out, list) or not out or getattr(out[-1], "type", "") != "ai":
        problems.append("it does not return the list of messages")
    return problems


def check_my_loop(loop):
    """Notebook 01, Exercise C1: test your run_loop with a scripted stand-in for the model (no model call, no cost).
    Returns yours if it passes, otherwise our run_loop, with a message."""
    if loop is run_loop:
        print("This is our run_loop from claims_lab.reference.block1.")
        return loop
    problems = _loop_problems(loop)
    if problems is None:
        print("Your run_loop does not use `bound`, so it cannot be tested offline. Using yours.")
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
    text = (messages[-1].text if messages else "").lower()
    if text.startswith("stopped after"):
        return "no decision (step limit)"
    if m := re.search(r"decision\W{0,5}(fast_track|request_info|decline|refer_adjuster)", text):
        return m.group(1)
    found = [(text.find(w), name) for name, words in _OUTCOME_WORDS.items() for w in words if w in text]
    return min(found)[1] if found else "unclear"
