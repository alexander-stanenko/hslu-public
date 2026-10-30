"""Block 4: the rule checks against the labels, the routing, the letter loop, the approval stretch task."""
import pytest
from langgraph.types import Command

import claims_lab as cl
from claims_lab.reference import block4 as B

JUDGEMENT = {"PKG-023", "PKG-025", "PKG-101"}  # exclusions and contradictions: no rule can decide these
EXTRA = {"PKG-008": {"invoice_date": "2026-10-08"}, "PKG-026": {"repair_type": "windscreen_replacement"},
         "PKG-104": {"repair_type": "bumper_repair"}}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")


def perfect_state(lab: dict, judgement: bool = False) -> dict:
    """The state after reading, if the model had extracted every fact correctly (facts from the label)."""
    fx, pid = lab["facts"], lab["package_id"]
    unnumbered = pid in ("PKG-021", "PKG-111")  # these packages state no policy number
    holder = None if not unnumbered else cl.query("SELECT c.name FROM policies p JOIN customers c ON p.customer_id=c.customer_id "
                                                 "WHERE p.policy_id=?", (fx["policy_id"],))[0]["name"]
    facts = B.Facts(policy_id=None if unnumbered else fx["policy_id"], customer_name=holder, claim_type=fx["claim_type"],
                    incident_date=fx["incident_date"], amount_chf=fx["amount_chf"], invoice_no=fx["invoice_no"],
                    documents_present=list(lab["documents"].values()), **EXTRA.get(pid, {}))
    state = {"package": cl.load_package(pid), "facts": facts, "judgement": B.Judgement(needs_judgement=judgement)}
    state.update(B.rule_checks(state))
    return state


@pytest.mark.parametrize("lab", [l for l in cl.labels() if l["split"] in ("dev", "heldout")], ids=lambda l: l["package_id"])
def test_rules_decide_every_rule_case_correctly(lab):
    state = perfect_state(lab)
    way = B.route(state)
    d = B.decide_by_rule(state)["decision"] if way == "decide_by_rule" else B.fast_track(state)["decision"]
    exp = lab["expected"]
    if lab["package_id"] in JUDGEMENT:
        assert way == "fast_track"  # the rules find nothing; only the screen can send these to the agent
        return
    assert d.decision == exp["decision"], d.reasoning
    if exp["decision"] == "fast_track":
        assert d.payout_chf == pytest.approx(exp["payout_chf"], abs=1)
    else:
        assert set(exp["issues"] + exp["missing_items"]) - {"possible_exclusion"} <= set(d.issues + d.missing_items)


def test_route_sends_judgement_to_the_agent():
    lab = next(l for l in cl.labels() if l["package_id"] == "PKG-023")
    assert B.route(perfect_state(lab, judgement=True)) == "claims_agent"
    lab = next(l for l in cl.labels() if l["package_id"] == "PKG-015")
    assert B.route(perfect_state(lab, judgement=True)) == "decide_by_rule"  # a failed rule decides first


def test_pipeline_runs_and_a_person_confirms_payouts(offline, capsys):
    app = B.build_pipeline()
    cl.show_graph(app)
    assert "rule_checks" in capsys.readouterr().out  # offline: the Mermaid text
    assert B.system(app)(cl.load_package("PKG-019")).decision == "decline"
    for answer, expected in (("approve", "fast_track"), ("reject", "refer_adjuster")):
        app = B.build_pipeline(approval_step=True)
        cfg = {"configurable": {"thread_id": answer}}
        out = app.invoke({"package": cl.load_package("PKG-014")}, cfg)
        assert out["__interrupt__"][0].value["payout_chf"] == pytest.approx(3120)
        out = app.invoke(Command(resume=answer), cfg)
        assert out["decision"].decision == expected


def test_letter_loop_stops_after_three_rounds(offline):
    loop = B.build_letter_loop()
    d = cl.ClaimDecision(decision="request_info", missing_items=["police_report"], reasoning="Police report missing.")
    out = loop.invoke({"package": cl.load_package("PKG-003"), "decision": d})
    assert 1 <= out["rounds"] <= B.MAX_ROUNDS and out["letter"]


def test_compare_marks_fixed_and_broken_claims(offline):
    app = B.build_pipeline()
    ids = ["PKG-012", "PKG-019"]
    a = cl.scoreboard(lambda pkg: cl.ClaimDecision(decision="fast_track", payout_chf=0, reasoning="x"), ids=ids, label="a", show=False)
    b = cl.scoreboard(B.system(app), ids=ids, label="b", show=False)
    m = cl.compare(a, b, names=("pay all", "hybrid")).set_index("package")
    assert m.loc["PKG-019", "change"] == "fixed"


def test_watch_prints_each_step_and_its_kind(offline, capsys):
    state = B.watch(B.build_pipeline(), cl.load_package("PKG-015"))
    out = capsys.readouterr().out
    assert "extract_facts  (model) read: policy P-10025" in out and "rule_checks    (code)" in out
    assert "class extract_facts,screen,claims_agent model" in out and " visited" in out  # offline: the Mermaid text
    assert state["decision"].decision == "refer_adjuster"


def test_screen_test_and_judgement_agent_run(offline):
    df = B.test_screen(B.make_screen("Is anything unusual?"))
    assert list(df["claim"]) == list(B.SCREEN_TESTS) and set(df["as expected"]) <= {"✔", "✘"}
    result = B.run_judgement_agent(B.judgement_agent(judgement_rules=""), cl.load_package("PKG-023"), ["seepage"])
    assert result["structured_response"].decision in ("fast_track", "request_info", "refer_adjuster", "decline")


def test_code_checks_catch_the_bad_letter():
    request = cl.ClaimDecision(decision="request_info", missing_items=["police_report"], reasoning="x")
    problems = B.code_checks(B.BAD_LETTER, request)
    assert any("sentences" in p for p in problems) and any("request_info" in p for p in problems)
    paid = cl.ClaimDecision(decision="fast_track", payout_chf=1950.0, reasoning="x")
    assert B.code_checks("Dear Ms Meier, we pay CHF 1'950.00 today. Kind regards.", paid) == []
    assert any("not stated" in p for p in B.code_checks("Dear Ms Meier, we will pay you soon.", paid))


def test_watch_shows_the_agent_steps_indented(offline, capsys):
    app = B.build_pipeline(route_fn=lambda state: "claims_agent")  # offline, the screen flags nothing: force the agent
    state = B.watch(app, cl.load_package("PKG-023"))
    lines = capsys.readouterr().out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("claims_agent   (model) the agent works"))
    calls = [l for l in lines[start + 1:] if l.startswith(" " * 17 + "calls ")]
    assert calls and "lookup_policy(policy_id=" in calls[0]
    assert not any("ClaimDecision" in l for l in lines)  # the final answer is on the claims_agent line, not a call
    end = next(i for i, l in enumerate(lines) if i > start and l.startswith("claims_agent   (model) "))
    assert end > start + len(calls) and state["decision"].decision


def test_unfinished_route_and_graph_fall_back_to_the_reference(offline, capsys):
    from langgraph.graph import StateGraph

    def unfinished(state):
        ...
    assert B.test_route(unfinished) is B.route
    out = capsys.readouterr().out
    assert "not finished" in out and "a clean claim        → fast_track      ✔" in out
    assert B.test_route(B.route) is B.route and "not finished" not in capsys.readouterr().out

    def fresh():  # the six steps but no edges, as the exercise cell is before the TODO is done
        g = StateGraph(B.ClaimState)
        for name in ("extract_facts", "screen", "rule_checks", "decide_by_rule", "fast_track"):
            g.add_node(name, getattr(B, name))
        g.add_node("claims_agent", B.make_claims_agent())
        return g
    app = B.check_graph(fresh())
    assert "Explore path, or B1 not finished" in capsys.readouterr().out
    assert B._edges(app) == B._edges(B.build_pipeline())
    g = fresh()
    g.add_edge("__start__", "extract_facts")
    B.check_graph(g)
    out = capsys.readouterr().out
    assert "missing edge: START → screen" in out and "missing edge: rule_checks → fast_track" in out


def test_empty_instructions_fall_back_to_lakesides(capsys):
    assert B.filled_in("", "MY_RULES") == B.JUDGEMENT_RULES and "MY_RULES is still empty" in capsys.readouterr().out
    assert B.filled_in(..., "MY_SCREEN") == B.SCREEN_INSTRUCTIONS
    capsys.readouterr()
    assert B.filled_in("Refer every possible exclusion.", "MY_RULES") == "Refer every possible exclusion."
    assert capsys.readouterr().out == ""


def test_unfinished_checker_falls_back_to_the_reference(offline, capsys):
    d = cl.ClaimDecision(decision="request_info", missing_items=["police_report"], reasoning="Police report missing.")

    def unfinished(state):
        ...
    check, after = B.test_checker(unfinished, unfinished, d)
    out = capsys.readouterr().out
    assert check is B.check_letter and after is B.after_check
    assert "check_letter is not finished" in out and "after_check is not finished" in out
    assert "Internal term 'request_info' used." in out and "passes: False" in out  # code checks catch the bad letter

    def lenient(state):
        return {"passes": True, "problems": []}
    check, after = B.test_checker(lenient, B.after_check, d)
    out = capsys.readouterr().out
    assert check is lenient and after is B.after_check and "passed the bad letter" in out
    loop = B.build_letter_loop(*B.test_checker(unfinished, unfinished, d))
    assert loop.invoke({"package": cl.load_package("PKG-003"), "decision": d})["letter"]
