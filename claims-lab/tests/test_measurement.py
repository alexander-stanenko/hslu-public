"""The scoreboard measures replayed runs at their recorded cost, retries rate limits, and matches issues strictly."""
import claims_lab as cl
from claims_lab.reference import block1
from claims_lab.scoreboard import _evidence

IDS = ["PKG-001", "PKG-002", "PKG-003"]


def test_replayed_run_reports_the_recorded_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")
    with cl.replay(tmp_path / "replay.db"):
        live = cl.scoreboard(block1.chatbot, ids=IDS, label="recording", show=False)
        again = cl.scoreboard(block1.chatbot, ids=IDS, label="replayed", show=False)
    assert live["replayed_calls"].sum() == 0 and again["replayed_calls"].sum() == len(IDS)
    assert list(again["llm_calls"]) == list(live["llm_calls"]) == [1, 1, 1]  # not 0: the recorded calls count
    last = cl.scoreboard_history(plot=False).iloc[-1]
    assert "replayed" not in last.index and last["seconds_per_pkg"] >= 0  # the replay reports the recorded time


def test_a_rate_limit_is_retried_once(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # no real waiting in the test
    failed = set()

    def flaky(pkg):
        if pkg.package_id not in failed:
            failed.add(pkg.package_id)
            raise RuntimeError("Error code: 429 - rate limit reached")
        return {"decision": "decline", "reasoning": "x"}
    df = cl.scoreboard(flaky, ids=IDS, show=False)
    assert df["error"].isna().all()


def test_failed_claims_are_left_out_of_the_accuracy():
    def half_broken(pkg):
        if pkg.package_id == "PKG-001":
            raise ValueError("boom")
        return {"decision": cl.label(pkg.package_id)["expected"]["decision"], "reasoning": "as labelled"}
    cl.scoreboard(half_broken, ids=IDS, show=False, label="half broken")
    s = cl.scoreboard_history(plot=False).iloc[-1]
    assert s["errors"] == 1 and s["accuracy"] == 1.0


def test_issue_matching_needs_a_word_start():
    inside = cl.ClaimDecision(decision="fast_track", reasoning="The garage recommended it; a turnkey repair we trust.")
    ended = cl.ClaimDecision(decision="decline", reasoning="The policy ended on 30 June, before the hail.")
    key = cl.ClaimDecision(decision="refer_adjuster", reasoning="The key was left in the ignition.")
    assert not _evidence("policy_inactive", inside) and _evidence("policy_inactive", ended)     # not "recommended"
    assert not _evidence("possible_exclusion", inside) and _evidence("possible_exclusion", key)  # not "turnkey", "trust"
    listed = cl.ClaimDecision(decision="request_info", missing_items=["police_report"], reasoning="x")
    assert _evidence("police_report", listed) and not _evidence("invoice", listed)
