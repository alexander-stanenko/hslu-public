"""The scoreboard history: a demo cell run again keeps one point; another run under the same label gets a number."""
import pandas as pd
import pytest

import importlib
S = importlib.import_module("claims_lab.scoreboard")


@pytest.fixture(autouse=True)
def empty_history(monkeypatch):
    monkeypatch.setattr(S, "HISTORY", [])
    monkeypatch.setattr(S, "RUNS", {})


def _summary(label, replayed, accuracy=0.75):
    return {"label": label, "replayed": replayed, "accuracy": accuracy, "unsafe_approvals": 2, "llm_calls_per_pkg": 3.1,
            "cost_usd_per_pkg": 0.004}


def test_a_replayed_run_repeated_replaces_its_point():
    S._record(_summary("checklist agent", True), pd.DataFrame())
    S._record(_summary("checklist agent", True), pd.DataFrame())
    assert [h["label"] for h in S.HISTORY] == ["checklist agent"]


def test_a_live_run_after_the_demo_gets_a_number():
    S._record(_summary("checklist agent", True), pd.DataFrame())
    S._record(_summary("checklist agent", False, 1.0), pd.DataFrame())
    S._record(_summary("checklist agent", False, 0.875), pd.DataFrame())
    S._record(_summary("hybrid pipeline", False), pd.DataFrame())
    assert [h["label"] for h in S.HISTORY] == ["checklist agent", "checklist agent, run 2", "checklist agent, run 3",
                                              "hybrid pipeline"]
    assert set(S.RUNS) == {"checklist agent", "hybrid pipeline"}
