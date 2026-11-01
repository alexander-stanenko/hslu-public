"""After class in notebook 05: the garage review. The invoices and their planted problems, the read-only tools, the code
check and the recorded table; no model call."""
import json
import re

import pytest

from claims_lab import course
from claims_lab.reference import garage as G


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))


def test_invoices_are_fixed_and_the_task_names_only_lindenmatt():
    assert G.build() == G.INVOICES and len(G.INVOICES) == 25
    assert sum(v["vendor"].startswith("Garage Lindenmatt") for v in G.INVOICES.values()) == 12
    assert "Lindenmatt" in G.TASK and "Speedfix" not in G.TASK


def test_planted_problems_are_in_the_invoices():
    lo, hi = G.PRICES["bumper_repair"]
    assert G.INVOICES["LM-1005"]["total_chf"] > 1.5 * hi
    labour = [l for l in G.INVOICES["LM-1009"]["text"].splitlines() if "Labour" in l]
    assert len(labour) == 2 and labour[0][4:] == labour[1][4:]   # the same line, at positions 2 and 3
    for lm, sf in G._REUSED.items():
        sn = re.search(r"SN (\S+)", G.INVOICES[lm]["text"]).group(1)
        assert sn in G.INVOICES[sf]["text"]
    others = {i: v for i, v in G.INVOICES.items() if i not in G.TRUTH}
    for iid, inv in others.items():   # no other invoice is far above its range
        assert inv["total_chf"] <= 1.5 * G.PRICES[inv["repair_type"]][1], iid


def test_tools_only_read():
    rows = json.loads(G.list_invoices.invoke({"vendor": "speedfix"}))
    assert [r["invoice_id"] for r in rows] == [f"SF-{2001 + i}" for i in range(7)]
    assert "INVOICE LM-1003" in G.read_invoice.invoke({"invoice_id": "lm-1003"})
    assert "No invoice" in G.read_invoice.invoke({"invoice_id": "XX-1"})
    sn = re.search(r"SN (\S+)", G.INVOICES["LM-1007"]["text"]).group(1)
    assert json.loads(G.search_invoices.invoke({"text": sn})) == ["LM-1007", "SF-2004"]
    assert json.loads(G.price_range.invoke({"repair_type": "bumper repair"}))["typical_max_chf"] == 2500
    assert G.price_range.invoke({"repair_type": "rocket"}) == "unknown type"


def test_serial_check_finds_every_reused_part_and_nothing_else():
    s = G.score(G.serial_check())
    assert s["found"] == 6 and s["at Speedfix"] == 3 and s["false alarms"] == 0
    assert s["missed"] == ["LM-1005", "LM-1009"]


def test_score_counts_hits_and_false_alarms():
    r = G.Report(findings=[G.Finding(invoice_id=" lm-1005 ", reason="x"), G.Finding(invoice_id="HU-3001", reason="y")])
    s = G.score(r)
    assert s["found"] == 1 and s["false alarms"] == 1 and len(s["missed"]) == 7


def test_designs_build_offline_and_review_rejects_unknown_design():
    assert G.one_agent() is not None and G.team() is not None
    with pytest.raises(ValueError):
        G.review("committee")


def test_recorded_table_has_every_design():
    df = G.versus_recorded()
    assert len(df) == len(course.GARAGE_REFERENCE_RUNS) == 4
    assert df.iloc[-1]["found (of 8)"] == "8 in every run"
