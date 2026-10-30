"""Dataset integrity: every label is backed by the rules and discoverable in the documents."""
import catalogue as cat
import rules

import claims_lab as cl
from claims_lab.data import DATA_DIR


def test_split_sizes():
    assert len(cl.list_packages("dev")) == 28
    assert len(cl.list_packages("heldout")) == 12
    assert len(cl.list_packages("demo")) == 1
    assert len(cl.faq()) == 8


def test_all_documents_exist_and_load():
    for lab in cl.labels():
        pkg = cl.load_package(lab["package_id"])
        assert set(pkg.documents) == set(lab["documents"])
        assert "email.txt" in pkg.documents


def test_oracle_agrees_with_every_label():
    history = cl.query("SELECT * FROM claims")
    customers = cl.query("SELECT * FROM customers")
    for case in cat.CASES:
        if case["split"] == "demo":
            continue
        got = rules.oracle(case, history, cat.WEATHER_EVENTS, customers)
        exp = case["expected"]
        assert (got["decision"], got["payout"], got["missing"], got["issues"]) == \
               (exp["decision"], exp["payout"], exp["missing"], exp["issues"]), case["key"]


def test_key_facts_are_in_the_documents():
    """Amounts and invoice numbers the oracle relies on must be readable in the package text."""
    for lab in cl.labels():
        text = cl.load_package(lab["package_id"]).text()
        f = lab["facts"]
        if f["invoice_no"]:
            assert f["invoice_no"] in text, lab["package_id"]
        if f["amount_chf"] and any(t in ("invoice", "estimate") for t in lab["documents"].values()):
            assert f"{f['amount_chf']:,.2f}".replace(",", "'") in text, lab["package_id"]


def test_policy_reference_rules():
    for lab in cl.labels():
        text = cl.load_package(lab["package_id"]).text()
        case = next(c for c in cat.CASES if c["key"] == lab["case_key"])
        if case.get("policy_ref_in_docs", True):
            assert lab["facts"]["policy_id"] in text, lab["package_id"]
        else:
            assert lab["facts"]["policy_id"] not in text, lab["package_id"]


def test_categories_balanced_in_dev():
    cats = {}
    for l in cl.labels("dev"):
        cats[l["category"]] = cats.get(l["category"], 0) + 1
    assert cats == {"routine": 12, "incomplete": 5, "decline": 3, "ambiguous": 3, "suspicious": 5}


def test_bundle_is_small():
    size = sum(p.stat().st_size for p in DATA_DIR.rglob("*") if p.is_file())
    assert size < 2_000_000


def test_packages_with_several_policies_follow_from_the_rules():
    """Block 5, Part E: every policy decision of the three complex packages agrees with the oracle, and the facts it
    relies on are in the documents."""
    import complex as cx
    labs = {l["package_id"]: l for l in cl.complex_labels()}
    assert cl.list_packages("complex") == ["PKG-201", "PKG-202", "PKG-203"]
    assert sum(len(l["parts"]) for l in labs.values()) == 7
    assert not set(labs) & set(cl.list_packages("all"))   # kept apart from the 41 scored packages
    history = cl.query("SELECT * FROM claims")
    customers = cl.query("SELECT * FROM customers")
    for x in cx.COMPLEX:
        lab, pkg = labs[x["package_id"]], cl.load_package(x["package_id"])
        assert set(pkg.documents) == set(lab["documents"])
        text = pkg.text()
        for part, lp in zip(x["parts"], lab["parts"]):
            got = rules.oracle(cx.part_case(x, part), history, cat.WEATHER_EVENTS, customers)
            assert (got["decision"], got["payout"], got["issues"]) == \
                   (lp["expected"]["decision"], lp["expected"]["payout_chf"], lp["expected"]["issues"]), lp["policy_id"]
            assert lp["policy_id"] in text and set(lp["documents"]) <= set(pkg.documents)
            assert f"{lp['amount_chf']:,.2f}".replace(",", "'") in text, lp["policy_id"]
            if lp["invoice_no"]:
                assert lp["invoice_no"] in text
        assert len(text) // 4 > 2 * 560   # more than twice the longest single-policy package


def test_a_working_copy_from_older_package_data_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))
    from claims_lab.data import db_path
    p = db_path()
    (tmp_path / "lakeside.db.version").write_text("older")
    import sqlite3
    con = sqlite3.connect(p); con.execute("DELETE FROM policies"); con.commit(); con.close()
    assert cl.query("SELECT count(*) AS n FROM policies WHERE policy_id = 'P-10061'")[0]["n"] == 1


def test_a_fresh_working_copy_is_made_once_when_threads_start_together(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from claims_lab import data
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path / "w"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda _: data.query("SELECT COUNT(*) AS n FROM policies")[0]["n"], range(16)))
    assert len(set(rows)) == 1 and rows[0] > 0


def test_the_flagged_but_harmless_roof_is_outside_the_scored_sets():
    import json
    from claims_lab import data
    pkg = data.load_package("PKG-221")
    assert "1987" in pkg.text() and "no wear and tear" in pkg.text().lower()
    assert "PKG-221" not in data.list_packages("all")          # the 28 + 12 + demo stay as they were
    assert len(data.list_packages("dev")) == 28 and len(data.list_packages("heldout")) == 12
    lab = json.loads((data.DATA_DIR / "labels" / "judgement.jsonl").read_text().splitlines()[0])
    assert lab["parts"][0]["expected"]["decision"] == "fast_track" and lab["parts"][0]["expected"]["payout_chf"] == 2400.0
    from claims_lab import tools
    assert tools.lookup_policy("P-10081")["product"] == "building"
