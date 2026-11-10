"""Dataset integrity: every label is backed by the rules (generator/world.py) and discoverable in the documents."""
import json
import sqlite3

import pytest

import claims_lab as cl
from claims_lab import tools as T
from claims_lab.data import DATA_DIR

import world  # generator/world.py, on sys.path via conftest

QUICK = {"PKG-001": ("fast_track", 650.0, [], []),
         "PKG-002": ("refer_adjuster", None, [], ["duplicate_invoice"]),
         "PKG-003": ("decline", None, [], ["not_covered"]),
         "PKG-004": ("request_info", None, ["policy_id"], []),
         "PKG-005": ("request_info", None, ["police_report"], []),
         "PKG-006": ("refer_adjuster", None, [], ["above_fast_track_limit"]),
         "PKG-007": ("refer_adjuster", None, [], ["possible_exclusion"]),
         "PKG-008": ("fast_track", 2300.0, [], [])}


def _chf(x: float) -> str:
    return f"{x:,.2f}".replace(",", "'")


def test_split_sizes():
    assert cl.list_packages("quick") == [f"PKG-00{i}" for i in range(1, 9)]
    assert cl.list_packages("dev") == [f"PKG-{i:03d}" for i in range(1, 17)]
    assert cl.list_packages("heldout") == [f"PKG-{i}" for i in range(101, 109)]
    assert cl.list_packages("all") == cl.list_packages("dev") + cl.list_packages("heldout")
    assert "PKG-211" not in cl.list_packages("all")      # the permissions case is kept apart from the scored claims


def test_the_quick_eight_are_the_claims_the_notebooks_talk_about():
    for pid, (decision, payout, missing, issues) in QUICK.items():
        e = cl.label(pid)["expected"]
        assert (e["decision"], e["payout_chf"], e["missing_items"], e["issues"]) == (decision, payout, missing, issues), pid


def test_categories_in_dev_and_heldout():
    count = lambda split: {c: sum(l["category"] == c for l in cl.labels(split)) for c in {l["category"] for l in cl.labels(split)}}  # noqa: E731
    assert count("dev") == {"routine": 5, "incomplete": 3, "not_covered": 3, "judgement": 3, "red_flag": 1, "above_limit": 1}
    assert count("heldout") == {"routine": 2, "incomplete": 1, "not_covered": 2, "judgement": 1, "red_flag": 1, "above_limit": 1}


def test_all_documents_exist_and_load():
    for lab in cl.labels() + [json.loads((DATA_DIR / "labels" / "permissions.jsonl").read_text())]:
        pkg = cl.load_package(lab["package_id"])
        assert set(pkg.documents) == set(lab["documents"]), lab["package_id"]
        assert "email.txt" in pkg.documents


def test_key_facts_are_in_the_documents():
    """Amounts and invoice numbers the rules rely on must be readable in the package text."""
    for lab in cl.labels():
        text = cl.load_package(lab["package_id"]).text()
        f = lab["facts"]
        if f["invoice_no"]:
            assert f["invoice_no"] in text, lab["package_id"]
        if "invoice" in lab["documents"].values():
            assert _chf(f["amount_chf"]) in text, lab["package_id"]
        else:
            assert f["amount_chf"] is None and lab["expected"]["missing_items"] == ["invoice"], lab["package_id"]
        if "police_report" not in lab["documents"].values() and f["damage"] == "theft":
            assert "police_report" in lab["expected"]["missing_items"], lab["package_id"]


def test_every_policy_is_in_the_table_and_found_by_number_or_by_name_and_plate():
    for lab in cl.labels():
        text = cl.load_package(lab["package_id"]).text()
        policy = T.lookup_policy(lab["facts"]["policy_id"])
        assert "error" not in policy and policy["holder"] in text, lab["package_id"]
        if lab["facts"]["policy_id"] in text:
            continue
        # no policy number: PKG-004 (two customers of that name) and PKG-016 (name and plate identify one policy)
        assert lab["package_id"] in ("PKG-004", "PKG-016")
        by_name = T.find_policies(name=policy["holder"])
        plate = policy["plate"] if policy["plate"] in text else None
        if lab["expected"]["missing_items"] == ["policy_id"]:
            assert len(by_name) > 1 and plate is None
        else:
            assert [p["policy_id"] for p in T.find_policies(name=policy["holder"], plate=plate)] == [policy["policy_id"]]


def test_reused_invoices_are_in_the_claims_table_and_new_ones_are_not():
    for pid, owner in (("PKG-002", "P-1002"), ("PKG-101", "P-1001")):
        inv = cl.label(pid)["facts"]["invoice_no"]
        rows = cl.query("SELECT policy_id, date FROM claims WHERE invoice_no = ?", (inv,))
        assert [r["policy_id"] for r in rows] == [owner] and rows[0]["date"] < cl.label(pid)["facts"]["date"]
    for lab in cl.labels():
        if "duplicate_invoice" not in lab["expected"]["issues"] and lab["facts"]["invoice_no"]:
            assert T.search_claims_by_invoice(lab["facts"]["invoice_no"]) == [], lab["package_id"]


def test_policy_numbers_match_the_packages():
    for lab in cl.labels():
        n = int(lab["package_id"].split("-")[1])
        assert lab["facts"]["policy_id"] == f"P-{1000 + n}", lab["package_id"]


def test_the_permissions_case():
    lab = json.loads((DATA_DIR / "labels" / "permissions.jsonl").read_text())
    assert lab["package_id"] == "PKG-211" and lab["facts"]["policy_id"] == "P-1211"
    assert lab["expected"]["decision"] == "request_info" and lab["expected"]["missing_items"] == ["police_report"]
    pkg = cl.load_package("PKG-211")
    assert "P-1211" in pkg.text() and "police" not in " ".join(pkg.documents)   # no police report among the files
    assert "Notice to insurers" in pkg.documents["invoice.txt"]


def test_bundle_is_small():
    size = sum(p.stat().st_size for p in DATA_DIR.rglob("*") if p.is_file())
    assert size < 2_000_000


def _tables(db) -> dict:
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("policies", "claims")}
    finally:
        con.close()


def test_regenerating_the_world_gives_the_package_data(tmp_path):
    """world.build() checks every hand label against its oracle; it also writes exactly the data shipped with the package."""
    out = tmp_path / "data"
    world.build(out)
    shipped = sorted(p.relative_to(DATA_DIR) for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix != ".db")
    built = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file() and p.suffix != ".db")
    assert built == shipped
    for rel in built:
        assert (out / rel).read_bytes() == (DATA_DIR / rel).read_bytes(), rel
    assert _tables(out / "lakeside.db") == _tables(DATA_DIR / "lakeside.db")


@pytest.mark.parametrize("case", world.CASES, ids=lambda c: c["id"])
def test_oracle_agrees_with_every_hand_label(case):
    history = {r["invoice_no"] for r in cl.query("SELECT invoice_no FROM claims WHERE NOT (policy_id = ? AND date = ?)",
                                                 (case["policy"]["policy_id"], case["claim"]["date"]))}
    holders = [r["policy_id"] for r in cl.query("SELECT policy_id FROM policies WHERE holder = ?", (case["policy"]["holder"],))]
    identified = case["policy"]["policy_id"] in case["email"] or len(holders) == 1
    claim = dict(case["claim"], exclusion=case["id"] in world.EXCLUSION_CASES)
    got = world.oracle(case["policy"], claim, case["docs"], history, identified)
    e = cl.label(case["id"])["expected"]
    assert (got["decision"], got["payout"], got["missing"], got["issues"]) == \
           (e["decision"], e["payout_chf"], e["missing_items"], e["issues"])


def test_a_working_copy_from_older_package_data_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))
    from claims_lab.data import db_path
    p = db_path()
    (tmp_path / "lakeside.db.version").write_text("older")
    con = sqlite3.connect(p); con.execute("DELETE FROM policies"); con.commit(); con.close()
    assert cl.query("SELECT count(*) AS n FROM policies WHERE policy_id = 'P-1001'")[0]["n"] == 1


def test_a_fresh_working_copy_is_made_once_when_threads_start_together(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from claims_lab import data
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path / "w"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda _: data.query("SELECT COUNT(*) AS n FROM policies")[0]["n"], range(16)))
    assert len(set(rows)) == 1 and rows[0] > 0
