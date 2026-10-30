"""Build the Lakeside Insurance dataset into src/claims_lab/data/.

Run:  python generator/build.py      (deterministic; seed 42)
Fails loudly if any hand label disagrees with the rule oracle.
"""
from __future__ import annotations

import json
import random
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import catalogue as cat
import complex as cx
import injected as ij
import judgement as jd
import rules
import templates as T

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "src" / "claims_lab" / "data"
SEED = 42

FIRST = ["Andrea", "Benjamin", "Chiara", "David", "Eva", "Florian", "Gabriela", "Hans", "Isabelle", "Jan", "Karin",
         "Lea", "Matthias", "Noemi", "Oliver", "Petra", "Raphael", "Sabine", "Tobias", "Ursula", "Valentin", "Vera"]
LAST = ["Ackermann", "Bachmann", "Christen", "Dubois", "Egger", "Fuchs", "Gisler", "Hess", "Imhof", "Jost", "Koller",
        "Lang", "Marti", "Näf", "Oswald", "Peter", "Rüegg", "Sutter", "Tanner", "Ulrich", "Vonlanthen", "Wenger"]
PLACES = [("6003", "Luzern"), ("6005", "Luzern"), ("6010", "Kriens"), ("3012", "Bern"), ("3007", "Bern"), ("8003", "Zürich"),
          ("8048", "Zürich"), ("8400", "Winterthur"), ("4056", "Basel"), ("5000", "Aarau"), ("9000", "St. Gallen"), ("6300", "Zug")]
VEHICLES = ["VW Polo (2019)", "Skoda Kamiq (2022)", "Toyota C-HR (2021)", "Renault Captur (2020)", "Hyundai Kona (2023)", "BMW X1 (2020)"]
CANTON = {"60": "LU", "61": "LU", "62": "LU", "30": "BE", "31": "BE", "80": "ZH", "84": "ZH", "40": "BS", "50": "AG", "90": "SG", "63": "ZG"}


def filler(rng: random.Random, n: int = 40):
    """Background customers, policies and old claims so the database looks like a real portfolio."""
    customers, policies, claims = [], [], []
    for i in range(n):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        pc, city = rng.choice(PLACES)
        c = cat.cust(name, f"{rng.choice(['Bahnhof', 'Dorf', 'See', 'Berg', 'Kirch'])}strasse {rng.randint(1, 120)}", pc, city)
        pid = f"P-2{i:04d}"
        product = rng.choice(["motor", "motor", "household", "building"])
        if product == "motor":
            plate = f"{CANTON.get(pc[:2], 'LU')} {rng.randint(10, 999)} {rng.randint(100, 999)}"
            p = cat.motor(pid, plate, rng.choice(VEHICLES), collision=rng.random() < .4, ded_partial=rng.choice([200, 300, 500]),
                          ded_glass=rng.choice([0, 200]))
        elif product == "household":
            p = cat.household(pid, sum_insured=rng.choice([40000, 60000, 80000]), theft_outside=rng.random() < .5, glass=rng.random() < .3)
        else:
            p = cat.building(pid, sum_insured=rng.choice([700_000, 900_000, 1_400_000]), glass=rng.random() < .5)
        customers.append(c)
        policies.append((c, p))
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            # old claims only (> 13 months before the dataset date) so filler never trips the frequency rule
            inc = date(2026, 9, 1) - timedelta(days=rng.randint(400, 1500))
            claims.append(dict(policy_id=pid, incident_date=inc.isoformat(), claim_type=rng.choice(["glass", "hail", "water_damage", "storm"]),
                               amount_chf=round(rng.uniform(300, 4000), -1), status=rng.choice(["paid", "paid", "paid", "declined"]),
                               invoice_no=f"H-{rng.randint(10000, 99999)}", vendor=rng.choice(cat.VENDORS)))
    return customers, policies, claims


def main():
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "inbox").mkdir(parents=True)
    (OUT / "labels").mkdir()
    shutil.copytree(ROOT / "terms", OUT / "terms")

    # ---------------------------------------------------------------- entities
    f_customers, f_policies, f_claims = filler(rng)
    case_entities = [(c["customer"], c["policy"]) for c in cat.CASES] + [(n["customer"], n["policy"]) for n in cat.NAMESAKES]
    all_entities = case_entities + f_policies
    customers = []
    for i, (cu, pol) in enumerate(all_entities):
        cu = dict(cu, customer_id=f"C-{1000 + i}", customer_since=f"{rng.randint(2008, 2024)}-0{rng.randint(1, 9)}-01")
        customers.append(cu)
        pol["customer_id"] = cu["customer_id"]
    history = []
    for case in cat.CASES:
        for h in case["history"]:
            history.append(dict(h, policy_id=case["policy"]["policy_id"]))
    history += f_claims
    for i, h in enumerate(sorted(history, key=lambda h: h["incident_date"])):
        h["claim_id"] = f"C-{h['incident_date'][:4]}-{i + 1:04d}"
        h.setdefault("note", "")

    # ---------------------------------------------------------------- database
    db = sqlite3.connect(OUT / "lakeside.db")
    db.executescript("""
    CREATE TABLE customers(customer_id TEXT PRIMARY KEY, name TEXT, street TEXT, postcode TEXT, city TEXT, email TEXT, customer_since TEXT);
    CREATE TABLE policies(policy_id TEXT PRIMARY KEY, customer_id TEXT, product TEXT, status TEXT, valid_from TEXT, valid_to TEXT,
                          modules TEXT, deductibles_chf TEXT, sum_insured_chf REAL, insured_object TEXT);
    CREATE TABLE claims(claim_id TEXT PRIMARY KEY, policy_id TEXT, incident_date TEXT, claim_type TEXT, amount_chf REAL, status TEXT,
                        invoice_no TEXT, vendor TEXT, note TEXT);
    CREATE TABLE price_ranges(repair_type TEXT PRIMARY KEY, typical_min_chf REAL, typical_max_chf REAL);
    CREATE TABLE weather_events(date TEXT, region TEXT, event TEXT, intensity TEXT, description TEXT);
    """)
    db.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
                   [(c["customer_id"], c["name"], c["street"], c["postcode"], c["city"], c["email"], c["customer_since"]) for c in customers])
    db.executemany("INSERT INTO policies VALUES (?,?,?,?,?,?,?,?,?,?)",
                   [(p["policy_id"], p["customer_id"], p["product"], p["status"], p["valid_from"], p["valid_to"], json.dumps(p["modules"]),
                     json.dumps(p["deductibles"]), p["sum_insured"], json.dumps(p["object"])) for _, p in all_entities])
    db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?)",
                   [(h["claim_id"], h["policy_id"], h["incident_date"], h["claim_type"], h["amount_chf"], h["status"], h["invoice_no"],
                     h["vendor"], h["note"]) for h in history])
    db.executemany("INSERT INTO price_ranges VALUES (?,?,?)", [(k, *v) for k, v in rules.PRICE_RANGES.items()])
    db.executemany("INSERT INTO weather_events VALUES (?,?,?,?,?)",
                   [(w["date"], w["region"], w["event"], w["intensity"], w["description"]) for w in cat.WEATHER_EVENTS])
    db.commit()
    db.close()

    # ---------------------------------------------------------------- packages
    dev = [c for c in cat.CASES if c["split"] == "dev"]
    held = [c for c in cat.CASES if c["split"] == "heldout"]
    demo = [c for c in cat.CASES if c["split"] == "demo"]
    rng.shuffle(dev)
    rng.shuffle(held)
    ids = {c["key"]: f"PKG-{i + 1:03d}" for i, c in enumerate(dev)}
    ids.update({c["key"]: f"PKG-{101 + i:03d}" for i, c in enumerate(held)})
    ids.update({c["key"]: f"PKG-{900 + i:03d}" for i, c in enumerate(demo)})

    oracle_customers = customers
    labels, mismatches = [], []
    for case in cat.CASES:
        pid = ids[case["key"]]
        pdir = OUT / "inbox" / pid
        pdir.mkdir()
        claim = case["claim"]
        hh, mm = rng.randint(6, 22), rng.randint(0, 59)
        files = {"email.txt": "email"}
        (pdir / "email.txt").write_text(T.render_email(case, rng, f"{hh:02d}:{mm:02d}"))
        n_scan, n_photo = 1, 0
        for doc in case["docs"]:
            if doc == "claim_form":
                name = rng.choice(["Schadenmeldung.txt", "claim_form_signed.txt", f"scan_{n_scan:03d}.txt"])
                text = T.render_claim_form(case, rng, rng.randint(0, 1))
            elif doc in ("invoice", "estimate"):
                prefix = "Rechnung" if doc == "invoice" else "Offerte"
                name = rng.choice([f"{prefix}_{claim['invoice_no']}.txt", f"scan_{n_scan:03d}.txt", f"{claim['vendor'].split()[0]}_{claim['invoice_no']}.txt"])
                text = T.render_invoice(case, rng, rng.randint(0, 2), estimate=(doc == "estimate"))
            elif doc == "police_report":
                name = rng.choice(["Polizeirapport.txt", f"scan_{n_scan:03d}.txt", "police_confirmation.txt"])
                text = T.render_police_report(case, rng, rng.randint(0, 1))
            elif doc == "photo_note":
                stem = f"IMG_{rng.randint(1000, 9999)}"
                name = f"{stem}.txt"
                text = T.render_photo_note(case, rng, n_photo, stem)
                n_photo += 1
            elif doc == "purchase_receipt":
                for j, item in enumerate(claim["items"]):
                    rname = rng.choice([f"Quittung_{j + 1}.txt", f"receipt_{j + 1}.txt", f"scan_{n_scan:03d}.txt"])
                    while rname in files:
                        rname = f"receipt_{j + 1}_{rng.randint(10, 99)}.txt"
                    (pdir / rname).write_text(T.render_receipt(item, case, rng, rng.randint(0, 1)))
                    files[rname] = "purchase_receipt"
                    n_scan += 1
                continue
            else:
                raise ValueError(doc)
            while name in files:
                name = name.replace(".txt", f"_{rng.randint(10, 99)}.txt")
            (pdir / name).write_text(text)
            files[name] = doc
            n_scan += 1
        meta = dict(package_id=pid, received_at=f"{claim['received']}T{hh:02d}:{mm:02d}:00", channel=rng.choice(["email", "email", "web_form"]),
                    sender=case["customer"]["email"], attachments=[f for f in files if f != "email.txt"])
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2))

        if case["split"] != "demo":
            got = rules.oracle(case, history, cat.WEATHER_EVENTS, oracle_customers)
            exp = case["expected"]
            if (got["decision"], got["payout"], got["missing"], got["issues"]) != (exp["decision"], exp["payout"], exp["missing"], exp["issues"]):
                mismatches.append((case["key"], exp, got))
        labels.append(dict(
            package_id=pid, split=case["split"], category=case["category"], documents=files,
            facts=dict(policy_id=case["policy"]["policy_id"], product=case["policy"]["product"], claim_type=claim["type"],
                       incident_date=claim["incident"], received_date=claim["received"], amount_chf=claim["amount"],
                       invoice_no=claim["invoice_no"], vendor=claim["vendor"]),
            expected=dict(decision=case["expected"]["decision"], payout_chf=case["expected"]["payout"],
                          missing_items=case["expected"]["missing"], issues=case["expected"]["issues"]),
            explanation=case["explanation"], case_key=case["key"]))

    labels.sort(key=lambda l: l["package_id"])
    with open(OUT / "labels" / "labels.jsonl", "w") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    with open(OUT / "labels" / "faq.jsonl", "w") as f:
        for q in cat.FAQ:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    mismatches += add_complex(customers, history)
    mismatches += add_complex(customers, history, ij.INJECTED, "permissions.jsonl", split="permissions",
                              category="instruction_in_document", first_customer=1910)
    mismatches += add_complex(customers, history, jd.JUDGEMENT, "judgement.jsonl", split="judgement",
                              category="flagged_but_harmless", first_customer=1920)

    if mismatches:
        for k, e, g in mismatches:
            print(f"MISMATCH {k}: expected {e} | oracle {g}")
        raise SystemExit(f"{len(mismatches)} label(s) disagree with the oracle")
    n = {s: sum(1 for l in labels if l["split"] == s) for s in ("dev", "heldout", "demo")}
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"built {len(labels)} packages {n} + {len(cx.COMPLEX)} with several policies, {len(ij.INJECTED)} with an instruction in a document, {len(jd.JUDGEMENT)} flagged but harmless, {len(customers)} customers, {len(history)} historic claims, {size / 1024:.0f} KB -> {OUT}")


def add_complex(customers: list[dict], history: list[dict], items: list | None = None, label_file: str = "complex.jsonl",
                split: str = "complex", category: str = "several_policies", first_customer: int = 1900) -> list:
    """Hand-written packages appended after everything else and without random draws, so the 40 scored packages and
    their database rows do not change: the three packages with several policies (generator/complex.py), the package
    with an instruction in a document (generator/injected.py) and the flagged but harmless roof (generator/judgement.py).
    Returns label mismatches."""
    db = sqlite3.connect(OUT / "lakeside.db")
    new_history, mismatches = [], []
    with open(OUT / "labels" / label_file, "w") as f:
        for i, x in enumerate(cx.COMPLEX if items is None else items):
            cu = dict(x["customer"], customer_id=f"C-{first_customer + i}", customer_since="2015-04-01")
            db.execute("INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
                       (cu["customer_id"], cu["name"], cu["street"], cu["postcode"], cu["city"], cu["email"], cu["customer_since"]))
            for p in x["policies"]:
                db.execute("INSERT INTO policies VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (p["policy_id"], cu["customer_id"], p["product"], p["status"], p["valid_from"], p["valid_to"],
                            json.dumps(p["modules"]), json.dumps(p["deductibles"]), p["sum_insured"], json.dumps(p["object"])))
            for j, h in enumerate(x["history"]):
                h = dict(h, claim_id=f"C-{h['incident_date'][:4]}-9{i}{j:02d}")
                db.execute("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?)",
                           (h["claim_id"], h["policy_id"], h["incident_date"], h["claim_type"], h["amount_chf"], h["status"],
                            h["invoice_no"], h["vendor"], h["note"]))
                new_history.append(h)
            pdir = OUT / "inbox" / x["package_id"]
            pdir.mkdir()
            for name, (_, text) in x["documents"].items():
                (pdir / name).write_text(text if name != "email.txt" else
                                         f"From: {cu['name']} <{cu['email']}>\nTo: claims@lakeside-insurance.example\n"
                                         f"Date: {x['received_at'][:10]} {x['received_at'][11:16]}\nSubject: {x.get('subject', 'Claim - several damages')}\n\n{text}\n")
            meta = dict(package_id=x["package_id"], received_at=x["received_at"], channel="email", sender=cu["email"],
                        attachments=[n for n in x["documents"] if n != "email.txt"])
            (pdir / "meta.json").write_text(json.dumps(meta, indent=2))
            parts = []
            for part in x["parts"]:
                case = cx.part_case(x, part)
                got = rules.oracle(case, history + new_history, cat.WEATHER_EVENTS, customers + [cu])
                exp = part["expected"]
                if (got["decision"], got["payout"], got["missing"], got["issues"]) != (exp["decision"], exp["payout"], exp["missing"], exp["issues"]):
                    mismatches.append((case["key"], exp, got))
                pol, claim = case["policy"], part["claim"]
                parts.append(dict(policy_id=pol["policy_id"], product=pol["product"], what=part["what"], claim_type=claim["type"],
                                  amount_chf=claim["amount"], invoice_no=claim["invoice_no"], documents=part["documents"],
                                  expected=dict(decision=exp["decision"], payout_chf=exp["payout"], missing_items=exp["missing"],
                                                issues=exp["issues"]), explanation=part["explanation"]))
            f.write(json.dumps(dict(package_id=x["package_id"], split=split, category=category, title=x["title"],
                                    documents={n: t for n, (t, _) in x["documents"].items()}, incident_date=x["parts"][0]["claim"]["incident"],
                                    parts=parts, case_key=x["key"]), ensure_ascii=False) + "\n")
    db.commit()
    db.close()
    return mismatches


if __name__ == "__main__":
    main()
