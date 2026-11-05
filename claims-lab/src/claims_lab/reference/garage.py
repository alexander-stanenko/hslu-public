"""After class in notebook 05: a garage under suspicion. An adjuster suspects that Garage Lindenmatt AG overcharges
Lakeside. Three designs review the same 25 invoices from six garages with the same four read-only tools:

    from claims_lab.reference import garage
    garage.show_invoices()                         # what there is to review, no model call
    run = garage.review("team")                    # "one agent", "team" or "workflow"; about 15 model calls for the team
    garage.versus_recorded(run)                    # your run next to our runs
    garage.serial_check()                          # the reused-part check, in code, no model call

The invoices are synthetic and fixed (build(seed=7)). Planted problems, in TRUTH: three Lindenmatt invoices bill a
part whose serial number Carrosserie Speedfix already billed for another car, one bumper repair is far above the usual
range, and one invoice bills the same labour line twice. The task names only Lindenmatt, so the three Speedfix
invoices are found only by following the serial numbers to the other garage.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..models import get_model, run_config

# ------------------------------------------------------------------ the invoices
PRICES = {"windscreen_replacement": (900, 1400), "bumper_repair": (800, 2500), "mirror_replacement": (250, 900),
          "body_repair": (800, 4000), "paint_repair": (600, 2500), "front_end_repair": (1500, 5000)}

CARS = ["VW Golf (2021)", "Skoda Octavia (2019)", "Toyota Yaris (2020)", "Seat Leon (2020)", "Audi A3 (2018)",
        "BMW 118i (2022)", "Renault Clio (2017)", "Ford Focus (2019)", "Opel Corsa (2021)", "Peugeot 308 (2020)",
        "Mazda 3 (2018)", "Hyundai i30 (2021)", "Kia Ceed (2022)", "VW Polo (2016)", "Volvo V60 (2019)"]

PARTS = {"windscreen_replacement": ("Windscreen, OEM quality", "WS"), "bumper_repair": ("Front bumper, primed", "BP"),
         "mirror_replacement": ("Door mirror, heated, left", "MR"), "body_repair": ("Rear door skin", "DS"),
         "paint_repair": ("Clear coat and paint set", "PT"), "front_end_repair": ("Headlamp unit, LED, right", "HL")}

VENDORS = {"LM": ("Garage Lindenmatt AG", "Industriestrasse 14, 6010 Kriens"),
           "SF": ("Carrosserie Speedfix GmbH", "Littauerboden 3, 6014 Luzern"),
           "HU": ("Garage Huber AG", "Baselstrasse 40, 6003 Luzern"),
           "AZ": ("AutoGlas Zentral GmbH", "Bahnhofstrasse 9, 6340 Baar"),
           "KR": ("Carrosserie Kriens AG", "Obernauerstrasse 2, 6010 Kriens"),
           "ZL": ("Garage Zugerland AG", "Chamerstrasse 77, 6300 Zug")}

TRUTH = {"LM-1003": "reused part (SF-2002)", "LM-1007": "reused part (SF-2004)", "LM-1010": "reused part (SF-2006)",
         "LM-1005": "price far above range", "LM-1009": "labour billed twice",
         "SF-2002": "reused part (LM-1003)", "SF-2004": "reused part (LM-1007)", "SF-2006": "reused part (LM-1010)"}

_REUSED = {"LM-1003": "SF-2002", "LM-1007": "SF-2004", "LM-1010": "SF-2006"}


def _serial(rng) -> str:
    return f"{rng.randint(10, 99)}{rng.choice('ABCDEFGHJK')}{rng.randint(1, 9)}-{rng.randint(1000, 9999)}"


def build(seed: int = 7) -> dict[str, dict]:
    """The 25 invoices, as {invoice_id: {invoice_id, vendor, repair_type, total_chf, date, text}}."""
    rng = random.Random(seed)
    lm_types = ["windscreen_replacement", "bumper_repair", "front_end_repair", "mirror_replacement", "bumper_repair",
                "paint_repair", "front_end_repair", "body_repair", "body_repair", "windscreen_replacement",
                "mirror_replacement", "paint_repair"]
    sf_types = ["body_repair", "front_end_repair", "paint_repair", "front_end_repair", "bumper_repair",
                "windscreen_replacement", "mirror_replacement"]
    plan = ([("LM", 1001 + i, t) for i, t in enumerate(lm_types)] + [("SF", 2001 + i, t) for i, t in enumerate(sf_types)]
            + [("HU", 3001, "windscreen_replacement"), ("AZ", 3002, "windscreen_replacement"), ("KR", 3003, "body_repair"),
               ("ZL", 3004, "bumper_repair"), ("HU", 3005, "mirror_replacement"), ("KR", 3006, "front_end_repair")])
    ids = [f"{p}-{n}" for p, n, _ in plan]
    serials = {iid: _serial(rng) for iid in ids}
    for lm, sf in _REUSED.items():   # the same physical part billed twice: the LM invoice copies the SF serial and part
        serials[lm] = serials[sf]
    invoices = {}
    for k, (p, n, rtype) in enumerate(plan):
        iid = f"{p}-{n}"
        if iid in _REUSED:
            rtype = plan[ids.index(_REUSED[iid])][2]
        lo, hi = PRICES[rtype]
        part_name, code = PARTS[rtype]
        part = round(rng.uniform(0.35, 0.5) * (lo + hi) / 2, -1)
        hours = rng.choice([2.5, 3.0, 3.5, 4.0, 4.5])
        rate = 128 if p == "LM" else rng.choice([118, 124, 132])
        lines = [(part_name, f"{code}-{rng.randint(100000, 999999)} / SN {serials[iid]}", 1, part),
                 (f"Labour, {hours} h at CHF {rate}", "", hours, hours * rate),
                 ("Consumables and disposal", "", 1, round(rng.uniform(40, 90), 0))]
        if rtype == "windscreen_replacement":
            lines.insert(2, ("ADAS camera calibration", "", 1, 106.0))
        if iid == "LM-1005":   # far above the typical maximum of 2,500
            lines[0] = (part_name, lines[0][1], 1, 2380.0)
            lines[1] = (f"Labour, 11.5 h at CHF {rate}", "", 11.5, 11.5 * rate)
        if iid == "LM-1009":   # the labour line appears twice
            lines.insert(2, lines[1])
        net = round(sum(l[3] for l in lines), 2)
        vat = round(net * 0.081, 2)
        name, addr = VENDORS[p]
        body = [f"INVOICE {iid}", f"{name}, {addr}", f"Date: 2026-{1 + k % 9:02d}-{10 + k % 18:02d}    Lakeside claim "
                f"C-2026-{n}    Policy P-{11000 + k * 7}", f"Vehicle: {CARS[(k * 4 + 3) % len(CARS)]}",
                f"Work: {rtype.replace('_', ' ')}", "",
                f"{'Pos':<4}{'Description':<34}{'Part no. / serial':<32}{'Qty':>5}{'CHF':>11}"]
        for i, (d, ref, q, amt) in enumerate(lines, 1):
            body.append(f"{i:<4}{d:<34}{ref:<32}{q:>5}{amt:>11,.2f}")
        body += ["", f"Net CHF {net:,.2f}    VAT 8.1% CHF {vat:,.2f}    Total CHF {net + vat:,.2f}",
                 "Payment within 30 days. Thank you for your trust."]
        invoices[iid] = {"invoice_id": iid, "vendor": name, "repair_type": rtype, "total_chf": round(net + vat, 2),
                         "date": body[2].split()[1], "text": "\n".join(body)}
    return invoices


INVOICES = build()

TASK = ("An adjuster suspects that Garage Lindenmatt AG overcharges Lakeside. Review Garage Lindenmatt's invoices, "
        "and follow anything they lead to, also at other garages. Report every invoice, of any garage, that looks wrong, "
        "with a short reason.")


class Finding(BaseModel):
    invoice_id: str
    reason: str = Field(description="one sentence")


class Report(BaseModel):
    findings: list[Finding] = Field(description="every invoice that looks wrong; empty if none")


# ------------------------------------------------------------------ the tools: the same read-only functions for every design
@tool
def list_invoices(vendor: str) -> str:
    """List the invoices of one garage (part of its name is enough): id, date, repair type and total."""
    rows = [{k: v[k] for k in ("invoice_id", "date", "repair_type", "total_chf")} for v in INVOICES.values()
            if vendor.lower() in v["vendor"].lower()]
    return json.dumps(rows)


@tool
def read_invoice(invoice_id: str) -> str:
    """The full text of one invoice, with its line items, part numbers and serial numbers."""
    inv = INVOICES.get(invoice_id.strip().upper())
    return inv["text"] if inv else f"No invoice {invoice_id}."


@tool
def search_invoices(text: str) -> str:
    """Search the text of all invoices of all garages, for example for a serial number. Returns matching invoice ids."""
    hits = [i for i, v in INVOICES.items() if text.lower() in v["text"].lower()]
    return json.dumps(hits[:10]) + (f" (and {len(hits) - 10} more)" if len(hits) > 10 else "")


@tool
def price_range(repair_type: str) -> str:
    """Typical minimum and maximum total in CHF for a repair type. Far above means more than 1.5 times the maximum."""
    r = PRICES.get(repair_type.strip().lower().replace(" ", "_"))
    return json.dumps({"repair_type": repair_type, "typical_min_chf": r[0], "typical_max_chf": r[1]}) if r else "unknown type"


TOOLS = [list_invoices, read_invoice, search_invoices, price_range]

# ------------------------------------------------------------------ the designs
AGENT_PROMPT = ("You are a claims investigator at Lakeside Insurance. Use the tools to read invoices. Look for anything "
                "that suggests overcharging or fraud, for example prices far above the usual range, items billed twice, "
                "or the same part (same serial number) billed on more than one invoice. Report only invoices you checked.")
# one agent gets the same working instructions as the team's reviewers, so the comparison measures the split
ONE_AGENT_PROMPT = AGENT_PROMPT + (" Read each invoice, check prices and serial numbers (search every serial number), "
                                   "and follow what you find to other garages.")
REVIEWER_PROMPT = AGENT_PROMPT + (" You get a few invoice ids. Read each of them, check prices and serial numbers "
                                  "(search every serial number), and report what looks wrong. Be brief.")
LEAD_PROMPT = ("You lead an investigation at Lakeside Insurance. You do not read invoices yourself. List the invoices, "
               "then give them to reviewers in batches of about four with review_invoices; reviewers can run at the same "
               "time. When reviewers report invoices at other garages, have those reviewed too. Then write the report.")
REVIEW_DESCRIPTION = ("Give a batch of invoices (ids) to a reviewer, who reads them, checks prices and serial numbers, "
                      "and reports findings.")


def one_agent():
    """One agent with all four tools."""
    return create_agent(get_model(), TOOLS, system_prompt=ONE_AGENT_PROMPT, response_format=Report)


def team(collected: list | None = None):
    """A lead agent that lists invoices and hands batches to reviewer subagents (subagents as tools). Each reviewer starts
    with an empty context and gets read_invoice, search_invoices and price_range. collected receives the reviewers'
    messages, so a run can count all calls."""
    collected = [] if collected is None else collected

    @tool("review_invoices", description=REVIEW_DESCRIPTION)
    def review_invoices(invoice_ids: list[str]) -> str:
        sub = create_agent(get_model(), [read_invoice, search_invoices, price_range], system_prompt=REVIEWER_PROMPT,
                           response_format=Report)
        r = sub.invoke({"messages": [("user", "Review these invoices: " + ", ".join(invoice_ids))]}, config=run_config(20))
        collected.extend(r["messages"])
        return r["structured_response"].model_dump_json()

    return create_agent(get_model(), [list_invoices, search_invoices, review_invoices], system_prompt=LEAD_PROMPT,
                        response_format=Report)


class Review(BaseModel):
    looks_wrong: bool
    reason: str = Field(description="one sentence; empty if nothing looks wrong")


def workflow(config: dict | None = None) -> Report:
    """Code lists Lindenmatt's invoices and asks one model call per invoice, with the price range, at the same time."""
    llm = get_model().with_structured_output(Review, method="json_schema")
    ids = [i for i, v in INVOICES.items() if v["vendor"].startswith("Garage Lindenmatt")]

    def one(iid):
        inv = INVOICES[iid]
        lo, hi = PRICES[inv["repair_type"]]
        return iid, llm.invoke([("system", AGENT_PROMPT),
                                ("user", f"{inv['text']}\n\nTypical total for {inv['repair_type']}: CHF {lo:,} to {hi:,}. "
                                         "Does this invoice look wrong?")], config=config)
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(one, ids))
    return Report(findings=[Finding(invoice_id=i, reason=r.reason) for i, r in out if r.looks_wrong])


def serial_check() -> Report:
    """The reused-part check in code, over all invoices of all garages: a serial number billed on more than one invoice.
    No model call."""
    by_serial = defaultdict(list)
    for iid, inv in INVOICES.items():
        for line in inv["text"].splitlines():
            if " / SN " in line:
                by_serial[line.split(" / SN ")[1].split()[0]].append(iid)
    return Report(findings=[Finding(invoice_id=i, reason=f"serial number {sn} also billed on {', '.join(o for o in ids if o != i)}")
                            for sn, ids in by_serial.items() if len(ids) > 1 for i in ids])


# ------------------------------------------------------------------ running and scoring
def score(report: Report) -> dict:
    found = {f.invoice_id.strip().upper() for f in report.findings}
    hits = found & set(TRUTH)
    return {"found": len(hits), "of": len(TRUTH), "at Speedfix": len([h for h in hits if h.startswith("SF")]),
            "false alarms": len(found - set(TRUTH)), "missed": sorted(set(TRUTH) - found), "hits": sorted(hits)}


DESIGNS = ("one agent", "team", "workflow")


def review(design: str = "team") -> dict:
    """One run of a design: its report, the score against TRUTH, model calls, cost and time."""
    from ..costs import CostMeter
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    meter, t0, msgs = CostMeter(max_calls=80, max_input_tokens=60_000), time.time(), []
    if design == "workflow":
        report = workflow({"callbacks": [meter]})
    else:
        agent = one_agent() if design == "one agent" else team(msgs)
        r = agent.invoke({"messages": [("user", TASK)]}, config=run_config(30, callbacks=[meter]))
        report, msgs = r["structured_response"], r["messages"] + msgs
    calls = [c for m in msgs if getattr(m, "type", "") == "ai" for c in (m.tool_calls or [])]
    result = {"design": design, "report": report, **score(report), "llm_calls": meter.calls + meter.replayed_calls,
              "cost_usd": round(meter.cost_usd + meter.replayed_cost_usd, 4), "seconds": round(time.time() - t0, 1),
              "invoices_read": len({c["args"].get("invoice_id", "").upper() for c in calls if c["name"] == "read_invoice"}),
              "searches": sum(c["name"] == "search_invoices" for c in calls)}
    show(result)
    return result


def show(result: dict) -> None:
    print(f"{result['design']}: found {result['found']} of {result['of']} ({result['at Speedfix']} of 3 at Speedfix), "
          f"{result['false alarms']} false alarms; {result['llm_calls']} model calls, ${result['cost_usd']:.3f}, "
          f"{result['seconds']:.0f} s; read {result['invoices_read']} invoices, {result['searches']} searches")
    for f in result["report"].findings:
        print(f"  {f.invoice_id:<8} {'yes' if f.invoice_id.strip().upper() in TRUTH else 'no '}  {f.reason[:120]}")
    if result["missed"]:
        print("  missed: " + ", ".join(f"{i} ({TRUTH[i]})" for i in result["missed"]))


def show_invoices():
    """The invoices per garage: what there is to review. No model call."""
    import pandas as pd
    rows = [{"invoice": v["invoice_id"], "garage": v["vendor"], "date": v["date"], "repair": v["repair_type"].replace("_", " "),
             "total CHF": f"{v['total_chf']:,.2f}"} for v in INVOICES.values()]
    return pd.DataFrame(rows)


def versus_recorded(result: dict | None = None):
    """Your run next to our runs of the three designs and of the code check (ranges over several runs)."""
    import pandas as pd

    from ..course import GARAGE_REFERENCE_RUNS
    rows = []
    if result is not None:
        rows.append({"design": f"{result['design']}, your run", "runs": 1, "found (of 8)": str(result["found"]),
                     "at Speedfix (of 3)": str(result["at Speedfix"]), "false alarms": str(result["false alarms"]),
                     "invoices read": str(result["invoices_read"]), "cost per run": result["cost_usd"],
                     "seconds per run": result["seconds"]})
    for r in GARAGE_REFERENCE_RUNS:
        rows.append({"design": r["label"] + " (our runs)", "runs": r["runs"], "found (of 8)": r["found"],
                     "at Speedfix (of 3)": r["at_speedfix"], "false alarms": r["false_alarms"],
                     "invoices read": r["invoices_read"], "cost per run": r["cost_usd_per_run"],
                     "seconds per run": r["seconds_per_run"]})
    return pd.DataFrame(rows)
