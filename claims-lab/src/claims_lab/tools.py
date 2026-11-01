"""Lakeside's systems as plain Python functions: what a company's existing APIs would be.

They return Python data (dicts, lists, numbers), so code can use them directly: the hybrid pipeline, the MCP servers,
the tests. The notebooks turn them into LangChain tools with `@tool` from langchain_core.tools, and each tool returns
`json.dumps(...)` of the result, so the model always gets text, an empty result included.
"""
from __future__ import annotations

import json

from . import data
from .retrieval import search_policy_terms as _search


def _policy_row(p: dict) -> dict:
    cust = data.query("SELECT name, street, postcode, city FROM customers WHERE customer_id=?", (p["customer_id"],))[0]
    return dict(policy_id=p["policy_id"], holder=cust["name"], address=f"{cust['street']}, {cust['postcode']} {cust['city']}",
                postcode=cust["postcode"], product=p["product"], status=p["status"], valid_from=p["valid_from"], valid_to=p["valid_to"],
                modules=json.loads(p["modules"]), deductibles_chf=json.loads(p["deductibles_chf"]),
                sum_insured_chf=p["sum_insured_chf"], insured_object=json.loads(p["insured_object"]))


# ------------------------------------------------------------------ read tools
def find_policies(name: str | None = None, plate: str | None = None, postcode: str | None = None) -> list[dict]:
    """Find policies when the policy number is unknown. Search by customer name (exact or partial), vehicle plate
    (e.g. 'LU 248 113') and/or postcode. Returns all matching policies; more than one match means the policy is not
    uniquely identified."""
    rows = data.query("SELECT p.* FROM policies p JOIN customers c ON p.customer_id=c.customer_id")
    out = []
    for p in rows:
        r = _policy_row(p)
        if name and name.lower() not in r["holder"].lower():
            continue
        if plate and plate.replace(" ", "").upper() != str(r["insured_object"].get("plate", "")).replace(" ", "").upper():
            continue
        if postcode and postcode != r["postcode"]:
            continue
        out.append(r)
    return out


def lookup_policy(policy_id: str) -> dict:
    """Look up a policy by its number (e.g. 'P-10001'). Returns holder, product (motor/household/building), status,
    validity dates, included modules (coverage), deductibles in CHF, sum insured in CHF and the insured object."""
    rows = data.query("SELECT * FROM policies WHERE policy_id=?", (policy_id.strip().upper(),))
    if not rows:
        return {"error": f"No policy with number {policy_id}. Use find_policies to search by name, plate or postcode."}
    return _policy_row(rows[0])


def get_claim_history(policy_id: str) -> list[dict]:
    """List all earlier claims on a policy (date, type, amount in CHF, status, invoice number, vendor).
    Use it to check claim frequency and repeated invoices."""
    return data.query("SELECT claim_id, incident_date, claim_type, amount_chf, status, invoice_no, vendor FROM claims "
                      "WHERE policy_id=? ORDER BY incident_date", (policy_id.strip().upper(),))


def search_claims_by_invoice(invoice_no: str) -> list[dict]:
    """Check whether an invoice number was already used in any earlier claim (across all policies).
    An empty list means the invoice number is new."""
    return data.query("SELECT claim_id, policy_id, incident_date, amount_chf, status, vendor FROM claims WHERE invoice_no=?",
                      (invoice_no.strip(),))


def get_price_range(repair_type: str) -> dict:
    """Typical price range in CHF for a repair type from Lakeside's price list. Known repair types:
    windscreen_chip_repair, windscreen_replacement, hail_dent_repair, front_end_repair, bumper_repair, paint_repair,
    mirror_replacement, body_repair, water_damage_drying, pipe_repair, roof_repair, glass_household."""
    rows = data.query("SELECT * FROM price_ranges WHERE repair_type=?", (repair_type,))
    if not rows:
        known = [r["repair_type"] for r in data.query("SELECT repair_type FROM price_ranges")]
        return {"error": f"Unknown repair type {repair_type!r}", "known_repair_types": known}
    return rows[0]


def check_weather(postcode: str, date: str) -> list[dict]:
    """Weather events (hail, storm, heavy_rain) recorded by the weather service for the region of a postcode
    (first two digits) on a date (YYYY-MM-DD). An empty list means no event was recorded."""
    return data.query("SELECT * FROM weather_events WHERE region=? AND date=?", (postcode.strip()[:2], date.strip()))


def search_policy_terms(query: str, product: str | None = None) -> list[dict]:
    """Search the general terms and conditions (motor, household, building) and the internal claims handling guideline.
    Returns the most relevant sections with their reference (e.g. 'motor §6'). product: optional filter
    'motor', 'household', 'building' or 'guideline'."""
    return [{k: h[k] for k in ("source", "title", "text")} for h in _search(query, product=product, k=3)]


def calc_payout(amount_chf: float, deductible_chf: float, sum_insured_chf: float | None = None) -> float:
    """Payout in CHF = claimed amount (capped at the sum insured, if any) minus the deductible, never below 0."""
    capped = min(amount_chf, sum_insured_chf) if sum_insured_chf else amount_chf
    return round(max(0.0, capped - deductible_chf), 2)


def read_package(package_id: str) -> str:
    """Read all documents of a claim package from the inbox (email and attachments)."""
    return data.load_package(package_id).text()


# ------------------------------------------------------------------ write tools (side effects!)
def request_documents(package_id: str, missing_items: list[str], message: str) -> str:
    """SEND an e-mail to the customer asking for missing documents or information. This contacts the customer."""
    sender = data.load_package(package_id).sender
    data.execute("INSERT INTO outbox(package_id, recipient, subject, body) VALUES (?,?,?,?)",
                 (package_id, sender, f"Your claim {package_id}: documents needed", message + "\nMissing: " + ", ".join(missing_items)))
    return f"E-mail sent to {sender} requesting: {', '.join(missing_items)}"


def record_decision(package_id: str, decision: str, payout_chf: float | None = None, note: str = "") -> str:
    """RECORD the final decision for a claim package in the claims system. fast_track triggers the payment."""
    data.execute("INSERT INTO decisions(package_id, decision, payout_chf, note) VALUES (?,?,?,?)",
                 (package_id, decision, payout_chf, note))
    return f"Decision recorded for {package_id}: {decision}" + (f", payout CHF {payout_chf:,.2f}" if payout_chf else "")


READ_FUNCTIONS = [find_policies, lookup_policy, get_claim_history, search_claims_by_invoice, get_price_range,
                  check_weather, search_policy_terms, calc_payout]
WRITE_FUNCTIONS = [request_documents, record_decision]
