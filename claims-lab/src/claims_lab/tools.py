"""Lakeside's systems as plain Python functions: what a company's existing APIs would be.

They return Python data (dicts, lists, numbers), so code can use them directly: the hybrid pipeline, the MCP servers,
the tests. The notebooks turn them into LangChain tools with `@tool` from langchain_core.tools, and each tool returns
`json.dumps(...)` of the result, so the model always gets text, an empty result included.
"""
from __future__ import annotations

import json

from . import data
from .retrieval import search as _search


def _policy_row(p: dict) -> dict:
    return dict(policy_id=p["policy_id"], holder=p["holder"], plate=p["plate"], car=p["car"],
                covers=[c.strip() for c in p["covers"].split(",")], deductible_chf=p["deductible_chf"],
                valid_from=p["valid_from"], valid_to=p["valid_to"])


# ------------------------------------------------------------------ read tools
def find_policies(name: str | None = None, plate: str | None = None) -> list[dict]:
    """Find policies when the policy number is unknown: by the holder's name (or part of it) and/or the car's plate
    (e.g. 'LU 248 113'). Returns every matching policy; more than one match means the policy is not identified yet."""
    out = []
    for p in data.query("SELECT * FROM policies"):
        if name and name.lower() not in p["holder"].lower():
            continue
        if plate and plate.replace(" ", "").upper() != p["plate"].replace(" ", "").upper():
            continue
        out.append(_policy_row(p))
    return out


def lookup_policy(policy_id: str) -> dict:
    """Look up a policy by its number (e.g. 'P-1001'). Returns the holder, the car and its plate, the kinds of damage
    the policy covers, the deductible in CHF and the dates between which the policy is valid."""
    rows = data.query("SELECT * FROM policies WHERE policy_id=?", (policy_id.strip().upper(),))
    if not rows:
        return {"error": f"No policy with number {policy_id}. Use find_policies to search by name or plate."}
    return _policy_row(rows[0])


def get_claim_history(policy_id: str) -> list[dict]:
    """List the earlier claims on a policy: date, kind of damage, amount in CHF, invoice number, garage and status."""
    return data.query("SELECT claim_id, date, damage, amount_chf, invoice_no, garage, status FROM claims "
                      "WHERE policy_id=? ORDER BY date", (policy_id.strip().upper(),))


def search_claims_by_invoice(invoice_no: str) -> list[dict]:
    """Check whether an invoice number was already used in an earlier claim, on any policy.
    An empty list means the invoice number is new."""
    return data.query("SELECT claim_id, policy_id, date, amount_chf, garage, status FROM claims WHERE invoice_no=?",
                      (invoice_no.strip(),))


def search_terms(query: str) -> list[dict]:
    """Search Lakeside's general terms for car insurance. Returns the most relevant sections with their number
    (e.g. '§3'): what is covered, deductible and payout, what is never covered, documents a claim needs."""
    return [{k: h[k] for k in ("source", "title", "text")} for h in _search(query, k=2)]


def calc_payout(amount_chf: float, deductible_chf: float) -> float:
    """Payout in CHF = the invoice amount minus the deductible, never below 0."""
    return round(max(0.0, amount_chf - deductible_chf), 2)


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


READ_FUNCTIONS = [find_policies, lookup_policy, get_claim_history, search_claims_by_invoice, search_terms, calc_payout]
WRITE_FUNCTIONS = [request_documents, record_decision]
