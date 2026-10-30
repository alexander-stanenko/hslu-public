"""Business rules of Lakeside Insurance AG and the oracle that turns case facts into expected labels.

The same rules are written in prose into terms/*.md and terms/claims_guideline.md, so an agent
can discover them through the policy-terms search tool.
"""
from __future__ import annotations

from datetime import date

COMPANY = "Lakeside Insurance AG"
FAST_TRACK_LIMIT_CHF = 5000
REPORTING_DEADLINE_DAYS = 30
FREQUENCY_WINDOW_DAYS = 365
FREQUENCY_THRESHOLD = 3          # claims within the window, including the new one
PRICE_TOLERANCE = 1.5            # amount > 1.5 x typical max -> red flag

COVERAGE = {
    "motor": {
        "liability": set(),
        "partial": {"glass", "hail", "storm", "theft", "fire", "animal_collision", "vandalism"},
        "collision": {"collision"},
    },
    "household": {
        "basic": {"fire", "hail", "storm", "water_damage", "burglary"},
        "theft_outside": {"theft_outside"},
        "glass": {"glass_breakage"},
    },
    "building": {
        "basic": {"fire", "hail", "storm", "water_damage"},
        "glass": {"glass_breakage"},
    },
}

# Each inner list is "one of these document types must be present".
REQUIRED_DOCS = {
    "glass": [("invoice", "estimate")],
    "hail": [("invoice", "estimate")],
    "storm": [("invoice", "estimate")],
    "collision": [("invoice", "estimate")],
    "fire": [("invoice", "estimate")],
    "water_damage": [("invoice", "estimate")],
    "glass_breakage": [("invoice", "estimate")],
    "vandalism": [("police_report",), ("invoice", "estimate")],
    "animal_collision": [("police_report",), ("invoice", "estimate")],
    "theft": [("police_report",)],
    "burglary": [("police_report",), ("purchase_receipt",)],
    "theft_outside": [("police_report",), ("purchase_receipt",)],
}
MISSING_ITEM_NAME = {
    ("invoice", "estimate"): "invoice_or_estimate",
    ("police_report",): "police_report",
    ("purchase_receipt",): "proof_of_ownership",
}

PRICE_RANGES = {  # repair_type: (typical_min_chf, typical_max_chf)
    "windscreen_chip_repair": (80, 250),
    "windscreen_replacement": (900, 1400),
    "hail_dent_repair": (1500, 12000),
    "front_end_repair": (1500, 5000),
    "bumper_repair": (800, 2500),
    "paint_repair": (600, 2500),
    "mirror_replacement": (250, 900),
    "body_repair": (800, 4000),
    "water_damage_drying": (1000, 6000),
    "pipe_repair": (400, 2500),
    "roof_repair": (600, 5000),
    "glass_household": (200, 900),
}

WEATHER_CLAIM_TYPES = {"hail": "hail", "storm": "storm"}


def d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def deductible_for(product: str, claim_type: str, deductibles: dict) -> float:
    if product == "motor":
        if claim_type == "glass":
            return deductibles["glass"]
        if claim_type == "collision":
            return deductibles["collision"]
        return deductibles["partial"]
    if product == "building" and claim_type in ("hail", "storm"):
        return deductibles["natural_hazards"]
    return deductibles["default"]


def covered(product: str, modules: dict, claim_type: str) -> bool:
    return any(claim_type in COVERAGE[product][m] for m, on in modules.items() if on)


def oracle(case: dict, all_claims: list[dict], weather: list[dict], customers: list[dict]) -> dict:
    """Compute the expected decision for a case from its structured facts (rules §3 of the spec)."""
    pol, claim = case["policy"], case["claim"]
    inc, rec = d(claim["incident"]), d(claim["received"])

    # 1. policy identification
    if not case.get("policy_ref_in_docs", True):
        same_name = [c for c in customers if c["name"] == case["customer"]["name"]]
        if len(same_name) != 1:
            return dict(decision="request_info", payout=None, missing=["policy_id"], issues=[])
    # 2. active on incident date
    if not (d(pol["valid_from"]) <= inc <= d(pol["valid_to"])):
        return dict(decision="decline", payout=None, missing=[], issues=["policy_inactive"])
    # 3. coverage
    if not covered(pol["product"], pol["modules"], claim["type"]):
        return dict(decision="decline", payout=None, missing=[], issues=["not_covered"])
    # 4. required documents
    present = set(case["docs"])
    missing = [MISSING_ITEM_NAME[g] for g in REQUIRED_DOCS[claim["type"]] if not present.intersection(g)]
    if missing:
        return dict(decision="request_info", payout=None, missing=missing, issues=[])
    # 5. red flags
    issues = []
    if claim.get("invoice_date") and d(claim["invoice_date"]) < inc:
        issues.append("invoice_before_incident")
    if claim.get("invoice_no") and any(c["invoice_no"] == claim["invoice_no"] for c in all_claims):
        issues.append("duplicate_invoice")
    recent = [c for c in all_claims if c["policy_id"] == pol["policy_id"]
              and 0 <= (inc - d(c["incident_date"])).days <= FREQUENCY_WINDOW_DAYS]
    if len(recent) + 1 >= FREQUENCY_THRESHOLD:
        issues.append("frequency")
    rt = claim.get("repair_type")
    if rt in PRICE_RANGES and claim["amount"] and claim["amount"] > PRICE_TOLERANCE * PRICE_RANGES[rt][1]:
        issues.append("amount_above_typical")
    if claim["type"] in WEATHER_CLAIM_TYPES:
        region = claim.get("location_postcode", case["customer"]["postcode"])[:2]
        if not any(w["date"] == claim["incident"] and w["region"] == region and w["event"] == WEATHER_CLAIM_TYPES[claim["type"]] for w in weather):
            issues.append("no_weather_event")
    if (rec - inc).days > REPORTING_DEADLINE_DAYS:
        issues.append("late_report")
    issues += case.get("flags", [])          # textual flags: possible_exclusion, inconsistent_documents
    # 6. fast-track limit
    if claim["amount"] and claim["amount"] > FAST_TRACK_LIMIT_CHF:
        issues.append("above_fast_track_limit")
    if issues:
        return dict(decision="refer_adjuster", payout=None, missing=[], issues=issues)
    # 7. fast track
    ded = deductible_for(pol["product"], claim["type"], pol["deductibles"])
    payout = max(0.0, min(claim["amount"], pol.get("sum_insured") or claim["amount"]) - ded)
    return dict(decision="fast_track", payout=round(payout, 2), missing=[], issues=[])
