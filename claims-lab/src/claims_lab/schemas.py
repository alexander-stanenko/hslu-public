"""Shared output schemas. Every claims system built in the course returns a ClaimDecision."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["fast_track", "request_info", "refer_adjuster", "decline"]
DocType = Literal["email", "claim_form", "invoice", "estimate", "police_report", "photo_note", "purchase_receipt", "other"]


class ClaimDecision(BaseModel):
    """The decision on one claim package."""
    decision: Decision = Field(description="fast_track = pay now; request_info = documents/info missing; "
                                           "refer_adjuster = red flag, possible exclusion or above CHF 5,000; decline = not covered or policy inactive")
    payout_chf: float | None = Field(default=None, description="Amount to pay in CHF (only for fast_track): claimed amount minus the applicable deductible")
    missing_items: list[str] = Field(default_factory=list, description="For request_info: what is missing, e.g. police_report, proof_of_ownership, invoice_or_estimate, policy_id")
    issues: list[str] = Field(default_factory=list, description="Red flags or reasons, e.g. duplicate_invoice, frequency, not_covered, policy_inactive")
    reasoning: str = Field(description="Short justification (2-4 sentences)")


class PolicyDecision(ClaimDecision):
    """The decision on one policy of a claim package that concerns several policies."""
    policy_id: str = Field(description="The policy this decision is for, e.g. P-10061")


class PolicyDecisions(BaseModel):
    """One decision per policy, for a claim package that concerns several policies (Lakeside opens one claim per policy)."""
    decisions: list[PolicyDecision] = Field(description="One entry for every policy the package concerns")


class DocInfo(BaseModel):
    """Classification and key facts of one document."""
    doc_type: DocType
    key_facts: str = Field(description="The facts that matter for the claim: dates, amounts (CHF), numbers, names")


class ClaimFacts(BaseModel):
    """Key facts extracted from a claim package."""
    policy_id: str | None = Field(default=None, description="Policy number like P-10001, if stated")
    customer_name: str | None = None
    claim_type: Literal["glass", "hail", "storm", "theft", "vandalism", "animal_collision", "collision", "water_damage",
                        "burglary", "theft_outside", "glass_breakage", "fire", "other"] = Field(
        description="glass = a vehicle's windscreen or windows (motor); glass_breakage = glass in a home or building "
                    "(windows, hob, shower); theft = vehicle theft; theft_outside = household items stolen away from home")
    incident_date: str | None = Field(default=None, description="YYYY-MM-DD")
    amount_chf: float | None = Field(default=None, description="Total claimed amount in CHF incl. VAT, if known")
    invoice_no: str | None = None
    invoice_date: str | None = Field(default=None, description="YYYY-MM-DD")
    vendor: str | None = None
    documents_present: list[DocType] = Field(default_factory=list)
