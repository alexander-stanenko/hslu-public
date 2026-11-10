"""Shared output schemas. Every claims system built in the course returns a ClaimDecision."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["fast_track", "request_info", "refer_adjuster", "decline"]
DocType = Literal["email", "invoice", "police_report"]


class ClaimDecision(BaseModel):
    """The decision on one claim package."""
    decision: Decision = Field(description="fast_track = pay now; request_info = something is missing; "
                                           "refer_adjuster = an adjuster must decide; decline = not insured")
    payout_chf: float | None = Field(default=None, description="Amount to pay in CHF (only for fast_track): the invoice amount minus the deductible")
    missing_items: list[str] = Field(default_factory=list, description="For request_info: what is missing, e.g. police_report, invoice, policy_id")
    issues: list[str] = Field(default_factory=list, description="Red flags or reasons, e.g. duplicate_invoice, not_covered")
    reasoning: str = Field(description="Short justification (2-4 sentences)")


class DocInfo(BaseModel):
    """Classification and key facts of one document."""
    doc_type: DocType
    key_facts: str = Field(description="The facts that matter for the claim: dates, amounts (CHF), numbers, names")


class ClaimFacts(BaseModel):
    """Key facts extracted from a claim package."""
    policy_id: str | None = Field(default=None, description="Policy number like P-1001, if stated")
    customer_name: str | None = None
    plate: str | None = Field(default=None, description="The car's plate, like LU 248 113, if stated")
    damage: Literal["glass", "hail", "theft", "collision", "other"] = Field(description="The kind of damage")
    damage_date: str | None = Field(default=None, description="The day of the damage, YYYY-MM-DD")
    amount_chf: float | None = Field(default=None, description="The invoice total in CHF, if there is an invoice")
    invoice_no: str | None = None
    garage: str | None = None
    documents_present: list[DocType] = Field(default_factory=list)
