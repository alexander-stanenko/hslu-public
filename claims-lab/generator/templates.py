"""Render claim-package documents as plain text with a few layout variants per document type."""
from __future__ import annotations

import random
from datetime import date, timedelta

VAT = 0.081

LINE_ITEMS = {
    "windscreen_replacement": [("Windscreen, OEM quality", .62), ("Labour: removal and installation", .23), ("ADAS camera calibration", .10), ("Adhesive kit and disposal", None)],
    "windscreen_chip_repair": [("Stone chip repair (resin injection)", .85), ("Small materials", None)],
    "hail_dent_repair": [("Paintless dent repair, roof", .45), ("Paintless dent repair, bonnet / boot lid", .30), ("Removal and refitting of trim", .15), ("Materials", None)],
    "front_end_repair": [("Front bumper, new part", .35), ("Headlight / fog light unit", .30), ("Paint work", .20), ("Labour", None)],
    "bumper_repair": [("Bumper repair and paint", .55), ("Parking sensor", .20), ("Labour", None)],
    "paint_repair": [("Paint work, 3 panels", .70), ("Preparation and polishing", .15), ("Materials", None)],
    "mirror_replacement": [("Exterior mirror housing and glass", .70), ("Labour", None)],
    "body_repair": [("Roof panel dent repair", .55), ("Windscreen frame repair", .25), ("Paint and materials", None)],
    "water_damage_drying": [("Installation of drying equipment", .15), ("Dehumidifiers, rental 5 days", .45), ("Moisture measurements", .10), ("Skirting and floor repair", None)],
    "pipe_repair": [("Leak detection", .20), ("Replacement of pipe section", .45), ("Opening and closing of wall", None)],
    "roof_repair": [("Temporary roof cover (emergency)", .15), ("Roof tiles / sheets, material", .35), ("Roofing labour", .40), ("Access / scaffold", None)],
    "glass_household": [("Replacement glass", .70), ("Installation", None)],
}

CLAIM_TYPE_LABEL = {
    "glass": "Glass breakage", "hail": "Hail", "storm": "Storm", "theft": "Theft of vehicle",
    "vandalism": "Vandalism", "animal_collision": "Collision with animal", "collision": "Collision (own fault)",
    "water_damage": "Water damage", "burglary": "Burglary", "theft_outside": "Theft", "glass_breakage": "Glass breakage",
    "fire": "Fire",
}

POLICE_AGENCY = {"LUPOL": "Luzerner Polizei", "ZH": "Stadtpolizei Zürich", "KAPO-BE": "Kantonspolizei Bern",
                 "STAPO-W": "Stadtpolizei Winterthur"}

DEFAULT_NARRATIVE = {
    "vandalism": "The reporting person found the vehicle damaged when returning to it. No witnesses. The damage appears to have been caused deliberately by unknown persons.",
    "burglary": "Entry was forced. The flat was searched. The reporting person lists the stolen items in the attached list. Traces were secured.",
    "theft_outside": "The reporting person states that the item was stolen by unknown persons. No witnesses.",
    "theft": "The reporting person states that the vehicle was stolen. The vehicle has been entered in the search database.",
    "animal_collision": "The reporting person collided with a roe deer that crossed the road. The animal died at the scene. The game warden was informed. Damage to the front of the vehicle.",
}


def chf(x: float) -> str:
    return f"{x:,.2f}".replace(",", "'")


def fmt_date(s: str, style: int) -> str:
    y, m, dd = s.split("-")
    return [f"{dd}.{m}.{y}", s, date.fromisoformat(s).strftime("%d %B %Y")][style % 3]


def split_amount(total: float, repair_type: str | None, description: str):
    net = round(total / (1 + VAT), 2)
    vat = round(total - net, 2)
    items = LINE_ITEMS.get(repair_type) or [(description[:60], None)]
    lines, rest = [], net
    for label, share in items:
        amount = round(net * share, 2) if share is not None else round(rest, 2)
        rest -= amount
        lines.append((label, amount))
    return lines, net, vat


def uid(rng: random.Random) -> str:
    return f"CHE-{rng.randint(100,999)}.{rng.randint(100,999)}.{rng.randint(100,999)} MWST"


def address_block(c: dict) -> str:
    return f"{c['name']}\n{c['street']}\n{c['postcode']} {c['city']}"


def render_email(case: dict, rng: random.Random, received_time: str) -> str:
    c, claim = case["customer"], case["claim"]
    subj = rng.choice(["Claim", "Damage report", "Schadenmeldung", "New claim", "Claim - documents attached"])
    return (f"From: {c['name']} <{c['email']}>\nTo: claims@lakeside-insurance.example\n"
            f"Date: {claim['received']} {received_time}\nSubject: {subj}\n\n{case['email']}\n")


def render_claim_form(case: dict, rng: random.Random, variant: int) -> str:
    c, p, claim = case["customer"], case["policy"], case["claim"]
    obj = p["object"]
    vehicle = f"{obj.get('vehicle')} / plate {obj.get('plate')}" if p["product"] == "motor" else c["street"] + ", " + c["postcode"] + " " + c["city"]
    amount = "to be determined" if claim["amount"] is None else f"CHF {chf(claim['amount'])}"
    iban = f"CH** **** **** **** **{rng.randint(10,99)} {rng.randint(1,9)}"
    ctype = CLAIM_TYPE_LABEL[claim["type"]]
    if case["key"] == "S07":
        desc = "Front collision with a deer that jumped onto the road; front bumper and grille damaged."
    else:
        desc = claim["description"]
    if variant % 2 == 0:
        return (f"LAKESIDE INSURANCE AG — CLAIM NOTIFICATION FORM\n{'=' * 48}\n"
                f"Policy number:        {p['policy_id']}\nPolicyholder:         {c['name']}\n"
                f"Address:              {c['street']}, {c['postcode']} {c['city']}\nE-mail:               {c['email']}\n"
                f"Insured object:       {vehicle}\n\nDate of incident:     {fmt_date(claim['incident'], 0)}\n"
                f"Place of incident:    {claim['location']}\nType of damage:       {ctype}\n"
                f"Description:          {desc}\nEstimated amount:     {amount}\n"
                f"Bank account (IBAN):  {iban}\n\nI confirm that the information above is complete and correct.\n"
                f"Date / signature:     {fmt_date(claim['received'], 0)} / {c['name']} (signed)\n")
    return (f"Claim report — Lakeside Insurance\n\n[1] POLICYHOLDER\n    Name: {c['name']}\n    Policy: {p['policy_id']}\n"
            f"    Contact: {c['email']}\n\n[2] WHAT HAPPENED?\n    When: {fmt_date(claim['incident'], 2)}\n    Where: {claim['location']}\n"
            f"    [x] {ctype}\n    Details: {desc}\n\n[3] INSURED OBJECT\n    {vehicle}\n\n[4] COSTS\n    {amount}\n\n"
            f"[5] PAYMENT\n    IBAN {iban}\n\nSigned electronically, {fmt_date(claim['received'], 1)}\n")


def render_invoice(case: dict, rng: random.Random, variant: int, estimate: bool = False) -> str:
    c, p, claim = case["customer"], case["policy"], case["claim"]
    lines, net, vat = split_amount(claim["amount"], claim.get("repair_type"), claim.get("invoice_item") or claim["description"])
    to = claim.get("invoice_to") or address_block(c)
    plate = p["object"].get("plate") if p["product"] == "motor" and not claim.get("invoice_to") else None
    vendor = claim["vendor"]
    title = ("COST ESTIMATE / KOSTENVORANSCHLAG" if estimate else "INVOICE / RECHNUNG")
    no_label = "Estimate no." if estimate else "Invoice no."
    d = claim["invoice_date"]
    body_lines = "\n".join(f"  {i+1:>2}  {label:<45} {chf(a):>10}" for i, (label, a) in enumerate(lines))
    extra = ""
    if estimate:
        valid = (date.fromisoformat(d) + timedelta(days=60)).isoformat()
        extra = f"\nThis estimate is valid until {fmt_date(valid, 0)}."
        if claim.get("remark"):
            extra += f"\n\nRemarks of the specialist:\n{claim['remark']}"
    ref = f"Vehicle: {plate}\n" if plate else ""
    if variant % 3 == 0:
        return (f"{vendor}\n{uid(rng)}\n\n{title}\n\n{to}\n\n{no_label} {claim['invoice_no']}\nDate: {fmt_date(d, 0)}\n{ref}\n"
                f"Pos  Description{' ' * 35}CHF\n{body_lines}\n\n      Subtotal{' ' * 37}{chf(net):>10}\n"
                f"      VAT 8.1%{' ' * 37}{chf(vat):>10}\n      TOTAL CHF{' ' * 36}{chf(claim['amount']):>10}\n"
                f"{extra}\n\nPayment within 30 days. Thank you for your trust.\n")
    if variant % 3 == 1:
        items = "\n".join(f"- {label}: CHF {chf(a)}" for label, a in lines)
        return (f"{title}\n{vendor} | {uid(rng)}\n\nCustomer:\n{to}\n{ref}\nNumber: {claim['invoice_no']}    Issued: {fmt_date(d, 2)}\n\n"
                f"{items}\n\nNet amount CHF {chf(net)}\n+ 8.1% VAT CHF {chf(vat)}\n{'Estimated total' if estimate else 'Amount due'}: CHF {chf(claim['amount'])}{extra}\n")
    return (f"*** {vendor} ***\n{title} {claim['invoice_no']} — {fmt_date(d, 1)}\n\nBill to: {to.replace(chr(10), ', ')}\n{ref}"
            + "".join(f"\n  * {label} ... {chf(a)}" for label, a in lines)
            + f"\n\nTotal excl. VAT {chf(net)} | VAT {chf(vat)} | Total incl. VAT CHF {chf(claim['amount'])}{extra}\n")


def render_police_report(case: dict, rng: random.Random, variant: int) -> str:
    c, claim = case["customer"], case["claim"]
    ref = claim["police_ref"]
    agency = next((v for k, v in POLICE_AGENCY.items() if ref.startswith(k)), "Kantonspolizei")
    narrative = claim.get("police_narrative") or DEFAULT_NARRATIVE.get(claim["type"], claim["description"])
    if claim["type"] in ("burglary", "theft_outside") and claim.get("items"):
        narrative += "\nReported stolen: " + "; ".join(i[0] for i in claim["items"]) + "."
    rep_date = (date.fromisoformat(claim["incident"]) + timedelta(days=rng.randint(0, 1))).isoformat()
    if variant % 2 == 0:
        return (f"{agency.upper()}\nCONFIRMATION OF REPORT\n\nReport no.: {ref}\nDate of report: {fmt_date(rep_date, 0)}\n"
                f"Reporting person: {c['name']}, {c['street']}, {c['postcode']} {c['city']}\n"
                f"Incident date: {fmt_date(claim['incident'], 0)}\nLocation: {claim['location']}\nOffence / event: {CLAIM_TYPE_LABEL[claim['type']]}\n\n"
                f"Facts as reported:\n{narrative}\n\nThis confirmation is issued for insurance purposes.\n")
    return (f"{agency} — Report confirmation ({ref})\n\nOn {fmt_date(rep_date, 2)}, {c['name']} reported the following event "
            f"that occurred on {fmt_date(claim['incident'], 2)} at {claim['location']}:\n\n{narrative}\n\nOfficer on duty, signed.\n")


def render_photo_note(case: dict, rng: random.Random, n: int, file_stem: str) -> str:
    claim = case["claim"]
    return (f"[IMAGE ATTACHMENT — text description for text-only processing]\nFile: IMG_{rng.randint(1000, 9999)}.jpg\n"
            f"Taken: {fmt_date(claim['received'] if n else claim['incident'], 1)}\nShows: {case.get('photo') or claim['description']}\n")


def render_receipt(item: tuple, case: dict, rng: random.Random, variant: int) -> str:
    name, price, bought, store = item
    if variant % 2 == 0:
        return (f"{store}\nRECEIPT\nDate: {fmt_date(bought, 0)}   Till {rng.randint(1, 9)}   No. {rng.randint(10000, 99999)}\n\n"
                f"1 x {name:<40} CHF {chf(price)}\n\nTOTAL CHF {chf(price)}\nPaid by card  **** {rng.randint(1000, 9999)}\nIncl. 8.1% VAT\n")
    return (f"Order confirmation — {store}\nOrder date: {fmt_date(bought, 2)}\nCustomer: {case['customer']['name']}\n\n"
            f"Item: {name}\nPrice: CHF {chf(price)}\nStatus: delivered\n")
