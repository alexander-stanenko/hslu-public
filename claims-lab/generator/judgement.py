"""A claim with a sign that needs judgement, which the judging agent should pay (Block 4, Part D1).

PKG-221: the storm of 18 October breaks tiles on a 39-year-old roof in Zurich. The customer worries that the age will be
called wear and tear: a sign that needs judgement (notebook 04 D1 hands it to the agent; Lakeside's screen, which also
reads the report, did not flag it in our checks). The roofer's report settles it: the roof was maintained,
and a fallen branch caused the damage. Nothing points to an exclusion, so the right decision is to pay. Next to PKG-023
(water seeping in for months: refer), it gives the judging agent both kinds of decision.
Appended like the complex packages (complex.py), without random draws, so no other data changes.
"""
from catalogue import building, cust

ROOF_EMAIL = """Hello,

in the storm on 18 October a large branch from our neighbours' lime tree fell onto our roof and broke a number of tiles.
Some rain came into the attic before the roofer covered the hole the next morning.

Our house was built in 1987 and the roof is still the original one, so I am a little worried that you will call this
wear and tear. The roofer's invoice and his short report are attached. Building policy P-10081.

Kind regards
Daniel Frei"""

ROOF_INVOICE = """DACHDECKER IMBODEN GMBH · Witikonerstrasse 15 · 8032 Zürich
INVOICE DI-5530                                             Date: 22.10.2026
Customer: Daniel Frei, Gladbachstrasse 31, 8044 Zürich
Object: residential building, Gladbachstrasse 31 · Storm damage of 18.10.2026

1  Emergency cover with tarpaulin, 19.10.2026                    CHF   380.00
1  Removal and disposal of the fallen branch                      CHF   450.00
1  Replacement of 14 roof tiles and 2 m of battens               CHF 2'270.00
1  Ladder lift                                                    CHF   300.00
   Total, incl. VAT                                               CHF 3'400.00
Payable within 30 days.
"""

ROOF_REPORT = """DACHDECKER IMBODEN GMBH · Short report on the storm damage, Gladbachstrasse 31, 8044 Zürich
Date of visit: 19.10.2026 (emergency cover), 21.10.2026 (repair)

Cause: a branch of about 4 m from the neighbouring lime tree fell onto the south side of the roof during the storm of
18.10.2026 and broke 14 tiles and two battens.
Condition of the roof: clay tiles from 1987, re-laid and inspected by our company in May 2023 (our job no. 23-118);
tiles and battens around the damaged area are sound, no porous or brittle tiles, no earlier leaks.
Assessment: the damage was caused by the falling branch only. We see no wear and tear or lack of maintenance.
"""

F221 = dict(cust("Daniel Frei", "Gladbachstrasse 31", "8044", "Zürich"))

JUDGEMENT = [
    dict(key="J01", package_id="PKG-221", received_at="2026-10-23T09:05:00", customer=F221, subject="Storm damage to our roof",
         title="A storm-damaged old roof, maintained: a sign that turns out harmless",
         policies=[building("P-10081", sum_insured=950_000)],
         history=[],
         documents={"email.txt": ("email", ROOF_EMAIL), "Rechnung_DI-5530.txt": ("invoice", ROOF_INVOICE),
                    "Bericht_Dachdecker.txt": ("other", ROOF_REPORT)},
         parts=[
             dict(policy_id="P-10081", what="roof",
                  claim=dict(type="storm", repair_type="roof_repair", incident="2026-10-18", received="2026-10-23",
                             amount=3400.00, vendor="Dachdecker Imboden GmbH", invoice_no="DI-5530",
                             invoice_date="2026-10-22", location_postcode="8044"),
                  documents=["Rechnung_DI-5530.txt", "Bericht_Dachdecker.txt"],
                  expected=dict(decision="fast_track", payout=2400.0, missing=[], issues=[]),
                  explanation="Storm recorded in Zurich on 18 October; the roof is old but maintained, and the roofer "
                              "attributes the damage to a falling branch, so no exclusion applies: 3,400 minus the "
                              "natural-hazard deductible of 1,000."),
         ]),
]
