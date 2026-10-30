"""Three claim packages with several policies (split 'complex', planning/19_story_complex_claims.md).

One event damages things insured under two or three policies, so a package gets one decision per policy. The
documents are hand-written and long (a surveyor's, a plumber's and a police report), so a package is about 3,000 to
4,000 tokens. Every part is checked against rules.oracle like the 40 scored cases. build.py writes these packages
after everything else, with no random draws, so the 40 packages and the database rows they rely on stay unchanged.
"""
from catalogue import building, cust, household, motor

# ---------------------------------------------------------------------------------------------- PKG-201
STORM_REPORT = """BAUEXPERTISE LIMMAT GMBH
Independent building surveyors · Hardturmstrasse 161 · 8005 Zürich

EXPERT REPORT ON STORM DAMAGE
Property: single-family house, Rebhaldenstrasse 9, 8044 Zürich
Client: Martina Frei (owner)
Inspection: 19 October 2026, 09:30 to 11:45
Report date: 22 October 2026
Reference: BL-26-0417

1. Assignment
On 19 October 2026 the owner asked us to record the damage caused by the storm of 18 October 2026 to the house and
its cellar, to state the cause of the damage and to give an opinion on the repair costs claimed. The vehicle parked in
the driveway was not part of the assignment; it is mentioned in section 6 only for completeness.

2. The property
The house was built in 1987: two storeys and a cellar, solid masonry, a pitched roof covered with clay tiles
(Ziegelei Rafz, double-interlocking), roof pitch about 35 degrees. According to the owner and the invoices shown to
us, the roof was fully renovated in spring 2019: new battens, a new underlay membrane, new tiles and new ridge tiles.
The cellar lies about 2.40 m below ground level. It is lit by two window wells (light wells) on the west side, each
covered with a steel grating. The cellar floor has a floor drain connected to the house's sewer line.

3. The weather on 18 October 2026
MeteoSwiss recorded a storm over the city of Zurich on the evening of 18 October 2026 with gusts of up to 120 km/h at
the Fluntern station and heavy rain of about 38 mm within three hours. The fire brigade of the city reported more than
400 calls that night, most of them for fallen trees, loose roof tiles and flooded cellars. Neighbours of the owner
confirmed to us that several gardens in Rebhaldenstrasse were affected by fallen branches.

4. Findings on the roof
4.1 On the west slope, 38 tiles were lifted off by the wind and fell into the garden; most of them broke. Two ridge
tiles at the western gable end were loose. The underlay membrane was intact; no water reached the attic, as we
confirmed by inspecting the attic floor and the insulation from inside.
4.2 The remaining tiles are firmly seated and show no frost damage, no porous surfaces and no moss growth. The battens
we could see through the gap are sound. The clips that hold every fourth row are present, as is usual for a roof
renovated in 2019.
4.3 The roofer (Dachdecker Bühler AG) covered the gap with a tarpaulin during the night of 18 October and replaced the
tiles and the ridge tiles on 20 October. We saw the replaced tiles on 19 October in the garden and the tarpaulin in
place. The pattern of the missing tiles (a band along the western eaves and the gable) is typical of wind suction on
the leeward side of a gable during strong gusts.
4.4 Assessment: the damage to the roof was caused by the storm. We found no sign of wear, ageing or missing
maintenance that contributed to it. The roof was in good condition before the storm.

5. Findings in the cellar
5.1 During the storm, rainwater ran from the sloping garden towards the west side of the house. Leaves and small
branches blown into the two window wells blocked their drains, the wells filled up, and water ran through the closed
but not watertight cellar windows into the cellar. The owner noticed the water at about 22:30 and pumped it out with a
borrowed submersible pump; the fire brigade could not come because of the number of calls.
5.2 On 19 October the water marks on the walls stood at 14 to 17 cm above the floor in the storage room and the
laundry room. The floor drain was working; it could not take the amount of water arriving.
5.3 We looked for signs of earlier water entering the cellar: tide marks, efflorescence, mould, swollen skirting
boards. We found none. The walls above the fresh water marks were dry (moisture meter readings below 2 % on all
measurement points at 30 cm height). The paint on the cellar walls is about five years old and undamaged above the
water line.
5.4 In the storage room the following contents stood on the floor and were soaked: a chest freezer (motor and
electronics no longer working after the flooding), a shelf unit made of chipboard with stored winter clothing in
cardboard boxes, two rolled carpets, a box with photo albums and books, and a children's bicycle trailer (the
fabric is stained and smells). The building restoration company Wasserschaden-Service Limmat GmbH recorded the
damaged contents on 20 October and made a replacement estimate, which we have seen.
5.5 Assessment: the water entered the cellar during the storm as surface water from heavy rain, through the window
wells. This is sudden damage caused by the storm. We found no sign that water had entered the cellar before.

6. Note on the vehicle
A large branch of the ash tree on the neighbouring plot broke off at about 21:00 and fell onto the car parked in the
driveway (Subaru Forester, ZH 412 877). We saw dents on the roof and the tailgate and a scratched rear wing. The
vehicle was not part of our assignment, and we give no opinion on the repair costs. The owner told us the car is
insured with Lakeside Insurance as well.

7. Repair costs
7.1 Roof: the invoice of Dachdecker Bühler AG (DB-2214, CHF 3,400.00) covers the night-time emergency cover with a
tarpaulin, 38 new tiles, two ridge tiles re-set in mortar, scaffolding at the western gable and the disposal of the
broken tiles. The quantities agree with our findings and the unit prices are usual for the Zurich area.
7.2 Contents in the cellar: the estimate of Wasserschaden-Service Limmat GmbH (RS-7781, CHF 2,600.00) lists the
freezer, the shelf unit, the clothing, the carpets and the bicycle trailer at replacement value. The books and photo
albums are not included; the owner wishes to keep them. In our opinion the estimate is plausible.
7.3 Drying of the cellar: not needed. The water stood for less than one night, and the masonry was dry above the
water line on 19 October. The owner ventilated the cellar and ran a dehumidifier.

8. Summary
The roof and the cellar were damaged by the storm of 18 October 2026 (wind and heavy rain). We found no earlier
damage, no wear and no missing maintenance that contributed. The costs claimed for the roof (CHF 3,400.00) and for
the contents (CHF 2,600.00) correspond to the damage we found.

Zürich, 22 October 2026
Bauexpertise Limmat GmbH
Reto Zollinger, dipl. Bauingenieur FH, certified building surveyor
"""

STORM_EMAIL = """Hello,

the storm on Saturday evening, 18 October, hit us three times at once, so I am sending everything together.

First, the wind took about 40 tiles off the west side of our roof. Dachdecker Bühler covered the gap that same night
and replaced the tiles on Monday. Their invoice is attached. The house is insured with you under P-10061.

Second, the heavy rain filled the two window wells on the west side and the water ran into the cellar. It stood
about 15 cm high in the storage room. The freezer is broken and the shelves, clothes, carpets and the children's bike
trailer are ruined. Wasserschaden-Service Limmat made an estimate for the contents (attached). Our household contents
are insured with you as well, policy P-10062.

Third, a big branch from the neighbour's ash tree came down on our car in the driveway. The roof and the tailgate
are dented. Garage Zelg made a cost estimate. The car is also with you, policy P-10063 (ZH 412 877).

Because I wanted to be sure about the cause, I asked a building surveyor to look at the house and the cellar; the
report is attached as well. The photos are described in the claim form.

Could you please let me know what you need from me?

Kind regards
Martina Frei
Rebhaldenstrasse 9, 8044 Zürich"""

STORM_FORM = """LAKESIDE INSURANCE AG — CLAIM NOTIFICATION FORM
================================================
Policyholder:         Martina Frei
Address:              Rebhaldenstrasse 9, 8044 Zürich
E-mail:               martina.frei@mail.example
Date of incident:     18.10.2026, about 21:00 to 23:00
Place of incident:    Rebhaldenstrasse 9, 8044 Zürich
Type of damage:       Storm

Damaged items (one line per policy):
  1. Building, policy P-10061: roof tiles blown off on the west side. Repaired. CHF 3'400.00 (invoice)
  2. Household contents, policy P-10062: cellar flooded through the window wells; freezer, shelves, clothing,
     carpets, bicycle trailer. CHF 2'600.00 (estimate)
  3. Car Subaru Forester, ZH 412 877, policy P-10063: branch of the neighbour's tree fell on roof and tailgate.
     CHF 2'900.00 (estimate)

Photos (described): roof with a band of missing tiles along the western eaves; cellar floor with standing water
and soaked boxes; car in the driveway with a large ash branch across the roof.

Bank account (IBAN):  CH** **** **** **** **37 4

I confirm that the information above is complete and correct.
Date / signature:     23.10.2026 / Martina Frei (signed)
"""

STORM_ROOF = """Dachdecker Bühler AG · Wehntalerstrasse 410 · 8046 Zürich
INVOICE DB-2214                                         Date: 21.10.2026
Customer: Martina Frei, Rebhaldenstrasse 9, 8044 Zürich
Object: single-family house, roof west slope, storm damage of 18.10.2026

Pos  Description                                            CHF
1    Emergency call-out night 18.10., cover with tarpaulin   480.00
2    Scaffolding west gable, assembly and removal            620.00
3    38 clay tiles (double-interlocking), supply and fit     1'520.00
4    2 ridge tiles re-set in mortar                          310.00
5    Disposal of broken tiles                                 150.00
6    Travel and small material                                320.00
                                                       ------------
     Total (VAT included)                                   3'400.00

Payable within 30 days. Thank you for your order.
"""

STORM_CONTENTS = """Wasserschaden-Service Limmat GmbH · Badenerstrasse 520 · 8048 Zürich
ESTIMATE RS-7781 — replacement of water-damaged household contents         Date: 22.10.2026
Customer: Martina Frei, Rebhaldenstrasse 9, 8044 Zürich
Recorded on site: 20.10.2026 (cellar flooded during the storm of 18.10.2026)

Item                                                     Replacement value CHF
Chest freezer 250 l (Bauknecht, 2021), not working after flooding         780.00
Shelf unit, chipboard, 2 m, swollen                                       240.00
Winter clothing in 6 cardboard boxes (jackets, shoes)                     880.00
2 carpets (wool, 170 x 240 and 120 x 180), soaked, smell                  420.00
Children's bicycle trailer (fabric stained, frame rusting)                280.00
                                                                   ------------
Total                                                                   2'600.00

Books and photo albums: not included at the customer's request.
"""

STORM_CAR = """Garage Zelg AG · Zelgstrasse 12 · 8044 Zürich
COST ESTIMATE GZ-3390                                    Date: 21.10.2026
Customer: Martina Frei · Vehicle: Subaru Forester (2018), ZH 412 877
Damage: fallen branch (storm 18.10.2026)

Roof panel: remove 3 dents, filler, paint                  1'350.00
Tailgate: dent repair, paint                                 980.00
Rear wing left: polish out scratches, spot repair            370.00
Small material, cleaning                                     200.00
                                                       ------------
Total (VAT included)                                       2'900.00

Valid 30 days. Repair time about 3 working days.
"""

# ---------------------------------------------------------------------------------------------- PKG-202
PIPE_REPORT = """SANITÄR WYSS AG · Haldenstrasse 44 · 6006 Luzern
Plumbing · heating · leak detection

WORK REPORT AND FINDINGS
Order no.: SW-26-1093
Customer: Daniel Gerber, Seeburgstrasse 31, 6006 Luzern
Object: single-family house (built 2004), kitchen and living room, ground floor
Call received: Thursday 23 October 2026, 18:40
On site: 23 October 2026 19:25 to 22:10; 24 October 08:00 to 12:30; 27 October 07:30 to 09:00

1. The call
At 18:40 the customer called our emergency number: water was running out under the kitchen units and spreading over
the floor of the kitchen and the adjoining living room. The customer had closed the main water valve in the cellar
before calling, as we advised on the phone.

2. Situation on arrival
When our technician arrived, about 2 to 3 cm of water stood on the kitchen floor and had run under the threshold
into the living room. The parquet in the living room was wet over an area of about 12 square metres near the kitchen.
A sofa and a rug standing on that part of the floor were soaked at their lower edge. The customer had already moved
smaller furniture away and put towels down. The ceiling of the cellar below the kitchen was dry; the water had not
gone through the floor slab.

3. Locating the leak
With the main valve closed, the water stopped coming. We removed the base panel of the sink unit and the
dishwasher. The water came out of the wall behind the dishwasher, about 40 cm above the floor. We opened the tiled
wall over a width of 60 cm. The cold-water pipe (multilayer composite pipe, 16 mm) runs vertically in the wall to the
sink. At the connection to the elbow behind the dishwasher, the pressed fitting had split lengthwise over about 2 cm.
Water under mains pressure (about 5 bar) sprayed from the crack into the wall cavity and down to the floor.

4. Cause
The split runs along the sleeve of the fitting. It is a sudden failure of the fitting, of a kind we see a few times
a year with fittings of this series from the early 2000s. The pipe and the other fittings we exposed are in good
condition. Around the leak, the wall cavity, the insulation and the back of the tiles were wet only on the surface;
there was no mould, no discoloration and no crumbling mortar, which would show if water had been escaping for a
longer time. The customer told us that the kitchen floor had been dry that morning. In our opinion the pipe failed
on 23 October, within an hour or two before the customer noticed the water.

5. Work carried out
5.1 23 October (evening): water removed with a wet vacuum; the wall cavity opened; the split fitting cut out and
replaced with a new pressed fitting and 30 cm of new pipe; pressure test at 10 bar for 30 minutes, no loss of
pressure; water supply restored. The dishwasher was tested and is working.
5.2 24 October: moisture measurement of the floor and the walls. The screed under the parquet in the living room
showed high moisture readings over about 12 square metres. We installed two condensation dryers and one fan for
technical drying of the screed and the wall cavity. Daily readings by the customer, with our instructions.
5.3 27 October: final moisture measurement after three days: the readings are within the normal range in the wall
cavity. The screed will need about ten more days of drying; the dryers stay on site and will be collected by us.
The wall will be closed and re-tiled after drying (separate order, not part of this report).

6. Parquet
After the drying, the parquet in the living room will be checked. In our experience, oak parquet on a dried screed
usually lies flat again. If it does not, the parquet layer will have to give a separate estimate.

7. Costs
Our invoice SW-6610 of 26 October 2026 covers the leak detection, the pipe repair with the pressure test, and the
technical drying including rental of the equipment for 14 days, the electricity flat rate and the measurements:
pipe repair CHF 1,400.00, drying CHF 2,200.00, total CHF 3,600.00.

8. Furniture
The sofa (three-seater, fabric) and the wool rug stood on the wet part of the floor. The customer said both are new.
We did not assess them; this is outside our trade.

9. Recommendation
We recommend checking the other fittings of the same series in the bathroom upstairs at the next opportunity. They
showed no sign of damage when we looked at the accessible ones on 24 October.

Luzern, 27 October 2026
Sanitär Wyss AG
Marco Wyss, master plumber (eidg. dipl. Sanitärinstallateur)
"""

PIPE_EMAIL = """Dear Sir or Madam,

on Thursday evening, 23 October, a water pipe burst in the wall behind our dishwasher. Before we noticed, the water
had spread over the kitchen and part of the living room. Sanitär Wyss came the same evening, opened the wall and
replaced the broken part, and they are now drying the floor. Their report and invoice are attached.

The water also soaked our sofa and the rug in the living room; both cannot be saved, the sofa smells and the rug has
shrunk at the edge. I attach the furniture invoice for the replacement.

The house is insured with you (P-10064) and our household contents as well (P-10065).

Thank you and kind regards,
Daniel Gerber
Seeburgstrasse 31, 6006 Luzern"""

PIPE_FORM = """Claim report — Lakeside Insurance

[1] POLICYHOLDER
    Name: Daniel Gerber
    Policies: P-10064 (building), P-10065 (household contents)
    Contact: daniel.gerber@mail.example

[2] WHAT HAPPENED?
    When: Thursday, 23 October 2026, about 18:30
    Where: Seeburgstrasse 31, 6006 Luzern, kitchen and living room
    [x] Water damage
    Details: Pipe burst in the kitchen wall behind the dishwasher. Water on the floor of kitchen and living room.

[3] DAMAGE
    Building: pipe repair and drying of the floor, CHF 3'600.00 (invoice Sanitär Wyss AG)
    Contents: sofa and rug, CHF 1'850.00 (invoice Möbel Hubacher AG)

[4] PAYMENT
    IBAN CH** **** **** **** **52 8

Signed electronically, 26.10.2026
"""

PIPE_INVOICE = """Sanitär Wyss AG · Haldenstrasse 44 · 6006 Luzern
INVOICE SW-6610                                         Date: 26.10.2026
Customer: Daniel Gerber, Seeburgstrasse 31, 6006 Luzern
Order SW-26-1093, burst pipe kitchen, 23.10.2026

Emergency call-out evening, leak detection, wall opened          420.00
Pipe repair: pressed fitting and 30 cm pipe replaced, test       980.00
  Subtotal pipe repair                                         1'400.00
Technical drying: 2 dryers, 1 fan, 14 days rental              1'650.00
Electricity flat rate, moisture measurements (3 visits)          550.00
  Subtotal drying                                              2'200.00
                                                          ------------
Total (VAT included)                                           3'600.00
"""

PIPE_FURNITURE = """Möbel Hubacher AG · Pilatusstrasse 18 · 6003 Luzern
INVOICE MH-5521                                         Date: 25.10.2026
Customer: Daniel Gerber, Seeburgstrasse 31, 6006 Luzern

Sofa "Linea", three-seater, fabric anthracite                  1'490.00
Rug, wool, 200 x 300, natural                                    360.00
                                                          ------------
Total (VAT included)                                           1'850.00

Delivery within 3 weeks. Paid by card, thank you.
"""

# ---------------------------------------------------------------------------------------------- PKG-203
BURGLARY_REPORT = """KANTONSPOLIZEI BERN
Regionalpolizei Bern · Waisenhausplatz 32 · 3011 Bern

CONFIRMATION OF REPORT — BURGLARY AND THEFT OF A MOTOR VEHICLE
Report no.: KAPO-BE-2026-118842
Date of report: 25.10.2026, 07:10
Reporting person: Nadia Rossi, Weissensteinstrasse 22, 3007 Bern
Incident: night of 24 to 25 October 2026, between about 01:00 and 06:30
Location: Weissensteinstrasse 22, 3007 Bern (terraced house with carport)
Offences: burglary (Art. 139 and 186 StGB), theft of a motor vehicle

1. Report
At 06:55 the reporting person called the emergency number 117 and stated that the terrace door of the house had been
forced during the night and that the car, which had stood in the carport, was gone. A patrol arrived at 07:05. The
report was recorded on site and completed at the police station on the same day.

2. Findings at the scene
2.1 The terrace door on the garden side (wooden frame, double glazing) was forced at the lock side. Pry marks about
2 cm wide are visible on the frame at the height of the lock and 30 cm below it. The lock bolt is bent. The door
could no longer be closed. The garden gate at the back of the plot was open; the reporting person states that it is
normally closed with a latch.
2.2 In the living room and the study, drawers and cupboards had been opened and their contents thrown on the floor.
In the bedroom on the upper floor, the jewellery box on the chest of drawers was open and empty. The household was
asleep on the upper floor (the reporting person and one child); nobody heard anything.
2.3 In the entrance area, the key board next to the front door was empty. According to the reporting person, it held
the key of the car, a spare house key and the key of the bicycle cellar.
2.4 The carport was empty. According to the reporting person, the car had been parked there at about 22:30 on
24 October and locked. No glass fragments or other traces were found in the carport.
2.5 The forensic service secured traces on the terrace door frame (tool marks) and on the jewellery box. The results
are pending and will not be sent to insurers.

3. Items reported stolen
- Laptop Apple MacBook Air 13" (2024), with charger, from the desk in the study
- Necklace, gold 18 ct with pendant (inherited, valued 2025)
- Wristwatch Tissot PRX (bought 2023)
- Car key (Audi), spare house key, bicycle cellar key
- Motor vehicle: Audi Q3 35 TFSI, grey metallic, first registration 03/2022, plate BE 305 118, about 41,000 km

The reporting person will hand in proof of ownership directly to the insurer.

4. Measures
The vehicle was entered in the national search system RIPOL and the Schengen Information System at 07:40. The plate
was entered as stolen. The reporting person was advised to have the lock of the front door changed because a house
key was taken, and to inform the insurer. Neighbours in Weissensteinstrasse 18 to 30 were asked whether they had seen
or heard anything; one neighbour heard a car start at about 04:30 but did not look out.

5. Similar cases
In the past four weeks, three similar burglaries with theft of a car key from a key board near the front door were
reported in the districts Holligen and Weissenstein. In two cases the car was found abroad later. The investigation
is joined with those cases.

6. Note for insurers
This confirmation is issued for insurance purposes at the request of the reporting person. If the vehicle is found,
the reporting person and the insurer named by the reporting person will be informed.

Kantonspolizei Bern, Regionalpolizei Bern
Sgt. M. Bähler, on duty (signed)
Bern, 25 October 2026
"""

BURGLARY_EMAIL = """Hello,

last night someone broke into our house through the terrace door while we were asleep upstairs. They took my laptop,
my grandmother's necklace and my watch, and they took the car key from the key board in the hall and drove off with
our car. The police came this morning; their confirmation is attached.

I attach the receipts and the valuation for the stolen things, and the purchase contract of the car. According to the
Eurotax valuation the garage did for me today, the car is worth about CHF 24,000 now.

Household insurance P-10066, car insurance P-10067 (Audi Q3, BE 305 118).

What happens next? Do I get a replacement car in the meantime?

Nadia Rossi
Weissensteinstrasse 22, 3007 Bern"""

BURGLARY_FORM = """LAKESIDE INSURANCE AG — CLAIM NOTIFICATION FORM
================================================
Policyholder:         Nadia Rossi
Address:              Weissensteinstrasse 22, 3007 Bern
E-mail:               nadia.rossi@mail.example
Date of incident:     25.10.2026 (night)
Place of incident:    Weissensteinstrasse 22, 3007 Bern
Type of damage:       Burglary / theft of vehicle

Damaged items (one line per policy):
  1. Household contents, policy P-10066: laptop, necklace, watch stolen. CHF 3'800.00 (receipts, valuation)
  2. Car Audi Q3 (2022), BE 305 118, policy P-10067: stolen from the carport with the key taken in the burglary.
     CHF 24'000.00 (Eurotax valuation)
Police report: Kantonspolizei Bern, KAPO-BE-2026-118842

Bank account (IBAN):  CH** **** **** **** **19 2

I confirm that the information above is complete and correct.
Date / signature:     27.10.2026 / Nadia Rossi (signed)
"""

BURGLARY_RECEIPTS = """PROOF OF OWNERSHIP — stolen items (copies)

1) Apple Store Bern, Marktgasse · Receipt 4471-2024-0388 · 14.03.2024
   MacBook Air 13" M3, 16 GB, 512 GB ............................ CHF 1'900.00

2) Goldschmied Keller, Spitalgasse 4, Bern · Valuation for insurance purposes · 02.06.2025
   Necklace, yellow gold 750, 45 cm, pendant with small sapphire; current replacement value CHF 1'250.00

3) Bucherer Bern · Receipt BE-23-55120 · 21.12.2023
   Tissot PRX 40 mm, steel ...................................... CHF 650.00

Total claimed for household contents: CHF 3'800.00
"""

BURGLARY_CAR = """PURCHASE CONTRACT — USED VEHICLE (copy)
Seller: Auto Zentrum Wankdorf AG, Papiermühlestrasse 71, 3014 Bern
Buyer: Nadia Rossi, Weissensteinstrasse 22, 3007 Bern
Vehicle: Audi Q3 35 TFSI, grey metallic, first registration 03/2022, VIN WAUZZZF3*N1******, 9,800 km
Price: CHF 31'500.00, paid in full on handover
Handover: 12.05.2023, with two keys and the service booklet
Signed: Bern, 12.05.2023, seller and buyer
"""

# ---------------------------------------------------------------------------------------------- the cases
F201 = cust("Martina Frei", "Rebhaldenstrasse 9", "8044", "Zürich")
F202 = cust("Daniel Gerber", "Seeburgstrasse 31", "6006", "Luzern")
F203 = cust("Nadia Rossi", "Weissensteinstrasse 22", "3007", "Bern")

COMPLEX = [
    dict(key="X01", package_id="PKG-201", received_at="2026-10-23T19:12:00", customer=F201,
         title="One storm, three policies",
         policies=[building("P-10061"), household("P-10062", sum_insured=80000),
                   motor("P-10063", "ZH 412 877", "Subaru Forester (2018)", partial=False)],
         history=[],
         documents={"email.txt": ("email", STORM_EMAIL), "Schadenmeldung_Sturm.txt": ("claim_form", STORM_FORM),
                    "Rechnung_DB-2214.txt": ("invoice", STORM_ROOF), "Offerte_RS-7781.txt": ("estimate", STORM_CONTENTS),
                    "Kostenvoranschlag_GZ-3390.txt": ("estimate", STORM_CAR),
                    "Gutachten_Bauexpertise.txt": ("expert_report", STORM_REPORT)},
         parts=[
             dict(policy_id="P-10061", what="roof",
                  claim=dict(type="storm", repair_type="roof_repair", incident="2026-10-18", received="2026-10-23", amount=3400.00,
                             vendor="Dachdecker Bühler AG", invoice_no="DB-2214", invoice_date="2026-10-21", location_postcode="8044"),
                  documents=["Schadenmeldung_Sturm.txt", "Rechnung_DB-2214.txt", "Gutachten_Bauexpertise.txt"],
                  expected=dict(decision="fast_track", payout=2400.0, missing=[], issues=[]),
                  explanation="Storm covered by building basic; storm event region 80 on 18 Oct; natural-hazard deductible 1,000 -> 3,400 - 1,000 = 2,400."),
             dict(policy_id="P-10062", what="cellar contents",
                  claim=dict(type="storm", repair_type=None, incident="2026-10-18", received="2026-10-23", amount=2600.00,
                             vendor="Wasserschaden-Service Limmat GmbH", invoice_no="RS-7781", invoice_date="2026-10-22",
                             location_postcode="8044"),
                  documents=["Schadenmeldung_Sturm.txt", "Offerte_RS-7781.txt", "Gutachten_Bauexpertise.txt"],
                  expected=dict(decision="fast_track", payout=2400.0, missing=[], issues=[]),
                  explanation="Storm and heavy rain covered by household basic (natural hazards); no earlier damp; deductible 200 -> 2,600 - 200 = 2,400."),
             dict(policy_id="P-10063", what="car",
                  claim=dict(type="storm", repair_type="body_repair", incident="2026-10-18", received="2026-10-23", amount=2900.00,
                             vendor="Garage Zelg AG", invoice_no="GZ-3390", invoice_date="2026-10-21", location_postcode="8044"),
                  documents=["Schadenmeldung_Sturm.txt", "Kostenvoranschlag_GZ-3390.txt"],
                  expected=dict(decision="decline", payout=None, missing=[], issues=["not_covered"]),
                  explanation="Liability-only motor policy: own-vehicle storm damage needs partial cover (as PKG-019)."),
         ]),
    dict(key="X02", package_id="PKG-202", received_at="2026-10-27T08:41:00", customer=F202,
         title="A burst pipe, building and contents",
         policies=[building("P-10064"), household("P-10065")],
         history=[dict(policy_id="P-10065", incident_date="2025-11-20", claim_type="water_damage", amount_chf=1850.0, status="paid",
                       invoice_no="MH-5521", vendor="Möbel Hubacher AG", note="")],
         documents={"email.txt": ("email", PIPE_EMAIL), "claim_form.txt": ("claim_form", PIPE_FORM),
                    "Bericht_Sanitaer_Wyss.txt": ("repair_report", PIPE_REPORT), "Rechnung_SW-6610.txt": ("invoice", PIPE_INVOICE),
                    "Rechnung_MH-5521.txt": ("invoice", PIPE_FURNITURE)},
         parts=[
             dict(policy_id="P-10064", what="pipe and drying",
                  claim=dict(type="water_damage", repair_type="water_damage_drying", incident="2026-10-23", received="2026-10-27",
                             amount=3600.00, vendor="Sanitär Wyss AG", invoice_no="SW-6610", invoice_date="2026-10-26"),
                  documents=["claim_form.txt", "Bericht_Sanitaer_Wyss.txt", "Rechnung_SW-6610.txt"],
                  expected=dict(decision="fast_track", payout=3100.0, missing=[], issues=[]),
                  explanation="Sudden escape of water covered by building basic; no sign of gradual leakage; deductible 500 -> 3,600 - 500 = 3,100."),
             dict(policy_id="P-10065", what="sofa and rug",
                  claim=dict(type="water_damage", repair_type=None, incident="2026-10-23", received="2026-10-27", amount=1850.00,
                             vendor="Möbel Hubacher AG", invoice_no="MH-5521", invoice_date="2026-10-25"),
                  documents=["claim_form.txt", "Rechnung_MH-5521.txt"],
                  expected=dict(decision="refer_adjuster", payout=None, missing=[], issues=["duplicate_invoice"]),
                  explanation="Invoice MH-5521 was already paid on the household claim of 20 Nov 2025 (as PKG-015)."),
         ]),
    dict(key="X03", package_id="PKG-203", received_at="2026-10-27T21:05:00", customer=F203,
         title="A break-in and a stolen car",
         policies=[household("P-10066"), motor("P-10067", "BE 305 118", "Audi Q3 (2022)")],
         history=[],
         documents={"email.txt": ("email", BURGLARY_EMAIL), "Schadenmeldung.txt": ("claim_form", BURGLARY_FORM),
                    "Polizeirapport.txt": ("police_report", BURGLARY_REPORT),
                    "Belege_Diebesgut.txt": ("purchase_receipt", BURGLARY_RECEIPTS),
                    "Kaufvertrag_Audi_Q3.txt": ("purchase_contract", BURGLARY_CAR)},
         parts=[
             dict(policy_id="P-10066", what="stolen items",
                  claim=dict(type="burglary", repair_type=None, incident="2026-10-25", received="2026-10-27", amount=3800.00,
                             vendor=None, invoice_no=None),
                  documents=["Schadenmeldung.txt", "Polizeirapport.txt", "Belege_Diebesgut.txt"],
                  expected=dict(decision="fast_track", payout=3600.0, missing=[], issues=[]),
                  explanation="Burglary with forced entry, police report and proof of ownership present; deductible 200 -> 3,800 - 200 = 3,600."),
             dict(policy_id="P-10067", what="car",
                  claim=dict(type="theft", repair_type=None, incident="2026-10-25", received="2026-10-27", amount=24000.00,
                             vendor=None, invoice_no=None),
                  documents=["Schadenmeldung.txt", "Polizeirapport.txt", "Kaufvertrag_Audi_Q3.txt"],
                  expected=dict(decision="refer_adjuster", payout=None, missing=[], issues=["above_fast_track_limit"]),
                  explanation="Vehicle theft covered by partial cover with a police report, but CHF 24,000 is above the fast-track limit of CHF 5,000."),
         ]),
]


def part_case(x: dict, part: dict) -> dict:
    """One part as a case for rules.oracle: its policy, its claim and the document types it can use."""
    policy = next(p for p in x["policies"] if p["policy_id"] == part["policy_id"])
    types = [x["documents"][name][0] for name in part["documents"]]
    return dict(key=f"{x['key']}/{part['policy_id']}", customer=x["customer"], policy=policy, claim=part["claim"], docs=types,
                flags=part.get("flags", []))
