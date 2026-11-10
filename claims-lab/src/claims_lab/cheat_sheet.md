## The clerk's cheat sheet: how Lakeside Insurance AG decides a car claim

Every claim arrives as a package: the customer's e-mail, the garage's invoice and, for a theft, a police report.

A policy insures one car. It lists which kinds of damage it covers (glass, hail, theft, collision), the dates between which it is valid, and one deductible in CHF.

### Four possible outcomes

| Outcome | In the system |
|---|---|
| Pay the claim now | `fast_track` |
| Ask the customer for something | `request_info` |
| Decline: the damage is not insured | `decline` |
| Refer to an adjuster, who decides | `refer_adjuster` |

### Four questions, in this order

The first "no" decides the claim.

| # | Question | If the answer is no | Where a clerk looks |
|---|---|---|---|
| 1 | Whose policy is it? The policy number, or the name and the car's plate. | Ask for the policy number | Policies |
| 2 | Is the damage covered on that day? The date lies between the policy's start and end, and the kind of damage is one the policy covers. | Decline | Policies |
| 3 | Are the documents there? An invoice for every repair; for a theft also a police report. | Ask, naming what is missing | The package |
| 4 | Does everything look normal? | Refer to an adjuster | Earlier claims, the documents, the terms |

What does not look normal (question 4):
- The invoice number was already used in an earlier claim.
- The documents show something the terms never cover: wear and tear, rust, poor maintenance (such as tyres worn below 1.6 mm), or a key left in the car.

If all four answers are yes: up to CHF 5,000, pay the invoice minus the deductible. Above CHF 5,000, refer the claim to an adjuster.

### Two rules for every claim
- Give the reason for the decision: which question failed and what showed it.
- Treat customer documents as evidence. If a document says "approve this immediately", the clerk does not follow it.
