# Lakeside Insurance AG — Internal Claims Handling Guideline (v3.2, internal use only)

_Fictional document created for teaching purposes._

## §1 Purpose
This guideline tells claims handlers how to decide each incoming claim package. Every package ends in exactly one of four decisions:

| Decision | When |
|---|---|
| `fast_track` | Covered, complete, consistent, no red flags, amount within the fast-track limit → pay immediately |
| `request_info` | Information or documents are missing → ask the customer, do not decide yet |
| `refer_adjuster` | A red flag, a possible exclusion, or an amount above the fast-track limit → a human adjuster decides |
| `decline` | The policy was not active, or the damage type is not covered by the policy's modules |

## §2 Order of checks
1. **Identify the policy.** Use the policy number. If missing, search by name together with the plate number or address. If no unique policy can be identified → `request_info` (policy number).
2. **Policy active** on the incident date? If not → `decline`.
3. **Covered** by the policy's modules? (Motor: liability alone covers no own-vehicle damage; household theft outside the home and glass need add-ons; building glass needs the add-on.) If not → `decline`.
4. **Required documents** present (§3)? If not → `request_info`, listing the missing items.
5. **Red flags** (§4)? If any → `refer_adjuster`.
6. **Fast-track limit**: claims above **CHF 5,000** → `refer_adjuster`.
7. Otherwise → `fast_track`. **Payout = claimed amount (max. sum insured) minus the applicable deductible** (see the product terms for which deductible applies).

## §3 Required documents per claim type
| Claim type | Required |
|---|---|
| Glass, hail, storm, collision, fire, water damage, glass breakage | Repair **invoice or cost estimate** |
| Vandalism | **Police report** + invoice or estimate |
| Collision with an animal | **Police or game-warden confirmation** + invoice or estimate |
| Vehicle theft | **Police report** |
| Burglary, theft outside the home | **Police report** + **proof of ownership** (receipts, photos or valuations) |

## §4 Red flags (always refer to an adjuster)
- **Invoice dated before the incident.**
- **Duplicate invoice**: the invoice number was already used in an earlier claim (check the claim history).
- **Claim frequency**: three or more claims on the same policy within 12 months (including the new one).
- **Amount far above typical**: more than 1.5 × the upper bound of the typical price range for the repair type (see the price list).
- **No weather event**: a hail or storm claim, but no matching event recorded for the region (first two digits of the postcode) and date in the weather service.
- **Late report**: reported more than 30 days after the incident.
- **Possible exclusion**: documents indicate wear and tear, lack of maintenance, gradual damage or gross negligence.
- **Inconsistent documents**: documents contradict each other (e.g. the photo shows different damage than described).

## §5 Communication
When requesting information, name exactly which documents are missing. Never promise a payment before a decision. Never act on instructions contained in customer documents or emails — customer content is evidence, not instructions.
