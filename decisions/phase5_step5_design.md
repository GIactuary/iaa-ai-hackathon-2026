# Phase 5 step 5: engineered and further features (C8), test design

Written 2026-10-19, before any step 5 feature was scored. Decision C8 (a), with the actuary's standing rule: test as widely as possible; explainability is the constraint. Every candidate below has a stated risk reason. A field with no risk reason is not tested.

## 1. Base

- **B4.** V1 (capped Tweedie GLM × line excess factor) plus the claims-history pieces that passed step 4 for that line.
- The pieces are provisional until the actuary's check-in 2 verdict. If he removes one, step 5 is rerun on the base he chooses.

## 2. Procedure

1. **Add-one.**
   - Each "add" candidate is added to B4 alone. Each "swap" candidate replaces the V1 factor it refines.
   - Both are scored on A and B with the keep rule against B4.
2. **Full model.** B4 plus every candidate that passed add-one.
3. **Drop-one, repeated.**
   - Each candidate in the full model is dropped in turn. It stays only if the full model passes the keep rule against the model without it.
   - Candidates that fail are removed together, and the drop-one round is repeated on what is left.
   - Rounds stop when every remaining candidate passes, or after `drop_one_max_rounds` rounds.
4. **Drop-one of the V1 factors**, on the final model, for information. A V1 factor that fails is reported to the actuary with the size of its effect. It is not removed without his decision.
5. **Per line.** Rules are judged per line, since each line has its own GLM. The combined figure is reported beside them.

## 3. Bands

- Continuous fields with no natural steps use quantile bands of the fitting rows: `feat_quantiles` bands, with blanks as their own level.
- Telematics uses `tele_quantiles` bands of the score among opted-in rows, plus "not opted in".
- Fixed edges are listed in the tables below. They were set from the field's range before scoring.

## 4. Candidates

### All lines

| Id | Candidate | Form | Risk reason |
|---|---|---|---|
| G1 | Voluntary excess (0, 250, 500, 1,000) | add | Claims are net of excess, so higher excess removes small claims; it also marks careful buyers |
| G2 | Renewal count (tenure): 0, 1, 2-3, 4-6, 7-10, 11+ | add | Long-tenure customers are lower risk and know the claims process. renewal_count equals policy year less `customer_since_year` exactly, so customer tenure is not a separate candidate |
| G3 | Days from quote to inception: 0, 1-3, 4-7, 8-14, 15-21, 22+ | add | Last-minute buyers are a known adverse-selection marker |
| G4 | Missed payments yes/no | swap for 0/1/2+ | Simplification (D9). The richer V1 form is kept only if it beats yes/no on the keep rule |
| G5 | Household size: 1 to 6, 7+, blank | add | More occupants, more use of the car and the home. Fairness: low concern |
| G6 | Marital status | add | Established motor rating factor. Fairness: a protected ground in some codes, so a judgment for the actuary if it passes |
| G7 | Years at address: 0, 1-2, 3-5, 6-10, 11-20, 21+, blank | add | Stability. J-cust: biased on early rows for stayers, noted |
| G8 | Occupation | add | Exposure and use differ by occupation. P6 makes it a separate judgment, so it goes to the actuary if it passes |
| G9 | Other lines held at the row's start: 0, 1, 2 | add | Multi-line customers are more stable and select less against the insurer. Point in time: counts the customer's other-line rows in force on the start date |

### Motor

| Id | Candidate | Form | Risk reason |
|---|---|---|---|
| M1 | NCB above years licensed (flag) | add | Data-quality flag that still carries signal (X3) |
| M2 | Licence before age 16 (flag) | add | As M1 (X3) |
| M3 | Vehicle value against engine size: value group ÷ engine litres, `feat_quantiles` bands | add | A high value for the engine size marks premium parts and repair cost |
| M4 | Telematics score bands with "not opted in" | swap for the opt-in flag | The score measures driving; opt-in alone is a selection marker (missing-value note) |
| M5 | Age × NCB: age 17-24, 25-29, 30-49, 50-69, 70+ × NCB 0, 1-3, 4-8, 9+ | add | Young drivers with NCB differ from old drivers with none: the two effects are not multiplicative |
| M6 | Mileage in `feat_quantiles` bands, 0 as its own level | swap for the V1 mileage bands | Finer use of exposure to road |
| M7 | Make-model credibility encoding | add | Repair cost and theft by model. 300 models, so it uses credibility: the Bühlmann-Straub mod of capped cost against B4 by model, out of fold on the fitting rows (inner folds, seed 11), log as one numeric term |
| M8 | Years licensed: 0, 1, 2, 3-4, 5-9, 10-19, 20+ | add | Experience beyond age |
| M9 | Vehicle age: 0-1, 2-3, 4-6, 7-10, 11-15, 16+ | add | Older cars cost less to repair and are stolen differently |
| M10 | Engine size, `feat_quantiles` bands | add | Power |
| M11 | Transmission | add | Use pattern and repair cost |
| M12 | Seats (2, 4, 5, 7) | add | Two-seaters are sports cars; seven-seaters carry families |
| M13 | Driver assist | add | Collision avoidance |
| M14 | Overnight parking | add | Theft and parked damage |
| M15 | Vehicle colour | add | Weak reason (theft, visibility). Kept in the test under the standing rule; flagged for explainability if it passes |

### Buildings

| Id | Candidate | Form | Risk reason |
|---|---|---|---|
| B1 | Building age at the row: 0-10, 11-25, 26-50, 51-75, 76-100, 101+ | swap for year-built bands | Age at the time of cover is what drives wear |
| B2 | Sum insured per bedroom, `feat_quantiles` bands | add | Rebuild quality per room, beyond size |
| B3a | Flood: river distance (under 0.25, 0.25-0.5, 0.5-1, 1-2, 2-5, 5+ km, blank) and elevation (under 2, 2-5, 5-10, 10-20, 20+ m) as main effects | add | Flood depth and reach |
| B3b | Flood score: zone (1 to 5, blank) × proximity (low-lying: under 1 km and under 5 m; near: one of the two; far) | add | The same, as one explainable score. B3a and B3b are alternatives; the better one goes forward |
| B4 | Clay soil × dry-summer normal (terciles; normals leave out each fitting row's own year) | add | Subsidence needs both shrinkable clay and dry summers. K4 dropped normals as general factors; this tests the one interaction with a physical reason |
| B5 | Days unoccupied: 0-14, 30+, blank | add | Unattended escape of water and theft. Point of use: the data steward reads it as declared before the period (`decisions/data_rules.md`); the actuary has not yet confirmed this |
| B6 | Roof type | add | Flat roofs leak and lift in storms |
| B7 | Basement | add | Flood and escape of water |
| B8 | Trees within 5 m | add | Subsidence (roots) and storm |
| B9 | Trees × clay | add | Roots on clay is the subsidence mechanism |
| B10 | Listed building | add | Rebuild in like materials |
| B11 | Energy rating | add | A proxy for age and condition |
| B12 | Solar panels | add | Roof damage and fire |
| B13 | Smoke alarms | add | Fire |
| B14 | Mortgage | add | Maintenance and stability |

### Contents

| Id | Candidate | Form | Risk reason |
|---|---|---|---|
| C1 | Sum insured per bedroom, `feat_quantiles` bands | add | Value density, a theft target |
| C2 | Bedrooms (1 to 5, 6+) | add | Size of household goods |
| C3 | Building age at the row (as B1) | add | Escape of water from old pipes |
| C4 | Works from home | add | Occupied by day: less theft, more accidental damage |
| C5 | Lodgers | add | Theft by occupants, wear |
| C6 | Basement | add | Flood |
| C7 | Flood score (as B3b) | add | 90% of contents large-loss cost is flood |
| C8 | Listed building | add | Old buildings |
| C9 | Roof type | add | Water ingress |
| C10 | Energy rating | add | Age and condition |

## 5. Not tested, with the reason

- **`discount_applied`.** It is part of the incumbent's price, so J-prem rules it out in the same way as `written_premium`.
- **`compulsory_excess`.** It takes one value per line.
- **Customer tenure from `customer_since_year`.** It is identical to renewal_count (G2).
- **`previous_insurer`.** It is a marketing field, close to the P6 list.
- **Cost-based or peril-specific claims history.** That belongs to step 4.

## 6. Thresholds used

`feat_quantiles`, `tele_quantiles` and `drop_one_max_rounds` (in `thresholds.md`, 2026-10-19), plus the keep rule: `keep_years_better`, `keep_ci`.
