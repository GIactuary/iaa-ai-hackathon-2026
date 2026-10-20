# Phase 6 check-in 6: referral and treatment (2026-10-20)

Run: `scripts/phase6_referral.py`, evidence in `outputs/phase6.json`; the row list is `data/work/phase6/referrals_2025.csv` (not published). Design and thresholds written and committed first: `decisions/phase6_design.md` (8550658). The frozen model's family predictions come from `scripts/freeze_build.py`, rerun to save them; the frozen file came out byte-identical.

## Result in one line

Model disagreement is not a warning sign in this data: rows where the families disagree by more than 30% were no less reliable out of time. Pricing them at the highest family price loses everywhere; pricing them at the W2 GLM passes in motor only. The size referrals are few, and their segments are not used by the models.

## R3: model disagreement (refer_disagreement = 0.3)

**2025 rows flagged:** motor 2,321 (0.9%), buildings 28,758 (15.6%), contents 31,631 (16.9%).

**Out of time** (each year scored with weights fitted on the other):

| Line | Flagged A, B | Deviance per unit of expected cost, flagged against the rest: A | B | Actual over expected on flagged rows: A, B |
|---|---|---|---|---|
| Motor | 3.4%, 4.9% | 0.199 against 0.271 | 0.290 against 0.290 | 0.85, 0.99 |
| Buildings | 33.2%, 12.0% | 0.301 against 0.337 | 0.266 against 0.333 | 0.96, 0.88 |
| Contents | 27.5%, 24.0% | 0.567 against 0.681 | 0.541 against 0.679 | 1.03, 1.00 |

Flagged rows carry the same or less error per unit of cost. The blend is not less reliable where the families disagree.

**Treatments of flagged rows against keeping the blend (T0):**

| Treatment | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| T1 highest family price | −2,431 [−12,708, 10,233], −5,743 | −21,908 [−79,463, 40,028], −44,523 | −28,019 [−44,850, −10,741], −23,598 | −52,357 [−108,749, 11,028], −73,864 |
| T2 the W2 GLM's price | +18,782 [7,077, 31,902], +1,478: passes | −23,002 [−37,396, −7,814], −19,483 | −1,066 [−7,748, 6,450], −10,449 | −5,286 [−26,477, 14,787], −28,453 |

## R1 and R2: size and thin segments

- **R1, beyond 1.25 times the segment maximum:** 1 buildings row in a property type and bedroom segment with no training rows; 1 contents row at 88,400 sum insured, 2.32 times its segment maximum (the phase 2 referral).
- **R2, segment under 100 training rows:** 49 buildings rows (segments of 22 and 93 rows) and 6 contents rows (one segment of 34).
- **The models do not use these segments.** Buildings and contents dropped property type and bedrooms at check-in 2 (L8). Sum insured is priced across the whole line, where every referred value is inside the training range.

## Ranges

Highest over lowest family price per 2025 row: median 1.07 in motor, 1.17 in buildings and 1.18 in contents; 99th percentile 1.47, 1.99 and 1.80. The blend lies inside the range on every motor and contents row and on 94% of buildings rows; the log-average mix is rescaled to the line total, which can move it just outside.

## Cards for the actuary (round 1)

- **Q1** (card P1 on the page). Treatment of model disagreement.
- **Q2** (card P2 on the page). Treatment of the size and thin-segment referrals.

## Verdicts (actuary, 2026-10-20, check-in 6 round 1)

Cards P1 and P2 on the page are register items Q1 and Q2: the AI first reused the IDs P1 and P2, which phase 0 decisions already hold, and renamed them when recording. Q1 (a), Q2 (a), with no notes. Exchange D26.

- **Applied by `scripts/apply_referral.py`**, which writes `blend_frozen_v2.csv` from `blend_frozen_v1.csv`. The 2,321 motor rows where the families disagree by more than 30% take the W2 GLM's price, a median 0.84 of the blend's. The motor line is then rescaled to its decided 2025 total, so the other motor rows rise by 0.13%. Buildings and contents are unchanged.
- **Review list:** every R1, R2 and R3 row of 2025 (62,767 rows) is in `data/work/phase6/referrals_2025.csv` for underwriting review. The R1 and R2 rows keep the blend price.
