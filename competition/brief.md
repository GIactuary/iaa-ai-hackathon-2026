# Competition brief: IAA AI Assisted Hackathon 2026, Part 1

Written at intake, 2026-10-17, from the actuary's copy of the Overview, Data and Rules pages, taken 2026-10-17 (`competition/kaggle pages.docx`, removed from the repository and its history before publication, register PUB-H, since the host's pages may not be redistributed), the notes in `overview.md`, `rules_notes.md`, `data_description.md`, and checks run on `data/raw/` (scripts kept in the session scratchpad; every number below can be regenerated from the raw files). The copied pages hold no separate actuarial pricing guide; the documentation standard we work to is `docs/FILING_TEMPLATE.md` (FSRA guidance, NAIC predictive-model white paper, CIA standards). If the actuary meant another guide, it is added to `competition/` and read before phase 1 closes.

## Conclusion first

This is a loss-cost (pure premium) problem across three lines, scored on Tweedie deviance with power 1.5 at row level. The target does not exist in the data as delivered: claims are valued at 31 December 2024, so training costs must be developed to ultimate and the level projected to 2025 before any model sees them. The data carries planted traps (two leakage columns, unit errors in exposure and sum insured, inconsistent codes, broken customer ids, one 11.9M reserve) and one large judgment: the weather file includes realised 2025 weather. Part 1 is half of the accuracy score and the start of a two-week story; Part 2 (a new market) is judged on how we detect and respond to change, so everything built now must be re-runnable on new data.

## 1. Task, target, metric

| Item | Reading |
|---|---|
| Task | For each 2025 policy row in `test_motor`, `test_buildings`, `test_contents` (626,639 rows), predict the expected ultimate claim cost in CU |
| Target definition | All claims with loss date inside the row's period, net of excess (paid and incurred are already net of excess), net of recoveries, excluding repudiated claims. The risk premium, not the premium charged |
| Metric | Tweedie D² with power 1.5: 1 − D(y, ŷ) / D(y, baseline), the share of Tweedie deviance explained relative to a constant-rate baseline. Higher is better |
| Scale implied | Row-level cost, so the prediction is rate × exposure. Short rows (test mean exposure 0.78 against about 1.0 in training) must be scaled by exposure |
| GLM family implied | Playbook row "pure premium or loss cost": Tweedie GLM, log link, log(exposure) offset. The classical alternative, Poisson frequency × gamma severity by peril, is a candidate in the same family and gives the cleaner rating story |
| Tree half implied | Gradient boosting with a Tweedie (or Poisson and gamma) objective, exposure as offset or weight |

What the metric rewards. Tweedie deviance at p = 1.5 sits between Poisson (count-like) and gamma (relative error). It penalises getting the overall level wrong on every row, so the 2025 base level (development plus trend) matters as much as the relativities. It is less dominated by single huge claims than squared error, but a large test claim on a row we price near zero still costs a lot. Predictions must be strictly positive: a zero prediction against a positive actual gives infinite deviance at p = 1.5.

## 2. Window, deadline, deliverables, submission

| Item | Reading |
|---|---|
| Part 1 window | Opened 2026-10-16. On 2026-10-17 the Overview showed start "a day ago" and close "6 days to go", so closing about 2026-10-23. Exact time and zone not published (rules 2.3 still shows "[DATE, TIME UTC]"); Kaggle deadlines are normally 23:59 UTC. Foundational rule 3.a makes the time zone our responsibility. **Plan to submit by 2026-10-23 12:00 UTC** |
| Part 2 | Separate Kaggle page, opens when Part 1 closes, about one week; new emerging market with different characteristics and data problems |
| Submission | One CSV, `row_id,prediction`, all three test files together, prediction ≥ 0 (we will keep it > 0). Five submissions a day; one final may be selected, else best public score |
| Final deliverables (end of Part 2, by form) | Team name and members; week 1 and week 2 prediction files; a video demo (first 5 minutes graded); a write-up (first 5 pages graded); the code under version control. Both must cover both parts |
| Team | Up to 3, same team for both parts, no change after the Part 1 merger deadline (date not yet published) |
| Prize eligibility | Members or students of a Full or Associate Member Association of the IAA |

Time plan consequence. With about six days left in Part 1, the model should be frozen by about 2026-10-20 and the remaining days spent on validation, explanation and preparing the Part 2 adaptation kit.

## 3. Judging, and how it differs from 2025

| 2026 category | Weight | 2025 nearest | 2025 weight |
|---|---|---|---|
| Pricing performance and adaptation (Part 1 and Part 2 private leaderboards equally, plus the panel's view of modelling choices, data quality treatment, limitations, uncertainty) | 40% | Private-leaderboard accuracy | 30% |
| Effective use of AI (through EDA, features, modelling, diagnostics, testing; shown improvement; rigorous verification of AI output) | 25% | Creative use of AI | 30% |
| Trust, validation and guardrails (validation, automated tests, diagnostics, governance, reproducibility, human review and escalation) | 20% | Interpretability | 20% |
| Communication and documentation (5-minute video, 5-page write-up, version-controlled repository) | 15% | Documentation and reproducibility | 20% |

What changed and what it means for us:

1. **Accuracy is half leaderboard, half judgment.** "Adaptation" and "treatment of data quality, limitations and uncertainty" are scored inside the 40%. The data traps found below are part of the test.
2. **Guardrails are a category of their own (20%).** Automated tests, escalation protocols and human review must be visible, not described. Our threshold file, hold-out lock and referral rule fit this directly.
3. **The deliverable is a video plus 5 pages plus a repository**, not a single Kaggle notebook. The 2025 notebook builder becomes the engine for the write-up and exhibits; the repository itself is judged. The data must stay out of any public repository (rules 2.5.b).
4. **AI use is judged on verification.** "Rigorous verification of AI output" is named. Exchange cards where the AI was wrong and the test caught it remain the strongest evidence.
5. **Two-part structure.** Part 2 rewards a pipeline that can detect what changed in new data. Build every check (data rules, screens, drift tests) as code that reruns on a new market.
6. **Screening is still AI first, panel final.**

## 4. Rules on external data, pretrained models, AI and LLM calls

| Topic | Rule (section) | Reading for us |
|---|---|---|
| External data | 2.7.a, 2.7.b: allowed if publicly available and equally accessible at no cost, or "reasonably accessible to all Participants and of minimal cost"; proprietary data not allowed | Allowed in principle; whether anything useful exists is checked in section 5 |
| AutoML, pretrained tools | 2.7.c: allowed with an appropriate licence | Allowed; no pretrained model is needed for tabular pricing |
| AI on the data | 2.5.a: "You may process the Competition Data with AI tools and services, including hosted large language models" | Our workflow is allowed |
| LLM calls inside the deliverable code | No rule found that forbids them | Our rule 6 stands: no live LLM calls in the scored pipeline, so a clean rerun gives the same numbers |
| Data security | 2.5.b: no redistribution outside the competition | `data/` stays out of any public repository |
| Open source | Foundational 6.c, and 2.6 for winners: OSI-approved licence not limiting commercial use | All packages in `requirements.txt` qualify (BSD, MIT, Apache) |
| Hand labelling | Foundational 4.b: no hand labelling or human prediction of test records | Referral treatments must be rules applied by code, not hand-set prices for named rows |
| Each stage needs a submission | Foundational 4.c: in a multi-stage competition "one or more valid Submissions may be required during each Competition stage" | Submit a valid Part 1 file early (a baseline on day 2), not only at the end |
| Winner's code | 2.9.a: training code, inference code and a description of the computing environment; 2.6.b: a method description detailed enough to reproduce | Matches our rule 6: one command regenerates every output |
| Licence | CC BY-NC 4.0 | Non-commercial; fine for the competition |

## 5. Data origin

Synthetic, from a bespoke generator. Evidence:

- Identifiers and codes are artificial: `surname` takes 600 values of the form "Surname143"; `make_model` is MM001 to MM300; `previous_insurer` INS01 to INS40; regions R01 to R12; currency "CU".
- The schema (11 linked tables: customers, addresses, annual weather by address, three policy lines, claims with peril and reporting dates) matches no public insurance dataset we know. The common public sets (French motor freMTPL2, Australian private motor, the pg15 training set, the Swedish motorcycle set) carry different columns, one line only and no linked weather or address tables. Matching rows on numeric columns, as done for Ames in 2025, has no candidate source to match against.
- Distributions are smooth in the way generators produce: vehicle colours are almost exactly equal shares (121,477 to 123,232 of 979,968); `driver_age` equals age from date of birth to within one year on every matched row.

Consequence for J-ext. There is no public source that holds the answers, unlike Ames in 2025. External data could only add general benchmarks (published claims inflation, for example), which do not map to a synthetic CU economy. Recommendation: no external data.

## 6. Shapes, IDs, dates

| File | Rows | Columns | Grain |
|---|---|---|---|
| customers | 545,240 | 18 | customer |
| addresses | 575,983 | 16 | address |
| weather | 2,879,915 | 9 | address × year, **2021 to 2025** |
| policies_motor | 979,968 | 39 | policy period, 2021 to 2024 |
| policies_buildings | 707,726 | 27 | policy period |
| policies_contents | 721,617 | 28 | policy period |
| claims | 163,150 | 12 | claim |
| test_motor / buildings / contents | 254,986 / 184,155 / 187,498 | 38 / 26 / 27 | policy period, 2025 |
| sample_submission | 626,639 | 2 | test row |

- **IDs:** row_id, policy_id, customer_id, address_id, claim_id, all read as text. `policy_id` persists across years: 85% of 2025 test policies were in force in 2024 (motor 200,063 of 236,008).
- **Dates:** period_start, period_end, loss_date, report_date. Rows never cross a calendar year. Validation must hold out the latest periods (rolling-origin by policy year), not random folds; random folds would also let a policy's 2023 row inform its own 2022 row.
- **Claims to rows:** 99.7% of claims match exactly one row by policy_id and loss date in period; none match two. The 414 unmatched claims are mostly December 2020 losses on policies that start on 2021-01-01 (outside any row's period, so outside the target by definition).
- **Accident-year picture.** Loss ratio = net incurred over earned premium (written premium × exposure), with each row's cost corrected for reporting completion (P10). Claims are uncapped, and the locked hold-out customers are excluded. Corrected on 2026-10-18 (exchange D13): the intake figures divided by annual premium without exposure, and showed 2024 undeveloped. They were motor 0.91, 0.83, 0.81, 0.61; buildings 1.17, 1.07, 1.07, 1.05; contents 0.75, 0.65, 0.67, 0.64.

| Line | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| Motor | 1.05 | 1.08 | 0.97 | 1.00 |
| Buildings | 1.33 | 1.29 | 1.32 | 1.36 |
| Contents | 0.84 | 0.81 | 0.85 | 0.84 |

- **Development.** Paid to incurred for accident year 2024: motor 0.73, buildings 0.72, contents 0.86. December 2024 motor claims reported: 1,409 against about 2,300 in a normal month. Subsidence claims reported for Nov and Dec 2024: 5 and 7 against about 27 a month. Reporting lag 95th percentile: motor 82 days, buildings 54, contents 33; maximum 1,361 days.
- **Trend.** Median net severity rises about 6% to 9% a year for most perils (escape of water 3,379 to 4,219 from 2021 to 2024; glass and hail 652 to 835). The insurer's 2025 premium rate (written premium weighted by exposure) is 2.6% (motor) to 5.6% (buildings) above 2024, in line with these trends. Corrected on 2026-10-18 (exchange D13): the intake figure of 36% to 44% divided total annual premium by total exposure, and test rows average 0.78 to 0.82 years of exposure against about 1.0 in training.

## 7. Leakage paths and inputs known only after the event

| Input | Evidence | Reading | Proposed treatment |
|---|---|---|---|
| `renewal_decision` | Training only. "declined_by_insurer" rows have a claim 95% (motor) to 99% (contents) of the time | Outcome at the end of the row | Never an input |
| `customer_service_contacts` | Rows with a claim average 3.3 contacts, without 0.4, every year. Training max 36, test max 6, test mean 0.40 | Training counts contacts during the period, including claim contacts; test counts something else | Not an input as recorded. Possibly rebuilt from a prior period if it can be shown to exist at pricing time |
| Weather for the policy year | `weather` holds 2025 for test addresses. Storm frequency in the windiest fifth of address-years is 1.9 times the calmest; flood 2.5 times on rainfall; subsidence 3.0 times on dry summers. 2025 mean wind index is the highest of the five years (4.89 against 4.35 to 4.60) | Realised weather is known only after the year. It is supplied for the test year and carries real signal | Judgment J-weather |
| `period_end`, `exposure` in test | Test rows already reflect 2025 cancellations and mid-term changes | Known only after the event, but given and needed to scale the row cost | Use as given (it defines the row); note it in limitations |
| Claim history of the same policy, customer or address | Available for 85% of test policies | Legitimate at pricing (prior years only) if built strictly from claims reported before the row starts | Candidate experience-rating feature, built point-in-time |
| `missed_payments_12m` | Claimants 0.134 against 0.10; same distribution in train and test. Data page: "Missed instalments in the previous 12 months" | A trailing measure, known at pricing | Candidate input; the train-test match is checked in phase 1 |
| `days_unoccupied` (buildings) | Data page: "Longest period the property was unoccupied", with no window stated | Could include days during the row, which are known only afterwards | Phase 1 check: compare its distribution on rows with and without a claim, and train against test, as done for contacts |
| `written_premium` | The insurer's current price | Legitimate input at pricing, but it embeds another model's view | Judgment J-prem |

Data quality traps found (to be classified in phase 1, not fixed yet):

- Exposure in days instead of years on about 1,500 to 1,600 training rows per line (574 motor rows show 365). Test exposure is clean.
- Sum insured 1,000 times too large on about 0.2% of rows (1,395 buildings, 1,402 contents in training; 370 and 360 in test). Dividing by 1,000 brings them inside the normal range.
- `construction_type` in 11 spellings. The data page documents four levels (brick, stone, timber, non-standard), but the data has two short codes, W/w and T/t, beside them. Whether W and T are both timber (wood, timber) or one is something else is a phase 1 check (claim rates, sum insured and region mix of each code).
- Motor `customer_id` lost leading zeros on 10% of rows (98,110); padding to 8 digits matches every one to the customers table.
- One open motor bodily injury claim reserved at 11,894,277 CU with 738,769 paid; the next largest claim of any kind is 1,118,868. Possibly a keying error (a digit too many), possibly a true catastrophe claim. A screen item.
- Missing values in a few percent of rows for occupation, marital status, household size, years at address, flood zone, distance to river, soil, roof, energy rating, vehicle age, engine size, mileage, parking, days unoccupied, smoke alarms, security level, high-value items. `telematics_score` is blank exactly when not opted in (absent by design).

Fairness and proxy flags for the rating plan: `surname`, `email_domain`, `marketing_segment`, `campaign_code`, `branch_id`, `marketing_consent`, `document_preference`; `credit_band` and `occupation` are restricted in some jurisdictions for some lines. A later judgment.

## 8. Candidate business story and intended use

**Story.** We are the pricing team of an established insurer writing private motor, home buildings and home contents in one developed market. The current rating plan (`written_premium`) is under review: buildings has run at a loss ratio of 1.29 to 1.36 in every year, motor at about 1.0, contents at about 0.84, and the 2025 premium rate is up only 2.6% to 5.6%. We are asked for a technical price: the expected ultimate cost of each 2025 policy row, built so that it can be filed, explained by line and peril, and moved to a new market.

**Intended use (candidate).** The technical risk premium used to (a) set 2025 base rates and relativities per line, (b) measure rate adequacy against the current price, policy by policy, and (c) refer risks the evidence cannot support to an underwriter.

**Costlier error.** Under-pricing. A price below the true cost writes loss-making business and, through comparison sites (35% of motor), attracts more of it (adverse selection). Over-pricing loses volume but not capital. Consequence: the model is calibrated to the expected value (the metric demands it), and the range and referral rules lean toward the under-pricing side.

## Judgments opened at intake

- **J1** Intended use and business story (above).
- **J-ext** External data: recommend none.
- **J-weather** Whether realised 2025 weather is an input.
- **J-prem** Whether the current premium is an input, an offset, or only the incumbent to beat.

See `decisions/judgment_register.md`.
