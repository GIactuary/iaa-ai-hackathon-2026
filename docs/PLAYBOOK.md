# Playbook: principles from raw data to submission

The 2026 task will differ from 2025, so this is a set of principles, each tested on the 2025 data. Every 2025 prize-winning method was rebuilt on the same 1,463 training houses and the same five-fold splits on three seeds, so the scores below are like for like. The master copy of this playbook is a document in the user's claude.ai project; this file is the working copy for the repo.

## What 2025 taught, on identical houses

| Entry | Approach | Cross-validated RMSLE | Private leaderboard |
|---|---|---|---|
| 3rd prize | One XGBoost + the 2,930 public Ames sales | 0.081 | 0.077 (1st of 49) |
| Our decided model | Forward-built GLM + LightGBM, 50/50 fixed | 0.090 | not submitted (closed) |
| 2nd prize | 8-model stack with XGBoost meta-model, blended with 5 single models | 0.0905 | 0.116 (10th) |
| Special mention | 7-model stack, AI-agent features | 0.092 (own folds) | 0.118 (18th) |
| 1st prize | LightGBM + XGBoost + CatBoost, tuned weights | 0.0925 | 0.114 (3rd) |
| 3rd prize method without the public sales | Same model, competition data only | 0.105 | |

Three lessons sit under every principle.

1. Model count does not buy accuracy. A 13-model stack tied a 2-model pair; three trees averaged beat the best single tree by about 1%, while one GLM plus one tree beat it by about 4%.
2. Data decisions beat modelling. The public source the synthetic data was generated from was worth more than all the modelling work in the field.
3. A handful of extreme records decided the leaderboard. Most teams scored about 0.09 in cross-validation and 0.115 on the leaderboard. Six test houses over 4,000 sq ft explain much of the gap: entries that priced two 5,000 sq ft new builds in a cheap neighbourhood near $700k finished mid-pack; the most accurate entry priced them near $190k.

## The principles, in working order

1. **Settle the use before the data.** Decide what the prediction is for, which error costs more, and what is known at the point of use. The metric then fixes the target scale: RMSLE means modelling log of the target; a deviance metric means the matching GLM family. In 2025 we framed the model as an insured-value check at homeowners underwriting, which ruled out variables known only at sale and tilted the prediction range upward.
2. **Know where the data came from and what the rules allow.** In hour one: is the data synthetic, from what public source, what does the external-data rule say (2025 section 6a allowed public data), may the notebook call LLM APIs, what must be submitted. Using an external source is a logged judgment for the actuary.
3. **Clean by meaning, and prove each reading.** Every blank is feature absent, zero, estimable, or truly unknown, taken from the data dictionary and confirmed by a cross-check (blank pool quality with pool area 0 means no pool). Never mean-impute an absent feature.
4. **Screen before modelling, and write a reason for every exclusion.** Fit a quick line (log target on size, quality, location), list records more than 4 standard deviations off, investigate each. In 2025 the screen put one sale first (z = −6.3, a partial sale of an unfinished new build); the winner also excluded a second sale with an eyeball rule, and our model priced that one within 0.4% of its actual price.
5. **Check the test inputs against the training evidence on day one.** Test features are public. Any test record beyond 1.25 times its segment's largest training value, or carrying a category never seen in training, goes on a referral list before a model exists. Exclusions and referrals are one decision: when a training record is excluded, test records like it have lost their only evidence.
6. **Design validation before fitting anything.** Five-fold cross-validation on three split seeds; paired bootstrap against the incumbent on the same rows; a 20% hold-out locked away and opened once. Write every threshold down first. Expect the hidden test to score worse than cross-validation when it holds extremes.
7. **Engineer features from the business, and keep them on evidence.** Start with what an appraiser or underwriter would compute (totals, ages, quality × size). A feature stays only if removing it worsens error on all three seeds. In 2025 three of eight engineered features (porch area and two presence flags) failed and were removed with no change in error.
8. **Choose model families, not model counts.** One GLM-type model and one boosted tree, errors checked for difference, averaged 50/50 on the link scale with weights fixed in advance. A third family earns a place only by beating the pair on a paired test with fresh splits. Tuning comes last: a 30-trial search did not beat hand-set settings.
9. **Make the GLM the rating plan and the trees a bounded adjustment.** Build the GLM forward: start from the base rate, add the variable that cuts cross-validated error most, stop when the next gains under 0.2%. Pool category levels with fewer than 5 records. A 50/50 log blend equals GLM price × √(tree price ÷ GLM price); in 2025 that adjustment stayed within ±5% for 90% of test houses. Explain the GLM by its relativities and the adjustment by Shapley values (LightGBM `pred_contrib`), ALE curves and permutation importance. A forward GLM cost about 1% in the pair against a Lasso and gives a plan with ~25 named variables.
10. **Treat the tail as a separate problem.** Refer records that are out of evidence (principle 5) or where the two models disagree by more than 30% (in 2025, the 2% of houses with the widest disagreement carried about 40% more error than the rest). Price them by a stated rule: cap size at the edge of the evidence; when the capped price still contradicts the only comparable record, rate from that comparable. A disagreement flag alone keeps the model price and goes to review.
11. **Report the honest number.** Out of fold only; show apparent against cross-validated against hold-out error. In 2025 the trees looked twice as accurate on fitted rows. When the hold-out is level, say so and rest the claim on the larger tests.

## When the 2026 task is a different problem

| If the task is | Target and GLM half | Tree half | What stays the same |
|---|---|---|---|
| A price or value, RMSLE | Log target, Gaussian GLM | LightGBM, squared error on the log | Principles 1 to 11 as written |
| Claim counts or frequency | Poisson GLM with exposure offset | LightGBM, Poisson objective, exposure as init score | Referral on out-of-evidence risks; Poisson deviance as the metric |
| Claim severity | Gamma GLM, log link | LightGBM, gamma objective | Large-loss screen and capping (principles 4 and 10) |
| Pure premium or loss cost | Tweedie GLM, log link | LightGBM, Tweedie objective | Double lift and A/E by segment against the incumbent |
| Yes or no (lapse, fraud, claim) | Logistic GLM | LightGBM, log-loss | Calibration by decile replaces the lift chart; ranges become probability bands |
| Time to event or development | Discuss with the actuary first | | Validation by time (later periods held out), not random folds |

If records are dated, random folds leak the future: hold out the latest periods instead.

## Traps

| Trap | How it bites | Guard |
|---|---|---|
| Inflated validation | Stacking scored on in-sample predictions (special mention claimed 0.058; honest was 0.092) | Out-of-fold only; show the spread across seeds |
| Eyeball outlier rules | A rule drawn from a scatter plot excluded a normal sale | Screen with a fitted line and write a reason for each exclusion |
| Noise-level features | Agent-proposed features kept on gains inside CV noise | Drop-one test on three seeds |
| Point-of-use leakage | Inputs known only after the event (sale type, sale condition) | Point-of-use test in phase 1 |
| The tail | A few extreme test records dominate the metric | Referral rule and stated treatment (principle 10) |
| Non-reproducible AI | LLM output differs on rerun | No live LLM calls in the notebook; log exchanges as data |
| Generic AI narrative | Reads as filler to human judges | Actuary writes the findings; AI critiques; log what changed |
| Over-engineering | Dashboards and agents that move no criterion | Every build item names the criterion it serves |
| Last-night churn | 14 submission runs on the final evening (special mention) | Freeze the model by mid-window |

## Notebook structure (IAA template headings, rate-filing content inside)

1. Summary: filing-style summary, numbers filled from the run.
2. Generative AI integration: the working loop, the decision ledger, the AI roles.
3. Step-by-step analysis, following `docs/FILING_TEMPLATE.md`: purpose, data, rating variables, development, validation, rating plan, rating example, impact and referrals, segments.
4. Findings and insights in plain language; the judgment register.
5. Reproducibility: seeds, versions, prompt library, transcript, the independent review.

## Time plan (assumes a 72-hour window; rescale once the rules are known)

| Hours | Work |
|---|---|
| 0 to 4 | Intake, rules, data origin, metric; data steward readings; baselines submitted |
| 4 to 12 | Screens, validation design, hold-out locked |
| 12 to 30 | Features, forward GLM, trees, blend, tail rule |
| 30 to 36 | Model frozen; validation report |
| 36 to 56 | Explanation exhibits, judgment register, AI log, findings |
| 56 to 66 | Independent review in a fresh context; fixes; clean rerun |
| 66 to 72 | Buffer; the actuary submits |
