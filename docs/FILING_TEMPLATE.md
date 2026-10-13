# Model documentation in Canadian rate-filing structure

Section 3 of the notebook follows this structure. Each section opens with what a reviewer would ask (grey box), shows its exhibits, and closes with a "how to read this" box that explains every technical term with a worked example in real numbers. References: FSRA rating and underwriting guidance (AU0142INT), FSRA standard filing requirements, the NAIC predictive model white paper and its 2024 guidance for non-GLM models, CIA Standards of Practice on documentation of judgment.

| Section | What the reviewer asks | Exhibits that answer it |
|---|---|---|
| 0. Summary | Every proposed change, its overall effect, the largest individual effects | Tiles: incumbent error, proposed error, share of records moving 5% or less, referred records |
| 1. Purpose and use | Model type and why it suits its role; who uses the output | Candidate uses, costlier error for each, what each changes (J1) |
| 2. Data | Sources, adjustments, missing values, exclusions with reasons, limitations | Target distribution raw and transformed; missing-value treatment by column; outlier screen with decisions; train-versus-test evidence check; limitations table |
| 3. Rating variables | Data dictionary, rationale, rejected variables, point-of-use and proxy concerns | Correlations; one-way of the main driver; engineered variables with drop-one results; point-of-use test (J2, J6) |
| 4. Model development | Model type, target, link and distribution, data split, variable selection | Cross-validation diagram; families and roles; best mix of each size; forward GLM ladder with selection stability; GLM against Lasso; learning curve; tuning history (J9) |
| 5. Validation | Fit on validation data, stability, why it beats the incumbent | Paired comparison with bootstrap intervals per seed; lift; double lift; apparent against cross-validated against hold-out; controls table; actual against predicted; prediction ranges by predicted-value fifth with coverage and the costly-side miss rate (J5) |
| 6. Rating plan | How the model produces rates; relativities against one-ways; interpretability tools | GLM relativities against filed-model ALE for the top variables; categorical relativities; adjustment-factor distribution; adjustment drivers (Shapley); permutation importance (J4) |
| 7. Rating example | The algorithm applied to a real record | One record: features, each model's price, the filed price and range; GLM worksheet (base × factors); tree adjustment broken down by variable |
| 8. Impact and referrals | Distribution of changes, largest swings, mitigation (Ontario 15% cap analogue) | Dislocation chart; referral list with trigger and treatment; largest changes (J3, J8) |
| 9. Segment testing | Calibration by segment; proxy and disparate-impact concerns | A/E by the main ordinal driver and by location, with counts (J7) |
| 10. Competitive comparison | How the proposal compares with competitors | Leaderboard context; other methods rerun on the same folds; predictions for the records that decide the metric |
| 11. Judgment register | Deviations from indications and the judgments behind the model | Table: reference, decision, evidence, options, recommendation, status |
| 12. Limitations, governance, AI use | Inventory, limitations, peer review | Limitations; versions and seeds; AI use table (stage, what AI proposed or tested, what the test showed, what the actuary decided) |

## Judgment register template

Each row: reference (J1, J2 ...), the decision in a few words, the evidence with the exhibit number, the options, the AI's recommendation, and the status. Status is "Needs your call" until the actuary decides; then "Decided" with the decision text. Typical judgments: intended use; variables unavailable at the point of use; treatment of excluded records' lookalikes; what is filed and how it is explained; range shape; location or other proxy-risk variables; thin segments; review threshold; blend or single model; external data.

## Plain-language conventions that worked

- RMSLE 0.09 reads as "predictions are typically about 9% off"; on a $300k house that is about ±$27k.
- A relativity of 1.09 reads as "priced 9% above an otherwise identical reference record".
- A/E of 1.07 reads as "actuals ran 7% above prediction, so the group is under-priced"; always show the count beside it.
- A 50/50 log blend reads as "credibility weighting with Z = 0.5 set in advance", and in dollars as the geometric mean of the two prices.
- Cross-validation reads as "a back-test run five times, like testing a reserving method on accident years it was not fitted to".
