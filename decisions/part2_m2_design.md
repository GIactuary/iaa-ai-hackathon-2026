# Part 2 milestone 2: test design, written before any run

28 October 2026. The actuary: "Of course, start. I want to get as close as possible, and I want you to give you the best analytical work." The check file scored 0.14266 on the public board. The gap diagnostic (`outputs/markets/part2/part2_gap_diagnostic.json`) ruled out the level, the insurer's premium, the contacts field, region and seasonality as the source of the gap. This design tests the remaining levers on the risk shape. Every choice goes through P2-V; the hold-out stays locked.

## What is scored

- **Rows:** emerging fitting rows (85% of customers), 2023-2024.
- **Target:** net cost to date, uncapped, as the competition scores it; predictions are multiplied by reporting completion, per peril where the candidate prices by peril.
- **Level:** each candidate's predictions are rescaled to the scored rows' actual total in each line (common level per line), so the tests measure the shape within lines, including where the large losses sit. The level is set separately by the level chain.
- **Pooled:** deviance (Tweedie 1.5) summed over the three lines, as the competition pools them.
- **P2-V:** V1 fit 2023, score 2024; V2 customer-grouped five-fold cross-validation over 2023-2024, seeds 11, 22, 33, each row's deviance averaged over the seeds. A candidate replaces the incumbent only if it lowers pooled deviance in both, and its 95% paired bootstrap interval on V2 (customers resampled within line, 2,000 draws) lies above zero.
- Parts fitted on Part 1 rows are fitted once; only the emerging parts (peril levels, layers) are refitted in each split.

## Candidates, tested in this order (each against the incumbent at that point)

1. **K0, incumbent:** the check file's shape: Part 1 W2 GLM on capped cost, one excess factor per line.
2. **K1, by peril:** one Tweedie GLM per peril on Part 1 rows (capped cost, the W2 factors), each peril's level set to the emerging fitting rows' completed capped cost for that peril, plus each peril's own excess factor (emerging own for motor, Part 1's shares for property, as P2-L). Tests whether pricing perils separately moves cost to the right rows.
3. **K2, large-loss perils by property:** K1, with the large-loss perils (buildings fire and flood, contents flood, motor bodily injury) priced as frequency times severity on uncapped cost: a Poisson frequency GLM and a gamma severity GLM, both on Part 1 rows with the W2 factors. Severity then follows sum insured and property type rather than being capped and spread.
4. **K3, address hazard:** K2 with the address's wildfire normal (buildings fire) and distance to river and elevation bands (flood) added to the large-peril frequency GLMs. The emerging wildfire index lies above Part 1's range (0.5 against 0.15), so the term extrapolates; the per-peril level calibration on emerging rows absorbs the average, and the test shows whether the slope holds.
5. **K4, best family per line:** the Part 1 LightGBM (as in the diagnostic) in place of the GLM where it wins, per line, and the 50/50 log blend.
6. **K5, gender in motor:** an emerging layer (gender, offset on the incumbent) in motor only (P2-S).
7. **K6, pooled challenger (P2-C (c)):** one GLM on Part 1 and emerging fitting rows together with a market factor.

Order matters only for what each is compared against; every candidate's result against the original K0 is reported as well.

Outputs: `scripts/part2_candidates.py`, `outputs/markets/part2/part2_candidates.json`.

## Round 1 results and addendum (28 October, written before round 2 runs)

Round 1 (`outputs/markets/part2/part2_candidates.json`): no candidate clears P2-V. Positive in both tests but with the cross-validation interval including zero: gender in motor (V1 +5,942 [2,654, 9,408]; V2 +7,895 [-2,113, 17,898]), the tree in contents (+10,195; +13,264 [-3,419, 30,448]). Address hazards: V2 +30,641 [-8,684, 71,227], V1 -2,049. By peril and fire and flood by property: no gain. The pooled intervals are about ±30,000 wide, set by a few large buildings claims.

Round 2 adds candidates that bring information the incumbent does not use. Each is tested against the incumbent (still K0), under the same rule:

8. **K7, the row's own-year weather.** The incumbent applies the target year's weather as one average per line (Y4). K7 applies each row's own year factor (its address's weather that year against its normal; the normal for a fitting row is the other experience year, so a row's own weather never enters its normal). Variants: (a) Part 1's elasticities on the Y4 perils; (b) elasticities fitted on the emerging fitting rows of each split, on the Y4 perils plus buildings fire with the wildfire index and buildings and contents flood with rainfall. The test-year file carries 2025 weather for every address, so the same factor is available for the 2025 rows. Y4 decided weather as one factor per line; a row-level factor would reopen Y4 and comes to the actuary as a card if it passes.
9. **K8, prior-year claims history.** A layer on the incumbent with the customer's claim count in the previous calendar year across all lines (none, 0, 1, 2 or more; rows with no prior-year row are "none"). For 2025 rows the previous year is 2024. All new-business test rows are "none".
10. **K9, everything positive together:** gender in motor, the tree in contents, address hazards, and whichever of K7 and K8 is positive in both tests.

## Round 2 results and addendum (28 October, written before round 3 runs)

Round 2 (`outputs/markets/part2/part2_candidates_round2.json`): K9+K7b passes P2-V against K0 (V1 +7,498; V2 +37,526 [298, 76,858]); history (K8) fails and hurts in V2. A direct check after round 2: emerging new business against renewal, actual over the Part 1 base's expected, is 1.37 [1.10, 1.66] on contents frequency (Part 1's W2 dropped new business from contents, L8), 0.97 on motor frequency, 0.88 on buildings. The test book is 76% new business against 14% in the scored 2024 rows, so this matters far more to the 2025 level than the 2024 test can show.

11. **K10, new business layer:** a new-business factor fitted on emerging fitting rows on top of K9+K7b, per line, tested against K9+K7b under P2-V. Reported with each line's fitted relativity.

## Paired public-board check (M2-B (a)), rule written before either file is uploaded

28 October. Two files differ only in the new-business factor: `part2_v2_nb-none_wx-row.csv` (no adjustment) and `part2_v2_nb-cred_wx-row.csv` (M2-N (b)). The actuary uploads both. **Rule:** keep the credibility-weighted factor unless its public score is lower than the unadjusted file's. Whatever the result, no further variant is chosen on the public board; the comparison and its outcome are reported in the write-up.

**Result (28 October).** Public scores: credibility-weighted 0.15312, no adjustment 0.15080, and a third file the actuary also uploaded, the full emerging figure, 0.14646 (the check file: 0.14266). Under the rule, the credibility-weighted factor stays. The ordering is the one credibility theory predicts: moving all the way to the thin emerging figure loses, ignoring it loses less, the credibility weight scores best. No further choice is made on the public board.

## Round 3 addendum (28 October, written before round 3 runs)

**Public-board noise, measured first** (2024 scored rows, 1,000 random thirds): one model's D² spans 0.107 to 0.165 (90%). Paired differences between two of our files: K9+K7b against K0 true +0.0016, on a third −0.0085 to +0.0102 (looks worse 39% of the time); gender in motor true +0.0013, on a third +0.0004 to +0.0021 (1%); tree in contents true +0.0022, −0.0017 to +0.0057 (17%). Proposed rule for board use, for the actuary: a variant is judged on the board only if its expected public gain exceeds twice its paired public-board spread measured this way; buildings large-loss changes are judged on our own validation only.

Round 3 candidates, each against the incumbent K9+K7b (the improved file's shape) under P2-V:

12. **K11, Y1 as decided (the premium as our fields predict it):** a LightGBM of log written premium on the fields both markets carry, fitted on Part 1 rows without claims; its prediction (lwp_hat) enters the Part 1 base GLM as one numeric term, in every line, replacing the base where the line's incumbent uses it (motor, buildings attritional perils) and as a tree input in contents. In Part 1 this was the post-mortem's largest gain (PM6 buildings +110,298 [62,978, 164,210]).
13. **K12, every factor credibility-weighted (the penalised form of P2-C (c)), amended before any run:** a layer on the incumbent with every W2 factor and region, fitted on emerging fitting rows with an L2 penalty that shrinks each emerging deviation toward zero (toward Part 1's relativity). The penalty strength is chosen from 1e-5 to 1 on fold 1 of the fitting rows' customer-grouped inner folds. This is pooling with the data choosing how far each factor moves, without refitting a million Part 1 rows in every split.
14. **K13, credibility layers:** region, make_model and occupation each as one credibility-weighted log mod against the incumbent (Bühlmann-Straub, `fields.credibility_encode`, out of fold), in a layer on K9+K7b.

**Round 3 results (28 October)** (`outputs/markets/part2/part2_candidates_round3.json`), against K9+K7b:
- K11, Y1 premium in the Part 1 base: out of time +5,783 [−6,069, 16,992]; cross-validation +2,588 [−10,784, 15,780]. Positive in both tests, interval includes zero: fails. Motor gains in both tests (+2,844; +5,981), buildings and contents are mixed. Premium model R² on Part 1 rows: motor 0.916, buildings 0.957, contents 0.917.
- K12, every factor credibility-weighted: −3,460; −10,984. The inner-fold choice of penalty sits at the strongest value tried (1.0) in most splits: the emerging data cannot support moving the factors away from the incumbent.
- K13, credibility-weighted codes: −43,451; −16,730. Fails.

**Motor-only Y1 premium (28 October, M2-R).** Measured before any upload (`scripts/part2_motor_premium_check.py`): against K9+K7b, out of time +2,844 [44, 5,469], cross-validation +5,981 [1,547, 10,441]. It passes P2-V on its own, so Y1 (decided (a), subject to the keep rule) enters motor. Expected public gain +0.0006 against twice its paired spread on a third (0.0008), so under M2-R the board cannot judge it; the decision rests on our own validation. Buildings and contents keep the K9+K7b shape (Y1 there failed in round 3).

## Tuning round (P2-F (a)), written before it runs, 28 October

Incumbent **I**: K9+K7b with motor's base replaced by the Y1 premium base (`part2_v2_nb-cred_wx-row_mprem`). Settings are chosen where each model is fitted (Part 1 rows), on fold 1 of five customer-grouped inner folds (seed 11), by Tweedie 1.5 deviance on capped cost at common level, as Part 1's step 7 did; the tuned model is then tested against I on emerging rows under P2-V.
- **Motor base GLM and buildings attritional peril GLMs:** training Tweedie power from the dated grid (1.1 to 1.9, step 0.1).
- **Contents tree:** leaves {15, 31, 63} × minimum rows per leaf {100, 400, 1600}, learning rate 0.05, early stopping 100.
- **T, all tuned settings together,** against I. Kept only if it passes P2-V.

**Tuning result (28 October)** (`outputs/markets/part2/part2_tuning.json`). Chosen on Part 1 rows: motor base power 1.4; buildings attritional perils 1.5 (accidental damage), 1.6 (escape of water, subsidence), 1.7 (storm); contents tree 15 leaves, 400 minimum rows, 152 trees. T against I: out of time +317 [−5,449, 6,025]; cross-validation +2,793 [−3,976, 9,604]. Fails P2-V; I keeps Part 1's settings. The gain sits in contents (+506; +3,701); motor and buildings are flat. Pooled D², I: 0.1396 out of time, 0.1448 cross-validation (K0: 0.1374, 0.1402).

## Motor credibility re-rate (K14), written before it runs, 28 October

The actuary asked whether re-rating with a GLM and credibility between Part 1 and the emerging data would gain. K12 tested this with one penalty shared by every factor in every line: it failed overall, and motor was the only line that gained (+759 out of time; +1,717 cross-validation). The actuary: "Test motor re-rate". K14 gives each motor factor its own credibility.

- **Incumbent I** in motor (K11 base with the gender layer and own-year weather); buildings and contents unchanged.
- **Factors:** every motor W2 factor as built for the emerging rows (less the year) plus region, as in K12. Gender stays as fitted in I.
- **Step 1, the emerging deviations:** on the fitting rows of each split, one Tweedie 1.5 GLM layer with I as the offset, all factors jointly. Its coefficients are each level's deviation from the incumbent's relativity, against the factor's most common level.
- **Step 2, their standard errors:** sandwich covariance clustered by customer (robust to a few large claims and to a customer's repeated rows).
- **Step 3, credibility per factor (empirical Bayes, the Bühlmann form):** for factor f with J non-base levels, the spread of true deviations τ²_f = max(0, mean over levels of (β̂² − se²)). Each level's credibility Z = τ²_f / (τ²_f + se²), and its deviation becomes Z × β̂. A factor whose deviations are no larger than their noise gets τ² = 0 and keeps Part 1's relativities. No penalty or other setting is chosen, so nothing is tuned on the scored rows.
- **Step 4:** prediction = I × exp(credibility-weighted deviations), rescaled to I's total on the fitting rows. Levels unseen in the fitting rows keep Part 1's relativity.
- **Rule:** K14 replaces I in motor only if it passes P2-V against I. The full-weight layer (Z = 1, every deviation as fitted) is scored beside it as the other end of the bracket; it is reported, not a candidate.
- **Board (M2-R):** the expected public gain and its paired spread on random thirds are reported, as for the motor premium.
- **Exhibit:** each factor's τ² and average Z from a fit on all fitting rows, for the write-up.

Outputs: `scripts/part2_motor_rerate.py`, `outputs/markets/part2/part2_motor_rerate.json`.

**K14 result (28 October)** (`outputs/markets/part2/part2_motor_rerate.json`). Against I: credibility-weighted re-rate out of time −3,473 [−11,067, 3,925], cross-validation −2,619 [−9,928, 4,664]: fails, I stays. The full-weight layer (every emerging deviation taken as fitted): −27,245 [−44,491, −9,658] and −38,983 [−62,282, −15,572], a clear loss. Pooled D², out of time / cross-validation: I 0.1396 / 0.1448, K14 0.1389 / 0.1445, full weight 0.1338 / 0.1407. Board (M2-R): expected public change −0.0003 to −0.0007. The bracket is the one credibility theory predicts when the complement is good and the new data are thin: Part 1's relativities score best, the credibility blend a little worse, full weight much worse. On all fitting rows, 11 of 17 factors get no credibility at all; cover type (mean Z 0.71) and fuel type (0.78) get the most, and no-claims years (0.28) shows why the test fails: its emerging deviations zigzag (0.85, 0.95, 1.23, 0.97, 1.19, 0.78 for 0 to 5 years) with standard errors of 0.11 to 0.16, noise that a factor-level spread estimate from a few levels partly mistakes for signal. K12's small motor gain was measured against the old motor base; the premium term now in I appears to have taken it up.

## Hold-out opening (P2-H), written before it is opened, 28 October

The actuary: "Freeze and open". The model is frozen as I (K9+K7b in buildings and contents; motor's base with the Y1 premium term, the gender layer and own-year weather). The hold-out is opened once, to confirm, not to choose.

- **Frame:** every emerging row, hold-out customers included. Part 1's parts are refitted on Part 1 rows exactly as before; they only predict for the larger frame. Every emerging part (peril levels, layers, weather responses) is fitted on non-hold-out rows only.
- **H1, out of time:** fit the non-hold-out 2023 rows, score the hold-out customers' 2024 rows. **H2:** fit all non-hold-out rows, score every hold-out row (2023 and 2024).
- **Metric:** Tweedie 1.5 deviance at a common level per line, as P2-V; D² per line and pooled; paired customer bootstrap (2,000 draws) for I against K0 (the check file's shape) and against K9+K7b (the improved file before the motor premium). Each line's level ratio before rescaling is reported for information.
- **Reading rule:** if I beats K0 on pooled deviance in at least one of H1 and H2, the final file is I. If I is worse than K0 in both, the actuary decides before any final file is built. Nothing else about the model changes on what the hold-out shows. About 520 motor, 80 buildings and 76 contents claims sit in the hold-out's 2024 rows, so buildings will dominate the swings.
- The new-business credibility factor (M2-N) and the level chain are set outside the shape tests, as in P2-V, and are not part of this check.

Outputs: `scripts/part2_holdout.py`, `outputs/markets/part2/part2_holdout.json`.

## Referral rule for the final file, written before it runs, 28 October

The thresholds are Part 1's, re-confirmed for the emerging market (P2-TH): `refer_size_ratio` 1.25, `refer_min_segment` 100, `refer_disagreement` 0.3.
- **R1, beyond the evidence:** a 2025 row whose size (buildings sum insured by property type and bedrooms; contents sum insured by tenure and bedrooms; motor vehicle value group and engine size on the whole book) exceeds 1.25 times the largest value in its segment among the final model's training rows, or sits in a segment with none. Training rows are every row the final model was fitted on: Part 1's developed rows (the risk shape) and every emerging row (levels and layers).
- **R2, thin segment:** the row's size segment holds fewer than 100 training rows.
- **R3, model disagreement:** does not apply. It compares families inside a blend, and the final model prices each line with one family (motor and buildings GLMs, contents tree). Recorded as not applicable rather than replaced with a new test.
- **Treatment:** Part 1's decided treatment (Q2 (a)): the row keeps the model price and is listed for underwriting review. Prices in the final file do not change; the list goes into the write-up. Carried over for the actuary to confirm.

Outputs: `scripts/part2_referral.py`, `outputs/markets/part2/part2_referrals.json`, the row list in `data/work/markets/part2/referrals_2025.csv` (not committed).

**Hold-out result (28 October, opened once)** (`outputs/markets/part2/part2_holdout.json`). Scored: H1 577 motor, 92 buildings, 65 contents claims; H2 1,168, 197 and 140. I against K0, deviance saved (positive: I better):

| Test | Motor | Buildings | Contents | Pooled [95%] |
|---|---|---|---|---|
| H1, hold-out 2024, fitted on 2023 | +1,971 | −769 | −2,189 | −987 [−9,648, 7,430] |
| H2, every hold-out row | +4,789 | −17,484 | +1,048 | −11,648 [−46,177, 11,635] |

Pooled D², K0 against I: H1 0.1508 against 0.1496; H2 0.1222 against 0.1154. The motor premium against K9+K7b: +635 and −1,792. Both pooled intervals include zero. Under the reading rule written before opening, I is worse than K0 in both, so the actuary decides before the final file is built (brief `docs/reviews/part2-holdout-decisions.html`, cards H-F and H-R).

**Referral result (28 October)** (`outputs/markets/part2/part2_referrals.json`). 10 of 209,063 2025 rows: 1 R1 (contents sum insured 117,400, 1.26 times its segment maximum), 9 R2 (5 buildings in a property type and bedrooms segment of 42 training rows, 4 contents in a tenure and bedrooms segment of 39). No motor row is referred.

**Decided (28 October, exchange D44).** H-F (b): the final file is the frozen model with buildings at the check file's shape, `part2_v2_nb-cred_wx-row_mprem_bchk_all.csv`. H-R (a): the 10 referred rows keep the model price and are listed for underwriting review.

## Level study after the final file (28 October, no board use)

The final file scored 0.15455 public (21st; top 0.17353). To see whether the level could explain the gap (`scripts/part2_level_study.py`, `outputs/markets/part2/part2_level_study.json`): the final shape fitted on 2023 predicts 2024, each line at its deviance-optimal level, pooled D² 0.1429. One line's level off by 10% costs 0.0008 (motor), 0.0004 (buildings), 0.0003 (contents); off by 25%, 0.0037, 0.0019, 0.0013. On 1,000 random thirds of the rows, the optimal level found on a third differs from the other two-thirds' by a standard deviation of 9% (motor), 28% (buildings) and 18% (contents). The level cannot explain a 0.019 gap unless it were off by far more than any evidence suggests, and a level read from the public third would carry most of that noise into the private rows.

## Boosted correction (K15), written before it runs, 28 October

The actuary: "do the boosted correction now", then documentation. K15 is the one lever not yet tested: a tree model trained on emerging rows that starts from the final model's prediction and learns only what that prediction misses (interactions and non-linear effects a GLM layer cannot express). It replaces the pooled challenger K6, which round 3 swapped for K12 to save run time.

- **Incumbent F (the final file's shape):** motor K11 (premium term, gender layer, own-year weather); buildings K0 (H-F (b)); contents K9+K7b (tree, own-year weather).
- **K15:** per line, a LightGBM (Tweedie 1.5) on capped cost with F, rescaled to the fitting rows' capped total, as the offset; prediction = F × the learned correction. Inputs: every field `fields.tree_fields` admits for the line (raw categories up to 11 levels and numeric fields; make_model and occupation left out, as their encodings failed in K13) plus the address's weather normals (leave-own-year-out). No row-year weather deviations, so buildings does not get back the row weather H-F removed. Settings fixed now: 15 leaves, 400 minimum rows per leaf, learning rate 0.02, 80% row and column sampling, up to 2,000 trees; the number of trees is chosen by early stopping (100 rounds) on fold 1 of five customer-grouped inner folds of the fitting rows (seed 11), then refitted on all fitting rows. A correction the data cannot support stops at few trees and stays near 1.
- **Rows:** the hold-out has served its purpose (opened once, decided in H-F), so P2-V now runs on every emerging customer: V1 fit 2023, score 2024; V2 customer-grouped five-fold cross-validation on seeds 11, 22, 33.
- **Rule:** P2-V per line, as H-F judged the lines: a line takes K15 only if it gains in both tests and its V2 95% paired customer-bootstrap interval lies above zero. Pooled results are reported as well.
- **Exhibit, whatever the result:** for each line, the number of trees chosen and the fields the correction uses most (mean absolute contribution to the log price), from a fit on all rows. This is the evidence for the study of what our model misses.

Outputs: `scripts/part2_boosted.py`, `outputs/markets/part2/part2_boosted.json`.
