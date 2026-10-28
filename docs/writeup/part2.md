# Part 2: the same pipeline in an emerging market

*Draft for the appointed actuary, 2026-10-28, sized for about 2 of the 5 graded pages. Every number comes from an evidence file in `outputs/markets/part2/` or a decision note in `decisions/`, named under each exhibit. Appendix references (A to C) are to material after page 5.*

## Summary

The emerging market has two years of data, a book a twentieth the size of Part 1's, and a 2025 test book that is 76% new business from an acquired broker portfolio. We priced it by borrowing Part 1's risk shape, fitted on a million developed-market rows, and letting the emerging data set the level and add only what it could prove. Every change had to beat the incumbent out of time and in cross-validation, with a 95% interval above zero. Every candidate change faced one rule; of about twenty tested, two passed. On customers locked away until the end, the borrowed shape held in motor and contents and failed in buildings, so the appointed actuary took buildings back to the plain Part 1 GLM. The final file scored 0.15455 on the public leaderboard, against 0.14266 for our first check file.

## 1. What changed, and what we borrowed

**The market (brief, `competition/part2_notes.md`).** 2023 and 2024 only, six regions, ages in bands, no credit score, mileage or telematics, occupation typed by the customer, local construction codes, and gender, which this market allows in pricing. The insurer takes on a broker portfolio at the start of 2025, so the training book (86% renewals) is not the book we price.

**Borrow the shape, learn the level (P2-C (b)).** Before any model, we tested whether two years of emerging data could carry their own risk shape. Out of time (fit 2023, score 2024), the Part 1 GLM's relativities beat a GLM fitted on emerging data in every line: motor D² 4.8% against 3.5%, contents 3.2% against −3.0%, buildings 11.5% against −144% (`part2_check_v1.json`). A few large buildings claims make a GLM fitted on one emerging year unusable. This is credibility at the level of a whole rating plan: the developed market is the complement, and the emerging data earns weight only where it shows it.

**The data rules carried over and stop on surprises.** Part 1's nine rules ran unchanged, with two additions: open claims net of their expected recoveries (R10, which the host confirmed is part of the target) and occupation trimmed and case-folded (R11). The construction map stopped the run on the new codes until it was written. Where emerging test blanks run at more than twice Part 1's rate, the blank is priced at the average of known levels: Part 1 had recovered most blanks from history, leaving levels such as a contents security blank at 0.005, which would have priced those contents rows near zero (contents D² −9.7% before the fix, 3.2% after).

## 2. The test, and what passed

**P2-V, written before the first candidate ran.** A change is kept only if it lowers deviance out of time (fit 2023, score 2024) and in customer-grouped five-fold cross-validation over both years on three seeds, and its cross-validation interval from 2,000 customer resamples lies above zero. 15% of customers (5,515) were locked away before any fit.

**Exhibit 5. Part 2 candidates against the incumbent, deviance saved, all lines pooled**

| Candidate | Out of time | Cross-validation [95%] | Kept |
|---|---|---|---|
| Fire and flood priced by property, with address hazards | −2,049 | +30,641 [−8,684, 71,227] | no, alone |
| Part 1's tree in contents | +10,195 | +13,264 [−3,419, 30,448] | no, alone |
| Gender in motor | +5,942 | +7,895 [−2,113, 17,898] | no, alone |
| The three together, with each row's own-year weather | +7,498 | +37,526 [298, 76,858] | **yes** |
| Insured's premium as our fields predict it, in motor | +2,844 | +5,981 [1,547, 10,441] | **yes** |
| Prior-year claims history | 0 | −15,348 | no |
| Region and occupation, credibility-encoded | −43,451 | −16,730 | no |
| Tuning on Part 1 rows | +317 | +2,793 [−3,976, 9,604] | no |
| Credibility re-rate of every factor (penalised GLM; per factor) | −3,460; −3,473 | −10,984; −2,619 | no |
| Boosted correction learned on emerging rows | −9,115 | −14,876 [−30,421, −943] | no |

*Sources: `part2_candidates*.json`, `part2_motor_rerate.json`, `part2_boosted.json`, `part2_tuning.json`; designs in `decisions/part2_m2_design.md`, each committed before its run.*

> **How to read this.** Each figure is Tweedie deviance saved on the emerging rows, each line at its own actual level. The winning combination saves 37,526 in cross-validation, about 0.004 of D²; its interval just clears zero. Most rows fail: the emerging data could not support moving Part 1's relativities, whether by a credibility-weighted GLM, a per-factor credibility re-rate, or a flexible tree on top of the final model. Three different methods reached the same answer, which is the strongest evidence in Part 2 that the borrowed shape was right.

**New business, by credibility (M2-N (b)).** The test book is 76% new business; the training book 14%. Each line's new-business relativity moves toward the emerging figure by Z = √(claims ÷ 1,082), the limited-fluctuation standard: motor (697 new-business claims) Z 0.80, factor 1.105; buildings (77) Z 0.27, 0.950; contents (102) Z 0.31, 1.094, with renewals rebalanced so each line's total is unchanged. The actuary allowed one paired check on the public board, decided before upload: the credibility-weighted file scored 0.15312, no adjustment 0.15080, the full emerging figure 0.14646. The ordering is the one credibility theory predicts.

**Gender (P2-S (a)).** Gender entered motor only, as a layer fitted on emerging data, and only because it passed as part of the combination. Appendix B shows its relativity, the effect on prices for otherwise identical drivers, and the price without it.

## 3. The hold-out, and a line-by-line call

The hold-out was opened once, on a rule written and committed before opening: if the final model beat the check file's shape in neither reading, the actuary decides.

**Exhibit 6. The frozen model against the check file's shape, deviance saved**

| Reading | Motor | Buildings | Contents | Pooled [95%] |
|---|---|---|---|---|
| Hold-out, 2024, fitted on 2023 | +1,971 | −769 | −2,189 | −987 [−9,648, 7,430] |
| Hold-out, every row | +4,789 | −17,484 | +1,048 | −11,648 [−46,177, 11,635] |
| Every customer, fit one year and score the next | +10,773 | −12,834 | +11,415 | |
| Every customer, cross-fitted | +19,797 | −2,754 | +14,818 | |

*Source: `part2_holdout.json`; the last two rows add the hold-out to the validation readings, which cover the other customers.*

> **How to read this.** Positive means the frozen model is better. Pooled, it fell short on the hold-out, within noise. By line, motor gains in every reading and contents in three of four; buildings loses in three of four, and across every customer it shows no gain (−12,834 and −2,754). Its fire and flood terms extrapolate: the emerging wildfire index runs at about 0.5, against at most 0.15 in Part 1.

The actuary kept motor and contents and took buildings back to the Part 1 GLM (H-F (b)), applying the keep rule line by line to all the evidence. This was decided after the hold-out was opened, and is reported as such. The public score moved from 0.15353 (the frozen model, fitted without the hold-out customers) to 0.15455 (the final file, refitted on every customer).

## 4. The price

**Level.** As in Part 1: a 2025 indication built peril by peril at normal weather, with Part 1's weather-normalised trends (two emerging years cannot fit a trend), times the 2025 weather. Excess factors: motor from its own large claims (1.007), property from Part 1's excess shares by peril (buildings 1.038, contents 1.023). 2025 weather factors 0.995, 1.017 and 0.972. Expected 2025 cost over written premium: motor 0.96, buildings 1.20, contents 0.77 (`part2_v2_nb-cred_wx-row_mprem_bchk_all.json`).

**Office premium (Appendix A).** For each of the 20 reference policies: risk premium, catastrophe reinsurance at 12% of property risk premium, 85 CU fixed expense, divided by what remains of each premium after variable expense (15%), commission (broker 20%, direct 0%), premium tax (8%) and profit (8%). [Formula and basis to be confirmed: OP-T, OP-B.] The expense table gives no comparison-site commission for this market; none of the 20 policies uses that channel, so no rate is assumed (P2-E (d)).

**Referral.** 10 of 209,063 rows are listed for underwriting review (one contents sum insured 1.26 times its segment's maximum, nine in thin segments); prices are unchanged (H-R (a)).

## 5. Limitations and what we learned

- **The board gap.** The leader's public score is 0.17353. Our 2025 public score matches our 2024 validation, so nothing broke on the test year. A level 10% wrong in one line would cost under 0.001 of D², and a level read from a third of the rows would be off by 9% (motor) to 28% (buildings), so level neither explains nor closes the gap (`part2_level_study.json`). Nothing learnable remains in the emerging data on top of our model (the boosted correction above). Whatever the leaders have that we lack sits in the base borrowed from Part 1 or in selection on a noisy third of the rows; one model's D² on a random third ranges from 0.107 to 0.165.
- **Buildings is the least certain line.** A few large claims swing any reading by ±20,000 of deviance; its shape is Part 1's, its excess load is Part 1's, and its new-business factor rests on 77 claims.
- **The new-business factor** is the largest assumption in the 2025 level and is credibility-weighted on thin data in property.
- **Realised 2025 weather** is used, which a real price set in 2024 would not have.
