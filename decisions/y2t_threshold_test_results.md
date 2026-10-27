# Y2-T threshold test: results

26 October 2026. Design `decisions/y2t_threshold_test_design.md`, committed before any run (00e5efb). Script `scripts/y2t_threshold_test.py`; evidence `outputs/y2t_threshold_test.json`; run logs in `data/work/y2t/`. All figures are claim-frequency (Poisson) deviance at a common level per line, summed over the three emerging lines; lower is better. Intervals are 95%, from a paired bootstrap resampling customers. The key totals were recomputed by separate code and match.

## Conclusion first

The level matters at one boundary only. A threshold at or above occupation's 80 levels is clearly worse in both designs. From 4 to 20, the designs cannot tell the thresholds apart: the differences are smaller than the spread between repeated fits. The pre-registered rule picks 4. The AI recommends 20 instead, and says why below; the actuary decides.

## Results

| Threshold | Encoded (emerging) | V1, out of time: over best [95%] | V2, cross-validation: over best [95%] |
|---|---|---|---|
| 4 | every code over 4 levels (adds age band, body type, soil, region, energy rating, colour) | +16 [−17, +50] | +21 [−15, +55] |
| 5 | make_model, occupation, colour, region, energy rating | +7 [−28, +43] | +8 [−22, +40] |
| 6 | make_model, occupation, colour, energy rating | +75 [+43, +108] | +11 [−17, +42] |
| 7 | make_model, occupation, colour | best | +8 [−13, +29] |
| 20 | make_model, occupation | +12 [−15, +40] | best |
| R0 | make_model and occupation left out | +47 [+12, +82] | +14 [−14, +40] |
| 100 | make_model only; occupation raw (80 levels) | +261 [+183, +339] | +321 [+240, +401] |
| 300 | nothing; make_model raw too | +369 [+274, +468] | +661 [+551, +766] |

How to read this: in V2, a threshold of 100 lets occupation into the trees raw. Its deviance is 321 above the best threshold, and the interval (240 to 401) is far from zero. The same model is also 307 worse than leaving occupation and make_model out altogether.

- **Repeated fits.** In V2, the same candidate's total moves by about 100 between seeds (threshold 20: 39,831, 39,840 and 39,934). The gaps among thresholds 4 to 20 are 7 to 21. Threshold 6's poor V1 result (+75) did not recur in V2 (+11), which points to fitting noise.
- **Threshold 20 against 4, seed by seed (V2):** 20 is lower in all three seeds, by 44, 16 and 1.
- **Per line (V2).** Motor gains most from encoding: 20 beats leaving the codes out by 17. In buildings and contents, threshold 20 and leaving the codes out are within 1.3 of each other, as expected for a field with no signal there.
- **Cleaned occupation (exploratory).** No measurable effect when encoded: V2 cleaned at 20 is +16.6 over R0, as typed +14.1. Raw, cleaning cuts the damage (−62 against −307 versus R0) but does not remove it.
- **Pre-registered selection.** Both designs pick threshold 4, so the consistency check passes.

## Why the AI recommends 20, not the rule's 4

The rule assumed that a lower threshold is the more economical model, because it encodes more codes. The results do not support that premise:
- **The gain comes from encoding the two codes with many levels.** Encoding the small codes as well (age band, body type, soil, region) buys nothing in either design.
- **Threshold 4 has nothing to show.** In V2 it is the weakest encoded regime and below leaving the codes out (−6.4). It is behind 20 in every seed.
- **It costs interpretability.** Encoding a five-level code such as age band replaces its levels with one smoothed number, so the trees' splits no longer show the bands.

So the point of diminishing returns runs the other way from what the rule assumed:
- Raising the threshold to 80 or above loses heavily.
- Lowering it below 20 gains nothing.
- 20 is the lowest threshold that encodes exactly the codes where raw entry is shown to do harm. It is the best threshold in V2 and within reach of the best in V1. It also falls inside Part 1's evidence: occupation raw at 20 levels passed there.

This departs from the pre-registered rule, and it is stated as such for the actuary's decision.

## Decision

The actuary chose 20 on 27 October 2026 (Y2-T brief round 2, finalized; register Y2-T, exchange D35). `code_levels_max` is dated in `decisions/thresholds_part2.md`.

## Part 1 run (completed 27 October, after the decision)

The first Part 1 run was slow because of a fault in the fold split, not the model fitting: scikit-learn's grouped split is quadratic on text customer ids, about 20 minutes per call on one core at 400,000 motor rows. `make_folds` now passes integer codes in sorted id order. That gives exactly the same folds (tested, and checked on real customer ids on both markets with all three seeds), and the run finished in 36 minutes as four parallel workers. The rerun's motor claim-count unit matches the old run's log line for line (same setting and trees for all seven candidates), and the emerging results recompute identically.

Part 1's codes: make_model 300 levels, occupation 20, region 12, vehicle colour 8, the rest 7 or fewer. Summed over the three lines; extra deviance over the best candidate, 95% interval; cost is capped cost (Tweedie 1.5).

| Threshold | Encoded (motor) | A count | A cost | B count | B cost |
|---|---|---|---|---|---|
| 4 | every code over 4 levels | +28 [−40, +99] | +69,927 [+560, +144,439] | +10 [−27, +48] | best |
| 5 | make_model, occupation, region, colour, energy rating | +40 [−29, +106] | +4,541 [−43,927, +53,633] | +30 [−13, +72] | +65,325 [+18,593, +109,582] |
| 7 | make_model, occupation, region, colour | +36 [−30, +100] | +22,516 [−1,532, +43,408] | best | +10,459 [−33,162, +55,194] |
| 11 | make_model, occupation, region | +9 [−58, +72] | best | +20 [−7, +45] | +8,403 [−37,561, +56,916] |
| 20 | make_model only; occupation and region raw | best | +141,386 [+60,091, +220,561] | +23 [−31, +71] | +88,928 [+27,466, +153,860] |
| R0 | make_model left out | +71 [+32, +111] | +132,142 [+53,194, +212,801] | +48 [−7, +96] | +58,125 [−111, +122,650] |
| 300 | nothing; make_model raw | +687 [+566, +802] | +528,866 [+420,940, +647,703] | +506 [+392, +618] | +330,257 [+248,025, +413,356] |

How to read this: in scoring A on capped cost, threshold 20 leaves occupation (20 levels) and region (12) raw. Its deviance is 141,386 above threshold 11, which encodes them, and the interval (60,091 to 220,561) is clear of zero. In scoring B the same comparison gives +88,928 against the best and +80,525 against threshold 11.

- **make_model raw is clearly worse in all four**, as in the emerging market.
- **On claim count, 4 to 20 cannot be told apart**, as in the emerging market.
- **On capped cost, leaving occupation (20 levels) and region (12) raw hurts in both scorings.** Threshold 20 is no better than leaving make_model out (R0). Encoding them (threshold 11 or lower) is best or within reach. Encoding the codes of 7 levels or fewer as well adds nothing consistent: 4 is best in B and clearly worse in A.
- **Pre-registered rule on Part 1:** picks 4, 5, 4 and 4 (A count, A cost, B count, B cost).

What this means for the emerging market: none of its codes has between 9 and 79 levels, so any threshold from 8 to 79 gives the same model as 20 (make_model and occupation encoded, every other code raw). Part 1's cost evidence supports that model: raw codes of 7 or 8 levels did no harm, and codes of 12 levels or more did. It contradicts one part of the reason recorded for the value 20, that "occupation raw at 20 levels passed in Part 1" (PM2m). On this test it did not, on capped cost.

**Decision (Y2-T2).** The actuary set `code_levels_max` to 11 on 27 October 2026 (exchange D37): encode every code with 12 or more levels, the value Part 1's cost evidence supports. The emerging model is the same as at 20.
