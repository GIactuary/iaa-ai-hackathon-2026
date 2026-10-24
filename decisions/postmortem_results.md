# Part 1 post-mortem: results

Run 2026-10-24 under `decisions/postmortem_design.md` (design and its amendment both committed before the runs they cover). Script `scripts/postmortem.py`; evidence `outputs/postmortem.json`; logs in `data/work/postmortem/`. Gains are Tweedie deviance saved against the reference, at a common level per line: scoring A with its 95% customer-bootstrap interval, then scoring B. "Passes" means the keep rule passes.

## Conclusion first

Most of the gap to the top is in **buildings**, and it is information already in our own fields that our models did not use: the incumbent's rating plan prices **flood geography** (river distance and elevation within flood zone) better than any model we built. Adding the part of the premium that our fields predict to the W2 GLM lifts our all-lines ranking D² by about 0.2 points on both scorings (A 10.51% to 10.71%, B 10.12% to 10.36%). The gap to first place was 0.28 points on Kaggle's scale; the two scales are not the same (Kaggle's baseline differs from ours), so the comparison is of size, not exact. The part of the premium our fields cannot predict adds nothing. Weather adds nothing. Giving the trees every legitimate field helps a little across lines, but no single line passes. Open motor claims without recoveries overstate the 2025 motor indication by 1.28%.

## Results

| Test | Line | A [95% interval] | B | Passes |
|---|---|---|---|---|
| PM1a own-year weather vs R | all lines | −62,438 [−149,559, 26,546] | −5,623 | No |
| PM1b weather and climate normal vs R | all lines | +26,097 [−45,111, 99,801] | +785 | No |
| PM2m every legitimate field except make-model vs R | all lines | +150,968 [35,279, 274,886] | +54,819 | Yes |
| | motor | +30,687 [−28,214, 92,259] | −37,798 | No |
| | buildings | +90,644 [−870, 198,552] | +49,428 | No (interval touches zero) |
| | contents | +29,637 [−11,993, 76,857] | +43,189 | No |
| PM3m PM2m plus weather vs R | all lines | +36,656 [−60,820, 139,841] | −3,487 | No |
| PM4a premium as a tree input vs R | buildings | +156,196 [86,228, 231,125] | +91,195 | Yes |
| | all lines | +173,330 [83,103, 261,375] | +89,987 | Yes |
| PM4b tree started from the premium vs R | buildings | +152,312 [55,511, 256,060] | +61,197 | Yes |
| PM4c W2 GLM plus log premium vs W2 GLM | buildings | +86,353 [32,282, 139,298] | +87,649 | Yes |
| | all lines | +89,610 [34,993, 146,710] | +87,151 | Yes |
| PM6 W2 GLM plus the premium our fields predict vs W2 GLM | buildings | +110,298 [62,978, 164,210] | +123,815 | Yes |
| | contents | +12,481 [448, 26,702] | +9,054 | Yes |
| | all lines | +118,566 [67,440, 174,232] | +133,454 | Yes |
| PM6 W2 GLM plus the premium remainder vs W2 GLM | all lines | −2,680 [−20,340, 14,202] | −8,995 | No |

PM4a and PM4b are flat in motor and contents. The reference R (LightGBM on the step 7 inputs) is itself below the W2 GLM (−272,101 on A, −118,238 on B, all lines), so the tree results are read against R, and the GLM results against the W2 GLM.

**How to read this.** Take PM6 in buildings. Adding one number to the W2 GLM, the premium as predicted from our own rating fields, saves 110,298 units of deviance on 2023 and 123,815 on 2024. The interval on 2023 runs from 62,978 to 164,210, well clear of zero. In ranking D² terms buildings moves from 6.90% to 7.47% on 2023 and from 6.63% to 7.26% on 2024.

## What the premium knows

- **It is a function of our fields.** A LightGBM regression of log premium on our inputs, fitted without any claims information, explains 97.5% of its variance in buildings on the scored rows (motor 95.4%, contents 95.8%). Its largest inputs in buildings are flood zone, sum insured, region and distance to river.
- **The gain sits in the predictable part.** PM6 splits the premium into what our fields predict and the remainder. The predicted part carries the whole gain; the remainder carries none. So the premium holds no information we lack, and it leaks nothing. What it holds is structure our models did not find.
- **Where the structure is (double lift, buildings).** Rows are sorted by how far the premium-informed GLM moves the W2 GLM's price, then split into tenths. In the tenth it raises most (about 22% up), actual cost was 1.28 times the W2 GLM's price on 2023 and 1.57 times on 2024; the premium-informed model was at 0.98 and 1.20. Those rows are closer to a river (median 1.85 km against about 2.4) and lower (6.6 m against about 8.3), and flood is 44% of their cost against about 28% overall. In the tenth it lowers most, the W2 GLM overpriced (0.60, 0.83).
- **Why our models missed it.** Flood claims are rare, so the interaction of zone, river distance and elevation is thin in four years of experience. Even the trees, which had river distance and elevation, did not learn it (PM4a still gains 156,196 over R). The incumbent's plan carries it with more credibility than our experience gives. In rate-filing terms, the current rating plan is a credible complement for the flood relativities where our own data is thin.

## Recoveries on open claims (PM5)

Only motor recovers: own damage 23.6% of incurred on closed claims, theft about 4%. Open claims carry none (host, Kaggle forum). Deducting the expected recovery at the peril's closed-claim rate: motor accident year 2024 net cost falls 1.52%, 2023 0.01%. The 2025 motor indicated capped loss cost falls **1.28%**, more than the 2024 share alone suggests, because the overstated 2024 severity also steepens the own-damage trend. Buildings and contents do not change. On the score this is small (deviance grows with the square of a level error); on the price it is a real overstatement.

## Where the AI was wrong

- **Weather.** Before the test, the AI ranked missing address weather as the second most likely cause of the gap. PM1a and PM1b found nothing.
- **The first PM2 design.** It let LightGBM split on `make_model`, a 300-level code that step 5 had already shown to be noise (M7). Motor lost 414,287 on 2023. The amendment removed it, written before the rerun.

## Open judgments for Part 2

Y1 to Y4 in `decisions/judgment_register.md`.
