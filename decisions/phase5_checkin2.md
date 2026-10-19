# Phase 5 check-in 2: claims history and features (2026-10-19)

Runs:
- `scripts/phase5_step4.py`: run 1 is `outputs/phase5_step4.json`. Run 2 used `P5_K=pg` and is `outputs/phase5_step4_pg.json`.
- `scripts/phase5_step5.py`: `outputs/phase5_step5.json`.

Designs, written before the runs: `decisions/phase5_step4_design.md` (with the section 7 amendment) and `decisions/phase5_step5_design.md`. Every test is out of time on non-hold-out customers, at a common level per line, under the keep rule. The decisions are register items L1 onward.

## Step 4: claims history

**The mod.** Policy history enters as log(mod) in each line's GLM:
- mod = (A + k) / (E + k);
- A = claims reported before the cutoff on the policy's earlier rows;
- E = class frequency × exposure × the share expected to be reported by the cutoff.

**k.** The run 2 Poisson-gamma k on the final fit is motor 2.76, buildings 2.30 and contents 2.54. The GLM slopes in the final fit, with declared history also in, are 0.86, 0.82 and 0.83.

| Piece added to V1 (run 2) | Motor: A saved [95%], B saved | Buildings | Contents | All lines | Keep |
|---|---|---|---|---|---|
| H-P own policy | +77,289 [41,027, 115,031], +103,319 | +27,158 [1,463, 55,687], +27,815 | +10,395 [4,394, 17,130], +11,909 | +114,842 [68,647, 164,656], +143,043 | All lines |
| H-C customer's other lines | −1,723, +1 | −6,061, +556 | −345, +610 | −8,129 [−16,894, 1,723], +1,166 | No |
| H-A other customers at the address | +292, +1,123 | −2,951, −3,911 | −1,187, −43 | −3,847, −2,831 | No |
| H-D declared claims before joining | +44,067 [9,976, 80,600], +54,683 | +34,659 [−7,845, 82,112], +45,916 | +4,023 [−14,343, 22,056], +28,820 | +82,749 [26,082, 142,422], +129,419 | Motor and all lines |
| H-R any prior repudiated claim | −69, −415 | −309, +3,445 | −958, +1,570 | −1,337, +4,599 | No |

Both pieces together, each dropped in turn:

| Test | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| H-D added to H-P | +34,182 [1,145, 69,444], +44,629: yes | +31,203 [−10,935, 78,451], +42,150: no | +3,117 [−15,216, 20,935], +27,554: no | +68,502 [12,293, 126,299], +114,334: yes |
| H-P added to H-D | +67,404 [34,454, 101,395], +93,266: yes | +23,702 [−408, 50,086], +24,049: no | +9,489 [3,670, 16,073], +10,644: yes | +100,595 [57,252, 146,590], +127,958: yes |

Ranking D² for all lines, V1 against V1 + H-P + H-D:
- scoring A: 10.02% to 10.34%;
- scoring B: 9.49% to 9.95%.

Forms of policy history against H-P, all lines (A saved [95%], B saved):

| Form | Result | Reading |
|---|---|---|
| Raw counts, with prior exposure years | −172,685 [−239,478, −109,656], −27,217 | Counts learned on short histories do not carry to longer ones; credibility does |
| Mod in ten bands | −141,146 [−200,141, −81,262], −58,433 | The smooth form wins |
| Average-rate E instead of class E | −25,419 [−36,813, −13,711], −3,574 | Class E wins |
| β fixed at 1, no refit | −3,960 [−23,725, 15,794], −5,803 | Not significantly worse |
| Split by peril group | +6,392 [−1,680, 14,349], −3,082 | No gain |

**Where the AI was wrong (D17).** The non-parametric k of the design was unstable in run 1:
- infinite for buildings on A;
- contents at 4.6 to 92.7.

Run 2 uses the Poisson-gamma moment estimate. The amendment was written down before run 2 was scored, and both runs are kept.

**Coverage on the 2025 test rows.** 72% of rows have policy history. 20% of motor rows have a claim in it, against 8% of buildings and 7% of contents rows.

## Step 5: features

Base: V1 plus H-P and H-D in every line (provisional on L2 and L3). 67 candidate tests: 24 motor, 24 buildings, 19 contents. Each was added alone; passers then went through drop-one rounds.

| Line | Passed add-one | Kept after drop-one | Final against base: A saved [95%], B saved | Ranking D², A | Ranking D², B |
|---|---|---|---|---|---|
| Motor | G1, M9 | G1, M9 | +79,907 [45,218, 112,951], +61,068 | 5.37% to 5.68% | 6.09% to 6.34% |
| Buildings | B9 | B9 | +6,574 [495, 13,878], +3,744 | 6.83% to 6.87% | 6.48% to 6.50% |
| Contents | none | none | +0 [0, 0], +0 | 6.46% to 6.46% | 5.79% to 5.79% |

**Kept.**
- G1, voluntary excess (motor).
- M9, vehicle age (motor).
- B9, trees within 5 m × clay (buildings).

**Near misses** (card L7). These save in both years but their A interval crosses zero:
- years licensed in motor;
- voluntary excess in property;
- the flood measures in buildings.

**Fairness fields.** Household size, marital status and occupation all fail, so no fairness judgment is needed on them now. Days unoccupied fails too, which makes its point-of-use question moot.

**M7, make-model.** The test was empty (D18):
- the non-parametric k on cost was infinite, so the encoding was 1 everywhere;
- a hand check found about 3% between-model spread in frequency and nothing in cost;
- M7 is reported as a true null.

**V1 factors under drop-one** (`outputs/phase5_step5.json`, `base_factor_drop`). Five rating factors lower deviance in both years when removed: body type (motor), property type and bedrooms (buildings), new business and property type (contents). Bicycle cover does too, but it defines cover (L8).

## Verdicts (actuary, 2026-10-19, check-in 2 round 1)

All eight cards: option (a), the recommendation, with no notes. Exchange D19.

## L8 joint-drop check: criterion written before it ran (2026-10-19)

The model is the line's model after check-in 2: V1 plus H-P and H-D, plus G1 and M9 in motor and B9 in buildings. The check drops these factors together:
- body type in motor;
- property type and bedrooms in buildings;
- new business and property type in contents.

The joint drop is confirmed for a line if the model without them has lower total deviance than the model with them in both scorings, A and B. The 95% interval on A is reported but not required: each factor already failed the keep rule as an addition. If a line fails, its factors stay and the line goes back to the actuary.

**Result (`scripts/phase5_l8_check.py`, `outputs/phase5_l8_check.json`).** The joint drop is confirmed in every line. Deviance saved by dropping, A [95%] and B:
- motor: +247 [−4,748, 4,866], +2,609;
- buildings: +5,648 [−15,775, 25,734], +25,331;
- contents: +4,299 [−10,927, 22,389], +3,617.

## Working base after check-in 2 (W2)

| Line | Model |
|---|---|
| Motor | V1 less body type, plus policy history (log mod, Poisson-gamma k), declared prior claims, voluntary excess and vehicle age |
| Buildings | V1 less property type and bedrooms, plus policy history, declared prior claims, and trees within 5 m × clay |
| Contents | V1 less new business and property type, plus policy history and declared prior claims |

Each line keeps the line excess factor and the K3 and K4 level. Candidates carried to step 7 (L4, L7):
- customer, address and repudiation history;
- years licensed (motor);
- voluntary excess (property);
- the flood measures (buildings).
