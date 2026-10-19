# Phase 5 step 4: claims history (C7), test design

Written 2026-10-19, before any history feature was built or scored. Decision C7 (a): history for the policy, the customer and the address, point in time, weighted by Bühlmann-Straub credibility, each piece kept only on the keep rule against the working base V1.

## 1. What a row may see

- **Cutoff.** A row starting on date t sees only claims reported before its cutoff c = min(t, data end). The data end is 1 January after the last fitting year: 2023-01-01 for scoring A, 2024-01-01 for scoring B, 2025-01-01 for the final fit and the 2025 test rows.
  - This mirrors pricing 2025: a test row starting in June 2025 knows claims reported up to 31 December 2024 only. About 20% of test rows start after 1 January.
  - Exposure counts only up to the cutoff, so a policy's own earlier 2025 row (no claims data) adds nothing.
- **Hold-out.** Hold-out customers' rows and claims are removed from every history count, including the address counts of other customers.
- **Claims counted.** Every claim reported before the cutoff, repudiated or not.
  - Counting by status would let later knowledge in: a claim reported in mid-2022 carries its status as at 31 December 2024.
  - The prior-repudiation piece (H-R below) uses status and carries this caveat.
- **Counts, not cost.** Incurred cost as at the cutoff is not in the data, only as at 31 December 2024. A cost-based history would use later case-reserve movements, so it is not built.

## 2. The experience mod

For each row and each scope, with A = claims reported before the cutoff and E = the claims expected over the same exposure:

- **E (expected claims).** The sum over the scope's earlier rows of class frequency × exposure before the cutoff × the share of claims expected to be reported by the cutoff.
  - Class frequency comes from a Poisson GLM per line on the V1 rating factors, fitted on the fitting years (claims of all statuses, offset exposure × count completion).
  - The reported share uses the line's count-weighted delay curve (the P10 curves). For a row covering days s to m before the cutoff it is the average of F(c − 1 − d) over those days. Seasonality is ignored here.
- **Credibility.** Z = E / (E + k), with E in expected claims. The mod is Z × A/E + (1 − Z) × 1 = (A + k) / (E + k).
  - A row with no history has E = 0, so Z = 0 and the mod is 1. No separate "no history" level is needed. That matters here, because every 2021 row has no history in this data whatever its true past.
  - A claim-free history gives a mod below 1 that falls as E grows.
- **k from the data.** The Bühlmann-Straub non-parametric estimator, on the fitting years, non-hold-out rows. One risk is one scope unit (policy, customer or address), one period is one row year. X = A/E with weight E, where A is all claims on the unit's rows in that year and E is class frequency × exposure × count completion at the valuation date.
  - s² = Σ w (X − X̄ᵢ)² / Σ (nᵢ − 1)
  - a = [Σ wᵢ (X̄ᵢ − X̄)² − (r − 1) s²] / [w − Σ wᵢ² / w]
  - k = s² / a
  - If a ≤ 0, the data shows no between-risk variance: k is infinite, the mod is 1 everywhere, and the piece is reported as null.
  - The Poisson-gamma moment estimate (s² = 1) is shown beside it as a check.
- **How the mod enters the model.** log(mod) is one numeric column in the capped Tweedie GLM, so the prediction is proportional to mod^β. β = 1 is pure experience rating on top of the class rate. β is fitted, so a scope that carries over only partly (another line, another occupant) is diluted by the data.

## 3. Scopes (disjoint)

| Piece | Scope | Units for k | Lines |
|---|---|---|---|
| H-P | The row's own policy, earlier rows | Policy, per line | All |
| H-C | The same customer's other policies, any line | Customer, all lines pooled | All |
| H-A | Other customers' rows at the same address: property claims for buildings and contents; motor claims for motor (household drivers) | Address | All (property is the plan; motor is the widened test) |
| H-D | Declared claims in the 5 years before joining (`prior_claims_5y`, customers table), as a factor 0, 1, 2, 3+ | n/a | All |
| H-R | Any repudiated claim reported before the cutoff on the policy or the customer's other policies (yes/no) | n/a | All |

## 4. Variants of H-P (widened under the standing rule)

1. **Raw counts** (C7 option c, for evidence only): prior claim count 0, 1, 2+ as a factor, with prior exposure years as a factor.
2. **Banded mod**: the mod in bands (quantile edges of fitting rows with history, `hist_mod_quantiles`) plus a "none" level, instead of log(mod).
3. **Average-rate E**: E from the line's average frequency instead of the class frequency.
4. **Pure multiplier**: V1 × mod, with no refit (β fixed at 1).
5. **By peril group**: separate mods for weather perils and the rest. Property weather is storm, flood and subsidence. For motor, own damage, third-party property and bodily injury are split from glass and theft. E is split by each group's share of claims in the fitting years.
6. **Customer split**: H-C divided into the same line and other lines.

## 5. Tests

- **Each piece against V1** on scorings A and B, at a common level per line, with the keep rule: better in both scorings, and the 95% customer-bootstrap interval on A above zero (2,000 draws). Per line and combined.
- **Each H-P variant against H-P.** A variant replaces H-P only if it passes the keep rule against H-P. The raw-counts variant is evidence on C7 and replaces nothing.
- **Combination.** V1 plus every piece that passed, then each piece dropped in turn from that combination. A piece stays only if dropping it fails the keep rule in reverse, that is, if the combination is better than the combination without it.
- **Reported.**
  - k, Z at typical history lengths, β per scope and line, and mod distributions.
  - The level is unchanged by construction: the off-balance re-levels each line (P11).
  - A worked example: one renewal with a claim, and one without.

## 6. Thresholds used

`hist_mod_quantiles` (written in `thresholds.md` on 2026-10-19), plus the keep rule already written: `keep_years_better`, `keep_ci`.

## 7. Amendment after run 1 (2026-10-19, written before run 2 was scored)

**What went wrong.** The non-parametric estimator of section 2 proved too inefficient for this data. Run 1 found:
- k infinite for buildings on scoring A, so the policy history there was null and its test meaningless;
- contents k at 25.1 (A), 4.6 (B) and 92.7 (final fit);
- motor k at 6.1, 5.0 and 5.7.

**Why.** A policy-year carries about 0.05 expected claims in property and 0.15 in motor. The non-parametric between-risk estimate subtracts about one unit of Poisson noise per policy from a signal of a × E per policy, so it is mostly noise. Within-policy variance is Poisson (s² from 1.02 to 1.06; 1.004 claims per incident, so claims do not cluster). The Poisson-gamma moment estimator weights policies by E², which is efficient in this rare-event regime. It gave stable values: motor 2.35, 2.62, 2.76; buildings 1.77, 2.21, 2.30; contents 2.08, 2.30, 2.54.

**Independent check.** The GLM slopes on log(mod) imply an effective k of about k ÷ β for small histories:
- motor: 3.5 (A), 3.4 (B), 3.4 (final);
- buildings: 3.0 (B), 3.1 (final);
- contents: 6.0 (A), 3.2 (B), 4.1 (final).

These sit close to the Poisson-gamma values and far from the non-parametric ones.

**Amendment.** Run 2 uses the Poisson-gamma moment estimate of k for every scope, with every other part of the design unchanged. Both runs are reported; run 1 stays in `outputs/phase5_step4.json` and run 2 is `outputs/phase5_step4_pg.json`. The choice of estimator goes to the actuary at check-in 2.
