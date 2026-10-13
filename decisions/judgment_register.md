# Judgment register

Status is "Needs your call" until the appointed actuary decides. Only the actuary changes a status to "Decided".

| Ref | Decision | Evidence (exhibit) | Options | AI recommendation | Status |
|---|---|---|---|---|---|
| J1 | Intended use and business story | | | | Needs your call |
| J-ext | External data | | Use the public source / do not | | Needs your call |
| J-model | How models combine: within a family, keep only the best candidate, or weight several | Exchange D1. 2025 evidence (identical folds): three tree models averaged beat the best single tree by about 1%; one GLM plus one tree beat it by about 4% | (a) best single model per family, weights across families only; (b) weighted members within each family, then weights across families | (a), with (b) tested as a challenger on a paired test; it keeps one explainable model per family | Decided by the actuary, 2026-10-13 (exchange D2): "a with b sounds right, go with it" |
