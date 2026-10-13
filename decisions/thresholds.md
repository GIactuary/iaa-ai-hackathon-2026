# Thresholds, written before the tests that use them

The code reads this table through `src/thresholds.py`. It reads the `Key` and `Value` columns; `Value` is a plain number (a share is written as a decimal, so 0.2% is 0.002). A threshold is refused by the code until its `Written on` date is filled in (YYYY-MM-DD), so nothing runs on a value that was not written down first. A blank `Value` means the threshold is still open.

The values below are the 2025 practice, carried forward unchanged. None is confirmed for 2026 until the actuary dates it.

| Key | Threshold | Value | Meaning | Used in | Written on | Rationale |
|---|---|---|---|---|---|---|
| screen_z | Outlier screen | 4 | Records more than 4 standard deviations from a quick fitted line are investigated | /screen | | 2025 practice |
| refer_size_ratio | Out-of-evidence referral | 1.25 | Size beyond 1.25 times the segment's largest training value | /screen, referral rule | | 2025 practice |
| refer_disagreement | Model-disagreement referral | 0.30 | GLM and tree predictions more than 30% apart (ratio of the larger to the smaller, minus 1) | referral rule | | 2025 practice |
| drop_one_seeds_worse | Feature kept | 3 | Removing the feature worsens error on this many of the 3 split seeds | drop-one test | | 2025 practice |
| glm_entry_gain | GLM variable entry | 0.002 | A variable enters if it lowers cross-validated error by at least 0.2% (relative, on the metric's scale) | forward GLM | | 2025 practice |
| glm_min_level_count | GLM level pooling | 5 | Category levels with fewer than 5 records are pooled | forward GLM | | 2025 practice |
| holdout_share | Hold-out | 0.20 | 20% of training records locked away, seed fixed, opened once | validation | | 2025 practice |
| range_lower_q | Prediction range, lower percentile | | Lower percentile of out-of-fold misses within each predicted-value fifth (2025: 0.08, tilted for the insured-value use) | prediction ranges | | Open: depends on the intended use (J1) and the range-shape judgment |
| range_upper_q | Prediction range, upper percentile | | Upper percentile of out-of-fold misses within each predicted-value fifth (2025: 0.98) | prediction ranges | | Open: as above |
