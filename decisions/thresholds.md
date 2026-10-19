# Thresholds, written before the tests that use them

The code reads this table through `src/thresholds.py`. It reads the `Key` and `Value` columns; `Value` is a plain number (a share is written as a decimal, so 0.2% is 0.002). A threshold is refused by the code until its `Written on` date is filled in (YYYY-MM-DD), so nothing runs on a value that was not written down first. A blank `Value` means the threshold is still open.

The rows are the thresholds the principles call for (docs/PLAYBOOK.md). Every value is set for 2026 with the actuary once the data and its intended use are known; the 2025 values are context in docs/LESSONS_2025.md, not defaults. Add rows as new thresholds come up.

| Key | Threshold | Value | Meaning | Used in | Written on | Rationale |
|---|---|---|---|---|---|---|
| screen_z | Outlier screen | | Records further than this many standard deviations from a quick fitted line are investigated | /screen | | |
| refer_size_ratio | Out-of-evidence referral | | A size beyond this multiple of the segment's largest training value is referred | /screen, referral rule | | |
| refer_disagreement | Model-disagreement referral | | Models further apart than this share are referred | referral rule | | |
| drop_one_seeds_worse | Feature kept | | Removing the feature worsens error on at least this many split seeds | drop-one test | | |
| holdout_share | Hold-out | | Share of training records locked away, seed fixed, opened once | validation | | |
| si_unit_multiple | Sum insured unit repair (data rule R2) | 10 | A sum insured above this multiple of the line's training 99th percentile is read as entered 1,000 times too large and divided by 1,000 | src/data_rules.py | 2026-10-18 | Proposed by the AI, to be confirmed by the actuary (register P8-R2). Written after the distribution was inspected, so the test is invariance, not a blind threshold. The flagged set is identical for every multiple from 5 to 50 on both lines (buildings 1,395 train and 370 test rows; contents 1,402 and 360), because the largest genuine value and the smallest erroneous one are 49 times apart for buildings and 24 times for contents. Divided by 1,000, the flagged values match the same policy's other rows (median ratio 1.00) |
