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
