---
description: Phase 2 screens. Outlier screen on training records and the train-versus-test evidence check, before any model.
---
0. Agree the screen thresholds with the actuary (`screen_z`, `refer_size_ratio` and any others needed) and write them, dated, in `decisions/thresholds.md` before running anything.
1. Fit a quick line of the target (on the metric's scale) on the main size, quality and location inputs. List records further off than `screen_z` standard deviations. For each, show its raw fields, its fitted value, and your reading of why. Recommend exclude or keep with a reason; looking unusual is not a reason.
2. For every test record, compare against the training evidence: size beyond `refer_size_ratio` times its segment's largest training value, category levels never seen in training, impossible combinations. Write the list to `decisions/referrals_v0.md`.
3. Cross-reference: which test records resemble any record you recommend excluding? Those lose their only evidence if it is excluded.

Present the tables, put the exclusion and referral decisions in the judgment register at "Needs your call", log the exchange with `/log-decision`, and wait for the actuary.
