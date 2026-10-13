---
description: Phase 2 screens. Outlier screen on training records and the train-versus-test evidence check, before any model.
---
1. Fit a quick line of the target (on the metric's scale) on the main size, quality and location inputs. List records more than 4 standard deviations off. For each, show its raw fields, its fitted value, and your reading of why. Recommend exclude or keep with a reason; looking unusual is not a reason.
2. For every test record, compare against the training evidence: size beyond 1.25 times its segment's largest training value, category levels never seen in training, impossible combinations. Write the list to `decisions/referrals_v0.md`.
3. Cross-reference: which test records resemble any record you recommend excluding? Those lose their only evidence if it is excluded.
4. Write the thresholds you used into `decisions/thresholds.md` before reporting results.

Present the tables, put the exclusion and referral decisions in the judgment register at "Needs your call", log the exchange with `/log-decision`, and wait for the actuary.
