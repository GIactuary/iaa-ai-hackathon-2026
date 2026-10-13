# Thresholds, written before the tests that use them

| Threshold | Value | Used in | Written on | Rationale |
|---|---|---|---|---|
| Outlier screen | 4 standard deviations from a quick fitted line | /screen | | 2025 practice |
| Out-of-evidence referral | 1.25 times the segment's largest training value | /screen, referral rule | | 2025 practice |
| Model-disagreement referral | GLM and tree more than 30% apart | referral rule | | 2025 practice |
| Feature kept | Removing it worsens error on all 3 seeds | drop-one test | | 2025 practice |
| GLM variable entry | Cross-validated gain of at least 0.2% | forward GLM | | 2025 practice |
| GLM level pooling | Levels with fewer than 5 records pooled | forward GLM | | 2025 practice |
| Hold-out | 20%, seed fixed, opened once | validation | | 2025 practice |
