---
description: Run the independent reviewer on the current notebook and record what it found.
---
Launch the `independent-reviewer` subagent on the current notebook. When it reports, write its findings to `outputs/review_<date>.md`, log the review as an exchange with `/log-decision`, and propose a fix for each blocking or material finding. Do not apply fixes that change a decided judgment without the actuary.
