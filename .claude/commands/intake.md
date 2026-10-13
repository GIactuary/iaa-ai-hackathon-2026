---
description: Hour-one intake. Read the competition files, settle the brief, and list the judgments for the actuary.
---
Read everything in `competition/` and list `data/raw/`. Then write `competition/brief.md` with:

1. Task, target, metric (and what it implies for the target scale and GLM family, per `docs/PLAYBOOK.md`, "When the 2026 task is a different problem").
2. Window, deadline with time zone, deliverables, submission mechanics, team rules.
3. Judging criteria and weights, and how they differ from 2025 (`docs/JUDGING_AND_AI_EVIDENCE.md`).
4. Rules on external data, pretrained models, and LLM calls inside the notebook, quoted with section numbers.
5. Data origin: synthetic or real; if synthetic, the likely public source. Check by matching rows on numeric columns, as done for Ames in 2025.
6. Shapes, ID columns, any date or time column (if present, validation must hold out the latest periods).
7. Leakage paths and inputs known only after the event.
8. A candidate business story for an insurer and the intended use, with the costlier error.

Then open `decisions/judgment_register.md` with J1 (intended use) and J-ext (external data) at "Needs your call", and log this exchange with `/log-decision`. Stop and wait for the actuary.
