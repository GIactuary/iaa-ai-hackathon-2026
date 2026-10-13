# Notebook

The final Kaggle notebook follows the IAA template headings (summary, generative AI integration, step-by-step analysis, findings, reproducibility) with the filing structure (`docs/FILING_TEMPLATE.md`) inside section 3. It is built, not written by hand:

```
python notebooks/build_notebook.py --evidence outputs/evidence.json [--execute] [--strict]
```

- Exchange cards come from `ai/ai_log.json` (the `entries` list); each card appears in the section its `section` field names (2, 3.1 to 3.10, 4 or 5).
- Numbers in the text are tokens such as `{{summary.proposed_cv|.4f}}`, filled from the evidence file. Missing values show as `[pending: ...]`.
- The `src/` modules are embedded with `%%writefile`, so the notebook runs the same tested code on Kaggle.
- Data-specific cells are marked PLACEHOLDER. `--strict` refuses to build while any remain; use it for the submission build.
- The last cell compares the run's evidence with the numbers quoted in the text.

`reference/2025/notebook_kit/build_notebook.py` is the 2025 version.
