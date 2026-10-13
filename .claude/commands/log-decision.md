---
description: Append an exchange card to ai/ai_log.json for an exchange that changed a decision.
argument-hint: short title of the exchange
---
Append one entry to `ai/ai_log.json` for the exchange "$ARGUMENTS", following the schema at the top of that file. Fill question, ai_answer, challenge, test, result, decision, effect, section and role from this conversation. Quote the actuary's words where possible. If the AI was wrong and the test caught it, say so plainly in ai_answer and result. Keep each field to one or two sentences. Do not invent a challenge or a test that did not happen; leave the field empty instead.
