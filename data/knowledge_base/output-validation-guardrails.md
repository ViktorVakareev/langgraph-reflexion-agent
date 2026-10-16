---
title: Output validation and guardrails
---
Never pass model output straight into code, a database or another tool. Validate it first: parse it into a typed schema (for example a Pydantic model with enums, ranges and patterns), reject extra fields and check cross-field rules.

When validation fails, a repair loop can send the validation errors back to the model and ask for a corrected answer within a small attempt budget. If it still fails, fall back to a safe default or a human.

Guardrails also cover content: allow-lists for actions, blocking personal data in outputs, and length limits. Log every rejected output; the log is a ready-made source of new test cases.
