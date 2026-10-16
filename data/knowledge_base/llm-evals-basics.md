---
title: LLM evaluation basics
---
An eval is a repeatable test of model or application quality on a fixed dataset. Start with a golden dataset of realistic inputs and expected outputs, including edge cases and known failures from production.

Choose the scorer to match the task. Exact match and schema validation suit structured output. Numeric tolerance suits calculations. Reference-based metrics such as F1 or ROUGE are cheap but miss meaning. Semantic similarity and LLM-as-judge handle free text but need calibration.

Track metrics per slice (topic, language, difficulty), not only the overall average, because regressions often hide in one slice. Keep the dataset versioned next to the code and run the eval in CI on every change to prompts, models or retrieval.

Separate validity from correctness: output can match the schema and still be wrong, so score both.
