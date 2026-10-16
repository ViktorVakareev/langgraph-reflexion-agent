---
title: Flaky tests and non-determinism
---
LLM outputs are non-deterministic even at temperature 0 because of sampling details, batching and provider-side changes. Tests that assert on exact wording become flaky.

Make unit tests deterministic by replacing the model with a fake: scripted responses, a fixed seed, or recorded responses replayed from cassettes. Keep real-model tests in a separate, clearly marked suite.

For real-model tests, assert on properties that must always hold (valid JSON, the right tool called, required facts present, no forbidden content) and use statistical thresholds: run N times and require a pass rate. Retrying until a test passes hides real problems; track retry counts and flakiness rates as metrics instead.
