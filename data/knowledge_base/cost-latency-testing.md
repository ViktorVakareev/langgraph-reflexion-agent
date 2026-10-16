---
title: Cost and latency testing
---
LLM features have budgets as well as quality targets. Measure tokens in and out, the number of model calls per request and latency percentiles (p50, p95) for each workflow.

Iterative agents multiply cost: every reflection or tool round adds model calls and usually grows the prompt, because the whole history is sent again. Cap the number of rounds, trim tool results before adding them to the history, and alert when calls per request exceed the expected maximum.

Add cost checks to CI with a fake model that counts calls, so a change that adds an extra round is caught before it reaches production.
