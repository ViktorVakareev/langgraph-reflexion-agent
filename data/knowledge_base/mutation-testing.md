---
title: Mutation testing
---
Mutation testing measures how good a test suite is, rather than how much code it executes. A tool such as mutmut changes the code in small ways (flipping a comparison, changing a constant, deleting a line) and runs the tests. If the tests still pass, the mutant survived, which reveals a missing or weak assertion.

High line coverage with many surviving mutants means tests execute code without checking results. For LLM applications, mutate the guardrails and validators: if the tests still pass when a safety check is removed, the safety suite is not really testing it.
