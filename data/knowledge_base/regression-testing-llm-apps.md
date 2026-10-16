---
title: Regression testing LLM applications
---
LLM applications regress when the prompt, the model version, the retrieval index or a tool changes. Pin model versions explicitly instead of using floating aliases, and record the version with every eval run.

Keep a regression suite of past failures: every bug found in production becomes a test case. Compare each candidate against the current baseline on the same dataset and block the release when a key metric drops beyond an agreed threshold.

Because outputs vary, compare distributions rather than single runs: run each case several times, and gate on pass rates with a margin rather than on one sample. Store full traces (inputs, retrieved context, tool calls, outputs) so a failing case can be reproduced and debugged.
