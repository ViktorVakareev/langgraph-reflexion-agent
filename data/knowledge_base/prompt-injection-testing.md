---
title: Testing for prompt injection
---
Prompt injection is untrusted text that tries to override an LLM application's instructions. Direct injection arrives in the user's message ("ignore previous instructions"); indirect injection hides in content the app reads, such as a web page, an email, a PDF or a tool result.

Build a regression suite of attack strings and run it on every prompt or model change. Cover role-play jailbreaks, instructions hidden in retrieved documents, payloads split across turns, encoded text (base64, homoglyphs) and requests to reveal the system prompt.

Assert on behaviour, not wording: the assistant must not call privileged tools, must not leak secrets or the system prompt, and must keep the original task. Canary tokens placed in the system prompt make leaks easy to detect with a simple string check.

Defence in depth matters more than any single filter: least-privilege tools, human approval for risky actions, separating instructions from data, and output validation before anything is executed.
