---
title: Testing tool-calling agents
---
For agents, assert on actions, not just the final text. Check which tools were called, with which arguments, in what order and how many times. Validate tool arguments against their schemas before execution.

Every loop needs an exit. Cap iterations, tool calls and total model calls, and record why the run stopped. A common bug is a loop whose stop rule counts something that does not grow on every pass, for example a counter that only increases when the model calls a tool; if the model answers in prose instead, the loop never ends.

Test error paths: a tool that times out, returns an error or returns empty results. The agent should recover or stop gracefully, not loop or invent results.
