---
title: Mocking LLM calls in pytest
---
Unit tests should never call a paid API. In LangChain, a fake chat model such as GenericFakeChatModel returns scripted AIMessages, including tool calls, so the real agent loop runs offline and deterministically.

Give scripted tool calls unique ids, because each ToolMessage must reference the id of the call it answers. Script failure paths too: malformed arguments, a refusal, a text reply where a tool call was expected and a tool that raises.

Use pytest fixtures to build the fake model and monkeypatch to replace network clients. Mark tests that need real keys with a custom marker such as live and skip them when the key is missing, so CI stays free and fast while the live suite runs on demand.
