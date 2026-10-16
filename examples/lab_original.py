"""The lab's Reflexion graph, run offline, and the problems this repo fixes.

    python examples/lab_original.py
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from reflexion_agent import KnowledgeBase, get_llm
from reflexion_agent.lab_pattern import run_lab

kb = KnowledgeBase()
QUESTION = "How do I test my chatbot for prompt injection?"
searched = set()


def search(query):
    hits = kb.search(query, 3)
    searched.update(h["source"] for h in hits)
    return hits


print("1) The stop rule counts ToolMessages, so every run is 4 rounds:")
msgs = run_lab(get_llm("offline"), search, QUESTION)
calls = [m for m in msgs if getattr(m, "tool_calls", None)]
print(f"   {len(msgs)} messages, {len(calls)} model calls, {sum(m.type == 'tool' for m in msgs)} search rounds "
      f"(this repo: 3 model calls when the answer is complete)\n")

print("2) Citations are never checked - a made-up source in a revision passes:")
noisy = run_lab(get_llm("offline", noisy=True), search, QUESTION)
first_revision = [m for m in noisy if getattr(m, "tool_calls", None)][1].tool_calls[0]["args"]
fake = [r for r in first_revision["references"] if r not in searched]
print(f"   revision 1 references: {first_revision['references']}")
print(f"   never retrieved: {fake} -> nothing flagged it\n")

print("3) Every raw search result stays in the history; the prompt grows each round:")
sizes, total = [], 0
for m in msgs:
    total += len(str(m.content)) + len(str(getattr(m, "tool_calls", "")))
    if m.type == "ai" and len(sizes) < 5:
        sizes.append(total)
print(f"   conversation size (characters) after each model reply: {sizes}\n")


class ProseReviser(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


print("4) If the revisor answers in prose, the loop never ends:")
first = AIMessage(content="", tool_calls=[{"name": "AnswerQuestion", "id": "c0", "args": {
    "answer": "draft", "reflection": {"missing": "x", "superfluous": "y"}, "search_queries": ["prompt injection"]}}])
model_calls = []
replies = iter([first] + [AIMessage(content="Here is my final answer in plain text.")] * 1000)


class Counting(ProseReviser):
    def _generate(self, *a, **k):
        model_calls.append(1)
        return super()._generate(*a, **k)


try:
    run_lab(Counting(messages=replies), search, QUESTION, recursion_limit=50)
except GraphRecursionError:
    print(f"   GraphRecursionError after {len(model_calls)} model calls (recursion_limit=50 here; LangGraph 1.x's "
          f"default is 10007, so about 5,000 paid calls)")
