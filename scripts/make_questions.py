"""Write data/questions.jsonl: research questions and the knowledge-base notes a good answer should cite.

    python scripts/make_questions.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    ("How do I test my chatbot for prompt injection?", ["prompt-injection-testing"]),
    ("How can I trust an LLM-as-judge score?", ["llm-as-judge"]),
    ("My LLM tests are flaky. How do I make them reliable?", ["flaky-tests-nondeterminism"]),
    ("How do I unit test a LangChain agent without calling a paid API?", ["mocking-llm-calls-pytest"]),
    ("How should I evaluate a RAG system's answers?", ["rag-evaluation"]),
    ("How do I catch regressions when we change the prompt or model?", ["regression-testing-llm-apps"]),
    ("What should I assert on when testing a tool-calling agent?", ["testing-tool-calling-agents"]),
    ("How do I validate LLM output before using it in code?", ["output-validation-guardrails"]),
    ("How can I tell if my test suite really checks the guardrails?", ["mutation-testing"]),
    ("How do I keep the cost of an iterative agent under control?", ["cost-latency-testing"]),
    ("What belongs in a red teaming safety suite?", ["red-teaming-safety-tests"]),
    ("How do I build a golden dataset for evals?", ["llm-evals-basics"]),
]

lines = [json.dumps({"id": f"q{i + 1:02d}", "question": q, "expected_sources": [f"kb://{s}" for s in srcs]})
         for i, (q, srcs) in enumerate(QUESTIONS)]
(ROOT / "data" / "questions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote data/questions.jsonl ({len(lines)} questions)")
