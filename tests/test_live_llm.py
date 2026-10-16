"""Live evaluation with a real LLM - skipped unless an API key is set.  pytest -m live -v

Scored on facts the code can verify: every reference was actually retrieved, the expected note is
cited, the answer is within the word limit, and the loop stays within its budget.
"""

import os

import pytest

from reflexion_agent import evaluate, get_llm, load_questions, summarize

PROVIDERS = [
    pytest.param("gemini", marks=pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")),
    pytest.param("openai", marks=pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
]
pytestmark = pytest.mark.live


@pytest.mark.parametrize("provider", PROVIDERS)
def test_citations_are_real_and_on_topic(provider):
    rows = evaluate(get_llm(provider), load_questions()[:6])
    s = summarize(rows)
    assert s["cited_ok"] == 1.0                       # the checks + feedback loop must enforce this
    assert s["expected"] >= 0.8
    assert all(r.llm_calls <= 3 for r in rows)        # 1 draft + max_revisions (2)
