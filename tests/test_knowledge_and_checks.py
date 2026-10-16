"""Local search and the deterministic citation checks."""

import json

import pytest

from reflexion_agent import KnowledgeBase, check_revision, word_count
from reflexion_agent.knowledge import tokens

RETRIEVED = {"kb://a": "A", "kb://b": "B"}


def test_every_question_retrieves_its_expected_note():
    kb = KnowledgeBase()
    for line in open("data/questions.jsonl", encoding="utf-8"):
        q = json.loads(line)
        assert q["expected_sources"][0] in [h["source"] for h in kb.search(q["question"], 3)], q["id"]


def test_search_shape_and_no_results():
    kb = KnowledgeBase()
    hits = kb.search("canary tokens system prompt leak", 2)
    assert len(hits) <= 2 and set(hits[0]) == {"source", "title", "content", "score"}
    assert hits[0]["source"] == "kb://prompt-injection-testing"
    assert kb.search("zzzz qqqq") == []
    assert len(kb.sources) == 12


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeBase(tmp_path)


def test_tokens_drop_stopwords_and_plurals():
    assert tokens("How do I test the Agents?") == ["test", "agent"]


def test_good_revision_passes():
    assert check_revision("Use fakes [1]. Pin versions [2].", ["kb://a", "kb://b"], RETRIEVED) == []


@pytest.mark.parametrize("answer, refs, code", [
    ("Use fakes [1].", ["kb://a", "kb://made-up"], "[unknown_source]"),
    ("Use fakes [3].", ["kb://a"], "[bad_citation]"),
    ("Use fakes.", ["kb://a"], "[uncited]"),
    ("Use fakes [1].", [], "[no_references]"),
    ("Use fakes [1] [2].", ["kb://a", "kb://a"], "[duplicate_reference]"),
    ("   ", ["kb://a"], "[empty]"),
])
def test_each_problem_is_reported(answer, refs, code):
    assert any(i.startswith(code) for i in check_revision(answer, refs, RETRIEVED))


def test_word_limit_ignores_citation_numbers():
    text = " ".join(["word"] * 10) + " [1]"
    assert check_revision(text, ["kb://a"], RETRIEVED, word_limit=10) == []
    assert check_revision(text, ["kb://a"], RETRIEVED, word_limit=9)[0].startswith("[too_long]")
    assert word_count("it's a well-known test") == 4
