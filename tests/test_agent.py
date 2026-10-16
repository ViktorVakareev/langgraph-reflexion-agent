"""The Reflexion graph: rounds, stop rules, citation checks, prose replies, budgets - offline and scripted."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from reflexion_agent import (KnowledgeBase, build_reflexion_graph, draft_answer, format_answer, get_llm, initial_state,
                             research)
from reflexion_agent.llm import FAKE_SOURCE

Q = "How do I test my chatbot for prompt injection?"


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def answer_call(i=0, queries=("prompt injection",)):
    return AIMessage(content="", tool_calls=[{"name": "AnswerQuestion", "id": f"a{i}", "args": {
        "answer": "Draft.", "reflection": {"missing": "sources", "superfluous": ""}, "search_queries": list(queries)}}])


def revise_call(i, answer="Use canary tokens [1].", refs=("kb://prompt-injection-testing",), missing="", queries=()):
    return AIMessage(content="", tool_calls=[{"name": "ReviseAnswer", "id": f"r{i}", "args": {
        "answer": answer, "reflection": {"missing": missing, "superfluous": ""}, "search_queries": list(queries),
        "references": list(refs)}}])


def run(*replies, **kw):
    return research(build_reflexion_graph(Scripted(messages=iter(replies))), Q, **kw)


def test_offline_run_completes_in_two_revisions():
    s = research(build_reflexion_graph(get_llm("offline")), Q)
    assert s["stop_reason"] == "complete" and s["revisions"] == 2 and s["llm_calls"] == 3 and s["searches"] == 3
    assert s["references"] == ["kb://prompt-injection-testing"] and s["issues"] == []
    assert "References:\n[1] kb://prompt-injection-testing - Testing for prompt injection" in format_answer(s)
    assert "no sources yet" in draft_answer(s)


def test_message_order_pairs_every_tool_call_with_a_tool_message():
    s = research(build_reflexion_graph(get_llm("offline")), Q)
    types = [type(m).__name__ for m in s["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert s["messages"][2].tool_call_id == s["messages"][1].tool_calls[0]["id"]


def test_noisy_made_up_source_is_caught_and_fixed():
    s = research(build_reflexion_graph(get_llm("offline", noisy=True)), Q)
    feedback = json.loads(s["messages"][4].content)["feedback"]
    assert any(f.startswith("[unknown_source]") and FAKE_SOURCE in f for f in feedback)
    assert any(f.startswith("[bad_citation]") for f in feedback)
    assert s["stop_reason"] == "complete" and FAKE_SOURCE not in s["references"]


def test_tool_results_are_trimmed():
    s = research(build_reflexion_graph(get_llm("offline")), Q)
    for hits in json.loads(s["messages"][2].content)["results"].values():
        assert all(len(h["content"]) <= 500 for h in hits)


def test_complete_after_one_revision_when_nothing_missing():
    s = run(answer_call(), revise_call(1))
    assert s["stop_reason"] == "complete" and s["revisions"] == 1 and s["llm_calls"] == 2


def test_budget_stops_a_model_that_always_wants_more():
    s = run(answer_call(), *(revise_call(i, missing="more", queries=["prompt injection"]) for i in range(1, 9)),
            max_revisions=3)
    assert s["stop_reason"] == "max_revisions" and s["revisions"] == 3 and s["llm_calls"] == 4


def test_prose_reply_is_nudged_and_bounded():
    prose = AIMessage(content="Here is my answer in plain text.")
    s = run(answer_call(), prose, revise_call(2), max_revisions=3)
    assert s["stop_reason"] == "complete" and s["revisions"] == 2
    assert any(isinstance(m, HumanMessage) and "ReviseAnswer tool" in m.content for m in s["messages"])


def test_prose_forever_ends_at_the_budget():
    s = run(answer_call(), *[AIMessage(content="plain text")] * 10, max_revisions=2)
    assert s["stop_reason"] == "max_revisions" and s["llm_calls"] == 3
    assert s["issues"][0].startswith("[no_tool_call]")


def test_invalid_arguments_are_reported():
    bad = AIMessage(content="", tool_calls=[{"name": "ReviseAnswer", "id": "r1", "args": {"answer": "x"}}])
    s = run(answer_call(), bad, max_revisions=1)
    assert s["stop_reason"] == "max_revisions" and s["issues"][0].startswith("[invalid]")


def test_unknown_source_from_a_real_model_is_never_accepted():
    s = run(answer_call(), *(revise_call(i, refs=["https://example.com/study"]) for i in range(1, 4)), max_revisions=2)
    assert s["stop_reason"] == "max_revisions" and s["issues"][0].startswith("[unknown_source]")


def test_too_many_queries_are_capped():
    queries = [f"prompt injection {w}" for w in ("canary", "jailbreak", "indirect", "base64", "tools")]
    s = run(answer_call(queries=queries[:3]), revise_call(1))
    assert s["searches"] == 3
    assert run(answer_call(queries=queries), revise_call(1))["searches"] == 3


def test_respond_without_tool_call_still_finishes():
    s = run(AIMessage(content="prose draft"), revise_call(1), max_revisions=1)
    assert s["answer"] == "Use canary tokens [1]." and s["searches"] == 0       # nothing was searched ...
    assert s["stop_reason"] == "max_revisions" and s["issues"][0].startswith("[unknown_source]")  # ... so no citation can pass


def test_initial_state_validation():
    assert initial_state(Q)["max_revisions"] == 2
    with pytest.raises(ValueError, match="empty"):
        initial_state("  ")
    with pytest.raises(ValueError, match="max_revisions"):
        initial_state(Q, max_revisions=0)


def test_word_limit_feedback():
    long = " ".join(["word"] * 30) + " [1]"
    s = run(answer_call(), revise_call(1, answer=long), revise_call(2), word_limit=20)
    assert s["stop_reason"] == "complete" and s["revisions"] == 2


def test_custom_search_source():
    class Fixed:
        def search(self, query, k=3):
            return [{"source": "kb://prompt-injection-testing", "title": "T", "content": "c", "score": 1}]
    s = research(build_reflexion_graph(Scripted(messages=iter([answer_call(), revise_call(1)])), Fixed()), Q)
    assert s["sources"] == {"kb://prompt-injection-testing": "T"}


def test_checkpoints_and_history(tmp_path):
    from reflexion_agent import get_checkpointer, history, mermaid
    db = str(tmp_path / "runs.sqlite")
    research(build_reflexion_graph(get_llm("offline"), checkpointer=get_checkpointer(db)), Q, thread_id="t")
    rows = history(build_reflexion_graph(get_llm("offline"), checkpointer=get_checkpointer(db)), "t")
    assert rows[-1]["next"] == [] and rows[-1]["values"]["stop_reason"] == "complete"
    text = mermaid(build_reflexion_graph(get_llm("offline"), KnowledgeBase()))
    assert all(n in text for n in ("respond", "execute_tools", "revise", "check", "nudge", "out_of_budget"))
