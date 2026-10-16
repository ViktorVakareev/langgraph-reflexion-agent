"""The lab's graph and its confirmed gotchas (LangGraph 1.2)."""

import warnings

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from reflexion_agent import KnowledgeBase, get_llm
from reflexion_agent.lab_pattern import final_answer, run_lab
from reflexion_agent.llm import FAKE_SOURCE

Q = "How do I test my chatbot for prompt injection?"
KB = KnowledgeBase()


def search(query):
    return KB.search(query, 3)


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_always_four_rounds_and_five_model_calls():
    msgs = run_lab(get_llm("offline"), search, Q)
    assert [m.type for m in msgs] == ["human"] + ["ai", "tool"] * 4 + ["ai"]
    assert final_answer(msgs)


def test_made_up_citation_is_not_checked():
    msgs = run_lab(get_llm("offline", noisy=True), search, Q)
    revision1 = [m for m in msgs if m.type == "ai"][1].tool_calls[0]["args"]
    assert FAKE_SOURCE in revision1["references"]        # accepted silently; the run carries on


def test_prose_reply_loops_until_recursion_error():
    first = AIMessage(content="", tool_calls=[{"name": "AnswerQuestion", "id": "c0", "args": {
        "answer": "d", "reflection": {"missing": "m", "superfluous": "s"}, "search_queries": ["q"]}}])
    model = Scripted(messages=iter([first] + [AIMessage(content="plain text")] * 100))
    with pytest.raises(GraphRecursionError):
        run_lab(model, search, Q, recursion_limit=30)


def test_final_answer_index_error_without_tool_call():
    with pytest.raises(IndexError):
        final_answer([AIMessage(content="prose")])


def test_message_graph_is_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from langgraph.graph import MessageGraph
        MessageGraph()
    assert any("MessageGraph is deprecated" in str(w.message) for w in caught)


def test_tavily_search_results_is_deprecated():
    """Skipped unless langchain-community is installed (it isn't a dependency of this repo)."""
    community = pytest.importorskip("langchain_community.tools.tavily_search")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        community.TavilySearchResults(max_results=1, tavily_api_key="not-a-real-key")
    assert any("langchain-tavily" in str(w.message) for w in caught)


def test_drawing_without_a_path_map_hides_the_loop():
    from reflexion_agent import mermaid
    from reflexion_agent.lab_pattern import build_lab_graph
    plain = mermaid(build_lab_graph(get_llm("offline"), search))
    mapped = mermaid(build_lab_graph(get_llm("offline"), search, path_map=True))
    assert "revisor -.-> execute_tools" not in plain and "revisor --> __end__" in plain
    assert "revisor -.-> execute_tools" in mapped
