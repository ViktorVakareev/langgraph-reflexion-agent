"""The offline stand-in: rules only, deterministic, never an API call."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from reflexion_agent import AnswerQuestion, OfflineResearcher, ReviseAnswer, get_llm
from reflexion_agent.llm import FAKE_SOURCE

Q = "How do I test my chatbot for prompt injection?"
HITS = {"results": {"q": [{"source": "kb://a", "title": "A", "content": "First fact. Second fact. Third."},
                          {"source": "kb://b", "title": "B", "content": "Other fact."},
                          {"source": "kb://a", "title": "A", "content": "More from A."}]}}


def revise(model, extra=()):
    msgs = [HumanMessage(Q), AIMessage(content="", tool_calls=[{"name": "AnswerQuestion", "args": {}, "id": "c0"}]),
            ToolMessage(json.dumps(HITS), tool_call_id="c0"), *extra]
    return model.bind_tools([ReviseAnswer]).invoke(msgs).tool_calls[0]


def test_responder_drafts_without_sources():
    call = get_llm("offline").bind_tools([AnswerQuestion]).invoke([HumanMessage(Q)]).tool_calls[0]
    assert call["name"] == "AnswerQuestion" and "[1]" not in call["args"]["answer"]
    assert call["args"]["reflection"]["missing"] and 1 <= len(call["args"]["search_queries"]) <= 2


def test_first_revision_cites_retrieved_passages_and_wants_more():
    args = revise(get_llm("offline"))["args"]
    assert args["references"] == ["kb://a", "kb://b"] and "[1]" in args["answer"] and "[2]" in args["answer"]
    assert args["reflection"]["missing"] and args["search_queries"]


def test_second_revision_is_complete():
    first = AIMessage(content="", tool_calls=[{"name": "ReviseAnswer", "args": {}, "id": "r1"}])
    args = revise(get_llm("offline"), [first, ToolMessage(json.dumps(HITS), tool_call_id="r1")])["args"]
    assert args["reflection"]["missing"] == "" and args["search_queries"] == [] and args["answer"].count("[1]") == 2


def test_noisy_adds_a_made_up_source_once():
    args = revise(get_llm("offline", noisy=True))["args"]
    assert args["references"][-1] == FAKE_SOURCE and f"[{len(args['references']) + 1}]" in args["answer"]
    fb = {"results": {}, "feedback": ["[unknown_source] x"]}
    fixed = revise(get_llm("offline", noisy=True), [ToolMessage(json.dumps(fb), tool_call_id="c0")])["args"]
    assert FAKE_SOURCE not in fixed["references"]


def test_reads_the_lab_payload_format_and_ignores_junk():
    lab = {"q": HITS["results"]["q"], "not-a-list": 3}
    msgs = [HumanMessage(Q), ToolMessage(json.dumps(lab), tool_call_id="x"), ToolMessage("not json", tool_call_id="y"),
            ToolMessage("[1, 2]", tool_call_id="z")]
    assert get_llm("offline").bind_tools([ReviseAnswer]).invoke(msgs).tool_calls[0]["args"]["references"][0] == "kb://a"


def test_no_results_and_no_tool():
    args = get_llm("offline").bind_tools([ReviseAnswer]).invoke([HumanMessage(Q)]).tool_calls[0]["args"]
    assert args["references"] == [] and "do not cover" in args["answer"]
    assert "needs a bound tool" in OfflineResearcher().invoke("hi").content


def test_long_passages_are_trimmed_to_the_word_budget():
    long = {"results": {"q": [{"source": f"kb://{i}", "title": "", "content": "word " * 120 + "."} for i in range(3)]}}
    msgs = [HumanMessage(Q), ToolMessage(json.dumps(long), tool_call_id="c0")]
    args = get_llm("offline").bind_tools([ReviseAnswer]).invoke(msgs).tool_calls[0]["args"]
    assert len(args["references"]) == 1


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_llm("watsonx")


@pytest.mark.parametrize("provider, cls", [("gemini", "ChatGoogleGenerativeAI"), ("openai", "ChatOpenAI")])
def test_real_providers_build_without_calling_the_api(provider, cls, monkeypatch):
    pytest.importorskip("langchain_google_genai" if provider == "gemini" else "langchain_openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    assert type(get_llm(provider)).__name__ == cls
