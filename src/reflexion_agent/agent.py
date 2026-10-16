"""The Reflexion agent as a LangGraph 1.x StateGraph.

    START -> respond -> execute_tools -> revise -> check -+-> done                 (checks pass, nothing missing)
                             ^                            +-> out_of_budget        (max_revisions used)
                             |                            +-> nudge -> revise      (model answered in prose)
                             +----------------------------+   (checks failed or more to research)

- ``respond``        first draft + self-critique + search queries (``AnswerQuestion`` tool call)
- ``execute_tools``  runs the queries on the search source; trimmed results (and any check feedback)
                     go back as one ``ToolMessage`` per tool call
- ``revise``         a revised answer with numbered citations (``ReviseAnswer`` tool call)
- ``check``          deterministic: references must be retrieved sources, citations must match, word limit
- ``nudge``          if the model replied in prose, ask it to call the tool (counts against the budget)

Unlike the lab, every pass through ``revise`` is counted, so the loop always ends.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from .checks import check_revision
from .knowledge import KnowledgeBase, Search
from .prompts import responder_prompt, revisor_prompt
from .schemas import AnswerQuestion, ReviseAnswer

SNIPPET_CHARS = 500
MAX_QUERIES = 3


def merge(left: Optional[dict], right: Optional[dict]) -> dict:
    return {**(left or {}), **(right or {})}


class ReflexionState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    max_revisions: int
    word_limit: int
    revisions: int
    llm_calls: int
    searches: int
    sources: Annotated[dict[str, str], merge]     # every source id retrieved so far -> title
    answer: str
    references: list[str]
    missing: str
    issues: list[str]
    stop_reason: str                                # "complete" | "max_revisions"


def tool_call(message: Any, name: str) -> Optional[dict]:
    for call in getattr(message, "tool_calls", None) or []:
        if call["name"] == name:
            return call
    return None


def build_reflexion_graph(llm: BaseChatModel, search: Optional[Search] = None, *, results_per_query: int = 3,
                          checkpointer: Any = None):
    search = search or KnowledgeBase()
    responder = responder_prompt | llm.bind_tools([AnswerQuestion], tool_choice="AnswerQuestion")
    revisors: dict[int, Any] = {}

    def revisor(limit: int):
        if limit not in revisors:
            revisors[limit] = revisor_prompt(limit) | llm.bind_tools([ReviseAnswer], tool_choice="ReviseAnswer")
        return revisors[limit]

    def respond(state: ReflexionState) -> dict:
        reply = responder.invoke({"messages": state["messages"]})
        call = tool_call(reply, "AnswerQuestion")
        return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1,
                "answer": (call or {}).get("args", {}).get("answer", reply.content if not call else "")}

    def execute_tools(state: ReflexionState) -> dict:
        last = state["messages"][-1]
        feedback = state.get("issues") or []
        messages, sources, searches = [], {}, state.get("searches", 0)
        for call in getattr(last, "tool_calls", None) or []:
            queries = [q for q in (call["args"].get("search_queries") or []) if isinstance(q, str) and q.strip()]
            results = {}
            for query in queries[:MAX_QUERIES]:                 # cap: the lab ran every query the model asked for
                hits = search.search(query, results_per_query)
                searches += 1
                results[query] = [{"source": h["source"], "title": h["title"], "content": h["content"][:SNIPPET_CHARS]}
                                  for h in hits]
                sources.update({h["source"]: h["title"] for h in hits})
            payload = {"results": results, **({"feedback": feedback} if feedback else {})}
            messages.append(ToolMessage(content=json.dumps(payload), tool_call_id=call["id"]))
        return {"messages": messages, "sources": sources, "searches": searches}

    def revise(state: ReflexionState) -> dict:
        limit = state.get("word_limit", 250)
        reply = revisor(limit).invoke({"messages": state["messages"]})
        return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1,
                "revisions": state.get("revisions", 0) + 1}

    def check(state: ReflexionState) -> dict:
        call = tool_call(state["messages"][-1], "ReviseAnswer")
        if call is None:
            return {"issues": ["[no_tool_call] Reply by calling the ReviseAnswer tool."]}
        try:
            parsed = ReviseAnswer.model_validate(call["args"])
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_url=False))
            return {"issues": [f"[invalid] {errors}"]}
        issues = check_revision(parsed.answer, parsed.references, state.get("sources", {}), state.get("word_limit", 250))
        return {"issues": issues, "answer": parsed.answer, "references": parsed.references,
                "missing": parsed.reflection.missing.strip()}

    def route(state: ReflexionState) -> str:
        issues = state.get("issues") or []
        complete = not issues and not state.get("missing")
        if complete:
            return "done"
        if state.get("revisions", 0) >= state.get("max_revisions", 2):
            return "out_of_budget"
        if issues and issues[0].startswith("[no_tool_call]"):
            return "nudge"
        return "research"

    def nudge(state: ReflexionState) -> dict:
        return {"messages": [HumanMessage(content="Reply by calling the ReviseAnswer tool with answer, reflection, "
                                                  "search_queries and references.")]}

    graph = StateGraph(ReflexionState)
    for name, fn in [("respond", respond), ("execute_tools", execute_tools), ("revise", revise), ("check", check),
                     ("nudge", nudge)]:
        graph.add_node(name, fn)
    graph.add_node("done", lambda s: {"stop_reason": "complete"})
    graph.add_node("out_of_budget", lambda s: {"stop_reason": "max_revisions"})
    graph.add_edge(START, "respond")
    graph.add_edge("respond", "execute_tools")
    graph.add_edge("execute_tools", "revise")
    graph.add_edge("revise", "check")
    graph.add_conditional_edges("check", route, {"done": "done", "out_of_budget": "out_of_budget",
                                                 "research": "execute_tools", "nudge": "nudge"})
    graph.add_edge("nudge", "revise")
    graph.add_edge("done", END)
    graph.add_edge("out_of_budget", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, max_revisions: int = 2, word_limit: int = 250) -> ReflexionState:
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    if max_revisions < 1:
        raise ValueError("max_revisions must be >= 1")
    return {"messages": [HumanMessage(content=question)], "question": question, "max_revisions": max_revisions,
            "word_limit": word_limit, "revisions": 0, "llm_calls": 0, "searches": 0, "sources": {}, "issues": []}


def research(app, question: str, max_revisions: int = 2, word_limit: int = 250, thread_id: str = "default") -> ReflexionState:
    """Run the loop to the end and return the final state."""
    return app.invoke(initial_state(question, max_revisions, word_limit),
                      {"configurable": {"thread_id": thread_id}, "recursion_limit": 100})


def draft_answer(state: ReflexionState) -> str:
    """The first (un-researched) answer, for before/after comparisons."""
    for m in state.get("messages", []):
        call = tool_call(m, "AnswerQuestion")
        if call:
            return call["args"].get("answer", "")
    return ""


def format_answer(state: ReflexionState) -> str:
    """The final answer with a References section built from the validated references."""
    refs = state.get("references") or []
    sources = state.get("sources", {})
    lines = [state.get("answer", "").strip()]
    if refs:
        lines += ["", "References:"] + [f"[{i}] {r} - {sources.get(r, 'not retrieved!')}" for i, r in enumerate(refs, 1)]
    return "\n".join(lines)
