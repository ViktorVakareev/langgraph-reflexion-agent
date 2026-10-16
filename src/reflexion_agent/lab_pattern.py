"""The lab's Reflexion graph, rebuilt faithfully so its behaviour can be shown and tested.

Kept as in the lab: the shared prompt template with a trailing system message, ``AnswerQuestion`` /
``ReviseAnswer`` bound as tools, ``execute_tools`` that runs every query and dumps the raw results into
a ``ToolMessage``, ``MessageGraph`` (deprecated in LangGraph 1.0) and ``event_loop`` with
``MAX_ITERATIONS = 4``.

Changed: the search is injected (the lab used Tavily, a paid API) and the persona is neutral. The lab
role-played two real, named public figures giving one-sided diet advice to a user who says they are
pre-diabetic with heart problems; that isn't reproduced here.

Confirmed on LangGraph 1.2:

1. The loop always runs 4 search rounds and 5 model calls, whatever the quality: ``event_loop`` only
   counts ``ToolMessage``s.
2. If the revisor answers in prose (no tool call), ``execute_tools`` adds no ``ToolMessage``, the count
   never grows, and the graph calls the revisor again and again until ``GraphRecursionError`` - a paid
   model call on every pass.
3. Citations are requested but never checked: a reference that was never retrieved passes.
4. Every raw search result stays in the history, so each revision sends a longer prompt.
5. Without a path map on the conditional edge, the drawn graph hides the loop.
"""

from __future__ import annotations

import json
import warnings
from typing import Callable, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END

from .schemas import AnswerQuestion, ReviseAnswer

MAX_ITERATIONS = 4

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior QA engineer who specialises in testing LLM applications.

        Your response must follow these steps:
        1. {first_instruction}
        2. Present the reasoning behind your advice.
        3. Reflect and critique your answer.
        4. After the reflection, **list 1-3 search queries separately** for researching the topic. Do not include them inside the reflection.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's question above using the required format."
    ),
])

first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
revise_instructions = """Revise your previous answer using the new information.
- Incorporate the previous critique.
- You MUST include numerical citations.
- Add a "References" section to the bottom of your answer in the form of:
- [1] https://example.com
- Keep response under 250 words with precision over volume.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)


def build_lab_graph(llm: BaseChatModel, search: Callable[[str], object], path_map: bool = False):
    """The lab's graph. ``search(query)`` replaces ``tavily_tool.invoke(query)``.

    The lab calls ``add_conditional_edges("revisor", event_loop)`` without a path map, so the drawn graph
    shows ``revisor --> END`` and hides the loop. ``path_map=True`` adds one (same behaviour).
    """
    initial_chain = first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])
    revisor_chain = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])

    def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
        last_ai_message = state[-1]
        tool_messages = []
        for tool_call in last_ai_message.tool_calls:
            if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
                call_id = tool_call["id"]
                search_queries = tool_call["args"].get("search_queries", [])
                query_results = {}
                for query in search_queries:
                    result = search(query)
                    query_results[query] = result
                tool_messages.append(ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id)
                )
        return tool_messages

    def event_loop(state: List[BaseMessage]) -> str:
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
        num_iterations = count_tool_visits
        if num_iterations >= MAX_ITERATIONS:
            return END
        return "execute_tools"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.graph import MessageGraph

        graph = MessageGraph()
    graph.add_node("respond", initial_chain)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revisor", revisor_chain)
    graph.add_edge("respond", "execute_tools")
    graph.add_edge("execute_tools", "revisor")
    if path_map:
        graph.add_conditional_edges("revisor", event_loop, {"execute_tools": "execute_tools", END: END})
    else:
        graph.add_conditional_edges("revisor", event_loop)
    graph.set_entry_point("respond")
    return graph.compile()


def run_lab(llm: BaseChatModel, search: Callable[[str], object], question: str, recursion_limit: int = 25) -> list:
    """``app.invoke(question)`` as in the lab; returns the list of messages."""
    return build_lab_graph(llm, search).invoke(HumanMessage(content=question), {"recursion_limit": recursion_limit})


def final_answer(responses: list) -> str:
    """The lab's way to read the result: the last message's tool call (IndexError if there is none)."""
    return responses[-1].tool_calls[0]["args"]["answer"]
