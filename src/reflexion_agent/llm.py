"""Provider switch: Gemini (free key, default), OpenAI, or an offline stand-in.

``OfflineResearcher`` is **not an LLM**. It plays the two roles by rules:

- bound to ``AnswerQuestion`` it writes a short, generic, uncited draft, a critique that says
  what is missing, and search queries built from the question;
- bound to ``ReviseAnswer`` it builds an answer from the passages in the ``ToolMessage`` search
  results, one sentence per source with ``[n]`` citations, and lists those source ids as references.
  The first revision still reports something missing (so a second research round runs), the second
  reports nothing missing.

``noisy=True`` makes the first revision cite a source that was never retrieved and a citation number
with no reference - the kind of fabricated citation real models produce - so the checks and the
feedback loop can be seen working. That behaviour is scripted, not learned.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .checks import word_count
from .knowledge import tokens

PROVIDERS = ("gemini", "openai", "offline")
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}
FAKE_SOURCE = "kb://2024-study-on-llm-testing"


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, noisy: bool = False,
            temperature: float = 0.2, **kwargs: Any) -> BaseChatModel:
    """Build the chat model. ``provider`` defaults to $LLM_PROVIDER, then "gemini".

    - gemini:  free API key from Google AI Studio in $GOOGLE_API_KEY ($GEMINI_MODEL)
    - openai:  $OPENAI_API_KEY ($OPENAI_MODEL), billed per token
    - offline: rule-based stand-in, no key, no network, no cost (``noisy`` only applies here)
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
                                      temperature=temperature, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
                          temperature=temperature, **kwargs)
    if provider == "offline":
        return OfflineResearcher(noisy=noisy)
    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def _sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def _topic(question: str) -> str:
    return question.strip().rstrip("?.!").strip()


class OfflineResearcher(BaseChatModel):
    """Rule-based responder + revisor for offline runs. Not an LLM."""

    noisy: bool = False
    bound: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "offline-rules"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "OfflineResearcher":
        return self.model_copy(update={"bound": [convert_to_openai_tool(t)["function"]["name"] for t in tools]})

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        question = str(humans[0].content) if humans else ""
        if "ReviseAnswer" in self.bound:
            name, args = "ReviseAnswer", self._revise(question, messages)
        elif "AnswerQuestion" in self.bound:
            name, args = "AnswerQuestion", self._respond(question)
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="(offline mode needs a bound tool)"))])
        call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])

    # ---------------------------------------------------------------- roles

    @staticmethod
    def _respond(question: str) -> dict:
        topic = _topic(question)
        keywords = [t for t in tokens(topic) if len(t) > 3][:3]
        queries = [topic] + ([" ".join(keywords) + " testing"] if keywords else [])
        return {
            "answer": (f"{topic}? Start by listing the main risks, write a few automated tests for them and run the "
                       "tests on every change. Review failures regularly and add new cases when something breaks. "
                       "This first draft is general and has no sources yet."),
            "reflection": {"missing": f"Concrete techniques, pass criteria and sources for: {topic}.",
                           "superfluous": "Generic advice that applies to any software project."},
            "search_queries": queries[:2],
        }

    def _revise(self, question: str, messages: list[BaseMessage]) -> dict:
        passages: list[tuple[str, str]] = []          # (source, text), first-seen order, no duplicates
        feedback: list[str] = []
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    payload = json.loads(m.content)
                except (TypeError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                feedback = payload.get("feedback", feedback)
                results = payload["results"] if "results" in payload else payload   # the lab's format: {query: hits}
                for hits in results.values():
                    if not isinstance(hits, list):
                        continue
                    for hit in hits:
                        if isinstance(hit, dict) and (hit["source"], hit["content"]) not in passages:
                            passages.append((hit["source"], hit["content"]))
        round_no = 1 + sum(1 for m in messages if isinstance(m, AIMessage)
                           and any(c["name"] == "ReviseAnswer" for c in m.tool_calls or []))
        chosen = passages[: (2 if round_no == 1 else 3)]
        refs: list[str] = []
        sentences = []
        for source, text in chosen:
            if source not in refs:
                refs.append(source)
            sentences.append(f"{_sentences(text, 2)} [{refs.index(source) + 1}]")
        if self.noisy and round_no == 1 and not feedback:
            sentences.append(f"A recent study confirms this approach [{len(refs) + 2}].")
            refs.append(FAKE_SOURCE)
        while sentences and word_count(" ".join(sentences)) > 230:
            sentences.pop()
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", " ".join(sentences))]
        refs = refs[: max(cited, default=0)]               # drop references whose sentences were cut
        topic = _topic(question)
        more = f"How to automate these checks in CI for: {topic}"
        return {
            "answer": " ".join(sentences) or "The search results do not cover this question.",
            "reflection": {"missing": "" if round_no >= 2 else more,
                           "superfluous": "" if round_no >= 2 else "Some background could be shorter."},
            "search_queries": [] if round_no >= 2 else [f"{topic} regression CI"],
            "references": refs,
        }
