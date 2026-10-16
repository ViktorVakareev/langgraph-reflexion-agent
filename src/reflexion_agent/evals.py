"""Score research runs on data/questions.jsonl - measured facts only, never wording.

- **complete**     the loop ended because the answer passed the checks and nothing was missing
- **cited ok**     every reference is a source the search actually returned (citation precision = 1)
- **expected**     the note a good answer should cite is among the references
- **in limit**     the answer is within the word limit
- **model calls / searches** what the run cost
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from .agent import build_reflexion_graph, research
from .checks import CITATION, word_count
from .knowledge import KnowledgeBase
from .lab_pattern import run_lab

DEFAULT_QUESTIONS = Path("data") / "questions.jsonl"


@dataclass
class Row:
    id: str
    complete: bool
    cited_ok: bool
    expected: bool
    in_limit: bool
    llm_calls: int
    searches: int
    references: list[str]


def load_questions(path: Optional[str | Path] = None) -> list[dict]:
    return [json.loads(line) for line in Path(path or DEFAULT_QUESTIONS).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evaluate(llm: BaseChatModel, questions: list[dict], word_limit: int = 250, max_revisions: int = 2,
             kb: Optional[KnowledgeBase] = None) -> list[Row]:
    kb = kb or KnowledgeBase()
    app = build_reflexion_graph(llm, kb)
    rows = []
    for q in questions:
        s = research(app, q["question"], max_revisions, word_limit, thread_id=q["id"])
        refs = s.get("references") or []
        retrieved = s.get("sources", {})
        rows.append(Row(q["id"], s.get("stop_reason") == "complete", bool(refs) and all(r in retrieved for r in refs),
                        any(e in refs for e in q["expected_sources"]),
                        word_count(CITATION.sub("", s.get("answer", ""))) <= word_limit,
                        s.get("llm_calls", 0), s.get("searches", 0), refs))
    return rows


def evaluate_lab(llm: BaseChatModel, questions: list[dict], word_limit: int = 250,
                 kb: Optional[KnowledgeBase] = None) -> list[Row]:
    """The lab's graph on the same questions. 'complete' has no meaning there: it always runs 4 rounds."""
    kb = kb or KnowledgeBase()
    rows = []
    for q in questions:
        retrieved: dict[str, str] = {}
        searches = 0

        def search(query: str) -> list[dict]:
            nonlocal searches
            searches += 1
            hits = kb.search(query, 3)
            retrieved.update({h["source"]: h["title"] for h in hits})
            return hits

        msgs = run_lab(llm, search, q["question"])
        calls = [m for m in msgs if getattr(m, "tool_calls", None)]
        args = calls[-1].tool_calls[0]["args"] if calls else {}
        refs = args.get("references") or []
        rows.append(Row(q["id"], False, bool(refs) and all(r in retrieved for r in refs),
                        any(e in refs for e in q["expected_sources"]),
                        word_count(CITATION.sub("", args.get("answer", ""))) <= word_limit,
                        len(calls), searches, refs))
    return rows


def summarize(rows: list[Row]) -> dict:
    n = len(rows)
    return {"n": n, "complete": sum(r.complete for r in rows) / n, "cited_ok": sum(r.cited_ok for r in rows) / n,
            "expected": sum(r.expected for r in rows) / n, "in_limit": sum(r.in_limit for r in rows) / n,
            "llm_calls": sum(r.llm_calls for r in rows), "searches": sum(r.searches for r in rows)}


