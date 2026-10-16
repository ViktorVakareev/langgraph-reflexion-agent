"""Free, local search: a keyword (BM25-style) index over the Markdown notes in data/knowledge_base/.

It stands in for the lab's paid Tavily web search. Every passage has a stable source id such as
``kb://prompt-injection-testing``, so the agent's citations can be checked against what was
actually retrieved. ``TavilySource`` is an optional adapter for real web search.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

DEFAULT_DIR = Path("data") / "knowledge_base"
STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "it", "how", "do", "i",
             "my", "what", "with", "can", "should", "when", "we", "you", "your", "that", "this", "as", "by", "at", "from",
             "if", "so", "not", "does", "which", "into", "than", "then", "its", "their", "there", "about", "best"}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str


class Search(Protocol):
    def search(self, query: str, k: int = 3) -> list[dict]: ...


class KnowledgeBase:
    """BM25 over paragraphs. ``search`` returns [{"source", "title", "content", "score"}]."""

    def __init__(self, directory: Optional[str | Path] = None, k1: float = 1.4, b: float = 0.75):
        self.passages: list[Passage] = []
        for path in sorted(Path(directory or DEFAULT_DIR).glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            title = re.search(r"^title:\s*(.+)$", raw, re.M)
            body = re.sub(r"^---.*?---\s*", "", raw, flags=re.S)
            for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
                self.passages.append(Passage(f"kb://{path.stem}", title.group(1).strip() if title else path.stem, para))
        if not self.passages:
            raise FileNotFoundError(f"No Markdown notes found in {directory or DEFAULT_DIR}")
        self.docs = [Counter(tokens(p.title + " " + p.text)) for p in self.passages]
        self.avg = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    @property
    def sources(self) -> dict[str, str]:
        return {p.source: p.title for p in self.passages}

    def _score(self, q: list[str], doc: Counter) -> float:
        length = sum(doc.values())
        return sum(self.idf.get(t, 0) * doc[t] * (self.k1 + 1) / (doc[t] + self.k1 * (1 - self.b + self.b * length / self.avg))
                   for t in q if t in doc)

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokens(query)
        scored = sorted(((self._score(q, d), i) for i, d in enumerate(self.docs)), key=lambda x: (-x[0], x[1]))
        results, seen = [], set()
        for score, i in scored:
            if score <= 0 or len(results) >= k:
                break
            p = self.passages[i]
            if (p.source, p.text[:40]) in seen:
                continue
            seen.add((p.source, p.text[:40]))
            results.append({"source": p.source, "title": p.title, "content": p.text, "score": round(score, 3)})
        return results


class TavilySource:  # pragma: no cover - needs a paid/credit-based key and network
    """Optional real web search (pip install langchain-tavily; TAVILY_API_KEY)."""

    def __init__(self, max_results: int = 3):
        from langchain_tavily import TavilySearch

        self.tool = TavilySearch(max_results=max_results)

    def search(self, query: str, k: int = 3) -> list[dict]:
        out = self.tool.invoke({"query": query})
        return [{"source": r["url"], "title": r.get("title", ""), "content": r.get("content", ""), "score": r.get("score", 0)}
                for r in out.get("results", [])[:k]]
