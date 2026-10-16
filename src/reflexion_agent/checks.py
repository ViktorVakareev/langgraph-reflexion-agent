"""Deterministic checks on a revised answer: the citations must be real and the answer within limits.

The lab asks for "numerical citations" and a "References" section but never checks them, so a model
can cite a URL it never retrieved. Here every reference must be a source the search actually returned.
"""

from __future__ import annotations

import re

CITATION = re.compile(r"\[(\d+)\]")


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'-]*\b", text))


def check_revision(answer: str, references: list[str], retrieved: dict[str, str], word_limit: int = 250) -> list[str]:
    """Return a list of problems ([] = all checks passed). Each starts with a short code in brackets."""
    issues = []
    if not answer.strip():
        return ["[empty] The answer is empty."]
    if not references:
        issues.append("[no_references] List the source ids that support the answer.")
    unknown = [r for r in references if r not in retrieved]
    if unknown:
        issues.append(f"[unknown_source] Not in the search results: {', '.join(unknown)}. Cite only retrieved sources.")
    if len(set(references)) != len(references):
        issues.append("[duplicate_reference] Each source should appear once in references.")
    cited = {int(n) for n in CITATION.findall(answer)}
    bad = sorted(n for n in cited if n < 1 or n > len(references))
    if bad:
        issues.append(f"[bad_citation] Citation number(s) {bad} have no matching reference (there are {len(references)}).")
    if references and not cited:
        issues.append("[uncited] Add numbered citations such as [1] in the answer text.")
    words = word_count(CITATION.sub("", answer))
    if words > word_limit:
        issues.append(f"[too_long] {words} words; the limit is {word_limit}.")
    return issues
