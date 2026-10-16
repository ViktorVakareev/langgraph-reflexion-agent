"""The structured output the model fills in (as tool calls), as in the lab - with limits added."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="What information is missing from the answer. Empty if nothing important is missing.")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    """Answer the question, critique your answer, and list search queries to improve it."""

    answer: str = Field(description="Main response to the question (about 150 words)")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: list[str] = Field(description="1-3 search queries to research what is missing", max_length=3)


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer using the search results, with numbered citations."""

    references: list[str] = Field(description="Source ids from the search results that support the answer, "
                                              "in citation order: [1] is the first item")
