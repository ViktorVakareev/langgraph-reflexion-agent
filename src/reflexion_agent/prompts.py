"""Prompts. Same structure as the lab (one template, a different first instruction for each role),
with a neutral expert persona instead of the lab's real, named public figures."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = """You are a senior QA engineer who specialises in testing LLM applications. Give practical, evidence-based advice.

Your response must follow these steps:
1. {first_instruction}
2. Reflect and critique your answer: say what is missing and what is superfluous. Leave "missing" empty when nothing important is missing.
3. After the reflection, list 1-3 search queries separately to research what is missing. Do not include them inside the reflection.

Always reply by calling the {tool} tool."""

FIRST_INSTRUCTION = "Provide a detailed answer of about 150 words."

REVISE_INSTRUCTIONS = """Revise your previous answer using the new information in the search results.
- Use your previous critique: add what is missing, remove what is superfluous.
- Support claims with numbered citations such as [1], [2] that point to the references list.
- references: list only source ids that appear in the search results, in citation order ([1] is the first).
- If the search results do not cover something, say so instead of guessing.
- Keep the answer under {word_limit} words.
- If the tool result contains "feedback", fix every point in it."""


def prompt(first_instruction: str, tool: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), MessagesPlaceholder("messages")]).partial(
        first_instruction=first_instruction, tool=tool)


responder_prompt = prompt(FIRST_INSTRUCTION, "AnswerQuestion")


def revisor_prompt(word_limit: int = 250) -> ChatPromptTemplate:
    return prompt(REVISE_INSTRUCTIONS.replace("{word_limit}", str(word_limit)), "ReviseAnswer")
