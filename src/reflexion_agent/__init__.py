"""A Reflexion research agent (LangGraph 1.x): draft, self-critique, search, revise with checked citations."""

from .agent import ReflexionState, build_reflexion_graph, draft_answer, format_answer, initial_state, research
from .checks import check_revision, word_count
from .evals import evaluate, evaluate_lab, load_questions, summarize
from .knowledge import KnowledgeBase
from .llm import PROVIDERS, OfflineResearcher, get_llm
from .persistence import get_checkpointer, history, mermaid
from .schemas import AnswerQuestion, Reflection, ReviseAnswer

__all__ = ["ReflexionState", "build_reflexion_graph", "draft_answer", "format_answer", "initial_state", "research",
           "check_revision", "word_count", "evaluate", "evaluate_lab", "load_questions", "summarize", "KnowledgeBase",
           "PROVIDERS", "OfflineResearcher", "get_llm", "get_checkpointer", "history", "mermaid", "AnswerQuestion",
           "Reflection", "ReviseAnswer"]
__version__ = "1.0.0"
