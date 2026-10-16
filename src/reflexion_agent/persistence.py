"""Checkpointers: in memory, or a free local SQLite file (langgraph-checkpoint-sqlite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer(db_path: Optional[str] = None) -> Any:
    """InMemorySaver, or SqliteSaver writing to ``db_path`` (every round survives restarts)."""
    if not db_path:
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))


def history(app, thread_id: str) -> list[dict]:
    """Every saved checkpoint of a thread, oldest first: step, next node(s), the values, its config."""
    config = {"configurable": {"thread_id": thread_id}}
    rows = [{"step": s.metadata.get("step"), "next": list(s.next), "values": dict(s.values), "config": s.config}
            for s in app.get_state_history(config)]
    return list(reversed(rows))


def mermaid(app) -> str:
    """The graph as Mermaid text (paste into GitHub Markdown or mermaid.live)."""
    return app.get_graph().draw_mermaid()
