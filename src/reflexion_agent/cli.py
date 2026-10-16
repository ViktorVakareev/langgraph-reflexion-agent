"""Command line.

    reflexion-agent ask "How do I test my chatbot for prompt injection?" -p offline -t
    reflexion-agent ask "..." -p offline --noisy -t        # the citation check catches a made-up source
    reflexion-agent ask "..." --max-revisions 3 --word-limit 200 --db runs.sqlite --thread demo
    reflexion-agent eval -p offline                        # 12 questions, scored on citations and cost
    reflexion-agent eval -p offline --lab                  # the lab's graph on the same questions
    reflexion-agent search "flaky tests" -k 3              # what the search tool returns
    reflexion-agent graph [--lab]
    reflexion-agent history --db runs.sqlite --thread demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

from .agent import build_reflexion_graph, format_answer, initial_state, tool_call
from .checks import CITATION, word_count
from .evals import evaluate, evaluate_lab, load_questions, summarize
from .knowledge import KnowledgeBase
from .lab_pattern import build_lab_graph
from .llm import PROVIDERS, get_llm
from .persistence import get_checkpointer, history, mermaid

_COLOR = sys.stdout.isatty() or bool(os.getenv("FORCE_COLOR"))


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _wrap(text: str, indent: str = "    ", width: int = 104) -> str:
    return "\n".join(indent + line for para in text.splitlines() for line in (textwrap.wrap(para, width) or [""]))


def _words(text: str) -> int:
    return word_count(CITATION.sub("", text))


def _search_source(name: str):
    if name == "tavily":  # pragma: no cover - needs TAVILY_API_KEY and langchain-tavily
        from .knowledge import TavilySource
        return TavilySource()
    return KnowledgeBase()


def _print(node: str, update: dict, revision: list[int]) -> None:
    if node == "respond":
        call = tool_call(update["messages"][-1], "AnswerQuestion")
        if not call:
            print(_c("1;33", "▶ respond        no tool call (prose reply)"))
            return
        a = call["args"]
        print(_c("1;36", f"▶ respond        draft, {_words(a.get('answer', ''))} words, no citations"))
        print(_c("2", _wrap("missing: " + a.get("reflection", {}).get("missing", ""))))
        print(_c("2", _wrap("queries: " + "; ".join(a.get("search_queries", [])))))
    elif node == "execute_tools":
        for m in update.get("messages", []):
            if isinstance(m, ToolMessage):
                payload = json.loads(m.content)
                for query, hits in payload["results"].items():
                    ids = ", ".join(dict.fromkeys(h["source"].replace("kb://", "") for h in hits)) or "(no results)"
                    print(_c("1;94", "→ execute_tools  ") + f"{query!r} -> {ids}")
                if payload.get("feedback"):
                    print(_c("1;35", "  feedback sent:  ") + "; ".join(i.split("]")[0] + "]" for i in payload["feedback"]))
    elif node == "revise":
        revision[0] += 1
        call = tool_call(update["messages"][-1], "ReviseAnswer")
        if not call:
            print(_c("1;33", f"▶ revise #{revision[0]}      prose reply, no tool call"))
            return
        a = call["args"]
        refs = ", ".join(r.replace("kb://", "") for r in a.get("references", []))
        print(_c("1;36", f"▶ revise #{revision[0]}      {_words(a.get('answer', ''))} words, references: ") + refs)
    elif node == "check":
        issues = update.get("issues") or []
        if issues:
            for issue in issues:
                print(_c("1;31", "✖ check          ") + issue)
        else:
            missing = update.get("missing") or ""
            print(_c("1;32", "✔ check          citations valid") + (_c("2", f"; still missing: {missing}") if missing else
                                                                       _c("2", "; nothing missing")))
    elif node == "nudge":
        print(_c("1;33", "↺ nudge          asked the model to call the tool"))


def cmd_ask(args) -> int:
    llm = get_llm(args.provider, args.model, noisy=args.noisy)
    app = build_reflexion_graph(llm, _search_source(args.search), checkpointer=get_checkpointer(args.db))
    config = {"configurable": {"thread_id": args.thread}, "recursion_limit": 100}
    question = " ".join(args.question)
    inputs = initial_state(question, args.max_revisions, args.word_limit)
    if args.trace:
        print(_c("1;97", "? question       ") + question)
        revision = [0]
        for chunk in app.stream(inputs, config, stream_mode="updates"):
            for node, update in chunk.items():
                _print(node, update or {}, revision)
        state = app.get_state(config).values
    else:
        state = app.invoke(inputs, config)
    ok = state.get("stop_reason") == "complete"
    head = "✔ complete" if ok else "⚠ stopped: revision budget used up"
    print(_c("1;32" if ok else "1;33", f"{head} - {state['revisions']} revision(s), {state['llm_calls']} model calls, "
                                       f"{state['searches']} searches"))
    if state.get("issues"):
        print(_c("1;31", "  unresolved: " + "; ".join(state["issues"])))
    print()
    print(format_answer(state))
    return 0 if ok else 2


def cmd_eval(args) -> int:
    llm = get_llm(args.provider, args.model, noisy=args.noisy)
    questions = load_questions(args.questions)
    rows = evaluate_lab(llm, questions, args.word_limit) if args.lab else evaluate(llm, questions, args.word_limit,
                                                                                   args.max_revisions)
    label = "lab graph" if args.lab else "reflexion graph"
    print(_c("1", f"{len(rows)} questions · {label} · provider: {args.provider}{' (noisy)' if args.noisy else ''}"))
    print(_c("1", f"{'id':<5}{'complete':>9}{'cited ok':>10}{'expected':>10}{'in limit':>10}{'calls':>7}{'searches':>10}"))
    yes = lambda b: _c("32", "yes") if b else _c("31", " no")  # noqa: E731
    for r in rows:
        complete = _c("2", "  -") if args.lab else yes(r.complete)
        print(f"{r.id:<5}{'':>6}{complete}{'':>7}{yes(r.cited_ok)}{'':>7}{yes(r.expected)}{'':>7}{yes(r.in_limit)}"
              f"{r.llm_calls:>7}{r.searches:>10}")
    s = summarize(rows)
    pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
    complete = "  -" if args.lab else pct(s["complete"])
    print(_c("1", f"{'all':<5}{complete:>9}{pct(s['cited_ok']):>10}{pct(s['expected']):>10}{pct(s['in_limit']):>10}"
                  f"{s['llm_calls']:>7}{s['searches']:>10}"))
    return 0


def cmd_search(args) -> int:
    for hit in KnowledgeBase().search(" ".join(args.query), args.k):
        print(_c("1;36", f"{hit['source']}") + _c("2", f"  score {hit['score']}  {hit['title']}"))
        print(_wrap(hit["content"]))
    return 0


def cmd_graph(args) -> int:
    llm = get_llm("offline")
    app = build_lab_graph(llm, lambda q: []) if args.lab else build_reflexion_graph(llm)
    print(mermaid(app))
    return 0


def cmd_history(args) -> int:
    app = build_reflexion_graph(get_llm("offline"), checkpointer=get_checkpointer(args.db))
    rows = history(app, args.thread)
    if not rows:
        print(f"No checkpoints for thread '{args.thread}'.")
        return 1
    for row in rows:
        v = row["values"]
        nxt = ", ".join(row["next"]) or "(done)"
        print(f"{_c('1;33', 'step ' + str(row['step']).rjust(2))}  next: {nxt:<15} revisions={v.get('revisions', 0)} "
              f"model_calls={v.get('llm_calls', 0)} searches={v.get('searches', 0)} messages={len(v.get('messages', []))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="reflexion-agent", description="Reflexion research agent with checked citations")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="research one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("-t", "--trace", action="store_true", help="print every round")
    ask.add_argument("--search", choices=("kb", "tavily"), default="kb", help="kb = local notes (free, default)")
    ask.add_argument("--db", default=os.getenv("CHECKPOINT_DB"), help="SQLite checkpoint file (default: in memory)")
    ask.add_argument("--thread", default="default")
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="score runs on data/questions.jsonl")
    ev.add_argument("--questions", default=None)
    ev.add_argument("--lab", action="store_true", help="run the lab's graph instead")
    ev.set_defaults(func=cmd_eval)

    for p in (ask, ev):
        p.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "gemini"))
        p.add_argument("-m", "--model", default=None)
        p.add_argument("--noisy", action="store_true", help="offline only: cite a made-up source in the first revision")
        p.add_argument("--max-revisions", type=int, default=2)
        p.add_argument("--word-limit", type=int, default=250)

    se = sub.add_parser("search", help="query the local knowledge base")
    se.add_argument("query", nargs="+")
    se.add_argument("-k", type=int, default=3)
    se.set_defaults(func=cmd_search)

    gr = sub.add_parser("graph", help="print the graph as Mermaid")
    gr.add_argument("--lab", action="store_true")
    gr.set_defaults(func=cmd_graph)

    hist = sub.add_parser("history", help="list the saved checkpoints of a run")
    hist.add_argument("--db", default=os.getenv("CHECKPOINT_DB"), required=not os.getenv("CHECKPOINT_DB"))
    hist.add_argument("--thread", default="default")
    hist.set_defaults(func=cmd_history)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
