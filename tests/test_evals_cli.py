"""Evals and the command line (offline; no API)."""

import pytest

from reflexion_agent import cli, evaluate, evaluate_lab, get_llm, load_questions, summarize


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    monkeypatch.setattr(cli, "_COLOR", False)


def test_offline_eval_numbers():
    rows = evaluate(get_llm("offline"), load_questions())
    s = summarize(rows)
    assert s["n"] == 12 and s["complete"] == s["cited_ok"] == s["expected"] == s["in_limit"] == 1
    assert s["llm_calls"] == 36 and s["searches"] == 36


def test_lab_eval_costs_more():
    s = summarize(evaluate_lab(get_llm("offline"), load_questions()[:3]))
    assert s["llm_calls"] == 15 and s["complete"] == 0


def test_cli_ask_trace(capsys):
    assert cli.main(["ask", "How do I test my chatbot for prompt injection?", "-p", "offline", "--noisy", "-t"]) == 0
    out = capsys.readouterr().out
    assert "▶ respond" in out and "→ execute_tools" in out and "✖ check          [unknown_source]" in out
    assert "feedback sent:  [unknown_source]; [bad_citation]" in out and "✔ complete - 2 revision(s), 3 model calls" in out


def test_cli_ask_plain_and_budget(capsys):
    assert cli.main(["ask", "How do I build a golden dataset for evals?", "-p", "offline"]) == 0
    assert "References:" in capsys.readouterr().out
    assert cli.main(["ask", "How do I build a golden dataset?", "-p", "offline", "--max-revisions", "1", "-t"]) == 2
    assert "revision budget used up" in capsys.readouterr().out


def test_cli_eval_search_graph_history(capsys, tmp_path):
    assert cli.main(["eval", "-p", "offline"]) == 0
    assert "all       100%" in capsys.readouterr().out
    assert cli.main(["eval", "-p", "offline", "--lab", "--questions", "data/questions.jsonl"]) == 0
    assert "lab graph" in capsys.readouterr().out
    assert cli.main(["search", "flaky", "tests", "-k", "1"]) == 0
    assert "kb://flaky-tests-nondeterminism" in capsys.readouterr().out
    assert cli.main(["graph"]) == 0 and "nudge" in capsys.readouterr().out
    assert cli.main(["graph", "--lab"]) == 0 and "revisor" in capsys.readouterr().out
    db = str(tmp_path / "r.sqlite")
    cli.main(["ask", "How do I mock LLM calls?", "-p", "offline", "--db", db, "--thread", "x"])
    capsys.readouterr()
    assert cli.main(["history", "--db", db, "--thread", "x"]) == 0 and "(done)" in capsys.readouterr().out
    assert cli.main(["history", "--db", db, "--thread", "none"]) == 1


def test_cli_trace_prints_prose_and_nudge(capsys, monkeypatch):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    replies = iter([AIMessage(content="prose draft"), AIMessage(content="prose again"), AIMessage(content="still prose")])
    monkeypatch.setattr(cli, "get_llm", lambda *a, **k: Scripted(messages=replies))
    assert cli.main(["ask", "Q?", "-t", "--max-revisions", "2"]) == 2
    out = capsys.readouterr().out
    assert "no tool call (prose reply)" in out and "prose reply, no tool call" in out and "↺ nudge" in out
    assert "unresolved: [no_tool_call]" in out


def test_python_dash_m_entry_point():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "reflexion_agent", "graph"], capture_output=True, text=True)
    assert result.returncode == 0 and "respond" in result.stdout
