"""Generate and execute notebooks/walkthrough.ipynb (offline provider by default).

    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Reflexion agent — walkthrough\n\n"
       "Draft → self-critique → search → revise with citations, in a LangGraph loop that checks every citation against "
       "what was actually retrieved. Runs **offline by default** (a rule-based stand-in, not an LLM: no key, no cost). "
       "Set `PROVIDER = \"gemini\"` (free key) or `\"openai\"` for a real model."),
    code("PROVIDER = \"offline\"  # \"gemini\" | \"openai\" | \"offline\"\n\n"
         "import json, os\nfrom pathlib import Path\nfrom dotenv import load_dotenv\nload_dotenv()\n"
         "if Path.cwd().name == \"notebooks\":\n    os.chdir(\"..\")   # data/ paths are relative to the repo root\n\n"
         "from reflexion_agent import (KnowledgeBase, build_reflexion_graph, research, draft_answer, format_answer,\n"
         "                             check_revision, get_llm, mermaid, evaluate, evaluate_lab, load_questions, summarize)\n"
         "llm = get_llm(PROVIDER)\nchecks = []"),
    md("## 1. The search tool: free, local, with stable source ids\nTwelve short notes on testing LLM applications, "
       "searched with BM25. It replaces the lab's paid Tavily web search."),
    code("kb = KnowledgeBase()\nfor hit in kb.search(\"flaky LLM tests\", 2):\n    print(hit[\"source\"], hit[\"score\"], \"-\", hit[\"content\"][:90])\n"
         "checks.append(kb.search(\"flaky LLM tests\", 1)[0][\"source\"] == \"kb://flaky-tests-nondeterminism\")"),
    md("## 2. The graph\n`respond → execute_tools → revise → check`, then done, another research round, a nudge "
       "(prose reply) or the budget stop."),
    code("app = build_reflexion_graph(llm, kb)\nprint(mermaid(app))"),
    md("## 3. One question, before and after"),
    code("state = research(app, \"How do I test my chatbot for prompt injection?\")\n"
         "print(\"DRAFT:\\n\" + draft_answer(state) + \"\\n\")\nprint(\"FINAL:\\n\" + format_answer(state))\n"
         "print(state[\"stop_reason\"], state[\"revisions\"], \"revisions,\", state[\"llm_calls\"], \"model calls\")\n"
         "checks.append(state[\"stop_reason\"] == \"complete\" and state[\"references\"])"),
    md("## 4. A made-up citation is caught\nWith `noisy=True` the offline model cites a source it never retrieved in its "
       "first revision. The check flags it, the feedback goes back with the next search results, and the revision fixes it."),
    code("noisy = research(build_reflexion_graph(get_llm(\"offline\", noisy=True), kb), \"How do I test my chatbot for prompt injection?\")\n"
         "feedback = json.loads(noisy[\"messages\"][4].content)[\"feedback\"]\nprint(\"feedback:\", feedback)\n"
         "print(\"final references:\", noisy[\"references\"])\n"
         "print(check_revision(\"Proven by a study [1].\", [\"kb://2024-study-on-llm-testing\"], noisy[\"sources\"]))\n"
         "checks.append(\"kb://2024-study-on-llm-testing\" not in noisy[\"references\"])"),
    md("## 5. Evaluate: this graph vs the lab's\nScored on citations that were really retrieved, the expected note being "
       "cited, the word limit, and cost."),
    code("ours = summarize(evaluate(llm, load_questions(), kb=kb))\nlab = summarize(evaluate_lab(llm, load_questions(), kb=kb))\n"
         "print(\"this repo:\", ours)\nprint(\"lab graph:\", lab)\nchecks.append(ours[\"llm_calls\"] < lab[\"llm_calls\"])"),
    code("print(f\"{sum(bool(c) for c in checks)}/{len(checks)} checks\", \"- all checks passed\" if all(checks) else \"- FAILED\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, ROOT / "notebooks" / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
