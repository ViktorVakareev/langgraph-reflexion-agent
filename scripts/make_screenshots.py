"""Regenerate the README images in docs/images/ (offline, reproducible).

    pip install -e ".[all]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed. The graph diagrams are drawn from
each compiled graph's own Mermaid export (app.get_graph().draw_mermaid()).

Mermaid is loaded from cdn.jsdelivr.net, or from a local file in $MERMAID_JS
(e.g. after `npm pack mermaid@11` and unpacking it: package/dist/mermaid.min.js).
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline",
       "COLUMNS": "120", "PYTHONWARNINGS": "ignore"}

PROMPT = r"PS C:\LAB\lab-building-a-reflexion-agent\langgraph-reflexion-agent>"
FONT_DIR = os.getenv("TERMINAL_FONT_DIR")  # optional: @fontsource/cascadia-mono "files" folder (npm pack)


def _font_face() -> str:
    """Embed Cascadia Mono (Windows Terminal's font, SIL OFL) if available; else fall back to Consolas."""
    if not FONT_DIR:
        return ""
    import base64
    faces = []
    for weight in (400, 700):
        path = Path(FONT_DIR) / f"cascadia-mono-latin-{weight}-normal.woff2"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            faces.append(f"@font-face{{font-family:'Cascadia Mono';font-weight:{weight};"
                         f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")
    return "".join(faces)


# Windows Terminal look: dark tab strip, PowerShell tab, min/max/close on the right, Campbell colours
TERMINAL_CSS = """
body{margin:0;padding:28px;background:#dfe3e8;font-family:'Segoe UI',system-ui,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border:1px solid #3b3b3b;border-radius:8px;overflow:hidden;
     box-shadow:0 10px 30px rgba(0,0,0,.35);background:#0c0c0c}
.bar{background:#202020;height:36px;display:flex;align-items:flex-end;padding-left:8px;position:relative}
.tab{background:#0c0c0c;color:#e6e6e6;font-size:12px;height:30px;padding:0 12px;display:flex;align-items:center;gap:8px;
     border-radius:6px 6px 0 0;max-width:620px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ps{display:inline-flex;align-items:center;justify-content:center;width:16px;height:14px;background:#2671be;color:#fff;
    font:bold 9px Consolas,monospace;border-radius:2px;flex:none}
.x{color:#9d9d9d;margin-left:8px;font-size:11px}
.plus{color:#bdbdbd;font-size:16px;padding:0 10px 6px}
.ctl{position:absolute;right:0;top:0;display:flex;height:32px}
.ctl span{width:46px;display:flex;align-items:center;justify-content:center;color:#e6e6e6;font-size:13px}
pre{margin:0;padding:14px 18px 18px;color:#cccccc;
    font:14px/1.5 'Cascadia Mono','Cascadia Code',Consolas,'DejaVu Sans Mono',monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#cccccc}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def windows_command(command: str) -> str:
    """Show commands the way they're typed in PowerShell on Windows (backslash paths)."""
    import re
    return re.sub(r"\b(examples|scripts|notebooks|data)/", r"\1\\", command)


def terminal_html(command: str, output: str, title: str) -> str:
    body = Ansi2HTMLConverter(inline=True, dark_bg=True).convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{_font_face()}{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><div class="tab"><span class="ps">&gt;_</span>{html.escape(title)}<span class="x">✕</span></div>
<span class="plus">+</span><span class="plus">⌄</span>
<div class="ctl"><span>&#x2014;</span><span>&#x2610;</span><span>✕</span></div></div>
<pre><span class="prompt">{html.escape(PROMPT)} </span>{html.escape(windows_command(command))}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


GRAPH_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-flex;gap:26px;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
.col{display:flex;flex-direction:column;align-items:center;min-width:240px}
h3{margin:0 0 4px;font-size:17px;color:#0f172a}p{margin:0 0 12px;color:#475569;font-size:13px;max-width:330px;text-align:center}
"""


def graphs_html() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    from reflexion_agent import build_reflexion_graph, get_llm, mermaid
    from reflexion_agent.lab_pattern import build_lab_graph

    cols = [
        ("The lab's MessageGraph", "always 4 search rounds and 5 model calls; a prose reply loops forever "
         "(drawn with a path map, which the lab omits)", build_lab_graph(get_llm("offline"), lambda q: [], path_map=True)),
        ("This repo", "check validates citations against retrieved sources; ends when complete, at the budget, "
         "or nudges a prose reply", build_reflexion_graph(get_llm("offline"))),
    ]
    body = "".join(f'<div class="col"><h3>{t}</h3><p>{html.escape(d)}</p><pre class="mermaid">{html.escape(mermaid(g))}</pre></div>'
                   for t, d, g in cols)
    return f'<html><head><meta charset="utf-8"><style>{GRAPH_CSS}</style></head><body><div class="card">{body}</div></body></html>'


def shoot_graphs(page) -> None:
    page.set_content(graphs_html())
    local = os.getenv("MERMAID_JS")
    page.add_script_tag(**({"path": local} if local else {"url": "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"}))
    page.evaluate("async () => { mermaid.initialize({startOnLoad: false, theme: 'default'}); await mermaid.run(); }")
    page.wait_for_timeout(300)
    page.locator(".card").screenshot(path=str(OUT / "graphs.png"))
    print("  ✓ graphs.png")


def main() -> None:
    py = sys.executable
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        shoot_graphs(page)

        shots = [
            ("trace.png", "Draft, critique, search, revise with citations - two rounds, then complete",
             'reflexion-agent ask "How do I test my chatbot for prompt injection?" -p offline -t', None),
            ("citation-check.png", "A made-up source in revision 1 is caught, sent back as feedback and fixed",
             'reflexion-agent ask "How do I test my chatbot for prompt injection?" -p offline --noisy -t', None),
            ("eval.png", "12 questions: citations really retrieved, expected note cited, word limit, cost",
             "reflexion-agent eval -p offline", None),
            ("eval-lab.png", "The lab's graph on the same questions: always 5 model calls per question",
             "reflexion-agent eval -p offline --lab", None),
            ("search.png", "The free local search tool (BM25 over Markdown notes)",
             'reflexion-agent search "flaky LLM tests" -k 2', None),
            ("lab-original.png", "The lab's graph: fixed 4 rounds, unchecked citations, growing prompt, endless prose loop",
             f"{py} examples/lab_original.py", "python examples/lab_original.py"),
            ("quickstart.png", "Quickstart", f"{py} examples/quickstart.py offline", "python examples/quickstart.py offline"),
            ("pytest-results.png", "Test suite", f"{py} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{py} -m pytest -q --color=yes -p no:cacheprovider --cov=reflexion_agent --cov-report=term-missing",
             "pytest --cov=reflexion_agent --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        nb_html = run(f"{py} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.set_content(nb_html, wait_until="networkidle")
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()


if __name__ == "__main__":
    main()
