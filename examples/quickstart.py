"""Research one question: draft -> critique -> search -> revise with checked citations.

    python examples/quickstart.py            # Gemini (free key in .env)
    python examples/quickstart.py offline    # no key, no cost
"""

import sys

from dotenv import load_dotenv

from reflexion_agent import build_reflexion_graph, draft_answer, format_answer, get_llm, research

load_dotenv()
app = build_reflexion_graph(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))
state = research(app, "My LLM tests are flaky. How do I make them reliable?")

print("Draft:\n" + draft_answer(state) + "\n")
print("Final:\n" + format_answer(state))
print(f"\n-> {state['stop_reason']}: {state['revisions']} revisions, {state['llm_calls']} model calls, "
      f"{state['searches']} searches")
