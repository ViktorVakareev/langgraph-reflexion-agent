from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    """data/knowledge_base and questions.jsonl are relative to the repo root; keep the environment out of the tests."""
    monkeypatch.chdir(ROOT)
    for var in ("LLM_PROVIDER", "CHECKPOINT_DB"):
        monkeypatch.delenv(var, raising=False)
    yield
