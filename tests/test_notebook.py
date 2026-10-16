"""The walkthrough notebook must run top to bottom (offline provider)."""

from pathlib import Path

import nbformat
import pytest

nbclient = pytest.importorskip("nbclient")
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "walkthrough.ipynb"


def test_walkthrough_notebook_executes():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=240, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    outputs = "".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []))
    assert "all checks passed" in outputs
