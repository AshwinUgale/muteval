"""The keyless promptfoo demo must run offline and expose the language-rule gap.

This is the "gift" a promptfoo user sees first: `muteval run --config
examples/promptfoo_offline/muteval_config.py` with NO API key, producing a real
survivor. If this breaks, the demo in the README/blog is a lie — so it's pinned.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("yaml")  # the promptfoo adapter needs PyYAML

from muteval import run_mutation_testing
from muteval.config import load_config

_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "examples",
    "promptfoo_offline",
    "muteval_config.py",
)


def _run(monkeypatch):
    # Prove it needs no key: run with the env var absent.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return run_mutation_testing(load_config(_CONFIG))


def test_demo_runs_keyless_and_scores(monkeypatch):
    result = _run(monkeypatch)
    # Baseline passed and the run earned a real (non-vacuous) score.
    assert result.status == "valid"
    assert result.score is not None
    assert result.effective_score is not None
    # The promptfoo assertions caught at least one regression (kills exist).
    assert result.killed >= 1


def test_demo_survivor_is_the_uncovered_language_rule(monkeypatch):
    result = _run(monkeypatch)
    gaps = result.real_survivors
    assert gaps, "expected at least one real survivor (the uncovered rule)"
    # The uncovered rule is the language one; it must surface as a HIGH survivor.
    descs = " ".join(o.mutant.description for o in gaps).lower()
    assert "reply in english" in descs
    assert any(o.severity == "high" for o in result.high_severity_survivors)
