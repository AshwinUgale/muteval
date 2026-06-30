"""Tests for the System-aware built-in openai_run (no API calls)."""

import muteval.runners as r
from muteval.system import System


def _capture(monkeypatch):
    cap = {}

    def fake_chat(messages, model, temperature=0.0):
        cap["model"] = model
        cap["system"] = messages[0]["content"]
        cap["user"] = messages[1]["content"]
        return "out"

    monkeypatch.setattr(r, "_chat", fake_chat)
    return cap


def test_system_mode_uses_mutated_context_and_model(monkeypatch):
    cap = _capture(monkeypatch)
    run = r.openai_run(model="gpt-4o-mini")
    out = run(System(prompt="SYS", context=("alpha doc", "beta doc"), model="gpt-4o"),
              {"question": "q?"})
    assert out == "out"
    assert cap["model"] == "gpt-4o"          # system.model honored (model-swap)
    assert cap["system"] == "SYS"
    assert "alpha doc" in cap["user"]         # mutated context injected


def test_system_mode_dropped_context_changes_user_message(monkeypatch):
    cap = _capture(monkeypatch)
    run = r.openai_run()
    # Simulate a drop_context_doc mutant: only one doc remains.
    run(System(prompt="S", context=("only beta",)), {"question": "q"})
    assert "only beta" in cap["user"]
    assert "alpha" not in cap["user"]


def test_prompt_mode_is_legacy(monkeypatch):
    cap = _capture(monkeypatch)
    run = r.openai_run(model="gpt-4o-mini")
    run("PROMPT", {"question": "q", "context": ["case doc"]})
    assert cap["model"] == "gpt-4o-mini"
    assert cap["system"] == "PROMPT"
    assert "case doc" in cap["user"]


def test_system_without_context_falls_back_to_case(monkeypatch):
    cap = _capture(monkeypatch)
    run = r.openai_run()
    run(System(prompt="S"), {"question": "q", "context": ["fallback doc"]})
    assert "fallback doc" in cap["user"]
