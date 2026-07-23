"""v0.4 'Provably honest': fill coverage on the modules the suite under-exercised
(adapter helpers, the stdlib OpenAI runner, terminal report states, scope edges).
These are behavior tests, not coverage theater — each asserts a real contract.
"""

from __future__ import annotations

import pytest

from muteval import EvalOutcome
from muteval.adapters import base
from muteval.report import format_probe_card, format_report
from muteval.runner import MutationResult


# --- adapters/base.py --------------------------------------------------------

def test_case_get_dict_object_and_none():
    assert base.case_get({"a": 1}, "a") == 1
    assert base.case_get({"a": 1}, "missing") is None
    assert base.case_get({"a": 1}, None) is None

    class C:
        x = 7

    assert base.case_get(C(), "x") == 7
    assert base.case_get(C(), "nope") is None


def test_scorer_to_eval_higher_is_better():
    ev = base.scorer_to_eval(lambda o, c: 0.8, threshold=0.7, name="q")
    out = ev("text", {})
    assert isinstance(out, EvalOutcome)
    assert out.passed and out.score == 0.8 and out.threshold == 0.7 and out.name == "q"
    assert base.scorer_to_eval(lambda o, c: 0.5, threshold=0.7, name="q")("t", {}).passed is False


def test_scorer_to_eval_lower_is_better():
    # e.g. a toxicity/latency score where lower passes.
    ev = base.scorer_to_eval(lambda o, c: 0.2, threshold=0.5, name="tox", higher_is_better=False)
    assert ev("t", {}).passed is True
    ev2 = base.scorer_to_eval(lambda o, c: 0.9, threshold=0.5, name="tox", higher_is_better=False)
    assert ev2("t", {}).passed is False


def test_named_sets_name_and_survives_unsettable():
    fn = lambda o, c: True
    assert base.named(fn, "nice").__name__ == "nice"
    # Something whose __name__ can't be set must not raise.
    assert base.named(object(), "x") is not None  # no exception


# --- runners.py (stdlib OpenAI run), no network via monkeypatch --------------

def test_openai_run_system_and_legacy_modes(monkeypatch):
    from muteval import System
    from muteval import runners

    captured = {}

    def fake_chat(messages, model, temperature=0.0):
        captured["messages"] = messages
        captured["model"] = model
        return "ANSWER"

    monkeypatch.setattr(runners, "_chat", fake_chat)
    run = runners.openai_run(model="gpt-4o-mini")

    # System mode: uses system.model + system.context.
    sys_target = System(prompt="be helpful", context=("doc A", "doc B"), model="gpt-4o")
    out = run(sys_target, {"question": "why?"})
    assert out == "ANSWER"
    assert captured["model"] == "gpt-4o"  # system.model wins
    user_msg = captured["messages"][1]["content"]
    assert "doc A" in user_msg and "why?" in user_msg

    # Legacy prompt mode: uses the default model + case context.
    out2 = run("bare prompt", {"question": "q", "context": ["ctx doc"]})
    assert captured["model"] == "gpt-4o-mini"
    assert "ctx doc" in captured["messages"][1]["content"]


def test_openai_run_question_key_fallbacks(monkeypatch):
    from muteval import runners

    captured = {}
    monkeypatch.setattr(runners, "_chat", lambda m, model, temperature=0.0: captured.setdefault("m", m) or "x")
    run = runners.openai_run()
    run("p", "just a string case")  # str case path
    assert "just a string case" in captured["m"][1]["content"]


def test_chat_requires_api_key(monkeypatch):
    from muteval import runners

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        runners._chat([{"role": "user", "content": "hi"}], "gpt-4o-mini")


def test_ssl_context_builds():
    from muteval import runners

    assert runners._ssl_context() is not None


def test_runner_text_helpers():
    from muteval import runners

    # _docs_to_text: list joins, string passes through, empty -> "".
    assert runners._docs_to_text(["a", "b"]) == "a\n\nb"
    assert runners._docs_to_text("solo") == "solo"
    assert runners._docs_to_text(None) == ""
    # _case_context: non-dict -> "", dict pulls + joins its context.
    assert runners._case_context("not a dict", "context") == ""
    assert runners._case_context({"context": ["x", "y"]}, "context") == "x\n\ny"
    # _case_question: dict with no matching key -> "", falls back to str() otherwise.
    assert runners._case_question({"nope": 1}, ("question",)) == ""
    assert runners._case_question(42, ("question",)) == "42"


# --- report.py terminal / invalid states ------------------------------------

def test_report_baseline_errored():
    txt = format_report(MutationResult(baseline_passed=True, baseline_error="kaboom"), use_color=False)
    assert "INVALID RUN" in txt and "kaboom" in txt


def test_report_baseline_failed():
    txt = format_report(MutationResult(baseline_passed=False), use_color=False)
    assert "baseline FAILED" in txt


def test_report_no_mutants():
    # baseline passes, but no outcomes -> total == 0.
    txt = format_report(MutationResult(baseline_passed=True, outcomes=[]), use_color=False)
    assert "NO MUTANTS" in txt


def test_probe_card_empty_and_populated():
    from muteval.probes.base import ProbeResult

    assert "No probes ran" in format_probe_card([], use_color=False)
    rs = [
        ProbeResult(name="statistical_adequacy", ok=True, summary="ok", detail="d"),
        ProbeResult(name="redundancy", ok=False, summary="two evals correlate", detail=""),
    ]
    card = format_probe_card(rs, use_color=False)
    assert "report card" in card and "statistical_adequacy" in card and "WARN" in card


# --- scope.py edges ----------------------------------------------------------

def test_scope_include_exclude_on_lines():
    from muteval.scope import make_scope

    scope = make_scope(include=r"cite")
    original = "Line one about cite.\nLine two about tone."
    # A mutant that changes the 'cite' line is kept; one changing 'tone' is dropped.
    assert scope.keep(original, original.replace("cite", "quote")) is True
    assert scope.keep(original, original.replace("tone", "voice")) is False


def test_strip_markers_extracts_ranges():
    from muteval.scope import strip_markers

    clean, ranges = strip_markers("fixed [[mutate]]MUTABLE[[/mutate]] tail")
    assert clean == "fixed MUTABLE tail"
    assert ranges == [(6, 13)]  # the MUTABLE region in clean coordinates
    assert clean[ranges[0][0]:ranges[0][1]] == "MUTABLE"
    # No markers -> everything mutable.
    assert strip_markers("no markers here") == ("no markers here", None)


def test_strip_markers_unterminated():
    from muteval.scope import strip_markers

    clean, ranges = strip_markers("keep [[mutate]]rest is mutable")
    assert clean == "keep rest is mutable"
    assert ranges and clean[ranges[0][0]:ranges[0][1]] == "rest is mutable"


def test_scope_ranges_only_keeps_edits_inside_region():
    from muteval.scope import strip_markers, make_scope

    clean, ranges = strip_markers("safe prefix [[mutate]]changeme[[/mutate]] safe suffix")
    scope = make_scope(ranges=ranges)
    # Edit inside the marked region -> kept.
    inside = clean.replace("changeme", "changedXX")
    assert scope.keep(clean, inside) is True
    # Edit in the protected prefix -> dropped.
    outside = clean.replace("safe prefix", "EDITED prefix")
    assert scope.keep(clean, outside) is False


def test_scope_exclude_drops_matching_lines():
    from muteval.scope import make_scope

    scope = make_scope(exclude=r"boilerplate")
    original = "Real instruction here.\nSome boilerplate footer."
    # Changing the excluded 'boilerplate' line is filtered out...
    assert scope.keep(original, original.replace("boilerplate", "legalese")) is False
    # ...changing the real line is kept.
    assert scope.keep(original, original.replace("Real", "Actual")) is True
