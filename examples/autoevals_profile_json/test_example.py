"""Exercise the actual optional Autoevals scorer and the mutation runner offline."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("autoevals", reason="install autoevals for the offline example tests")

from autoevals import Score

from muteval import run_mutation_testing
from muteval.adapters.base import scorer_to_eval
from muteval.config import load_config
from muteval.runner import select_mutants

_CONFIG = Path(__file__).with_name("muteval_config.py")


@pytest.fixture
def demo(monkeypatch):
    for key in ("OPENAI_API_KEY", "BRAINTRUST_API_KEY", "MUTEVAL_PROFILE_SUITE"):
        monkeypatch.delenv(key, raising=False)
    return runpy.run_path(str(_CONFIG))


def test_real_autoevals_catches_both_changed_survivors(demo):
    weak = run_mutation_testing(demo["make_config"]("weak"))
    strong = run_mutation_testing(demo["make_config"]("strong"))

    assert weak.status == strong.status == "valid"
    assert weak.baseline_passed and strong.baseline_passed
    assert weak.total == strong.total == 2
    assert weak.score == 0.0
    assert len(weak.real_survivors) == 2
    assert all(outcome.output_changed is True for outcome in weak.real_survivors)
    assert strong.score == 1.0
    assert strong.killed == 2
    assert not strong.survivors
    assert all(o.failing_eval == "profile_exact_match" for o in strong.outcomes)


def test_mutated_prompt_reaches_renderer_and_reference_stays_fixed(demo):
    case = demo["CASES"][0]
    original_case = json.loads(json.dumps(case))
    baseline = demo["render_profile"](demo["PROMPT"], case)
    outputs = [
        demo["render_profile"](mutant.prompt, case)
        for mutant in select_mutants(demo["make_config"]())
    ]

    assert json.loads(baseline) == {"name": "Ari", "city": None}
    assert [json.loads(output) for output in outputs] == [
        {"name": "Ari"},
        {"name": "Ari", "city": "Atlantis"},
    ]
    assert case == original_case


def test_json_key_order_is_not_a_regression_and_score_is_preserved(demo):
    cfg = demo["make_config"]("strong")
    outcome = cfg.evals[1]('{ "city": null, "name": "Ari" }', demo["CASES"][0])
    assert outcome.passed
    assert outcome.score == outcome.threshold == 1.0
    assert outcome.name == "profile_exact_match"


@pytest.mark.parametrize("output", ["not JSON", "{}"])
def test_bad_baseline_has_no_trusted_score(demo, output):
    cfg = demo["make_config"]("strong", run=lambda prompt, case: output)
    result = run_mutation_testing(cfg)
    assert result.status == "baseline_failed"
    assert result.score is result.effective_score is None
    assert result.total == 0


@pytest.mark.parametrize("score", [None, float("nan")])
def test_missing_or_nonfinite_score_is_an_error_not_a_kill(demo, score):
    result = Score(name="unavailable", score=score)
    cfg = demo["make_config"]("strong")
    cfg.evals = [
        scorer_to_eval(
            lambda output, case: demo["_require_score"](result),
            threshold=1.0,
            name="unavailable",
        )
    ]
    cfg.eval_names = ["unavailable"]
    run = run_mutation_testing(cfg)
    assert run.status == "baseline_errored"
    assert run.baseline_error
    assert run.score is run.effective_score is None
    assert run.total == 0


def test_scorer_error_is_not_converted_to_a_score(demo):
    result = Score(name="failed", score=1.0)
    result.error = RuntimeError("scorer unavailable")
    with pytest.raises(RuntimeError, match="scorer reported an error"):
        demo["_require_score"](result)


def test_skipped_mutant_invalidates_run_instead_of_shrinking_it_silently(demo):
    def score(output, case):
        if json.loads(output).get("city") == "Atlantis":
            return demo["_require_score"](Score(name="skipped", score=None))
        return demo["_profile_exact_score"](output, case)

    cfg = demo["make_config"]("strong")
    cfg.evals[1] = scorer_to_eval(score, threshold=1.0, name="profile_exact_match")
    result = run_mutation_testing(cfg)
    assert result.baseline_passed
    assert result.status == "partial_errors"
    assert result.total == 2
    assert result.killed == 1
    assert result.errored == 1
    assert not result.survivors
    assert "skipped this case" in next(o.error for o in result.outcomes if o.errored)


def test_cli_config_selects_strong_suite(monkeypatch, demo):
    monkeypatch.setenv("MUTEVAL_PROFILE_SUITE", "strong")
    cfg = load_config(_CONFIG)
    assert cfg.eval_names == ["valid_json", "profile_exact_match"]
    assert run_mutation_testing(cfg).score == 1.0


def test_unknown_suite_fails_closed(demo):
    with pytest.raises(ValueError, match="must be 'weak' or 'strong'"):
        demo["make_config"]("strnog")
