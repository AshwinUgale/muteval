"""Gates & validity — no run may be scored or gated on evidence it doesn't have.

From a full audit: unresolved ties were never gated (all-tied runs passed CI or
crashed it; 1 resolved of 17 read "100%"), the judge parser turned "0/10" into
a perfect pass, a NaN metric score counted as a kill, a truthy non-verdict
(0.2, "fail") counted as a pass, the baseline was graded once while mutants were
judged by majority, and kills on output the original itself produces inflated
the effective score to 100% for a system that ignores its prompt.
"""

import itertools
import math
import re

import pytest

from muteval import MutEvalConfig, checks, run_mutation_testing
from muteval.checks import _parse_judge_score
from muteval.cli import main
from muteval.evals import EvalOutcome, coerce_outcome
from muteval.report import format_report, result_to_dict
from muteval.runner import (
    BASELINE_FAILED,
    NO_CONFIDENT_SCORE,
    NO_EVALUATED_MUTANTS,
    NO_SCORED_MUTANTS,
    PARTIAL_UNRESOLVED,
    VALID,
)

PROMPT = "- You must cite the order ID.\n- Never promise refunds.\n- Always be polite."


class _Pattern:
    """An eval whose verdicts follow a fixed prefix, then a repeating cycle."""

    def __init__(self, prefix, cycle):
        self.it = itertools.chain(prefix, itertools.cycle(cycle))

    def __call__(self, output, case):
        return next(self.it)


def _cfg(ev, runs=1, prompt=PROMPT, **kw):
    return MutEvalConfig(
        prompt=prompt,
        cases=[{"x": 1}],
        run=lambda p, c: "ok",  # ignores the prompt: no mutant changes behavior
        evals=[ev],
        runs_per_mutant=runs,
        **kw,
    )


# --- judge score parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "reply, score",
    [
        ("8/10", 0.8),
        ("0/10", 0.0),  # used to parse as 1.0 — a zero that PASSED
        ("3/10", 0.3),
        ("Rating: 2 out of 10", 0.2),
        ("Score: 2 ... 9 of the 10 items", 0.2),
        ("1", 0.1),  # a 1 out of 10, not 1.0
        ("10", 1.0),
        ("8\n\nNote: step 3 was skipped", 0.8),  # first number, not the last
        ("I'd give it a 9", 0.9),
        ("7.5", 0.75),
    ],
)
def test_judge_reply_parses_the_asked_for_score(reply, score):
    assert math.isclose(_parse_judge_score(reply), score)


@pytest.mark.parametrize("reply", ["", "   ", "no idea", "11", "-3", "15/10"])
def test_unusable_judge_reply_raises_not_kills(reply):
    with pytest.raises(ValueError):
        _parse_judge_score(reply)


def test_llm_judge_empty_reply_errors_the_mutant(monkeypatch):
    monkeypatch.setattr(checks, "_openai_chat_stdlib", lambda *a, **k: "")
    ev = checks.llm_judge("is it polite")
    with pytest.raises(ValueError):
        ev("hello", {})
    monkeypatch.setattr(checks, "_openai_chat_stdlib", lambda *a, **k: "0/10")
    assert ev("hello", {}).passed is False


# --- eval return values ----------------------------------------------------------


@pytest.mark.parametrize("bad", [0.2, 1, 0, "fail", "", b"x", {"score": 1}])
def test_non_verdict_return_values_raise(bad):
    with pytest.raises(TypeError):
        coerce_outcome(bad)


def test_verdict_like_return_values_are_accepted():
    assert coerce_outcome(True).passed is True
    assert coerce_outcome(None).passed is False  # re.search() no-match idiom
    assert coerce_outcome(re.search("a", "abc")).passed is True
    assert coerce_outcome(["hit"]).passed is True  # findall idiom
    o = coerce_outcome({"pass": False, "score": 0.3, "reason": "why"})
    assert o.passed is False and o.score == 0.3 and o.detail == "why"


def test_non_finite_score_raises():
    with pytest.raises(ValueError):
        coerce_outcome(EvalOutcome(passed=False, score=float("nan"), threshold=0.5))
    with pytest.raises(ValueError):
        coerce_outcome({"passed": True, "score": float("inf")})


def test_nan_metric_errors_mutants_instead_of_killing_them():
    def metric(output, case):
        score = 0.9 if output == "ok" else float("nan")
        return EvalOutcome(passed=score >= 0.5, score=score, threshold=0.5)

    cfg = MutEvalConfig(
        prompt=PROMPT,
        cases=[{"x": 1}],
        run=lambda p, c: "ok" if p == PROMPT else p,  # baseline scores 0.9
        evals=[metric],
    )
    r = run_mutation_testing(cfg, operators=["flip_negation"])
    assert r.killed == 0 and r.errored == len(r.outcomes) > 0
    assert r.status == NO_EVALUATED_MUTANTS and r.score is None


# --- the baseline is judged by the mutants' rule ---------------------------------


def test_baseline_graded_runs_per_mutant_times():
    # 2 of 3 baseline runs fail: by the strict-majority rule the ORIGINAL would
    # be "killed", so every kill it hands a mutant is noise -> invalid.
    r = run_mutation_testing(_cfg(_Pattern([True, False, False], [True]), runs=3))
    assert r.status == BASELINE_FAILED and r.score is None
    assert math.isclose(r.baseline_pass_rate, 1 / 3)
    assert "itself be 'killed'" in format_report(r, use_color=False)


def test_baseline_that_ties_is_not_a_pass():
    r = run_mutation_testing(_cfg(_Pattern([True, False], [True]), runs=2))
    assert r.status == BASELINE_FAILED and r.baseline_pass_rate == 0.5


def test_single_run_baseline_is_unchanged():
    calls = itertools.count()

    def ev(output, case):
        next(calls)
        return True

    run_mutation_testing(_cfg(ev), operators=["drop_few_shot_example"])  # 0 mutants
    assert next(calls) == 1  # graded exactly once, as before


# --- noise kills -----------------------------------------------------------------


def test_kills_on_unchanged_output_are_noise_not_detection():
    # The system ignores the prompt; a flaky judge still "kills" mutants 2/3 of
    # the time. None of that is detection.
    ev = _Pattern([True, True, True], [False, False, True])
    r = run_mutation_testing(_cfg(ev, runs=3), operators=["weaken_modals"])
    assert r.status == VALID and r.killed > 0
    assert len(r.noise_kills) == r.killed  # every kill matched the baseline
    assert r.score == 1.0  # the raw score is what happened...
    assert r.effective_score is None  # ...but nothing changed behavior
    out = format_report(r, use_color=False)
    assert "noise kill" in out
    assert result_to_dict(r)["noise_kills"] == r.killed


def test_default_run_has_no_noise_kills():
    cfg = MutEvalConfig(
        prompt=PROMPT,
        cases=[{"x": 1}],
        run=lambda p, c: p,  # output tracks the prompt
        evals=[lambda o, c: "must" in o],
    )
    r = run_mutation_testing(cfg, operators=["weaken_modals"])
    assert r.killed > 0 and r.noise_kills == []
    assert all(o.output_changed is True for o in r.outcomes if o.killed)


# --- unresolved ties are gated ----------------------------------------------------


def test_all_ties_is_no_confident_score():
    ev = _Pattern([True, True], [False, True])  # every mutant ties 1-1
    r = run_mutation_testing(_cfg(ev, runs=2), operators=["weaken_modals"])
    assert r.resolved == 0 and r.status == NO_CONFIDENT_SCORE


def test_mostly_ties_is_partial_unresolved_by_default():
    # First mutant fails both runs (killed), the rest tie: 1 resolved of N.
    ev = _Pattern([True, True, False, False], [False, True])
    r = run_mutation_testing(_cfg(ev, runs=2), operators=["weaken_modals"])
    assert r.resolved == 1 and r.score == 1.0  # the old "100%"...
    assert r.status == PARTIAL_UNRESOLVED  # ...is no longer a valid run
    assert "INVALID for CI" in format_report(r, use_color=False)

    ev = _Pattern([True, True, False, False], [False, True])
    ok = run_mutation_testing(
        _cfg(ev, runs=2, max_unresolved_rate=1.0), operators=["weaken_modals"]
    )
    assert ok.status == VALID


def test_max_unresolved_rate_is_validated():
    with pytest.raises(ValueError):
        _cfg(lambda o, c: True, max_unresolved_rate=1.5)


# --- which mutants ran ------------------------------------------------------------


def test_robustness_only_vs_regression_errored():
    cfg = MutEvalConfig(
        prompt="- You should cite sources.",
        cases=[{"x": 1}],
        run=lambda p, c: p,
        evals=[lambda o, c: True],
    )
    r = run_mutation_testing(cfg, operators=["paraphrase_instruction"])
    assert r.status == NO_SCORED_MUTANTS

    def run(prompt, case):  # every REGRESSION mutant (it drops "must not") errors
        if "must not" not in prompt:
            raise RuntimeError("boom")
        return prompt

    cfg = MutEvalConfig(
        prompt="- You should cite sources.\n- You must not guess.",
        cases=[{"x": 1}],
        run=run,
        evals=[lambda o, c: True],
        max_error_rate=1.0,
    )
    r = run_mutation_testing(cfg, operators=["paraphrase_instruction", "flip_negation"])
    assert r.robustness and r.evaluated == 0
    assert r.status == NO_EVALUATED_MUTANTS  # NOT "only robustness ran"


# --- CLI gates --------------------------------------------------------------------

_TIED = f"""
import itertools
from muteval import MutEvalConfig
_it = itertools.chain([True, True], itertools.cycle([False, True]))
config = MutEvalConfig(
    prompt={PROMPT!r}, cases=[{{"x": 1}}], run=lambda p, c: "ok",
    evals=[lambda o, c: next(_it)], runs_per_mutant=2, operators=["weaken_modals"],
)
"""

_NOISE = f"""
import itertools
from muteval import MutEvalConfig
_it = itertools.chain([True] * 3, itertools.cycle([False, False, True]))
config = MutEvalConfig(
    prompt={PROMPT!r}, cases=[{{"x": 1}}], run=lambda p, c: "ok",
    evals=[lambda o, c: next(_it)], runs_per_mutant=3, operators=["weaken_modals"],
)
"""


def _write(tmp_path, body):
    p = tmp_path / "cfg.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    "gate", [["--fail-under", "80"], ["--fail-on-severity", "high"], []]
)
def test_all_tied_run_exits_2_under_every_gate(tmp_path, capsys, gate):
    code = main(["run", "--config", _write(tmp_path, _TIED), "--no-color", *gate])
    assert code == 2  # not 0 (green CI), not a TypeError crash
    assert "tied" in capsys.readouterr().err


def test_noise_kills_cannot_pass_fail_under(tmp_path, capsys):
    code = main(
        ["run", "--config", _write(tmp_path, _NOISE), "--no-color", "--fail-under", "50"]
    )
    assert code == 1  # raw score is 100%, but no kill was detection
    assert "noise kills" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0.8", "150", "-1"])
def test_fail_under_must_be_a_percent(tmp_path, capsys, value):
    code = main(
        ["run", "--config", _write(tmp_path, _NOISE), "--no-color", "--fail-under", value]
    )
    assert code == 2
    assert "--fail-under" in capsys.readouterr().err
