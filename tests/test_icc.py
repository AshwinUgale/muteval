"""v0.6: ICC(2,1) for numeric-judge reliability — behavior + edge cases.

(The gold-standard cross-check vs pingouin lives in test_stats_reference.py.)
"""

from __future__ import annotations

import math

from muteval import EvalOutcome, MutEvalConfig, System
from muteval.probes.judge_reliability import judge_reliability
from muteval.stats import icc


def test_perfect_agreement_is_one():
    # Raters identical per subject, subjects differ -> ICC == 1.
    m = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]]
    assert math.isclose(icc(m), 1.0, abs_tol=1e-9)


def test_no_between_subject_variance_is_low():
    # Subjects identical, raters differ -> almost no reliable signal (ICC <= 0-ish).
    m = [[1.0, 5.0, 9.0], [1.0, 5.0, 9.0], [1.0, 5.0, 9.0]]
    val = icc(m)
    assert val is not None and val < 0.1


def test_noisy_is_between_zero_and_one():
    m = [[1.0, 2.0, 1.0], [5.0, 4.0, 6.0], [9.0, 8.0, 9.0], [2.0, 3.0, 2.0]]
    val = icc(m)
    assert 0.0 < val < 1.0


def test_degenerate_returns_none():
    assert icc([[1.0, 1.0]]) is None            # n < 2 subjects
    assert icc([[1.0], [2.0]]) is None           # k < 2 raters
    assert icc([[1.0, 2.0], [3.0]]) is None       # ragged
    assert icc([[3.0, 3.0], [3.0, 3.0]]) is None  # zero total variance


def test_judge_reliability_reports_icc_for_scored_judge():
    # A deterministic scored judge: identical score every run, cases differ ->
    # ICC == 1. (Two cases -> two subjects, runs -> raters.)
    def scored(output, case):
        return EvalOutcome(passed=True, score=case["s"], threshold=0.5, name="j")

    cfg = MutEvalConfig(
        system=System(prompt="answer.", model="gpt-4o-mini"),
        cases=[{"q": "a", "s": 0.9}, {"q": "b", "s": 0.2}, {"q": "c", "s": 0.6}],
        run=lambda system, case: "out",
        evals=[scored], eval_names=["j"],
    )
    r = judge_reliability(cfg, runs=3)
    assert "j" in r.metrics["icc_by_eval"]
    assert math.isclose(r.metrics["icc_by_eval"]["j"], 1.0, abs_tol=1e-9)


def test_judge_reliability_omits_icc_for_boolean_judge():
    # A pass/fail judge with no score -> no ICC entry (nothing numeric to correlate).
    def boolean(output, case):
        return True

    cfg = MutEvalConfig(
        prompt="answer.", cases=[{"q": "a"}, {"q": "b"}],
        run=lambda p, c: "out", evals=[boolean], eval_names=["b"],
    )
    r = judge_reliability(cfg, runs=3)
    assert r.metrics["icc_by_eval"] == {}
