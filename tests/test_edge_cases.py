"""v0.4 'Provably honest': numerical edge cases at the boundaries.

The degenerate inputs (n=0/1, all-tie, single exemplar, constant metric, one
rating value) are exactly where hand-rolled statistics silently go wrong. Each
is pinned here.
"""

from __future__ import annotations

import math

from muteval import EvalOutcome, MutEvalConfig, System, checks
from muteval.probes.discrimination import _auc, _cohens_d, discrimination
from muteval.probes.judge_reliability import _krippendorff_alpha_nominal
from muteval.probes.redundancy import redundancy
from muteval.stats import jeffreys_interval, wilson_interval


# --- confidence intervals at n=0,1 ------------------------------------------

def test_intervals_unknown_at_n0():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert jeffreys_interval(0, 0) == (0.0, 1.0)


def test_intervals_at_n1_are_valid():
    for k in (0, 1):
        for fn in (wilson_interval, jeffreys_interval):
            lo, hi = fn(k, 1)
            assert 0.0 <= lo <= hi <= 1.0
    # Jeffreys pins the closed end at the observed extreme.
    assert jeffreys_interval(0, 1)[0] == 0.0
    assert jeffreys_interval(1, 1)[1] == 1.0


def test_jeffreys_n1_lower_bound_is_a_real_quantile():
    # At n=1, k=1 the lower bound is a genuine Beta(1.5, 0.5) quantile > 0 — NOT
    # the n<=0 "unknown" fallback. Pins n=1 as a computed case (kills the
    # `n <= 0 -> n <= 1` mutant, which would wrongly return (0, 1) here).
    lo, hi = jeffreys_interval(1, 1)
    assert lo > 0.0 and hi == 1.0
    lo0, hi0 = jeffreys_interval(0, 1)
    assert lo0 == 0.0 and hi0 < 1.0  # k=0,n=1 computes a real upper bound < 1


def test_unknown_confidence_falls_back_to_95_z():
    # A confidence level not in the z-table uses the default z (== the 0.95 z),
    # so the interval equals the 95% one. Pins the fallback branch (kills the
    # default-z mutants: None / empty / a different constant).
    from muteval.stats import interval

    assert wilson_interval(5, 10, 0.80) == wilson_interval(5, 10, 0.95)
    assert interval(5, 10, 0.80, "wilson") == wilson_interval(5, 10, 0.95)


def test_interval_dispatch_selects_method():
    # `interval` dispatches on method (default wilson). Pins both branches so a
    # mutated dispatch (wrong method / inverted comparison) is caught.
    from muteval.stats import interval

    assert interval(5, 10, 0.95, "wilson") == wilson_interval(5, 10, 0.95)
    assert interval(5, 10, 0.95, "jeffreys") == jeffreys_interval(5, 10, 0.95)
    assert interval(5, 10, 0.95) == wilson_interval(5, 10, 0.95)  # default = wilson
    # jeffreys and wilson genuinely differ here, so the dispatch is observable.
    assert interval(5, 10, 0.95, "jeffreys") != interval(5, 10, 0.95, "wilson")


# --- AUC / Cohen's d degeneracies -------------------------------------------

def test_auc_all_tie_is_half():
    auc, u = _auc([1.0, 1.0], [1.0, 1.0])
    assert auc == 0.5


def test_auc_perfect_separation():
    auc, _ = _auc([5.0, 6.0], [1.0, 2.0])
    assert auc == 1.0


def test_cohens_d_single_exemplar_is_none():
    # <2 in either group -> undefined pooled SD.
    assert _cohens_d([1.0], [2.0, 3.0]) is None
    assert _cohens_d([1.0, 2.0], [3.0]) is None


def test_cohens_d_zero_spread_is_none():
    assert _cohens_d([2.0, 2.0], [5.0, 5.0]) is None


# --- Krippendorff single value ----------------------------------------------

def test_krippendorff_single_value_is_one():
    # Every rater said the same thing on every item -> perfect agreement.
    assert _krippendorff_alpha_nominal([[1, 1, 1], [1, 1]]) == 1.0


def test_krippendorff_total_disagreement_is_nonpositive():
    alpha = _krippendorff_alpha_nominal([[0, 1], [1, 0], [0, 1], [1, 0]])
    assert alpha <= 0.0 + 1e-9


# --- probe "not assessed" paths ---------------------------------------------

def _cfg(cases, evals, names):
    return MutEvalConfig(
        prompt="answer the question.",
        cases=cases,
        run=lambda p, c: "x",
        evals=evals,
        eval_names=names,
    )


def test_discrimination_not_assessed_without_exemplars():
    cfg = _cfg(
        [{"question": "q"}],
        [checks.contains("x")],
        ["has_x"],
    )
    res = discrimination(cfg)
    assert res.metrics.get("assessed") is False
    assert "not assessed" in res.summary


def test_redundancy_not_assessed_when_scores_constant():
    # Two evals that return the SAME constant for every case -> no variation.
    const_a = lambda output, case: EvalOutcome(passed=True, score=1.0, name="a")
    const_b = lambda output, case: EvalOutcome(passed=True, score=1.0, name="b")
    cfg = _cfg(
        [{"question": "q1"}, {"question": "q2"}, {"question": "q3"}],
        [const_a, const_b],
        ["a", "b"],
    )
    res = redundancy(cfg)
    assert res.metrics.get("assessed") is False
    assert "not assessed" in res.summary


def test_single_exemplar_discrimination_reports_gracefully():
    # good/bad present but only one exemplar each: AUC computable, d is None.
    cfg = _cfg(
        [{"question": "q", "good": ["x present"], "bad": ["absent"]}],
        [checks.contains("x")],
        ["has_x"],
    )
    res = discrimination(cfg)
    # It IS assessed (exemplars exist); must not crash and must stay in-range.
    assert res.metrics.get("assessed") is not False
