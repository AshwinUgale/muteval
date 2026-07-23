"""v0.6 gate: every probe's headline signal is MONOTONIC in injected severity,
and hits its endpoints. Deterministic, no keys.

If a probe's number didn't move smoothly with how-bad-the-input-is, it couldn't
be trusted to rank suites — this pins that property for each one.
"""

from __future__ import annotations

import math

import pytest

from muteval.probes.discrimination import _auc
from muteval.probes.judge_bias import position_bias, verbosity_bias
from muteval.probes.redundancy import _spearman
from muteval.probes.threshold_calibration import _verdict
from muteval.stats import cohens_kappa, icc, interval


def _nonincreasing(xs):
    return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))


def _nondecreasing(xs):
    return all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))


# --- discrimination: AUC falls as good/bad overlap grows ---------------------

def test_discrimination_auc_monotonic_in_overlap():
    good = [10.0, 11.0, 12.0, 13.0]
    aucs = [_auc(good, [1.0 + s, 2.0 + s, 3.0 + s, 4.0 + s])[0] for s in (0, 4, 8, 10, 14)]
    assert _nonincreasing(aucs)
    assert aucs[0] == 1.0            # fully separated
    assert aucs[-1] <= 0.5 + 1e-9    # fully overlapped / inverted


# --- judge_reliability (ICC): reliability falls as run-to-run spread grows ----

def test_reliability_icc_monotonic_in_noise():
    subjects = [1.0, 4.0, 7.0, 10.0, 13.0]
    iccs = []
    for d in (0.0, 0.5, 1.5, 3.0, 6.0):  # deterministic within-subject spread
        matrix = [[s - d, s + d, s - d / 2, s + d / 2] for s in subjects]
        iccs.append(icc(matrix))
    assert _nonincreasing(iccs)
    assert iccs[0] == 1.0             # no noise -> perfect reliability


# --- redundancy (Spearman): correlation rises as metrics align ---------------

def test_redundancy_spearman_monotonic_in_alignment():
    a = list(range(12))
    corrs = []
    # move from anti-aligned to aligned by swapping fewer and fewer pairs
    for swaps in (6, 4, 2, 1, 0):
        b = list(a)
        for i in range(swaps):
            b[i], b[len(b) - 1 - i] = b[len(b) - 1 - i], b[i]
        corrs.append(_spearman(a, b))
    assert _nondecreasing(corrs)
    assert math.isclose(corrs[-1], 1.0, abs_tol=1e-9)  # fully aligned


# --- statistical_adequacy (interval width): shrinks as n grows ---------------

@pytest.mark.parametrize("method", ["wilson", "jeffreys"])
def test_adequacy_interval_width_monotonic_in_n(method):
    widths = [(hi - lo) for lo, hi in
              (interval(round(0.9 * n), n, method=method) for n in (5, 10, 40, 160, 640))]
    assert _nonincreasing(widths)


# --- threshold_calibration: verdict crosses ok as the line moves --------------

def test_threshold_calibration_verdict_gradient():
    good, bad = [0.9, 0.85], [0.2, 0.25]  # separate at ~0.55
    verdicts = [_verdict(good, bad, t)["verdict"] for t in (0.1, 0.3, 0.55, 0.8, 0.95)]
    # too_lenient below the band, ok inside, too_strict above
    assert verdicts[0] == "too_lenient"
    assert verdicts[2] == "ok"
    assert verdicts[-1] == "too_strict"


# --- judge_bias (position): bias rate tracks the injected bias fraction --------

def _positional_judge(fraction, n):
    """Picks position A for the first `fraction` of pairs (biased), content
    otherwise (order-invariant)."""
    cutoff = fraction * n

    def judge(a, b, case):
        if case["i"] < cutoff:
            return "A"
        return "A" if "CORRECT" in a else "B" if "CORRECT" in b else "tie"

    return judge


def test_judge_bias_position_monotonic_in_injected_bias():
    n = 10
    pairs = [("the CORRECT one", "a wrong one", {"i": i}) for i in range(n)]
    rates = [position_bias(_positional_judge(f, n), pairs) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert _nondecreasing(rates)
    assert rates[0] == 0.0 and rates[-1] == 1.0


# --- human_agreement (Cohen's kappa): rises as rater agreement grows ----------

def test_human_agreement_kappa_monotonic():
    # 10 items, machine alternates T/F; human matches the first k, flips the rest.
    machine = [i % 2 == 0 for i in range(10)]
    kappas = []
    for k in (2, 4, 6, 8, 10):
        human = [machine[i] if i < k else not machine[i] for i in range(10)]
        kappas.append(cohens_kappa(machine, human))
    assert _nondecreasing(kappas)
    assert math.isclose(kappas[-1], 1.0, abs_tol=1e-9)  # full agreement
