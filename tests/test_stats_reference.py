"""v0.4 'Provably honest': cross-check the hand-rolled math against gold-standard
reference libraries. muteval's core is dependency-free, so every statistic is
re-implemented by hand — these tests prove those implementations match the
established libraries to tight tolerance. Skipped when the [verify] extras are
absent (they are test-only, never a runtime dependency).
"""

from __future__ import annotations

import math

import pytest

from muteval.probes.discrimination import _auc, _cohens_d
from muteval.probes.judge_reliability import _krippendorff_alpha_nominal
from muteval.probes.redundancy import _spearman
from muteval.stats import _beta_ppf, _betai, jeffreys_interval, wilson_interval

# Skip the whole module unless the reference libraries are installed.
statsmodels = pytest.importorskip("statsmodels.stats.proportion")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_special = pytest.importorskip("scipy.special")
sk_metrics = pytest.importorskip("sklearn.metrics")
krippendorff = pytest.importorskip("krippendorff")
pingouin = pytest.importorskip("pingouin")
np = pytest.importorskip("numpy")


CASES = [(0, 1), (1, 1), (1, 10), (5, 10), (9, 10), (19, 20), (0, 5), (5, 5), (50, 200), (137, 400)]


@pytest.mark.parametrize("k,n", CASES)
def test_wilson_matches_statsmodels(k, n):
    lo, hi = wilson_interval(k, n, 0.95)
    rlo, rhi = statsmodels.proportion_confint(k, n, alpha=0.05, method="wilson")
    assert math.isclose(lo, rlo, abs_tol=1e-6)
    assert math.isclose(hi, rhi, abs_tol=1e-6)


# Interior points only: at the extremes (k=0 or k=n) muteval follows the
# Brown-Cai-DasGupta (2001) convention of pinning the closed end to 0/1, whereas
# statsmodels returns the raw equal-tailed Beta quantile. That divergence is
# intentional and separately pinned in test_edge_cases.py.
@pytest.mark.parametrize("k,n", [(k, n) for (k, n) in CASES if 0 < k < n])
def test_jeffreys_matches_statsmodels_interior(k, n):
    lo, hi = jeffreys_interval(k, n, 0.95)
    rlo, rhi = statsmodels.proportion_confint(k, n, alpha=0.05, method="jeffreys")
    assert math.isclose(lo, rlo, abs_tol=1e-5)
    assert math.isclose(hi, rhi, abs_tol=1e-5)


@pytest.mark.parametrize(
    "good,bad",
    [
        ([3.0, 4.0, 5.0], [1.0, 2.0]),
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        ([1.0, 1.0, 2.0], [1.0, 2.0, 2.0]),  # ties across groups
        ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0]),
    ],
)
def test_auc_matches_sklearn(good, bad):
    auc, _ = _auc(good, bad)
    y_true = [1] * len(good) + [0] * len(bad)
    y_score = list(good) + list(bad)
    ref = sk_metrics.roc_auc_score(y_true, y_score)
    assert math.isclose(auc, ref, abs_tol=1e-9)


@pytest.mark.parametrize(
    "good,bad",
    [
        ([3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0]),
        ([10.0, 12.0, 9.0, 11.0], [5.0, 6.0, 4.0, 7.0]),
    ],
)
def test_cohens_d_matches_pingouin(good, bad):
    d = _cohens_d(good, bad)
    ref = float(pingouin.compute_effsize(good, bad, eftype="cohen"))
    assert d is not None
    assert math.isclose(d, ref, abs_tol=1e-6)


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0]),
        ([1.0, 2.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 3.0, 5.0]),  # ties
        ([5.0, 3.0, 8.0, 1.0, 9.0, 2.0], [4.0, 4.0, 7.0, 2.0, 8.0, 3.0]),
    ],
)
def test_spearman_matches_scipy(a, b):
    ours = _spearman(a, b)
    ref = scipy_stats.spearmanr(a, b).statistic
    assert math.isclose(ours, ref, abs_tol=1e-9)


@pytest.mark.parametrize(
    "items",
    [
        [[1, 1], [2, 2], [1, 2], [3, 3], [2, 1]],
        [[0, 0, 1], [1, 1, 1], [2, 2, 2], [0, 1, 0], [2, 2, 1]],
    ],
)
def test_krippendorff_nominal_matches_library(items):
    ours = _krippendorff_alpha_nominal(items)
    # Library wants reliability_data as raters x units; our items are units x raters.
    reliability_data = np.array(items, dtype=float).T
    ref = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
    assert math.isclose(ours, ref, abs_tol=1e-6)


# --- numerical internals: the incomplete-beta engine ------------------------
# These pin _betai / _betacf / _beta_ppf directly (not just the end-to-end
# interval), which is what kills the mutmut survivors clustered in the
# continued-fraction internals. See docs/MUTMUT.md.

_ABX = [
    (0.5, 0.5, 0.3),
    (2.0, 3.0, 0.4),
    (5.5, 2.5, 0.7),
    (10.0, 20.0, 0.35),
    (1.0, 1.0, 0.9),
    (50.0, 50.0, 0.5),
    (3.0, 7.0, 0.05),
    (7.0, 3.0, 0.95),
    (100.5, 0.5, 0.999),
    (0.5, 100.5, 0.001),
]


@pytest.mark.parametrize("a,b,x", _ABX)
def test_betai_matches_scipy_betainc(a, b, x):
    # _betai is the regularized incomplete beta I_x(a,b) == scipy.special.betainc.
    assert math.isclose(_betai(a, b, x), float(scipy_special.betainc(a, b, x)), abs_tol=1e-9)


@pytest.mark.parametrize("a,b", [(0.5, 0.5), (2.0, 3.0), (5.5, 2.5), (10.0, 20.0), (50.0, 50.0)])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99])
def test_beta_ppf_matches_scipy_betaincinv(p, a, b):
    # _beta_ppf inverts I_x(a,b) == scipy.special.betaincinv. Bisection is exact
    # to ~2^-100; loosen only enough to absorb betai's own ~3e-12 tolerance.
    ours = _beta_ppf(p, a, b)
    ref = float(scipy_special.betaincinv(a, b, p))
    assert math.isclose(ours, ref, abs_tol=1e-6)
