"""v0.4 'Provably honest': property-based invariants (Hypothesis).

For random inputs, the statistics helpers and the runner must never produce an
impossible value (a probability outside [0,1], a CI that doesn't contain its
point estimate, a rank vector that isn't a permutation, killed>evaluated, ...).
These are the invariants the whole product's honesty rests on.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from muteval.probes.discrimination import _auc, _cohens_d
from muteval.probes.judge_reliability import _krippendorff_alpha_nominal
from muteval.probes.redundancy import _pearson, _ranks, _spearman
from muteval.stats import (
    jeffreys_interval,
    min_samples_for_lower_bound,
    min_samples_for_precision,
    wilson_interval,
)

# --- confidence intervals ----------------------------------------------------

_n_and_k = st.integers(min_value=0, max_value=10_000).flatmap(
    lambda n: st.tuples(st.just(n), st.integers(min_value=0, max_value=n))
)


@settings(max_examples=400)
@given(_n_and_k, st.sampled_from([0.90, 0.95, 0.99]))
def test_wilson_bounds_and_contains_point(nk, conf):
    n, k = nk
    lo, hi = wilson_interval(k, n, conf)
    assert 0.0 <= lo <= hi <= 1.0
    if n > 0:
        p = k / n
        assert lo - 1e-9 <= p <= hi + 1e-9  # Wilson interval contains the MLE


@settings(max_examples=400)
@given(_n_and_k, st.sampled_from([0.90, 0.95, 0.99]))
def test_jeffreys_bounds_and_contains_point(nk, conf):
    n, k = nk
    lo, hi = jeffreys_interval(k, n, conf)
    assert 0.0 <= lo <= hi <= 1.0
    if n > 0:
        p = k / n
        # Jeffreys can pull slightly off the raw MLE at the extremes; allow slack.
        assert lo - 1e-6 <= p <= hi + 1e-6


@settings(max_examples=200)
@given(_n_and_k)
def test_higher_confidence_never_narrows(nk):
    n, k = nk
    lo90, hi90 = wilson_interval(k, n, 0.90)
    lo99, hi99 = wilson_interval(k, n, 0.99)
    assert (hi99 - lo99) >= (hi90 - lo90) - 1e-9


@settings(max_examples=100)
@given(
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.0, max_value=0.49),
)
def test_min_samples_lower_bound_range(rate, target):
    n = min_samples_for_lower_bound(rate, target)
    assert 1 <= n <= 100_000
    if rate <= target:
        assert n == 100_000  # unprovable -> capped


@settings(max_examples=100)
@given(st.floats(min_value=0.01, max_value=0.99), st.floats(min_value=0.02, max_value=0.5))
def test_min_samples_precision_range(rate, half_width):
    n = min_samples_for_precision(rate, half_width)
    assert 1 <= n <= 100_000


# --- ranks / correlation -----------------------------------------------------

_floatlist = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)


@settings(max_examples=300)
@given(_floatlist)
def test_ranks_are_a_valid_average_rank_permutation(v):
    r = _ranks(v)
    assert len(r) == len(v)
    # Sum of average ranks == sum of 1..n regardless of ties.
    assert math.isclose(sum(r), len(v) * (len(v) + 1) / 2.0, rel_tol=0, abs_tol=1e-6)
    assert all(1.0 <= x <= len(v) for x in r)
    # Order is preserved: larger value -> not-smaller rank.
    for i in range(len(v)):
        for j in range(len(v)):
            if v[i] < v[j]:
                assert r[i] <= r[j] + 1e-9


@settings(max_examples=200)
@given(st.integers(min_value=2, max_value=40))
def test_spearman_pearson_in_range(n):
    a = list(range(n))
    b = list(range(n))
    assert math.isclose(_spearman(a, b), 1.0, abs_tol=1e-9)
    assert math.isclose(_spearman(a, list(reversed(b))), -1.0, abs_tol=1e-9)
    assert -1.0 - 1e-9 <= _pearson(a, b) <= 1.0 + 1e-9


# --- AUC / effect size / alpha ----------------------------------------------

_pair = st.tuples(
    st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=20),
    st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=20),
)


@settings(max_examples=300)
@given(_pair)
def test_auc_in_unit_interval(pair):
    good, bad = pair
    auc, u = _auc(good, bad)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= u <= len(good) * len(bad)


@settings(max_examples=200)
@given(_pair)
def test_cohens_d_finite_or_none(pair):
    good, bad = pair
    d = _cohens_d(good, bad)
    assert d is None or math.isfinite(d)


@settings(max_examples=200)
@given(st.lists(st.lists(st.integers(0, 3), min_size=2, max_size=5), min_size=1, max_size=30))
def test_krippendorff_alpha_upper_bounded_by_one(items):
    alpha = _krippendorff_alpha_nominal(items)
    assert alpha <= 1.0 + 1e-9  # alpha can be negative, never exceeds 1
