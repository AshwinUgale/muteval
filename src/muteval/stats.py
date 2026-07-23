"""Small statistics helpers (dependency-free).

A mutation score is a proportion (killed / evaluated), so a bare percentage
hides how uncertain it is — 19/20 is not "95%", it's 95% with a wide interval.
We report a Wilson score confidence interval, which is well-behaved for small n
and near 0/1 (unlike the naive normal/Wald interval).
"""

from __future__ import annotations

import math
from typing import Tuple

# z for common two-sided confidence levels (full precision: norm.ppf(1-alpha/2),
# so the hand-rolled Wilson interval matches reference libraries to ~1e-9).
_Z = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion, in [0, 1].

    Returns (low, high). With n == 0 the proportion is unknown -> (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _Z.get(confidence, _Z[0.95])  # unknown level -> precise 95% z
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta (Numerical Recipes)."""
    fpmin, eps, maxit = 1e-300, 3e-12, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), dependency-free."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b) via bisection (I_x is increasing in x)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def jeffreys_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Jeffreys (Beta-Binomial) credible interval for a binomial proportion.

    The Bayesian interval with a Beta(1/2, 1/2) prior — the small-n recommendation
    from Brown/Cai/DasGupta (2001) and the LLM-eval-specific "Don't Use the CLT in
    LLM Evals" (Bowyer et al., ICML 2025). Degrades more gracefully than Wilson at
    very small n. Returns (low, high) in [0, 1]; (0, 1) when n == 0.
    """
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    a, b = successes + 0.5, n - successes + 0.5
    lo = 0.0 if successes == 0 else _beta_ppf(alpha / 2.0, a, b)
    hi = 1.0 if successes == n else _beta_ppf(1.0 - alpha / 2.0, a, b)
    return (max(0.0, lo), min(1.0, hi))


def interval(successes: int, n: int, confidence: float = 0.95, method: str = "wilson"):
    """Dispatch to a confidence-interval method: 'wilson' (default) or 'jeffreys'."""
    if method == "jeffreys":
        return jeffreys_interval(successes, n, confidence)
    return wilson_interval(successes, n, confidence)


def min_samples_for_lower_bound(
    observed_rate: float, target: float, confidence: float = 0.95, cap: int = 100000
) -> int:
    """Smallest n whose Wilson lower bound exceeds `target`, at `observed_rate`.

    e.g. min_samples_for_lower_bound(0.95, 0.90) ~ how many cases you'd need,
    while observing ~95%, to defend "better than 90%".
    """
    if observed_rate <= target:
        return cap
    for n in range(1, cap + 1):
        low, _ = wilson_interval(round(observed_rate * n), n, confidence)
        if low > target:
            return n
    return cap


def min_samples_for_precision(
    rate: float, half_width: float, confidence: float = 0.95, cap: int = 100000
) -> int:
    """Smallest n whose Wilson CI half-width <= `half_width` at `rate`.

    e.g. how many cases you need for a +/-10% interval around an observed rate.
    """
    for n in range(1, cap + 1):
        low, high = wilson_interval(round(rate * n), n, confidence)
        if (high - low) / 2 <= half_width:
            return n
    return cap
