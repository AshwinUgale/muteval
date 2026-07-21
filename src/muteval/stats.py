"""Small statistics helpers (dependency-free).

A mutation score is a proportion (killed / evaluated), so a bare percentage
hides how uncertain it is — 19/20 is not "95%", it's 95% with a wide interval.
We report a Wilson score confidence interval, which is well-behaved for small n
and near 0/1 (unlike the naive normal/Wald interval).
"""

from __future__ import annotations

import math
from typing import Tuple

# z for common two-sided confidence levels.
_Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion, in [0, 1].

    Returns (low, high). With n == 0 the proportion is unknown -> (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _Z.get(confidence, 1.9600)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


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
