"""v0.4 'Provably honest': Monte-Carlo coverage of the confidence intervals.

We claim a "95% interval" on the mutation score, so we simulate to check the
intervals actually cover the true proportion ~95% of the time — the difference
between "looks rigorous" and "is rigorous".

Honest nuance (Brown, Cai & DasGupta 2001, *Interval Estimation for a Binomial
Proportion*): NO binomial interval achieves exact 95% coverage at finite n — the
discreteness of the binomial makes coverage oscillate ("saw-tooth") with n and p,
dipping below and rising above nominal, especially at small n and extreme p.
Wilson and Jeffreys are the *recommended* small-n intervals precisely because
they stay close to nominal and never collapse. So the meaningful, achievable
guarantees are:

  1. no cell *severely* undercovers (coverage floor), and
  2. coverage averages close to nominal across the grid.

A naive "coverage in [0.93,0.97] on every (p,n) cell" gate is NOT achievable and
would be dishonest to assert — this test encodes what the mathematics actually
allows.
"""

from __future__ import annotations

import random

import pytest

from muteval.stats import jeffreys_interval, wilson_interval

GRID_P = (0.5, 0.8, 0.95, 0.99)
GRID_N = (10, 30, 100)
DRAWS = 4000
SEED = 20260701

# Floors/ceilings that reflect the literature, not wishful thinking.
CELL_FLOOR = 0.90          # never severely undercover
CELL_CEILING = 0.995       # never absurdly conservative
MEAN_LO, MEAN_HI = 0.935, 0.975  # average coverage ~ nominal


def _coverage(method, p, n, draws=DRAWS, seed=SEED):
    lut = [method(k, n, 0.95) for k in range(n + 1)]  # one interval per distinct k
    rng = random.Random(seed)
    hit = 0
    for _ in range(draws):
        k = sum(1 for _ in range(n) if rng.random() < p)
        lo, hi = lut[k]
        if lo <= p <= hi:
            hit += 1
    return hit / draws


@pytest.mark.slow
@pytest.mark.parametrize("method", [wilson_interval, jeffreys_interval])
def test_ci_coverage_is_honest(method):
    cells = {}
    for p in GRID_P:
        for n in GRID_N:
            cells[(p, n)] = _coverage(method, p, n)

    worst = min(cells.values())
    best = max(cells.values())
    mean = sum(cells.values()) / len(cells)

    # 1. No cell severely undercovers (the property that matters most: the
    #    interval must not be falsely narrow).
    assert worst >= CELL_FLOOR, f"severe undercoverage: {cells}"
    # 2. No cell is absurdly conservative.
    assert best <= CELL_CEILING, f"pathological overcoverage: {cells}"
    # 3. On average, coverage sits near the nominal 95%.
    assert MEAN_LO <= mean <= MEAN_HI, f"mean coverage {mean:.3f} off-nominal: {cells}"
