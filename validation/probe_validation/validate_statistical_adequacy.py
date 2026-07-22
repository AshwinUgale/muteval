"""Controlled validation for the statistical_adequacy probe.

Ground truth is the sample size itself: a tiny suite can't defend its pass rate
(wide CI -> WARN); a large one can (narrow CI -> PASS). Deterministic, no keys.
"""

from muteval import MutEvalConfig


def _suite(n_pass, n_fail):
    cases = [{"p": 1}] * n_pass + [{"p": 0}] * n_fail
    return MutEvalConfig(
        prompt="x", cases=cases, run=lambda p, c: "o",
        evals=[lambda o, c: c["p"] == 1],
    )


def small_config():
    """8 cases -> CI too wide -> WARN."""
    return _suite(8, 0)


def large_config():
    """200 cases -> CI tight -> PASS."""
    return _suite(200, 0)
