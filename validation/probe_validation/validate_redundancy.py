"""Controlled validation for the redundancy probe.

The ground truth is CONSTRUCTED — we know exactly which metrics are redundant, so
we can assert the probe flags them and leaves independent metrics alone. Crucially
it also proves the Spearman upgrade: a monotonic-but-nonlinear duplicate is caught
by Spearman (rank) correlation but MISSED by plain Pearson. Deterministic, no keys.
"""

from muteval import MutEvalConfig
from muteval.evals import EvalOutcome

# Controlled score vectors (verified):
#   spearman(LIN, MONO) = 1.00  but  pearson(LIN, MONO) = 0.785 (< 0.9)
#   spearman/pearson(LIN, INDEP) ~ 0    (INDEP, IND2 mutually ~uncorrelated too)
_LIN = [1, 2, 3, 4, 5, 6, 7, 8]
_DUP = list(_LIN)                                  # identical to LIN
_MONO = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.9, 1.0]  # monotonic, very nonlinear
_INDEP = [8, 1, 5, 2, 7, 4, 3, 6]
_IND2 = [3, 6, 1, 8, 4, 2, 7, 5]


def _metric(key):
    def ev(output, case):
        return EvalOutcome(passed=True, score=case[key])
    ev.__name__ = key
    return ev


def _build(columns):
    keys = list(columns)
    n = len(next(iter(columns.values())))
    cases = [{k: columns[k][i] for k in keys} for i in range(n)]
    return MutEvalConfig(
        prompt="p", cases=cases, run=lambda p, c: "x",
        evals=[_metric(k) for k in keys], eval_names=keys,
    )


def redundant_config():
    """lin, dup, mono are one construct; indep is independent."""
    return _build({"lin": _LIN, "dup": _DUP, "mono": _MONO, "indep": _INDEP})


def distinct_config():
    """Three mutually near-uncorrelated metrics — no redundancy."""
    return _build({"a": _LIN, "b": _INDEP, "c": _IND2})
