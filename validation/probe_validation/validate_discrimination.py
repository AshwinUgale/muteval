"""Controlled validation for the discrimination probe.

Constructed ground truth: exemplar output strings encode their own score
("s=0.9"), so we control the exact good/bad score distributions. Proves the probe
flags a metric that can't separate good from bad, and — crucially — that the AUC
upgrade catches an overlapping metric with a big *mean gap* that the old raw-gap
probe would have passed. Deterministic, no keys.
"""

from muteval import MutEvalConfig
from muteval.evals import EvalOutcome


def _num(output):
    return float(output.split("=")[1])


def _parser_eval():
    def ev(output, case):
        v = _num(output)
        return EvalOutcome(passed=v >= 0.5, score=v)
    ev.__name__ = "parsed"
    return ev


def _cfg(good, bad):
    case = {"good": [f"s={v}" for v in good], "bad": [f"s={v}" for v in bad]}
    return MutEvalConfig(
        prompt="p", cases=[case], run=lambda p, c: "x",
        evals=[_parser_eval()], eval_names=["parsed"],
    )


def discriminating_config():
    """Good clearly outranks bad -> AUC 1.0."""
    return _cfg(good=[1.0, 0.9, 0.8, 1.0], bad=[0.0, 0.1, 0.2, 0.0])


def nondiscriminating_config():
    """Identical scores for good and bad -> AUC 0.5 (coin flip)."""
    return _cfg(good=[0.5, 0.5, 0.5, 0.5], bad=[0.5, 0.5, 0.5, 0.5])


def overlapping_large_gap_config():
    """One good outlier inflates the MEAN gap to 2.5, but ranks overlap -> AUC 0.625.
    The old raw-gap probe (threshold 0.3) would PASS this; AUC correctly fails it."""
    return _cfg(good=[10.0, 0.0, 0.0, 0.0], bad=[0.0, 0.0, 0.0, 0.0])
