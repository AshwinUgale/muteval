"""Controlled validation for the judge_reliability probe.

Injects judges with a KNOWN noise level (seeded): a deterministic judge is
perfectly reliable (0 flips, alpha=1); a coin-flip judge is unreliable (high flip
rate, alpha ~ 0). Deterministic given the seed, no keys.
"""

import random

from muteval import MutEvalConfig
from muteval.evals import EvalOutcome


def _reliable_eval():
    def ev(output, case):
        return EvalOutcome(passed=True)      # always the same verdict
    ev.__name__ = "reliable"
    return ev


def _noisy_eval(seed=0):
    rng = random.Random(seed)

    def ev(output, case):
        return EvalOutcome(passed=rng.random() < 0.5)  # coin flip -> max noise
    ev.__name__ = "noisy"
    return ev


def _cfg(evals, names, n=16):
    return MutEvalConfig(
        prompt="p", cases=[{"i": i} for i in range(n)], run=lambda p, c: "x",
        evals=evals, eval_names=names,
    )


def reliable_config():
    return _cfg([_reliable_eval()], ["reliable"])


def noisy_config():
    return _cfg([_noisy_eval(0)], ["noisy"])
