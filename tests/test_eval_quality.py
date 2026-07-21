"""muteval's central claim, as a test — across MULTIPLE domains.

Loads each controlled eval-quality experiment and asserts the EFFECTIVE mutation
score rises monotonically with suite coverage and hits both endpoints
(empty suite = 0%, complete suite = 100%). Enforced on more than one domain so
the relationship isn't a single-example fluke. If this breaks, muteval's score
has stopped meaning what we say it means.
"""

import importlib.util
import pathlib

import pytest

from muteval import MutEvalConfig, run_mutation_testing

_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "validation" / "eval_quality_experiment"
)
_EXPERIMENTS = [
    _DIR / "run_experiment.py",              # domain 1: support bot
    _DIR / "run_experiment_codereview.py",   # domain 2: code review
]


def _load(path):
    spec = importlib.util.spec_from_file_location(f"exp_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective_scores(path):
    m = _load(path)
    return [
        run_mutation_testing(
            MutEvalConfig(
                system=m.SYSTEM, cases=[m.CASE], run=m.run,
                evals=evals, eval_names=names,
            )
        ).effective_score
        for evals, names in m.SUITES.values()
    ]


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda p: p.stem)
def test_score_rises_monotonically_with_suite_quality(path):
    scores = _effective_scores(path)
    assert all(b >= a for a, b in zip(scores, scores[1:])), (
        f"{path.stem}: not monotonic: {scores}"
    )


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda p: p.stem)
def test_empty_suite_scores_zero(path):
    assert _effective_scores(path)[0] == 0.0


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda p: p.stem)
def test_complete_suite_scores_one(path):
    assert _effective_scores(path)[-1] == 1.0
