"""muteval's central claim, as a test.

Loads the controlled eval-quality experiment and asserts the EFFECTIVE mutation
score rises monotonically with suite coverage and hits both endpoints
(empty suite = 0%, complete suite = 100%). If this ever breaks, muteval's score
has stopped meaning what we say it means — so it's a test, not a demo.
"""

import importlib.util
import pathlib

from muteval import MutEvalConfig, run_mutation_testing

_EXP = (
    pathlib.Path(__file__).resolve().parent.parent
    / "validation" / "eval_quality_experiment" / "run_experiment.py"
)


def _load_experiment():
    spec = importlib.util.spec_from_file_location("eval_quality_exp", _EXP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective_scores():
    m = _load_experiment()
    scores = []
    for _label, (evals, names) in m.SUITES.items():
        cfg = MutEvalConfig(
            system=m.SYSTEM, cases=[m.CASE], run=m.run, evals=evals, eval_names=names
        )
        scores.append(run_mutation_testing(cfg).effective_score)
    return scores


def test_score_rises_monotonically_with_suite_quality():
    scores = _effective_scores()
    assert all(b >= a for a, b in zip(scores, scores[1:])), f"not monotonic: {scores}"


def test_empty_suite_scores_zero():
    assert _effective_scores()[0] == 0.0


def test_complete_suite_scores_one():
    assert _effective_scores()[-1] == 1.0
