"""Tests for the deepeval adapter.

These use a stub metric and a custom test_case_factory so they run without
deepeval installed — we're testing the adapter's wiring, not deepeval itself.
The adapter now returns an EvalOutcome carrying score + threshold.
"""

from types import SimpleNamespace

from muteval.evals import EvalOutcome
from muteval.adapters.deepeval import metric_to_eval, metrics_to_evals


class StubMetric:
    """Mimics the deepeval metric interface: measure() + is_successful()."""

    def __init__(self, score, threshold=0.5):
        self.score = score
        self.threshold = threshold
        self.seen = None

    def measure(self, test_case):
        self.seen = test_case
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


def _factory(output, case):
    # Stand-in for a deepeval LLMTestCase.
    return SimpleNamespace(input=case.get("input"), actual_output=output)


def test_passing_metric_yields_outcome():
    ev = metric_to_eval(StubMetric(score=0.9), test_case_factory=_factory)
    outcome = ev("some output", {"input": "q"})
    assert isinstance(outcome, EvalOutcome)
    assert outcome.passed is True
    assert outcome.score == 0.9
    assert outcome.threshold == 0.5


def test_failing_metric_yields_false():
    ev = metric_to_eval(StubMetric(score=0.1), test_case_factory=_factory)
    outcome = ev("some output", {"input": "q"})
    assert outcome.passed is False
    # A failing metric just below threshold still carries its score.
    assert outcome.score == 0.1


def test_outcome_is_truthy_for_pass():
    # EvalOutcome.__bool__ mirrors `passed`, so it drops into bool contexts.
    ev = metric_to_eval(StubMetric(score=0.9), test_case_factory=_factory)
    assert bool(ev("out", {"input": "q"})) is True


def test_factory_receives_output_and_case():
    metric = StubMetric(score=1.0)
    ev = metric_to_eval(metric, test_case_factory=_factory)
    ev("the actual output", {"input": "the question"})
    assert metric.seen.actual_output == "the actual output"
    assert metric.seen.input == "the question"


def test_metrics_to_evals_wraps_each():
    metrics = [StubMetric(0.9), StubMetric(0.1)]
    evals = metrics_to_evals(metrics, test_case_factory=_factory)
    assert len(evals) == 2
    results = [ev("out", {"input": "q"}).passed for ev in evals]
    assert results == [True, False]


def test_construction_is_lazy_no_deepeval_needed():
    # Building the eval must NOT import deepeval (only invoking with the
    # default factory would). Construction here uses no factory and must
    # still not raise.
    ev = metric_to_eval(StubMetric(0.9))
    assert callable(ev)
