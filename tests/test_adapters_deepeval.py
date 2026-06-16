"""Tests for the deepeval adapter.

These use a stub metric and a custom test_case_factory so they run without
deepeval installed — we're testing the adapter's wiring, not deepeval itself.
"""

from types import SimpleNamespace

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


def test_passing_metric_yields_true():
    ev = metric_to_eval(StubMetric(score=0.9), test_case_factory=_factory)
    assert ev("some output", {"input": "q"}) is True


def test_failing_metric_yields_false():
    ev = metric_to_eval(StubMetric(score=0.1), test_case_factory=_factory)
    assert ev("some output", {"input": "q"}) is False


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
    results = [ev("out", {"input": "q"}) for ev in evals]
    assert results == [True, False]


def test_construction_is_lazy_no_deepeval_needed():
    # Building the eval must NOT import deepeval (only invoking with the
    # default factory would). Construction here uses no factory and must
    # still not raise.
    ev = metric_to_eval(StubMetric(0.9))
    assert callable(ev)
