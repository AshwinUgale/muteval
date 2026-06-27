"""Tests for the RAGAS adapter.

These inject a stub sample_factory + score_fn so they run without ragas
installed — we're testing the adapter's wiring, not ragas itself.
"""

from muteval.evals import EvalOutcome
from muteval.adapters.ragas import metric_to_eval, metrics_to_evals


class StubMetric:
    """Mimics a ragas metric: name + single_turn_score(sample)."""

    def __init__(self, score, name="stub"):
        self._score = score
        self.name = name
        self.seen = None

    def single_turn_score(self, sample):
        self.seen = sample
        return self._score


def _sample_factory(output, case):
    return {"response": output, "user_input": case.get("input")}


def test_score_at_or_above_threshold_passes():
    ev = metric_to_eval(
        StubMetric(0.8), threshold=0.7, sample_factory=_sample_factory
    )
    out = ev("ans", {"input": "q"})
    assert isinstance(out, EvalOutcome)
    assert out.passed is True
    assert out.score == 0.8
    assert out.threshold == 0.7


def test_score_below_threshold_fails():
    ev = metric_to_eval(
        StubMetric(0.5), threshold=0.7, sample_factory=_sample_factory
    )
    assert ev("ans", {"input": "q"}).passed is False


def test_sample_factory_receives_output_and_case():
    metric = StubMetric(1.0)
    ev = metric_to_eval(metric, sample_factory=_sample_factory)
    ev("the answer", {"input": "the question"})
    assert metric.seen["response"] == "the answer"
    assert metric.seen["user_input"] == "the question"


def test_score_fn_override_used():
    # If a metric lacks single_turn_score, an explicit score_fn drives it.
    ev = metric_to_eval(
        object(),
        threshold=0.5,
        sample_factory=_sample_factory,
        score_fn=lambda sample: 0.9,
    )
    assert ev("x", {"input": "q"}).score == 0.9


def test_metrics_to_evals_wraps_each():
    evals = metrics_to_evals(
        [StubMetric(0.9), StubMetric(0.1)],
        threshold=0.5,
        sample_factory=_sample_factory,
    )
    assert [ev("o", {"input": "q"}).passed for ev in evals] == [True, False]
