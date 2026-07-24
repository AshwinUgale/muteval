"""Use your existing **RAGAS** metrics as muteval evals.

RAGAS is the other widely used RAG-evaluation library. Unlike deepeval, RAGAS
metrics produce a raw score in [0, 1] and don't carry a pass/fail threshold —
so you supply one. The adapter scores each (mutated) output with the metric and
passes when ``score >= threshold``, returning an
:class:`~muteval.evals.EvalOutcome` with the score and threshold attached for
near-miss reporting.

Example::

    from ragas.metrics import Faithfulness, AnswerRelevancy
    from muteval import MutEvalConfig
    from muteval.adapters.ragas import metrics_to_evals

    evals = metrics_to_evals(
        [Faithfulness(), AnswerRelevancy()],
        threshold=0.7,
        input_key="question",
        retrieval_context_key="context",
        reference_key="expected",
    )

    config = MutEvalConfig(
        prompt=SYSTEM_PROMPT,
        cases=[{"question": "...", "context": ["doc1", "doc2"], "expected": "..."}],
        run=my_run_fn,
        evals=evals,
    )

Install with ``pip install "muteval[ragas]"``. RAGAS's public API has shifted
across versions; this adapter targets the ``SingleTurnSample`` /
``single_turn_score`` interface (ragas >= 0.2). If your version differs, pass a
``sample_factory`` and/or ``score_fn`` to override the wiring.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from muteval.adapters.base import case_get
from muteval.evals import EvalFn, EvalOutcome


def _default_sample_factory(
    input_key: str,
    retrieval_context_key: Optional[str],
    reference_key: Optional[str],
) -> Callable[[str, Any], Any]:
    """Build a factory that turns (output, case) into a ragas SingleTurnSample."""

    def factory(output: str, case: Any) -> Any:
        # Imported lazily so muteval core never depends on ragas.
        try:
            from ragas.dataset_schema import SingleTurnSample
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'ragas adapter needs ragas: pip install "muteval[ragas]"'
            ) from exc

        ctx = case_get(case, retrieval_context_key)
        if ctx is not None:
            ctx = list(ctx)
        return SingleTurnSample(
            user_input=case_get(case, input_key),
            response=output,
            retrieved_contexts=ctx,
            reference=case_get(case, reference_key),
        )

    return factory


def _default_score_fn(metric: Any) -> Callable[[Any], float]:
    def score(sample: Any) -> float:
        return float(metric.single_turn_score(sample))

    return score


def metric_to_eval(
    metric: Any,
    *,
    threshold: float = 0.5,
    input_key: str = "input",
    retrieval_context_key: Optional[str] = None,
    reference_key: Optional[str] = None,
    sample_factory: Optional[Callable[[str, Any], Any]] = None,
    score_fn: Optional[Callable[[Any], float]] = None,
) -> EvalFn:
    """Wrap a single RAGAS metric as a muteval eval.

    Args:
        metric: A ragas metric instance (exposing ``single_turn_score(sample)``).
        threshold: Score at/above which the check passes.
        input_key / retrieval_context_key / reference_key: Which case keys map to
            the sample's ``user_input`` / ``retrieved_contexts`` / ``reference``.
        sample_factory: Advanced override — ``(output, case) -> sample``.
        score_fn: Advanced override — ``(sample) -> float``. Defaults to
            ``metric.single_turn_score``.

    Returns:
        An eval function ``(output, case) -> EvalOutcome``.
    """
    factory = sample_factory or _default_sample_factory(
        input_key, retrieval_context_key, reference_key
    )
    scorer = score_fn or _default_score_fn(metric)
    label = getattr(metric, "name", None) or type(metric).__name__

    def _eval(output: str, case: Any) -> EvalOutcome:
        sample = factory(output, case)
        score = float(scorer(sample))
        return EvalOutcome(
            passed=score >= threshold,
            score=score,
            threshold=threshold,
            name=label,
        )

    _eval.__name__ = label
    return _eval


def metrics_to_evals(metrics: List[Any], **kwargs: Any) -> List[EvalFn]:
    """Wrap a list of RAGAS metrics as muteval evals (see metric_to_eval)."""
    return [metric_to_eval(m, **kwargs) for m in metrics]
