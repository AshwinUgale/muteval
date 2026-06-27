"""Use your existing **deepeval** metrics as muteval evals.

The hard part of an eval suite is the metrics — LLM-as-judge rubrics, G-Eval
criteria, faithfulness/relevancy checks. If you've already written those in
deepeval, you shouldn't have to rewrite them as ``(output, case) -> bool``
functions just to run muteval. This adapter wraps a deepeval metric so muteval
can reuse it directly.

What muteval still needs from you: a ``run(prompt, case)`` that regenerates the
system output from the (mutated) prompt, and the ``cases``. deepeval only
stores the *metrics* and cached outputs, not how to call your system — and
muteval's whole job is to re-run your system with a degraded prompt.

Each wrapped eval returns an :class:`~muteval.evals.EvalOutcome` carrying the
metric's ``score`` and ``threshold``, so muteval can report survivors that only
*barely* passed as near misses.

Example::

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from muteval import MutEvalConfig
    from muteval.adapters.deepeval import metrics_to_evals

    metrics = [AnswerRelevancyMetric(threshold=0.7), FaithfulnessMetric()]
    evals = metrics_to_evals(
        metrics,
        input_key="question",
        retrieval_context_key="context",
    )

    config = MutEvalConfig(
        prompt=SYSTEM_PROMPT,
        cases=[{"question": "...", "context": ["doc1", "doc2"]}],
        run=my_run_fn,
        evals=evals,
        eval_names=[type(m).__name__ for m in metrics],
    )
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from muteval.adapters.base import case_get
from muteval.evals import EvalFn, EvalOutcome


def _default_test_case_factory(
    input_key: str,
    expected_output_key: Optional[str],
    retrieval_context_key: Optional[str],
    context_key: Optional[str],
) -> Callable[[str, Any], Any]:
    """Build a factory that turns (output, case) into a deepeval LLMTestCase."""

    def factory(output: str, case: Any) -> Any:
        # Imported lazily so muteval core never depends on deepeval.
        from deepeval.test_case import LLMTestCase

        user_input = case_get(case, input_key)
        if user_input is None:
            # Fall back to treating a non-dict case as the raw input.
            user_input = case if not isinstance(case, dict) else None

        return LLMTestCase(
            input=user_input,
            actual_output=output,
            expected_output=case_get(case, expected_output_key),
            retrieval_context=case_get(case, retrieval_context_key),
            context=case_get(case, context_key),
        )

    return factory


def metric_to_eval(
    metric: Any,
    *,
    input_key: str = "input",
    expected_output_key: Optional[str] = None,
    retrieval_context_key: Optional[str] = None,
    context_key: Optional[str] = None,
    test_case_factory: Optional[Callable[[str, Any], Any]] = None,
) -> EvalFn:
    """Wrap a single deepeval metric as a muteval eval.

    Args:
        metric: A deepeval metric instance (anything with ``.measure(tc)`` and
            ``.is_successful()``; ``.score`` and ``.threshold`` are used for
            near-miss reporting if present).
        input_key / expected_output_key / retrieval_context_key / context_key:
            Which keys on each ``case`` dict map to the deepeval test-case
            fields. Only ``input_key`` is required; the rest are optional and
            only needed by metrics that use them.
        test_case_factory: Advanced override — a ``(output, case) -> test_case``
            callable. If given, the *_key args are ignored. Mainly for testing
            or custom field mapping.

    Returns:
        An eval function ``(output, case) -> EvalOutcome``.
    """
    factory = test_case_factory or _default_test_case_factory(
        input_key, expected_output_key, retrieval_context_key, context_key
    )
    label = getattr(metric, "__name__", type(metric).__name__)

    def _eval(output: str, case: Any) -> EvalOutcome:
        test_case = factory(output, case)
        metric.measure(test_case)
        return EvalOutcome(
            passed=bool(metric.is_successful()),
            score=getattr(metric, "score", None),
            threshold=getattr(metric, "threshold", None),
            name=label,
        )

    _eval.__name__ = label
    return _eval


def metrics_to_evals(
    metrics: List[Any],
    **kwargs: Any,
) -> List[EvalFn]:
    """Wrap a list of deepeval metrics as muteval evals (see metric_to_eval)."""
    return [metric_to_eval(m, **kwargs) for m in metrics]
