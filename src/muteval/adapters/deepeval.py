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

# A muteval eval: (output_text, case) -> bool  (True == passed)
EvalFn = Callable[[str, Any], bool]


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

        def pull(key: Optional[str]):
            if key is None:
                return None
            if isinstance(case, dict):
                return case.get(key)
            return getattr(case, key, None)

        user_input = pull(input_key)
        if user_input is None:
            # Fall back to treating a non-dict case as the raw input.
            user_input = case if not isinstance(case, dict) else None

        return LLMTestCase(
            input=user_input,
            actual_output=output,
            expected_output=pull(expected_output_key),
            retrieval_context=pull(retrieval_context_key),
            context=pull(context_key),
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
            ``.is_successful()``).
        input_key / expected_output_key / retrieval_context_key / context_key:
            Which keys on each ``case`` dict map to the deepeval test-case
            fields. Only ``input_key`` is required; the rest are optional and
            only needed by metrics that use them.
        test_case_factory: Advanced override — a ``(output, case) -> test_case``
            callable. If given, the *_key args are ignored. Mainly for testing
            or custom field mapping.

    Returns:
        An eval function ``(output, case) -> bool``.
    """
    factory = test_case_factory or _default_test_case_factory(
        input_key, expected_output_key, retrieval_context_key, context_key
    )

    def _eval(output: str, case: Any) -> bool:
        test_case = factory(output, case)
        metric.measure(test_case)
        return bool(metric.is_successful())

    _eval.__name__ = getattr(metric, "__name__", type(metric).__name__)
    return _eval


def metrics_to_evals(
    metrics: List[Any],
    **kwargs: Any,
) -> List[EvalFn]:
    """Wrap a list of deepeval metrics as muteval evals (see metric_to_eval)."""
    return [metric_to_eval(m, **kwargs) for m in metrics]
