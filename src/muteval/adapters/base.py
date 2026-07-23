"""The adapter contract — how to make any eval framework's metrics muteval evals.

muteval grades whatever eval suite you already have. An *adapter* is the thin
shim that turns a third-party framework's metric into a muteval eval function
``(output, case) -> EvalOutcome``. Every adapter follows the same shape, so
adding a new framework is mostly copy-and-adjust:

1. Expose ``metric_to_eval(metric, **field_map) -> EvalFn`` and a plural
   ``metrics_to_evals(metrics, **field_map) -> list[EvalFn]``.
2. Import the third-party library **lazily** (inside the returned eval), so the
   muteval core stays dependency-free and ``import``-ing the adapter never
   requires the framework to be installed.
3. Map each ``case`` field to the framework's test-case fields via ``*_key``
   arguments (see ``_case_get``).
4. Return an ``EvalOutcome`` carrying ``passed`` **and** ``score``/``threshold``
   when the framework exposes them — that's what powers near-miss reporting.

``scorer_to_eval`` below covers the most common pattern in one helper: a metric
that yields a numeric score compared against a threshold.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from muteval.evals import EvalFn, EvalOutcome


def case_get(case: Any, key: Optional[str]) -> Any:
    """Pull ``key`` from a case (dict or object). ``None`` key returns ``None``."""
    if key is None:
        return None
    if isinstance(case, dict):
        return case.get(key)
    return getattr(case, key, None)


def scorer_to_eval(
    score_fn: Callable[[str, Any], float],
    *,
    threshold: float,
    name: str,
    higher_is_better: bool = True,
) -> EvalFn:
    """Wrap a ``(output, case) -> float`` scorer as a threshold-based muteval eval.

    Use this for any framework (or a homegrown metric) that produces a numeric
    score. The resulting eval returns an ``EvalOutcome`` with the score and
    threshold attached, so survivors can be reported as near misses.
    """

    def _eval(output: str, case: Any) -> EvalOutcome:
        score = float(score_fn(output, case))
        passed = score >= threshold if higher_is_better else score <= threshold
        return EvalOutcome(passed=passed, score=score, threshold=threshold, name=name)

    return _eval


def named(eval_fn: EvalFn, name: str) -> EvalFn:
    """Attach a ``__name__`` to an eval for nicer report labels."""
    try:
        eval_fn.__name__ = name
    except (AttributeError, TypeError):
        pass
    return eval_fn


__all__ = ["case_get", "scorer_to_eval", "named", "EvalFn", "EvalOutcome", "List"]
