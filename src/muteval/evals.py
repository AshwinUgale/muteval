"""Eval return values: a plain bool, or a richer scored ``EvalOutcome``.

v0 evals returned ``bool`` — pass or fail. That throws away the most useful
signal an LLM-as-judge metric produces: *how close* the call was. A mutant that
your suite passes with a faithfulness score of 0.71 against a 0.70 threshold is
a near miss — your eval almost caught the regression. Collapsing that to ``True``
hides it.

``EvalOutcome`` carries the score and threshold so the runner can surface those
near misses in the survivor report. Evals may still return a bare ``bool``;
``coerce_outcome`` normalizes either form, so nothing existing breaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union


@dataclass
class EvalOutcome:
    """The result of one eval check.

    Attributes:
        passed: Whether the check passed (True == passed).
        score: Optional raw score the check produced (e.g. an LLM-judge score).
        threshold: Optional pass/fail threshold the score was compared against.
        name: Optional label for reporting.
        detail: Optional human-readable note (e.g. why it failed).
    """

    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    name: Optional[str] = None
    detail: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.passed)

    @property
    def margin(self) -> Optional[float]:
        """``score - threshold`` when both are known, else ``None``.

        A small positive margin on a *passing* check is a near miss: the eval
        barely caught (or barely missed catching) the regression.
        """
        if self.score is None or self.threshold is None:
            return None
        return self.score - self.threshold


# An eval check: (output_text, case) -> bool | EvalOutcome  (truthy == passed).
EvalResult = Union[bool, EvalOutcome]
EvalFn = Callable[[str, Any], EvalResult]


def coerce_outcome(value: EvalResult, name: Optional[str] = None) -> EvalOutcome:
    """Normalize an eval's return value to an ``EvalOutcome``.

    Accepts an ``EvalOutcome`` (passed through, gaining ``name`` if it had none),
    a promptfoo-style ``{"pass"/"passed": bool, "score": ...}`` dict, or a
    truthy/falsy value such as a bool, an ``re.Match``/``None``, or a list.

    Raises ``TypeError`` for values whose truthiness is NOT a verdict: a number
    (``0.2`` is a score — truthy, so it would "pass"), a string (``"fail"`` is
    truthy), or a dict without a pass key (``{"passed": False}``-shaped typos).
    Raises ``ValueError`` for a non-finite score (``nan > threshold`` is False,
    so an unparseable metric would silently count as a KILL). Raising makes the
    mutant *errored* (fail closed), never a fake verdict.
    """
    if isinstance(value, EvalOutcome):
        if name and not value.name:
            value.name = name
        _check_finite(value, name)
        return value
    if isinstance(value, dict):
        key = "passed" if "passed" in value else "pass" if "pass" in value else None
        if key is None:
            raise TypeError(
                f"eval {name or ''} returned a dict without a 'pass'/'passed' key; "
                "return a bool or EvalOutcome(passed=..., score=...)"
            )
        score = value.get("score")
        outcome = EvalOutcome(
            passed=bool(value[key]),
            score=float(score) if isinstance(score, (int, float)) else None,
            threshold=value.get("threshold"),
            name=name,
            detail=value.get("reason"),
        )
        _check_finite(outcome, name)
        return outcome
    if isinstance(value, (str, bytes)) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        raise TypeError(
            f"eval {name or ''} returned {type(value).__name__} {value!r}, which is "
            "not a verdict (a number is a score; any non-empty string is truthy). "
            "Return a bool, or EvalOutcome(passed=score >= threshold, score=score)."
        )
    return EvalOutcome(passed=bool(value), name=name)


def _check_finite(outcome: EvalOutcome, name: Optional[str]) -> None:
    score = outcome.score
    if score is not None and not math.isfinite(score):
        raise ValueError(
            f"eval {name or outcome.name or ''} returned a non-finite score "
            f"({score!r}); refusing to treat it as a verdict"
        )
