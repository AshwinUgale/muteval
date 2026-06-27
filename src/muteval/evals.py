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

    Accepts an ``EvalOutcome`` (passed through, gaining ``name`` if it had none)
    or anything truthy/falsy (wrapped as ``EvalOutcome(passed=bool(value))``).
    """
    if isinstance(value, EvalOutcome):
        if name and not value.name:
            value.name = name
        return value
    return EvalOutcome(passed=bool(value), name=name)
