"""v0.6: judge-bias panel — is your A/B judge deciding on content or on artifacts?

Three well-documented LLM-judge failure modes, each measured by re-asking the
same comparison under a controlled transformation and checking whether the
verdict SHOULD have stayed the same:

* **position bias** — swap which answer is shown first; a fair judge names the
  same underlying answer both ways. The flip rate is the position-bias score.
* **verbosity bias** — pad the substantively-equal answer with filler; a fair
  judge should call it a tie (or not systematically prefer the longer one),
  averaged over both presentation orders to cancel position bias.
* **self-preference** — label which model produced each answer; a fair judge's
  verdict shouldn't move when the "own-model" label is attached. Reported as
  "not assessed" unless a self/other labeling is supplied.

There is no composite score — each is a separately-interpretable rate in [0, 1],
0 = unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from muteval.judge import TIE, WIN_A, WIN_B, normalize_verdict

# A comparison pair: (output_a, output_b, case)
Pair = Tuple[str, str, Any]


@dataclass
class BiasPanel:
    position_bias: Optional[float]
    verbosity_bias: Optional[float]
    self_preference: Optional[float]
    detail: Dict[str, Any] = field(default_factory=dict)

    def ok(self, threshold: float = 0.1) -> bool:
        """True if every ASSESSED bias is at/below threshold."""
        return all(
            v is None or v <= threshold
            for v in (self.position_bias, self.verbosity_bias, self.self_preference)
        )


def _winner(judge, a: str, b: str, case: Any):
    """The underlying answer the judge picked (a or b), or None on a tie."""
    v = normalize_verdict(judge(a, b, case))
    if v == WIN_A:
        return a
    if v == WIN_B:
        return b
    return None


def position_bias(judge, pairs: Sequence[Pair]) -> Optional[float]:
    """Fraction of pairs whose winner FLIPS when the presentation order is
    swapped. A fair judge is order-invariant. None if every pair tied."""
    flips = n = 0
    for a, b, case in pairs:
        w1 = _winner(judge, a, b, case)
        w2 = _winner(judge, b, a, case)  # swapped order
        if w1 is None or w2 is None:
            continue
        n += 1
        if w1 != w2:
            flips += 1
    return (flips / n) if n else None


def verbosity_bias(judge, pairs: Sequence[Pair]) -> Optional[float]:
    """``pairs`` are (short, long, case) where ``long`` is ``short`` padded with
    filler (same substance). Fraction of judgements that prefer the LONGER answer,
    averaged over both presentation orders (so position bias cancels). 0.5 would
    be neutral for a coin-flip judge; a strong verbosity bias approaches 1.0."""
    prefer_long = n = 0
    for short, long, case in pairs:
        for a, b, long_is in ((short, long, WIN_B), (long, short, WIN_A)):
            v = normalize_verdict(judge(a, b, case))
            if v == TIE:
                continue
            n += 1
            if v == long_is:
                prefer_long += 1
    return (prefer_long / n) if n else None


def self_preference(judge, labeled_pairs: Optional[Sequence[Tuple[str, str, Any, str]]]) -> Optional[float]:
    """``labeled_pairs`` are (own_output, other_output, case, _) where the first
    is from the judge's own model. Fraction that pick the own-model answer,
    averaged over both orders. None (not assessed) if no labeled pairs given."""
    if not labeled_pairs:
        return None
    prefer_own = n = 0
    for own, other, case, _ in labeled_pairs:
        for a, b, own_is in ((own, other, WIN_A), (other, own, WIN_B)):
            v = normalize_verdict(judge(a, b, case))
            if v == TIE:
                continue
            n += 1
            if v == own_is:
                prefer_own += 1
    return (prefer_own / n) if n else None


def run_judge_bias_panel(
    judge,
    pairs: Sequence[Pair],
    verbosity_pairs: Optional[Sequence[Pair]] = None,
    self_pref_pairs: Optional[Sequence[Tuple[str, str, Any, str]]] = None,
) -> BiasPanel:
    """Assemble the full panel. ``pairs`` drive position bias; optional
    ``verbosity_pairs`` (short,long,case) and ``self_pref_pairs`` add the other
    two lenses (else they report None = not assessed)."""
    pos = position_bias(judge, pairs)
    verb = verbosity_bias(judge, verbosity_pairs) if verbosity_pairs else None
    selfp = self_preference(judge, self_pref_pairs)
    return BiasPanel(
        position_bias=pos, verbosity_bias=verb, self_preference=selfp,
        detail={"n_pairs": len(pairs)},
    )
