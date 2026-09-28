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

There is no composite score — each is a separately-interpretable bias in
[0, 1], 0 = unbiased, 1 = maximally biased. Verbosity and self-preference are
TWO-SIDED: neutral is preferring neither side (a tie, or a 50/50 split), and
always preferring the SHORTER answer (or always the OTHER model's) is as much a
bias as the reverse. The raw preference rates and their direction are in
``BiasPanel.detail``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

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
        """True unless an ASSESSED bias is above ``threshold``. The two-sided
        lenses also need the evidence to exclude neutral: their preference
        rate's 95% interval must not contain 0.5 — a fair coin-flip judge over
        40 verdicts lands at 0.42 by chance alone, which is noise, not bias."""
        if self.position_bias is not None and self.position_bias > threshold:
            return False
        for bias, key in (
            (self.verbosity_bias, "prefers_longer"),
            (self.self_preference, "prefers_own_model"),
        ):
            if bias is None or bias <= threshold:
                continue
            rate = self.detail.get(f"{key}_rate")
            n = self.detail.get(f"{key}_n")
            if rate is None or not n:
                return False
            from muteval.stats import wilson_interval

            lo, hi = wilson_interval(round(rate * n), n)
            if not lo <= 0.5 <= hi:
                return False
        return True


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


def _preference_rate(judge, pairs, first_label: str) -> Optional[float]:
    """Fraction of judgements (both presentation orders) preferring the FIRST
    element of each pair; a TIE counts as half (a fair verdict on substantively
    equal answers). None only when there are no pairs."""
    score = n = 0.0
    for pair in pairs:
        first, second, case = pair[0], pair[1], pair[2]
        for a, b, first_is in ((first, second, WIN_A), (second, first, WIN_B)):
            v = normalize_verdict(judge(a, b, case))
            n += 1
            if v == TIE:
                score += 0.5
            elif v == first_is:
                score += 1
    return (score / n) if n else None


def _preference_n(pairs) -> int:
    """Number of verdicts behind a preference rate (each pair, both orders)."""
    return 2 * len(pairs) if pairs else 0


def _two_sided(rate: Optional[float]) -> Optional[float]:
    return None if rate is None else round(abs(rate - 0.5) * 2, 10)


def verbosity_preference(judge, pairs: Sequence[Pair]) -> Optional[float]:
    """Fraction of judgements preferring the LONGER answer (ties = 0.5), over
    both orders. 0.5 is neutral; 1.0 always-longer; 0.0 always-shorter."""
    return _preference_rate(judge, [(lg, sh, c) for sh, lg, c in pairs], "long")


def verbosity_bias(judge, pairs: Sequence[Pair]) -> Optional[float]:
    """``pairs`` are (short, long, case) where ``long`` is ``short`` padded with
    filler (same substance). TWO-SIDED bias in [0, 1]: 0 = prefers neither
    (ties, or a 50/50 split), 1 = always prefers one side — the longer OR the
    shorter answer. (It used to report only the longer-preference rate against a
    0.1 threshold, so an always-SHORTER judge passed and a fair coin-flip failed.)"""
    return _two_sided(verbosity_preference(judge, pairs))


def own_model_preference(
    judge, labeled_pairs: Optional[Sequence[Tuple[str, str, Any, str]]]
) -> Optional[float]:
    """Fraction of judgements picking the own-model answer (ties = 0.5), over
    both orders. 0.5 neutral. None (not assessed) if no labeled pairs given."""
    if not labeled_pairs:
        return None
    return _preference_rate(judge, labeled_pairs, "own")


def self_preference(
    judge, labeled_pairs: Optional[Sequence[Tuple[str, str, Any, str]]]
) -> Optional[float]:
    """``labeled_pairs`` are (own_output, other_output, case, _) where the first
    is from the judge's own model. TWO-SIDED bias in [0, 1]: 0 = the model label
    doesn't move the verdict, 1 = always picks one side (own OR other). None
    (not assessed) if no labeled pairs given."""
    return _two_sided(own_model_preference(judge, labeled_pairs))


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
    verb_rate = verbosity_preference(judge, verbosity_pairs) if verbosity_pairs else None
    own_rate = own_model_preference(judge, self_pref_pairs)
    return BiasPanel(
        position_bias=pos,
        verbosity_bias=_two_sided(verb_rate),
        self_preference=_two_sided(own_rate),
        detail={
            "n_pairs": len(pairs),
            "prefers_longer_rate": verb_rate,  # 0.5 neutral
            "prefers_longer_n": _preference_n(verbosity_pairs),
            "prefers_own_model_rate": own_rate,  # 0.5 neutral
            "prefers_own_model_n": _preference_n(self_pref_pairs),
        },
    )
