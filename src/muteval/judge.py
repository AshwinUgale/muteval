"""v0.6: a minimal pairwise-judge protocol for the bias panel.

An LLM-as-judge used for A/B comparisons is only trustworthy if its verdict comes
from the CONTENT, not from the presentation. A ``PairwiseJudge`` takes two
candidate outputs plus the case and names a winner. The bias panel
(``probes.judge_bias``) probes such a judge for the three well-documented
failure modes: position, verbosity, and self-preference bias.
"""

from __future__ import annotations

from typing import Any, Callable

# (output_a, output_b, case) -> "A" | "B" | "tie"
PairwiseJudge = Callable[[str, str, Any], str]

WIN_A = "A"
WIN_B = "B"
TIE = "tie"


def normalize_verdict(v: str) -> str:
    """Coerce a judge's raw string to one of A / B / tie."""
    s = (v or "").strip().lower()
    if s in ("a", "1", "first", "left"):
        return WIN_A
    if s in ("b", "2", "second", "right"):
        return WIN_B
    return TIE
