"""v0.6: the verify loop — prove a proposed eval actually closes a gap.

Finding a survivor is a diagnosis; the differentiator is *closing* it and proving
the fix works. Given a survivor and one or more candidate evals, this module
re-runs the pipeline and keeps only candidates that are VERIFIED:

  * the candidate FAILS on the survivor's mutant  (so it would now catch it), AND
  * the candidate PASSES on the baseline system   (so it doesn't break a green run).

A candidate that only does one of those is worthless (a check that fails on the
baseline turns every run red; a check that passes on the mutant catches nothing).
Requiring BOTH is what makes a suggested fix trustworthy — no LLM required to
verify, so the claim "this eval closes the gap" is proven, not asserted.

Candidate generation (from the survivor, via an LLM) is the `[llm]` follow-up;
this module accepts candidates from anywhere (hand-written, `checks.*`, or an
LLM), so the *verification* is usable today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from muteval.evals import EvalFn, coerce_outcome


@dataclass
class VerifiedFix:
    """A candidate eval that provably closes a survivor's gap."""

    eval: EvalFn
    name: str
    killed_case: Any = None  # a case on which the candidate catches the mutant


def _passes_all(config, system, candidate) -> bool:
    """True iff the candidate passes on EVERY case for this system."""
    for case in config.cases:
        out = config.invoke(system, case)
        if not coerce_outcome(candidate(out, case)).passed:
            return False
    return True


def _first_catch(config, system, candidate):
    """Return the first case on which the candidate FAILS (catches a regression),
    or None if it passes everywhere."""
    for case in config.cases:
        out = config.invoke(system, case)
        if not coerce_outcome(candidate(out, case)).passed:
            return case
    return None


def verify_fix(config, mutant_system, candidate) -> tuple[bool, bool]:
    """Return ``(kills_mutant, keeps_baseline)`` for a candidate eval."""
    keeps_baseline = _passes_all(config, config.system, candidate)
    kills_mutant = _first_catch(config, mutant_system, candidate) is not None
    return kills_mutant, keeps_baseline


def suggest_and_verify(config, survivor, candidates: Iterable[EvalFn]) -> List[VerifiedFix]:
    """Keep only the candidate evals that provably close ``survivor``'s gap.

    ``survivor`` may be a ``MutantOutcome`` or a ``Mutant`` (anything exposing a
    ``.system``, directly or via ``.mutant``). Returns the verified candidates,
    each of which kills the mutant AND keeps the baseline green.
    """
    mutant = getattr(survivor, "mutant", survivor)
    mutant_system = mutant.system
    verified: List[VerifiedFix] = []
    for cand in candidates:
        name = getattr(cand, "__name__", "candidate")
        keeps_baseline = _passes_all(config, config.system, cand)
        if not keeps_baseline:
            continue  # a fix that reddens the baseline is not a fix
        caught = _first_catch(config, mutant_system, cand)
        if caught is not None:
            verified.append(VerifiedFix(eval=cand, name=name, killed_case=caught))
    return verified
