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

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

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


# --- LLM candidate generation (safe: structured specs, never exec'd code) -----

# The LLM may only propose checks from this fixed vocabulary. Each maps to a
# real muteval check with STRING arguments — no code is generated or executed, so
# a malicious/hallucinated response can't do anything but build a benign check.
_ALLOWED_TYPES = ("contains", "not_contains", "regex_matches", "llm_judge")

_GEN_PROMPT = """A mutation to a system's prompt caused a regression its eval suite did NOT catch.

Mutation: {description}
Baseline (good) output: {baseline!r}
Mutated (bad) output:   {mutant!r}

Propose up to {n} eval checks that would PASS on the baseline output and FAIL on
the mutated output — so the suite would catch this regression next time.

Respond with ONLY a JSON array of objects, each one of:
  {{"type": "contains", "value": "<substring the GOOD answer has>"}}
  {{"type": "not_contains", "value": "<substring the BAD answer has>"}}
  {{"type": "regex_matches", "value": "<regex the GOOD answer matches>"}}
  {{"type": "llm_judge", "rubric": "<one-sentence rubric>", "threshold": 0.7}}
No prose, just the JSON array."""


def _spec_to_eval(spec: dict) -> Optional[EvalFn]:
    """Map a structured spec to a real check. Unknown/malformed -> None."""
    from muteval import checks

    t = spec.get("type")
    if t not in _ALLOWED_TYPES:
        return None
    try:
        if t == "contains":
            ev = checks.contains(str(spec["value"]))
        elif t == "not_contains":
            ev = checks.not_contains(str(spec["value"]))
        elif t == "regex_matches":
            ev = checks.regex_matches(str(spec["value"]))
        else:  # llm_judge
            ev = checks.llm_judge(str(spec["rubric"]), threshold=float(spec.get("threshold", 0.7)))
    except (KeyError, ValueError, TypeError, re.error):
        return None
    ev.__name__ = f"suggested_{t}"
    return ev


def parse_specs(raw: str) -> List[EvalFn]:
    """Parse an LLM response (a JSON array of specs) into candidate evals,
    tolerating ```json fences. Silently drops anything not in the vocabulary."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for spec in data:
        if isinstance(spec, dict):
            ev = _spec_to_eval(spec)
            if ev is not None:
                out.append(ev)
    return out


def _sample_outputs(config, survivor):
    """The baseline vs mutant output pair to feed the generator (from the
    survivor if captured, else computed on the first case)."""
    base = getattr(survivor, "baseline_output", None)
    mut = getattr(survivor, "mutant_output", None)
    if base is not None and mut is not None:
        return base, mut
    mutant = getattr(survivor, "mutant", survivor)
    case = config.cases[0]
    return config.invoke(config.system, case), config.invoke(mutant.system, case)


def generate_candidates(
    config, survivor, chat: Optional[Callable[[str, str], str]] = None,
    model: str = "gpt-4o-mini", n: int = 3,
) -> List[EvalFn]:
    """Ask an LLM to propose candidate checks that would catch ``survivor``.

    ``chat(prompt, model) -> str`` defaults to muteval's stdlib OpenAI call
    (needs OPENAI_API_KEY); inject your own for offline use/testing. The response
    is parsed as structured specs — no code is executed.
    """
    if chat is None:
        from muteval.checks import _openai_chat_stdlib

        chat = _openai_chat_stdlib
    base, mut = _sample_outputs(config, survivor)
    prompt = _GEN_PROMPT.format(description=getattr(getattr(survivor, "mutant", survivor),
                                                    "description", "(unknown)"),
                                baseline=base, mutant=mut, n=n)
    return parse_specs(chat(prompt, model))


def autofix(config, survivor, chat=None, model: str = "gpt-4o-mini", n: int = 3) -> List[VerifiedFix]:
    """Generate candidate fixes with an LLM, then return only the VERIFIED ones
    (each provably catches the mutant AND keeps the baseline green)."""
    return suggest_and_verify(config, survivor, generate_candidates(config, survivor, chat, model, n))
