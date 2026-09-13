"""Keyless Autoevals wiring demo; the profile renderer is deliberately synthetic.

The mutation target is the prompt, not the cases, references, or output. This
controlled renderer follows two literal rules so deleting either rule changes
its output. Replace ``render_profile`` with your real pipeline for real evidence.
"""

from __future__ import annotations

import json
import math
import os

from muteval import MutEvalConfig, checks
from muteval.adapters.base import scorer_to_eval

KEYS_RULE = '- Include both "name" and "city" keys in a JSON object.'
UNKNOWN_RULE = "- Use null for an unknown city; do not guess."
PROMPT = f"Extract a profile from the supplied facts.\n{KEYS_RULE}\n{UNKNOWN_RULE}"

# Entirely invented inputs. The runner reads facts, never the expected answer.
CASES = [
    {"facts": {"name": "Ari"}, "expected": {"name": "Ari", "city": None}},
    {
        "facts": {"name": "Bea", "city": "Linden"},
        "expected": {"name": "Bea", "city": "Linden"},
    },
]


def render_profile(prompt: str, case: dict) -> str:
    """Deterministic stand-in for an LLM; reads the supplied, mutated prompt."""
    facts = case["facts"]
    profile = {"name": facts["name"]}
    if KEYS_RULE in prompt:
        profile["city"] = facts.get("city")
    if UNKNOWN_RULE not in prompt and "city" not in facts:
        profile["city"] = "Atlantis"  # A deliberate unsupported inference.
    return json.dumps(profile)


def _require_score(result) -> float:
    # Autoevals uses None for a skipped score. Treat that as unavailable
    # evidence, not a failed eval (0) or a pass. The engine records the error.
    error = getattr(result, "error", None)  # Deprecated in current Autoevals.
    if error is not None:
        raise RuntimeError("Autoevals scorer reported an error") from error
    if result.score is None:
        raise ValueError("Autoevals scorer skipped this case (score is None)")
    value = float(result.score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Autoevals scorer must return a finite score between 0 and 1")
    return value


def _profile_exact_score(output: str, case: dict) -> float:
    # Optional dependency: importing this config and running the weak suite
    # need no Autoevals installation. No LLM scorer or network call is involved.
    from autoevals import ExactMatch

    # Compare JSON content, not whitespace or object-key ordering. Do not
    # normalize away missing keys, nulls, extra fields, or incorrect values.
    actual = json.dumps(json.loads(output), sort_keys=True, allow_nan=False)
    expected = json.dumps(case["expected"], sort_keys=True, allow_nan=False)
    return _require_score(ExactMatch()(output=actual, expected=expected))


def make_config(suite: str = "weak", run=render_profile) -> MutEvalConfig:
    """Compare the same mutations using JSON-only or exact-profile checks."""
    if suite not in ("weak", "strong"):
        raise ValueError("MUTEVAL_PROFILE_SUITE must be 'weak' or 'strong'")
    evals = [checks.is_json()]
    names = ["valid_json"]
    if suite == "strong":
        evals.append(
            scorer_to_eval(
                _profile_exact_score, threshold=1.0, name="profile_exact_match"
            )
        )
        names.append("profile_exact_match")
    return MutEvalConfig(
        prompt=PROMPT,
        cases=CASES,
        run=run,
        evals=evals,
        eval_names=names,
        operators=["drop_instruction_lines"],
        scope_include=r"^- ",  # Only the two rules, not the introductory sentence.
    )


config = make_config(os.environ.get("MUTEVAL_PROFILE_SUITE", "weak"))
