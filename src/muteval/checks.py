"""Framework-free eval checks — start grading in two lines, no eval lib needed.

The adapters (deepeval, ragas, ...) let you reuse metrics you already wrote.
But the fastest way to *try* muteval is to not need any of that. These factories
return ready-made eval functions ``(output, case) -> EvalOutcome`` for the most
common checks: substring presence, regex, JSON validity, exact match, and a
generic LLM-as-judge.

Example::

    from muteval import MutEvalConfig
    from muteval import checks

    config = MutEvalConfig(
        prompt=SYSTEM_PROMPT,
        cases=[{"input": "order status?", "order_id": "X1"}],
        run=my_run_fn,
        evals=[
            checks.contains_case("order_id"),       # output mentions the order id
            checks.not_contains("refund"),          # never promises a refund
        ],
    )
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from muteval.evals import EvalFn, EvalOutcome


def _case_get(case: Any, key: str) -> Any:
    if isinstance(case, dict):
        return case.get(key)
    return getattr(case, key, None)


def contains(substring: str, *, case_sensitive: bool = False) -> EvalFn:
    """Pass iff ``substring`` appears in the output."""

    needle = substring if case_sensitive else substring.lower()

    def _eval(output: str, case: Any) -> EvalOutcome:
        hay = output if case_sensitive else output.lower()
        return EvalOutcome(passed=needle in hay, name=f"contains({substring!r})")

    return _eval


def not_contains(substring: str, *, case_sensitive: bool = False) -> EvalFn:
    """Pass iff ``substring`` does NOT appear in the output (a guardrail check)."""

    needle = substring if case_sensitive else substring.lower()

    def _eval(output: str, case: Any) -> EvalOutcome:
        hay = output if case_sensitive else output.lower()
        return EvalOutcome(passed=needle not in hay, name=f"not_contains({substring!r})")

    return _eval


def contains_case(key: str, *, case_sensitive: bool = False) -> EvalFn:
    """Pass iff the value at ``case[key]`` appears in the output.

    Handy for "the answer must cite the order id / account number / etc." where
    the expected value lives on each case.
    """

    def _eval(output: str, case: Any) -> EvalOutcome:
        value = _case_get(case, key)
        if value is None:
            return EvalOutcome(
                passed=False, name=f"contains_case({key!r})", detail="missing key"
            )
        value = str(value)
        hay = output if case_sensitive else output.lower()
        needle = value if case_sensitive else value.lower()
        return EvalOutcome(passed=needle in hay, name=f"contains_case({key!r})")

    return _eval


def regex_matches(pattern: str, *, flags: int = 0) -> EvalFn:
    """Pass iff the output matches ``pattern`` (``re.search`` semantics)."""

    compiled = re.compile(pattern, flags)

    def _eval(output: str, case: Any) -> EvalOutcome:
        return EvalOutcome(
            passed=compiled.search(output) is not None,
            name=f"regex_matches({pattern!r})",
        )

    return _eval


def is_json() -> EvalFn:
    """Pass iff the output parses as JSON."""

    def _eval(output: str, case: Any) -> EvalOutcome:
        try:
            json.loads(output)
            ok = True
        except (ValueError, TypeError):
            ok = False
        return EvalOutcome(passed=ok, name="is_json")

    return _eval


def equals(expected_key: str = "expected", *, strip: bool = True) -> EvalFn:
    """Pass iff the output equals ``case[expected_key]`` (exact match)."""

    def _eval(output: str, case: Any) -> EvalOutcome:
        expected = _case_get(case, expected_key)
        a, b = output, ("" if expected is None else str(expected))
        if strip:
            a, b = a.strip(), b.strip()
        return EvalOutcome(passed=a == b, name=f"equals({expected_key!r})")

    return _eval


def llm_judge(
    rubric: str,
    *,
    judge: Optional[Callable[[str], float]] = None,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
    input_key: str = "input",
) -> EvalFn:
    """A generic LLM-as-judge check.

    Pass a ``judge(prompt) -> float in [0, 1]`` callable to stay dependency-free
    and deterministic in tests. If you don't, a minimal OpenAI-backed judge is
    used (requires ``openai`` and ``OPENAI_API_KEY``), asking the model to score
    the output against ``rubric`` from 0 to 1.
    """

    judge_fn = judge or _default_openai_judge(model)

    def _eval(output: str, case: Any) -> EvalOutcome:
        question = _case_get(case, input_key)
        prompt = (
            f"You are grading an AI system's output against a rubric.\n"
            f"Rubric: {rubric}\n\n"
            f"User input: {question}\n\n"
            f"Output to grade:\n{output}\n\n"
            f"Return ONLY a number from 0 to 1 for how well the output satisfies "
            f"the rubric."
        )
        score = float(judge_fn(prompt))
        return EvalOutcome(
            passed=score >= threshold,
            score=score,
            threshold=threshold,
            name="llm_judge",
        )

    return _eval


def _default_openai_judge(model: str) -> Callable[[str], float]:
    def _judge(prompt: str) -> float:
        from openai import OpenAI  # imported lazily; only needed if used

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "0").strip()
        match = re.search(r"[0-1](?:\.\d+)?", text)
        return float(match.group(0)) if match else 0.0

    return _judge
