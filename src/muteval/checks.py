"""Framework-free eval checks — start grading in two lines, no eval lib needed.

The adapters (deepeval, ragas, ...) let you reuse metrics you already wrote.
But the fastest way to *try* muteval is to not need any of that. These factories
return ready-made eval functions ``(output, case) -> EvalOutcome`` for the most
common checks: substring presence, regex, JSON validity, exact match, and a
generic LLM-as-judge.

The LLM judge calls the OpenAI REST API using ONLY the Python standard library —
no ``openai`` package, no heavy dependencies. muteval itself is dependency-free,
so the whole zero-framework path is just ``pip install muteval`` + an API key.

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

import json
import os
import re
import ssl
import urllib.request
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
    """Pass iff the value at ``case[key]`` appears in the output."""

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
    base_url: Optional[str] = None,
    input_key: str = "input",
) -> EvalFn:
    """A generic LLM-as-judge check.

    Pass a ``judge(prompt) -> float in [0, 1]`` callable to stay dependency-free
    and deterministic in tests. If you don't, a built-in judge calls an
    OpenAI-**compatible** chat API using only the Python standard library (no
    ``openai`` package, no extra installs) — it needs ``OPENAI_API_KEY`` set.

    Point it at ANY OpenAI-compatible endpoint (Groq, Gemini's OpenAI-compat API,
    GitHub Models, Ollama, a local vLLM/server) via ``base_url=`` or the
    ``OPENAI_BASE_URL`` env var, e.g. ``base_url="https://api.groq.com/openai/v1"``
    with ``model="openai/gpt-oss-20b"``. The built-in judge asks for a plain 0-10
    score (not a structured/`json_schema` response), so it works on models that
    don't support strict structured outputs. Install ``certifi`` if your Python's
    SSL store can't verify the endpoint.
    """
    judge_fn = judge or _default_openai_judge(model, base_url)

    def _eval(output: str, case: Any) -> EvalOutcome:
        question = _case_get(case, input_key)
        prompt = (
            f"You are grading an AI system's output against a rubric.\n"
            f"Rubric: {rubric}\n\n"
            f"User input: {question}\n\n"
            f"Output to grade:\n{output}\n\n"
            f"Return ONLY an integer from 0 to 10 (10 = perfect) for how well "
            f"the output satisfies the rubric."
        )
        score = float(judge_fn(prompt))
        return EvalOutcome(
            passed=score >= threshold,
            score=score,
            threshold=threshold,
            name="llm_judge",
        )

    _eval.is_llm = True  # expensive: ordered AFTER cheap deterministic checks
    return _eval


def cites_source(pattern: str = r"[A-Za-z]+[-_]?\d+", *, min_count: int = 1) -> EvalFn:
    """Pass iff the output cites >= ``min_count`` source ids matching ``pattern``,
    **regardless of bracket style** — it finds the bare id inside ``[id]``,
    ``(id)``, full-width ``【id】``, or none. Rule-based (no LLM).

    Bakes in a common gotcha: models cite with whatever brackets they like, so
    matching the id token itself is far more robust than matching ``\\[id\\]``.
    Pass your own ``pattern`` (e.g. ``r"doc-\\d+"``) for your id scheme.
    """
    rx = re.compile(pattern)

    def _eval(output: str, case: Any) -> EvalOutcome:
        n = len(rx.findall(output or ""))
        return EvalOutcome(
            passed=n >= min_count,
            score=float(n),
            name="cites_source",
            detail=f"{n} citation(s) matching {pattern!r}",
        )

    return _eval


def grounded(
    context_key: str = "context",
    *,
    judge: Optional[Callable[[str], float]] = None,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
) -> EvalFn:
    """LLM-judge preset: pass iff the output is **grounded** in
    ``case[context_key]`` — it uses only facts from the context and invents
    nothing; an honest "I don't know" when the answer isn't present counts as
    grounded. This is the eval muteval most often suggests for grounding/
    abstention survivors — now a one-liner. Judge notes: see ``llm_judge``.
    """
    judge_fn = judge or _default_openai_judge(model, base_url)

    def _eval(output: str, case: Any) -> EvalOutcome:
        ctx = _case_get(case, context_key)
        ctx_text = "\n".join(ctx) if isinstance(ctx, (list, tuple)) else str(ctx or "")
        prompt = (
            "Rate how well the ANSWER is grounded in the CONTEXT. It must use only "
            "facts from the context and invent nothing. If the context does not "
            "contain the answer and the reply honestly says it doesn't know, that "
            "is fully grounded.\n\n"
            f"CONTEXT:\n{ctx_text}\n\nANSWER:\n{output}\n\n"
            "Return ONLY an integer from 0 to 10 (10 = fully grounded)."
        )
        score = float(judge_fn(prompt))
        return EvalOutcome(
            passed=score >= threshold, score=score, threshold=threshold, name="grounded"
        )

    _eval.is_llm = True  # expensive: ordered AFTER cheap deterministic checks
    return _eval


_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _judge_endpoint(base_url: Optional[str] = None) -> str:
    """Resolve the chat-completions endpoint from base_url / OPENAI_BASE_URL.

    Accepts an OpenAI-style base ("…/v1") and appends "/chat/completions", or a
    full endpoint (already ending in "/chat/completions"), so any OpenAI-compatible
    provider works: OpenAI, Groq, Gemini (openai-compat), GitHub Models, Ollama…
    """
    base = base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
    endpoint = base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    return endpoint


def _openai_chat_stdlib(prompt: str, model: str, base_url: Optional[str] = None) -> str:
    """Call an OpenAI-compatible chat completions endpoint with only the stdlib."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it, or pass your own judge=... callable."
        )
    try:
        import certifi  # optional; fixes SSL verification on bare Pythons

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the system trust store
        ctx = ssl.create_default_context()

    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _judge_endpoint(base_url),
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""


def _default_openai_judge(model: str, base_url: Optional[str] = None) -> Callable[[str], float]:
    def _judge(prompt: str) -> float:
        text = _openai_chat_stdlib(prompt, model, base_url).strip()
        # Parse the LAST number; normalize a 0-10 integer to [0, 1]; clamp.
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return 0.0
        val = float(nums[-1])
        if val > 1:
            val = val / 10.0
        return max(0.0, min(1.0, val))

    return _judge
