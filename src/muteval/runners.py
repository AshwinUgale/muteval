"""Built-in ``run`` helpers so users don't have to write one.

muteval's one irreducible requirement is a ``run(prompt, case) -> output`` —
how to call the system with a (mutated) prompt. For the common "prompt + chat
model" case, you shouldn't have to write that. ``openai_run`` provides it using
only the standard library (no ``openai`` SDK), so the CLI can run a full
mutation test from just a prompt file + a cases file + flags.

Custom pipelines (RAG retrievers, agents, non-OpenAI models) still use a Python
config with their own ``run`` — this just removes the boilerplate for the
majority case.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any, Callable, Sequence

_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Case keys we'll look for, in order, to find the user's question/input.
_QUESTION_KEYS = ("question", "input", "query", "prompt", "text")
# Case key holding retrieved context (str or list of str), if any.
_CONTEXT_KEY = "context"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # optional; fixes SSL on bare Pythons

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _chat(messages: list, model: str, temperature: float = 0.0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages}
    ).encode("utf-8")
    req = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""


def _case_question(case: Any, question_keys: Sequence[str]) -> str:
    if isinstance(case, str):
        return case
    if isinstance(case, dict):
        for k in question_keys:
            if case.get(k):
                return str(case[k])
        return ""
    return str(case)


def _case_context(case: Any, context_key: str) -> str:
    if not isinstance(case, dict):
        return ""
    ctx = case.get(context_key)
    if not ctx:
        return ""
    if isinstance(ctx, (list, tuple)):
        return "\n\n".join(str(c) for c in ctx)
    return str(ctx)


def openai_run(
    model: str = "gpt-4o-mini",
    *,
    question_keys: Sequence[str] = _QUESTION_KEYS,
    context_key: str = _CONTEXT_KEY,
    temperature: float = 0.0,
) -> Callable[[str, Any], str]:
    """Return a ``run(prompt, case)`` that calls an OpenAI chat model.

    The mutated ``prompt`` becomes the system message; the case's question
    (first present of ``question_keys``) plus any retrieved ``context`` becomes
    the user message. Stdlib only — needs ``OPENAI_API_KEY`` (and ``certifi`` if
    your Python's SSL store is bare).
    """

    def run(prompt: str, case: Any) -> str:
        question = _case_question(case, question_keys)
        context = _case_context(case, context_key)
        user = (f"Context:\n{context}\n\nQuestion: {question}" if context else question)
        return _chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=temperature,
        )

    return run
