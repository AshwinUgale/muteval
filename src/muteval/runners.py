"""Built-in ``run`` helpers so users don't have to write one.

muteval's one irreducible requirement is a ``run -> output`` — how to call the
system with a (mutated) prompt/context. For the common "prompt + chat model"
case, you shouldn't have to write that. ``openai_run`` provides it using only
the standard library (no ``openai`` SDK).

It is **System-aware**: muteval calls ``run`` with either

  * a prompt string — legacy ``run(prompt, case)`` mode, or
  * a ``System`` — when the config is built with ``system=...`` (RAG/agent mode),

and ``openai_run`` handles both. In System mode it uses the mutated
``system.prompt``, the mutated ``system.context`` as the retrieval corpus, and
``system.model`` — so context-drop and model-swap mutations actually flow
through to the output.

Custom pipelines (your own retriever/agent) still pass their own ``run`` in a
Python config; this just removes the boilerplate for the majority case.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any, Callable, Optional, Sequence

_DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

_QUESTION_KEYS = ("question", "input", "query", "prompt", "text")
_CONTEXT_KEY = "context"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # optional; fixes SSL on bare Pythons

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _endpoint(base_url: Optional[str]) -> str:
    """Resolve the chat-completions URL from base_url / OPENAI_BASE_URL, so the
    system under test can run on ANY OpenAI-compatible provider (OpenAI, Groq,
    Gemini-compat, GitHub Models, Ollama, a local server)."""
    base = base_url or os.environ.get("OPENAI_BASE_URL")
    if not base:
        return _DEFAULT_ENDPOINT
    e = base.rstrip("/")
    return e if e.endswith("/chat/completions") else e + "/chat/completions"


def _chat(messages: list, model: str, temperature: float = 0.0,
          base_url: Optional[str] = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages}
    ).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(base_url),
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


def _docs_to_text(docs: Optional[Sequence[str]]) -> str:
    if not docs:
        return ""
    if isinstance(docs, (list, tuple)):
        return "\n\n".join(str(d) for d in docs)
    return str(docs)


def _case_context(case: Any, context_key: str) -> str:
    if not isinstance(case, dict):
        return ""
    return _docs_to_text(case.get(context_key))


def openai_run(
    model: str = "gpt-4o-mini",
    *,
    base_url: Optional[str] = None,
    question_keys: Sequence[str] = _QUESTION_KEYS,
    context_key: str = _CONTEXT_KEY,
    temperature: float = 0.0,
) -> Callable[[Any, Any], str]:
    """Return a System-aware ``run`` that calls an OpenAI chat model (stdlib).

    Called with a ``System`` (system mode) it uses ``system.prompt``, the
    mutated ``system.context`` (falling back to the case's own context if the
    system carries none), and ``system.model`` (falling back to ``model``).
    Called with a prompt string (legacy mode) it uses that prompt + the case's
    context + ``model``.
    """

    def run(target: Any, case: Any) -> str:
        # Imported lazily to avoid a hard import cycle at module load.
        from muteval.system import System

        if isinstance(target, System):
            prompt = target.prompt
            use_model = target.model or model
            context = _docs_to_text(target.context) or _case_context(case, context_key)
        else:
            prompt = target
            use_model = model
            context = _case_context(case, context_key)

        question = _case_question(case, question_keys)
        user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        # Only pass base_url when set, so stubs of _chat (in tests) stay compatible.
        extra = {"base_url": base_url} if base_url else {}
        return _chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            model=use_model,
            temperature=temperature,
            **extra,
        )

    return run


def _prompt_of(target: Any) -> str:
    """The (possibly mutated) prompt string from a run target (System or str)."""
    from muteval.system import System

    return target.prompt if isinstance(target, System) else target


def callable_run(spec: str) -> Callable[[Any, Any], str]:
    """Use an existing function as the run, imported by dotted path.

    ``spec`` is ``"package.module:function"``. The callable is invoked as
    ``fn(prompt, case) -> str`` with the (possibly mutated) prompt, so you can
    point muteval at a pipeline you already have — no ``run()`` wrapper, no config
    file (``muteval run --target mypkg.app:answer --prompt-file p.txt --cases c.jsonl``).
    NOTE: a ``--target`` function only sees the prompt, so context/model mutations
    won't reach it — for RAG/agent mutation, write a config with
    ``run(system, case)`` (the CLI warns when you combine ``--target`` with
    ``--context``/``--mutate-model``).
    """
    module_path, sep, attr = spec.partition(":")
    if not sep or not module_path or not attr:
        raise ValueError(
            f"--target must be 'package.module:function', got {spec!r}"
        )
    import importlib

    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} did not resolve to a callable")

    def run(target: Any, case: Any) -> str:
        return fn(_prompt_of(target), case)

    return run


_OUTPUT_KEYS = ("output", "text", "content", "answer", "response", "completion")


def http_run(
    url: str, *, timeout: int = 60, headers: Optional[dict] = None
) -> Callable[[Any, Any], str]:
    """Drive an HTTP endpoint as the system under test.

    POSTs JSON to ``url`` carrying the (mutated) ``prompt`` and the ``case``, plus
    ``context`` / ``model`` / ``tools`` when the config is in System mode — so
    retrieval-corruption and model-swap mutations actually reach the endpoint
    instead of being silently inert. Pass ``headers=`` for auth
    (``{"Authorization": "Bearer ..."}``). The response may be plain text, or JSON
    carrying the text under any of: output / text / content / answer / response /
    completion. Lets muteval test a deployed pipeline without importing it.
    """
    from muteval.system import System

    base_headers = {"Content-Type": "application/json"}
    if headers:
        base_headers.update(headers)

    def run(target: Any, case: Any) -> str:
        payload: dict = {"prompt": _prompt_of(target), "case": case}
        if isinstance(target, System):
            if target.context is not None:
                payload["context"] = list(target.context)
            if target.model is not None:
                payload["model"] = target.model
            if getattr(target, "tools", None) is not None:
                payload["tools"] = list(target.tools)
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=base_headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return raw  # plain-text response
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in _OUTPUT_KEYS:
                if k in data:
                    return str(data[k])
        return raw

    return run
