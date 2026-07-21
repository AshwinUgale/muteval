"""Use your existing promptfoo suite as a muteval target.

Point muteval at a `promptfooconfig.yaml`: the prompt becomes the mutation
target, each `tests` entry becomes a case, and promptfoo assertions are
translated to muteval evals. So muteval can ask "would your promptfoo asserts
catch a prompt regression?"

Supported assertions: contains, icontains, not-contains, equals, regex, and
llm-rubric / model-graded-* (-> muteval's stdlib LLM judge). Unsupported types
(javascript, python, custom) are skipped.

Needs PyYAML: pip install "muteval[promptfoo]".
"""

from __future__ import annotations

import re
from pathlib import Path

from muteval.config import MutEvalConfig


def _render(template: str, variables: dict) -> str:
    """Minimal {{ var }} substitution (promptfoo uses nunjucks; we cover the
    common variable case)."""
    def repl(m):
        return str(variables.get(m.group(1).strip(), m.group(0)))

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", repl, template or "")


def _assertion_check(assertion: dict):
    """Translate ONE promptfoo assertion to a muteval eval, or None if unsupported."""
    typ = str(assertion.get("type", "")).lower().strip()
    val = assertion.get("value")

    if typ == "contains":
        return lambda o, c: str(val) in o
    if typ == "icontains":
        return lambda o, c: str(val).lower() in o.lower()
    if typ in ("not-contains", "notcontains", "not_contains"):
        return lambda o, c: str(val) not in o
    if typ in ("not-icontains", "not_icontains"):
        return lambda o, c: str(val).lower() not in o.lower()
    if typ == "equals":
        return lambda o, c: o.strip() == str(val).strip()
    if typ in ("regex", "matches"):
        return lambda o, c: re.search(str(val), o) is not None
    if typ.startswith("llm-rubric") or typ.startswith("model-graded"):
        from muteval import checks

        judge = checks.llm_judge(str(val))
        return lambda o, c: bool(judge(o, c))
    return None  # javascript / python / custom -> not translatable


def _suite_eval(output, case) -> bool:
    """A case passes iff all of its (supported) promptfoo assertions pass."""
    for a in case.get("_asserts", []):
        chk = _assertion_check(a)
        if chk is not None and not chk(output, case):
            return False
    return True


def _prompt_from(data) -> str:
    prompts = data.get("prompts")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("promptfoo config has no `prompts`")
    p = prompts[0]
    if isinstance(p, dict):
        p = p.get("raw") or p.get("content") or p.get("id") or ""
    p = str(p)
    if p.startswith("file://"):
        p = Path(p[len("file://"):]).read_text(encoding="utf-8")
    return p


def _make_run(model):
    from muteval.checks import _openai_chat_stdlib

    def run(prompt, case):
        return _openai_chat_stdlib(_render(prompt, case), model)

    return run


def config_from_promptfoo_dict(data, model: str = "gpt-4o-mini", run=None) -> MutEvalConfig:
    """Build a MutEvalConfig from an already-parsed promptfoo config dict."""
    prompt = _prompt_from(data)
    tests = data.get("tests") or []
    default_asserts = (data.get("defaultTest") or {}).get("assert") or []
    cases = []
    for tst in tests:
        case = dict(tst.get("vars") or {})
        case["_asserts"] = list(default_asserts) + list(tst.get("assert") or [])
        cases.append(case)
    if not cases:
        raise ValueError("promptfoo config has no `tests`")
    return MutEvalConfig(
        prompt=prompt,
        cases=cases,
        run=run or _make_run(model),
        evals=[_suite_eval],
        eval_names=["promptfoo_asserts"],
    )


def from_promptfoo(path, model: str = "gpt-4o-mini", run=None) -> MutEvalConfig:
    """Load a promptfooconfig.yaml and return a MutEvalConfig."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'promptfoo adapter needs PyYAML: pip install "muteval[promptfoo]"'
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_promptfoo_dict(data, model=model, run=run)
