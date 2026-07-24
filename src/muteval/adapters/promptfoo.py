"""Use your existing promptfoo suite as a muteval target.

Point muteval at a `promptfooconfig.yaml`: the prompt becomes the mutation
target, each `tests` entry becomes a case, and promptfoo assertions are
translated to muteval evals. So muteval can ask "would your promptfoo asserts
catch a prompt regression?"

Supported assertions (translated to graded muteval evals): contains, icontains,
not-contains, not-icontains, equals, regex, and llm-rubric / model-graded-*
(-> muteval's stdlib LLM judge). Unsupported types (is-json, javascript, python,
cost, latency, custom, …) are SKIPPED — muteval prints which types it skipped,
and refuses a case whose assertions are *all* unsupported (rather than passing it
vacuously and inflating the score). One eval is emitted per assertion TYPE
(`promptfoo:contains`, `promptfoo:llm-rubric`, …) so the survivor report and
severity stay per-check.

Any OpenAI-compatible provider works for the model under test via ``base_url=``
or ``OPENAI_BASE_URL``. Needs PyYAML: pip install "muteval[promptfoo]".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from muteval.config import MutEvalConfig

# Assertion types muteval can translate into a graded eval.
_SUPPORTED_TYPES = {
    "contains", "icontains", "not-contains", "not-icontains",
    "equals", "regex", "llm-rubric", "model-graded",
}


def _render(template: str, variables: dict) -> str:
    """Minimal {{ var }} substitution (promptfoo uses nunjucks; we cover the
    common variable case)."""
    def repl(m):
        return str(variables.get(m.group(1).strip(), m.group(0)))

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", repl, template or "")


def _norm_type(assertion: dict) -> str:
    """Canonical assertion-type key (folds aliases) for grouping + support test."""
    t = str(assertion.get("type", "")).lower().strip()
    if t in ("not-contains", "notcontains", "not_contains"):
        return "not-contains"
    if t in ("not-icontains", "not_icontains"):
        return "not-icontains"
    if t in ("regex", "matches"):
        return "regex"
    if t.startswith("llm-rubric"):
        return "llm-rubric"
    if t.startswith("model-graded"):
        return "model-graded"
    return t


def _assertion_check(assertion: dict, base_url=None):
    """Translate ONE promptfoo assertion to a check fn, or None if unsupported."""
    typ = _norm_type(assertion)
    val = assertion.get("value")

    if typ == "contains":
        return lambda o, c: str(val) in o
    if typ == "icontains":
        return lambda o, c: str(val).lower() in o.lower()
    if typ == "not-contains":
        return lambda o, c: str(val) not in o
    if typ == "not-icontains":
        return lambda o, c: str(val).lower() not in o.lower()
    if typ == "equals":
        return lambda o, c: o.strip() == str(val).strip()
    if typ == "regex":
        return lambda o, c: re.search(str(val), o) is not None
    if typ in ("llm-rubric", "model-graded"):
        from muteval import checks

        judge = checks.llm_judge(str(val), base_url=base_url)
        return lambda o, c: bool(judge(o, c))
    return None  # is-json / javascript / python / custom -> not translatable


def _type_eval(typ: str, base_url=None):
    """A muteval eval for ONE assertion type: passes iff every assertion of that
    type on the case passes (and iff there is none of that type)."""
    def _eval(output, case) -> bool:
        for a in case.get("_asserts", []):
            if _norm_type(a) == typ:
                chk = _assertion_check(a, base_url)
                if chk is not None and not chk(output, case):
                    return False
        return True

    return _eval


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


def _make_run(model, base_url=None):
    from muteval.checks import _openai_chat_stdlib

    def run(prompt, case):
        return _openai_chat_stdlib(_render(prompt, case), model, base_url)

    return run


def config_from_promptfoo_dict(
    data, model: str = "gpt-4o-mini", run=None, base_url=None
) -> MutEvalConfig:
    """Build a MutEvalConfig from an already-parsed promptfoo config dict.

    Emits one eval per translatable assertion TYPE, warns about skipped types,
    and refuses a case whose assertions are all unsupported.
    """
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

    supported: set = set()
    skipped: set = set()
    for ci, case in enumerate(cases):
        asserts = case["_asserts"]
        translatable = 0
        for a in asserts:
            t = _norm_type(a)
            if t in _SUPPORTED_TYPES:
                supported.add(t)
                translatable += 1
            elif t:
                skipped.add(t)
        if asserts and translatable == 0:
            kinds = ", ".join(sorted({_norm_type(a) for a in asserts if _norm_type(a)}))
            raise ValueError(
                f"promptfoo case #{ci + 1}: all {len(asserts)} assertion(s) are "
                f"unsupported types ({kinds}) — muteval can't grade it, and passing "
                "it would inflate the score. Add a translatable assert "
                "(contains/regex/llm-rubric) or a muteval check."
            )
    if not supported:
        raise ValueError(
            "promptfoo config has no translatable assertions (supported: "
            "contains, icontains, not-contains, equals, regex, llm-rubric)."
        )
    if skipped:
        print(
            f"muteval: promptfoo — skipped {len(skipped)} unsupported assertion "
            f"type(s): {', '.join(sorted(skipped))} (not graded). Add a muteval "
            "check for those behaviors if you rely on them.",
            file=sys.stderr,
        )

    types = sorted(supported)
    evals = [_type_eval(t, base_url) for t in types]
    names = [f"promptfoo:{t}" for t in types]
    return MutEvalConfig(
        prompt=prompt,
        cases=cases,
        run=run or _make_run(model, base_url),
        evals=evals,
        eval_names=names,
    )


def from_promptfoo(path, model: str = "gpt-4o-mini", run=None, base_url=None) -> MutEvalConfig:
    """Load a promptfooconfig.yaml and return a MutEvalConfig."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'promptfoo adapter needs PyYAML: pip install "muteval[promptfoo]"'
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_promptfoo_dict(data, model=model, run=run, base_url=base_url)
