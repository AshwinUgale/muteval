"""Use your existing promptfoo suite as a muteval target.

Point muteval at a `promptfooconfig.yaml`: the prompt becomes the mutation
target, each `tests` entry becomes a case, and promptfoo assertions are
translated to muteval evals. So muteval can ask "would your promptfoo asserts
catch a prompt regression?"

Supported assertions (translated to graded muteval evals): contains, icontains,
not-contains, not-icontains, contains-any/-all, icontains-any/-all, equals,
not-equals, starts-with, regex, is-json, and llm-rubric / model-graded-* (->
muteval's stdlib LLM judge). External test files (`tests: file://cases.csv` /
`.jsonl` / `.json` / `.yaml`) are loaded; code-function / remote test generators
get a clear error. Unsupported assert types (javascript, python, cost, latency,
custom, …) are SKIPPED (muteval prints which). A case whose
assertions are *all* unsupported is dropped with a warning — never passed vacuously —
and the run fails closed only if nothing in the whole suite is translatable. One eval
is emitted per assertion TYPE (`promptfoo:contains`, `promptfoo:llm-rubric`, …) so the
survivor report and severity stay per-check.

The model under test is read from the promptfoo `providers:` block (first provider)
unless you pass an explicit model, so muteval mutates *your* prompt against *your*
model — not a default. A provider muteval can't call directly (e.g. a native Anthropic
endpoint) falls back to gpt-4o-mini with a warning. Any OpenAI-compatible provider
works via ``base_url=`` / ``OPENAI_BASE_URL``. Needs PyYAML: pip install
"muteval[promptfoo]".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from muteval.config import MutEvalConfig

# Assertion types muteval can translate into a graded eval.
_SUPPORTED_TYPES = {
    "contains",
    "icontains",
    "not-contains",
    "not-icontains",
    "contains-any",
    "contains-all",
    "icontains-any",
    "icontains-all",
    "equals",
    "not-equals",
    "starts-with",
    "regex",
    "is-json",
    "llm-rubric",
    "model-graded",
}


def _as_list(val):
    """List-valued assertions (contains-any/all) accept a YAML list OR a
    comma-separated string, matching promptfoo."""
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return [s.strip() for s in str(val).split(",") if s.strip()]


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
    if typ == "is-json":
        from muteval import checks

        return checks.is_json()
    if typ in ("llm-rubric", "model-graded"):
        from muteval import checks

        judge = checks.llm_judge(str(val), base_url=base_url)
        return lambda o, c: bool(judge(o, c))
    if typ == "contains-any":
        items = _as_list(val)
        return lambda o, c: any(s in o for s in items)
    if typ == "contains-all":
        items = _as_list(val)
        return lambda o, c: all(s in o for s in items)
    if typ == "icontains-any":
        items = [s.lower() for s in _as_list(val)]
        return lambda o, c: any(s in o.lower() for s in items)
    if typ == "icontains-all":
        items = [s.lower() for s in _as_list(val)]
        return lambda o, c: all(s in o.lower() for s in items)
    if typ == "not-equals":
        return lambda o, c: o.strip() != str(val).strip()
    if typ == "starts-with":
        return lambda o, c: o.lstrip().startswith(str(val))
    return None  # javascript / python / custom -> not translatable


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


_CODE_EXT = (".py", ".js", ".ts", ".mjs", ".cjs")


def _prompt_from(data, base_dir=Path(".")) -> str:
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
        ref = p[len("file://") :]
        if "*" in ref or "?" in ref:
            raise ValueError(
                f"promptfoo prompt uses a glob ({p}); muteval mutates a single prompt. "
                "Point it at a config with one prompt file, or inline the prompt."
            )
        head = ref.split(":", 1)[0].lower()
        if ":" in ref and head.endswith(_CODE_EXT):
            raise ValueError(
                f"promptfoo prompt comes from a code function ({p}); muteval can't "
                "execute it. Use a plain text/markdown prompt file or inline the prompt."
            )
        fp = base_dir / ref
        if not fp.exists():
            raise ValueError(f"promptfoo prompt file not found: {p}")
        p = fp.read_text(encoding="utf-8")
    return p


_EXPECT_PREFIXES = {
    "contains",
    "icontains",
    "not-contains",
    "regex",
    "equals",
    "starts-with",
}


def _expected_to_assert(val: str) -> dict:
    """Translate a promptfoo CSV `__expected` cell into an assertion. A known
    prefix (``contains: …``) is honored; a bare value defaults to equals."""
    if ":" in val:
        pre, _, rest = val.partition(":")
        if pre.strip().lower() in _EXPECT_PREFIXES:
            return {"type": pre.strip().lower(), "value": rest.strip()}
    return {"type": "equals", "value": val}


def _obj_to_test(obj) -> dict:
    """A loaded row/object -> a promptfoo-style test dict."""
    if isinstance(obj, dict) and ("vars" in obj or "assert" in obj):
        return obj
    return {"vars": obj if isinstance(obj, dict) else {"input": obj}}


def _load_external_tests(ref: str, base_dir: Path) -> list:
    """Load a promptfoo ``tests: file://…`` reference into a list of test dicts.

    Supports CSV / JSONL / JSON / YAML. Refuses (with a clear message) the ones
    muteval can't evaluate: remote URLs and code-function generators.
    """
    if "://" in ref and not ref.startswith("file://"):
        scheme = ref.split("://", 1)[0]
        raise ValueError(
            f"promptfoo tests use a '{scheme}://' source ({ref}); muteval can't "
            "fetch/load it. Export the cases to a local CSV/JSONL/JSON/YAML file."
        )
    path = ref[len("file://") :] if ref.startswith("file://") else ref
    head = path.split(":", 1)[0].lower()
    if (":" in path and head.endswith(_CODE_EXT)) or head.endswith(_CODE_EXT):
        raise ValueError(
            f"promptfoo tests come from code ({ref}); muteval can't execute it. "
            "Export the cases to CSV/JSONL/JSON/YAML or inline them."
        )
    fp = base_dir / path
    if not fp.exists():
        raise ValueError(f"promptfoo tests file not found: {ref}")
    low = path.lower()
    text = fp.read_text(encoding="utf-8")
    if low.endswith(".csv"):
        import csv
        import io

        out = []
        for row in csv.DictReader(io.StringIO(text)):
            row = {k: v for k, v in row.items() if k is not None}
            asserts = []
            for key in ("__expected", "expected"):
                v = row.pop(key, None)
                if v is not None and str(v).strip():
                    asserts.append(_expected_to_assert(str(v)))
            t: dict = {"vars": row}
            if asserts:
                t["assert"] = asserts
            out.append(t)
        return out
    if low.endswith(".jsonl"):
        import json

        return [
            _obj_to_test(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]
    if low.endswith(".json"):
        import json

        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
        return [_obj_to_test(x) for x in items]
    if low.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
        items = data if isinstance(data, list) else [data]
        return [_obj_to_test(x) for x in items]
    raise ValueError(
        f"promptfoo tests file type not supported: {ref} "
        "(use .csv / .jsonl / .json / .yaml)."
    )


def _load_external_obj(ref: str, base_dir: Path, what: str) -> dict:
    """Load a ``file://`` YAML/JSON reference into a dict (e.g. an external
    ``defaultTest``). Remote/unsupported sources get a clear error."""
    if "://" in ref and not ref.startswith("file://"):
        scheme = ref.split("://", 1)[0]
        raise ValueError(
            f"promptfoo {what} uses a '{scheme}://' source ({ref}); muteval can't "
            "load it. Inline it or use a local YAML/JSON file."
        )
    path = ref[len("file://") :] if ref.startswith("file://") else ref
    fp = base_dir / path
    if not fp.exists():
        raise ValueError(f"promptfoo {what} file not found: {ref}")
    low = path.lower()
    text = fp.read_text(encoding="utf-8")
    if low.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text) or {}
    if low.endswith(".json"):
        import json

        return json.loads(text)
    raise ValueError(
        f"promptfoo {what} file type not supported: {ref} (use .yaml / .json)."
    )


# Provider families muteval's OpenAI-compatible client can call at the default endpoint.
_OPENAI_NATIVE = {"openai", "azureopenai", "azure"}


def _provider_info(data):
    """Return (provider_id, family, model) from the first provider, else (None, None, None).

    Handles promptfoo id forms like ``openai:gpt-4o``, ``openai:chat:gpt-4o``,
    ``anthropic:messages:claude-3-5-sonnet``, ``ollama:llama3.1`` — the family is the
    first segment and the model is the last.
    """
    provs = data.get("providers")
    if not provs:
        return (None, None, None)
    p = provs[0] if isinstance(provs, list) else provs
    pid = (p.get("id") or p.get("label") or "") if isinstance(p, dict) else str(p)
    pid = pid.strip()
    if not pid:
        return (None, None, None)
    parts = pid.split(":")
    family = parts[0].lower()
    model = parts[-1] if len(parts) > 1 else None
    return (pid, family, model)


def _resolve_model(data, model, base_url):
    """Pick the model under test. Explicit ``model`` wins; otherwise read the
    promptfoo ``providers:`` block so muteval runs the model the suite actually uses."""
    if model:  # explicit override always wins
        return model
    pid, family, prov_model = _provider_info(data)
    if not pid:  # no providers block — keep the historical default
        return "gpt-4o-mini"
    if family in _OPENAI_NATIVE or base_url:
        chosen = prov_model or "gpt-4o-mini"
        via = " (via --base-url)" if base_url and family not in _OPENAI_NATIVE else ""
        print(
            f"muteval: promptfoo — model under test '{chosen}' from providers{via}.",
            file=sys.stderr,
        )
        return chosen
    print(
        f"muteval: promptfoo — provider '{pid}' isn't an OpenAI-compatible endpoint "
        "muteval can call directly; running 'gpt-4o-mini' instead. Pass --base-url "
        "(OpenAI-compatible) and --model to test your real provider.",
        file=sys.stderr,
    )
    return "gpt-4o-mini"


def _make_run(model, base_url=None):
    from muteval.checks import _openai_chat_stdlib

    def run(prompt, case):
        return _openai_chat_stdlib(_render(prompt, case), model, base_url)

    return run


def config_from_promptfoo_dict(
    data, model=None, run=None, base_url=None, base_dir=None
) -> MutEvalConfig:
    """Build a MutEvalConfig from an already-parsed promptfoo config dict.

    Emits one eval per translatable assertion TYPE, warns about skipped types,
    and *drops* (does not fail on) a case whose assertions are all unsupported —
    failing closed only if nothing in the whole suite is translatable. ``model``
    is auto-read from the ``providers:`` block when not given explicitly. External
    ``tests: file://…`` (CSV/JSONL/JSON/YAML) are loaded relative to ``base_dir``.
    """
    base_dir = Path(base_dir) if base_dir is not None else Path(".")
    prompt = _prompt_from(data, base_dir)

    raw_tests = data.get("tests")
    if isinstance(raw_tests, str):
        raw_tests = [raw_tests]
    tests = []
    for entry in raw_tests or []:
        if isinstance(entry, str):  # tests: file://cases.csv|.jsonl|.json|.yaml
            tests.extend(_load_external_tests(entry, base_dir))
        else:
            tests.append(entry)

    default_test = data.get("defaultTest") or {}
    if isinstance(default_test, str):  # defaultTest: file://shared/defaultTest.yaml
        default_test = _load_external_obj(default_test, base_dir, "defaultTest")
    default_asserts = (default_test or {}).get("assert") or []
    raw_cases = []
    for tst in tests:
        case = dict(tst.get("vars") or {})
        case["_asserts"] = list(default_asserts) + list(tst.get("assert") or [])
        raw_cases.append(case)
    if not raw_cases:
        raise ValueError("promptfoo config has no `tests`")

    supported: set = set()
    skipped: set = set()
    cases = []
    all_unsupported_cases = 0
    for case in raw_cases:
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
            # muteval can't grade this case — drop it (never pass it vacuously) and
            # keep going, rather than aborting the whole suite.
            all_unsupported_cases += 1
            continue
        cases.append(case)

    if all_unsupported_cases:
        print(
            f"muteval: promptfoo — dropped {all_unsupported_cases} case(s) whose "
            "assertions are all unsupported types (can't be graded, so not counted).",
            file=sys.stderr,
        )
    if not cases or not supported:
        raise ValueError(
            "promptfoo config has no translatable assertions to grade (supported: "
            "contains(-any/-all), icontains(-any/-all), not-contains, equals, "
            "not-equals, starts-with, regex, is-json, llm-rubric). "
            "Nothing to mutation-test — add a translatable assert or a muteval check."
        )
    if skipped:
        print(
            f"muteval: promptfoo — skipped {len(skipped)} unsupported assertion "
            f"type(s): {', '.join(sorted(skipped))} (not graded). Add a muteval "
            "check for those behaviors if you rely on them.",
            file=sys.stderr,
        )

    if run is None:
        run = _make_run(_resolve_model(data, model, base_url), base_url)

    types = sorted(supported)
    evals = [_type_eval(t, base_url) for t in types]
    names = [f"promptfoo:{t}" for t in types]
    return MutEvalConfig(
        prompt=prompt,
        cases=cases,
        run=run,
        evals=evals,
        eval_names=names,
    )


def from_promptfoo(path, model=None, run=None, base_url=None) -> MutEvalConfig:
    """Load a promptfooconfig.yaml and return a MutEvalConfig."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'promptfoo adapter needs PyYAML: pip install "muteval[promptfoo]"'
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_promptfoo_dict(
        data, model=model, run=run, base_url=base_url, base_dir=Path(path).parent
    )
