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
    "not-starts-with",
    "regex",
    "not-regex",
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


def _lookup(variables: dict, dotted: str):
    """``a.b`` -> variables["a"]["b"] (nunjucks-style), else a flat key."""
    if dotted in variables:
        return variables[dotted]
    cur = variables
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _render(template: str, variables: dict) -> str:
    """Minimal {{ var }} / {{ a.b }} substitution (promptfoo uses nunjucks; we
    cover variable references). An unknown variable is left as written."""

    def repl(m):
        value = _lookup(variables, m.group(1).strip())
        return m.group(0) if value is None else str(value)

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", repl, template or "")


def _render_value(value, variables: dict):
    """Render {{var}} in an ASSERTION value (a string, or each item of a list)
    against the case's vars — promptfoo does, so `not-contains: "{{forbidden}}"`
    checks the forbidden word, not the literal text "{{forbidden}}" (which
    guarded nothing: it always passed)."""
    if isinstance(value, str):
        return _render(value, variables)
    if isinstance(value, (list, tuple)):
        return [_render(v, variables) if isinstance(v, str) else v for v in value]
    return value


def _norm_type(assertion: dict) -> str:
    """Canonical assertion-type key (folds aliases) for grouping + support test."""
    t = str(assertion.get("type", "")).lower().strip()
    if t in ("not-contains", "notcontains", "not_contains"):
        return "not-contains"
    if t in ("not-icontains", "not_icontains"):
        return "not-icontains"
    if t in ("regex", "matches"):
        return "regex"
    if t in ("not-regex", "not-matches"):
        return "not-regex"
    if t.startswith("llm-rubric"):
        return "llm-rubric"
    if t.startswith("model-graded"):
        return "model-graded"
    return t


def _vars(case) -> dict:
    return (
        {k: v for k, v in case.items() if k != "_asserts"}
        if isinstance(case, dict)
        else {}
    )


def _json_equal(output: str, expected) -> bool:
    import json

    try:
        return json.loads(output) == expected
    except (TypeError, ValueError):
        return False


def _schema_ok(output: str, schema) -> bool:
    """is-json with a JSON schema: parse, then check the schema's top-level
    ``type`` and ``required`` keys (a pragmatic subset; the full schema is
    validated when the optional ``jsonschema`` package is installed)."""
    import json

    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return False
    if not isinstance(schema, dict):
        return True
    try:
        import jsonschema

        try:
            jsonschema.validate(data, schema)
            return True
        except jsonschema.ValidationError:
            return False
    except ImportError:
        pass
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool}
    if kind in types and not isinstance(data, types[kind]):
        return False
    if kind in ("number", "integer") and (
        isinstance(data, bool) or not isinstance(data, (int, float))
    ):
        return False
    required = schema.get("required") or []
    return not (
        required and not (isinstance(data, dict) and all(k in data for k in required))
    )


def _assertion_check(assertion: dict, base_url=None, grader_model=None):
    """Translate ONE promptfoo assertion to a check fn ``(output, case)`` —
    returning a bool, or an EvalOutcome for graded ones — or None if
    unsupported. Values are rendered against the case's vars at check time."""
    typ = _norm_type(assertion)
    raw = assertion.get("value")

    def val(c):
        return _render_value(raw, _vars(c))

    if typ == "contains":
        return lambda o, c: str(val(c)) in o
    if typ == "icontains":
        return lambda o, c: str(val(c)).lower() in o.lower()
    if typ == "not-contains":
        return lambda o, c: str(val(c)) not in o
    if typ == "not-icontains":
        return lambda o, c: str(val(c)).lower() not in o.lower()
    if typ == "equals":
        if isinstance(raw, (dict, list)):  # promptfoo compares objects as JSON
            return lambda o, c: _json_equal(o, raw)
        return lambda o, c: o.strip() == str(val(c)).strip()
    if typ == "regex":
        return lambda o, c: re.search(str(val(c)), o) is not None
    if typ == "is-json":
        return lambda o, c: _schema_ok(o, raw)
    if typ in ("llm-rubric", "model-graded"):
        from muteval import checks

        threshold = assertion.get("threshold")
        judge_kwargs = {"base_url": base_url}
        grader = assertion.get("_grader") or grader_model  # assertion's provider wins
        if grader:
            judge_kwargs["model"] = grader
        if isinstance(threshold, (int, float)):
            judge_kwargs["threshold"] = float(threshold)

        def _graded(o, c):
            judge = checks.llm_judge(str(val(c)), **judge_kwargs)
            # promptfoo vars use arbitrary names (question / query / …), but
            # llm_judge reads case["input"] — so an un-`input` suite showed the
            # judge "User input: None" (#36). Keep an existing `input` var; else
            # synthesize one from all the case's vars so the judge sees the real
            # input instead of nothing.
            if isinstance(c, dict) and c.get("input") is None:
                vars_only = _vars(c)
                c = {**c, "input": "\n".join(f"{k}: {v}" for k, v in vars_only.items())}
            return judge(o, c)  # an EvalOutcome: the score survives for near misses

        _graded.is_llm = True  # type: ignore[attr-defined]
        return _graded
    if typ == "contains-any":
        return lambda o, c: any(s in o for s in _as_list(val(c)))
    if typ == "contains-all":
        return lambda o, c: all(s in o for s in _as_list(val(c)))
    if typ == "icontains-any":
        return lambda o, c: any(s.lower() in o.lower() for s in _as_list(val(c)))
    if typ == "icontains-all":
        return lambda o, c: all(s.lower() in o.lower() for s in _as_list(val(c)))
    if typ == "not-equals":
        if isinstance(raw, (dict, list)):
            return lambda o, c: not _json_equal(o, raw)
        return lambda o, c: o.strip() != str(val(c)).strip()
    if typ == "starts-with":
        return lambda o, c: o.lstrip().startswith(str(val(c)))
    if typ == "not-starts-with":
        return lambda o, c: not o.lstrip().startswith(str(val(c)))
    if typ == "not-regex":
        return lambda o, c: re.search(str(val(c)), o) is None
    return None  # javascript / python / custom -> not translatable


def _type_eval(typ: str, base_url=None, grader_model=None):
    """A muteval eval for ONE assertion type: passes iff every assertion of that
    type on the case passes (and iff there is none of that type). A graded
    assertion's score is kept (the closest passing one, for near-miss
    reporting)."""
    from muteval.evals import EvalOutcome, coerce_outcome

    def _eval(output, case):
        closest = None
        for a in case.get("_asserts", []):
            if _norm_type(a) != typ:
                continue
            chk = _assertion_check(a, base_url, grader_model)
            if chk is None:
                continue
            outcome = coerce_outcome(chk(output, case))
            if not outcome.passed:
                return outcome
            if outcome.margin is not None and (
                closest is None or outcome.margin < closest.margin
            ):
                closest = outcome
        return closest if closest is not None else EvalOutcome(passed=True)

    if typ in ("llm-rubric", "model-graded"):
        _eval.is_llm = True  # type: ignore[attr-defined]
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
    if _is_chat_prompt(p):
        raise ValueError(
            "this promptfoo prompt is a chat message list (JSON [{role, content}, "
            "...]); muteval mutates a single prompt text and would mangle the JSON "
            "(and send it as one user message). Point a muteval config at the "
            "system message's text instead."
        )
    return p


def _is_chat_prompt(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped.startswith("["):
        return False
    import json

    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(m, dict) and "role" in m for m in data)
    )


# promptfoo's CSV `__expected` prefixes -> assertion type. Code / similarity
# prefixes map to their (unsupported) types, so they're SKIPPED with a warning
# — they used to fall through to `equals "<whole cell>"`, failing the baseline.
_EXPECT_PREFIXES = {
    "contains": "contains",
    "icontains": "icontains",
    "not-contains": "not-contains",
    "not-icontains": "not-icontains",
    "contains-any": "contains-any",
    "contains-all": "contains-all",
    "icontains-any": "icontains-any",
    "icontains-all": "icontains-all",
    "regex": "regex",
    "equals": "equals",
    "not-equals": "not-equals",
    "starts-with": "starts-with",
    "is-json": "is-json",
    "llm-rubric": "llm-rubric",
    "grade": "llm-rubric",
    "javascript": "javascript",
    "fn": "javascript",
    "eval": "javascript",
    "python": "python",
    "similar": "similar",
}


def _expected_to_assert(val: str) -> dict:
    """Translate a promptfoo CSV `__expected` cell into an assertion. A known
    prefix (``contains: …``, ``grade: …``, bare ``is-json``) is honored; a bare
    value defaults to equals (as in promptfoo)."""
    cell = val.strip()
    if cell.lower() == "is-json":
        return {"type": "is-json"}
    if ":" in cell:
        pre, _, rest = cell.partition(":")
        typ = _EXPECT_PREFIXES.get(pre.strip().lower())
        if typ:
            return {"type": typ, "value": rest.strip()}
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
            # __expected, __expected1, __expected2, ... (and a plain `expected`)
            keys = [k for k in row if re.fullmatch(r"__expected\d*", k)] + (
                ["expected"] if "expected" in row else []
            )
            for key in sorted(keys):
                v = row.pop(key, None)
                if v is not None and str(v).strip():
                    asserts.append(_expected_to_assert(str(v)))
            # Other `__` columns (__description, __metric, __threshold, ...) are
            # promptfoo metadata, not prompt variables.
            row = {k: v for k, v in row.items() if not k.startswith("__")}
            t: dict = {"vars": row}
            if asserts:
                t["assert"] = asserts
            out.append(t)
        return out
    if low.endswith(".jsonl"):
        import json

        return [
            _obj_to_test(json.loads(line)) for line in text.splitlines() if line.strip()
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


def _load_file_value(value, base_dir: Path):
    """promptfoo loads ``file://`` values (a var, an assertion value) from disk:
    text as-is, .json / .yaml parsed. Code references (.py/.js) are left alone —
    they belong to javascript/python assertions, which muteval skips."""
    if not (isinstance(value, str) and value.startswith("file://")):
        return value
    ref = value[len("file://") :]
    if ref.split(":", 1)[0].lower().endswith(_CODE_EXT):
        return value
    fp = base_dir / ref
    if not fp.exists():
        raise ValueError(f"promptfoo file reference not found: {value}")
    text = fp.read_text(encoding="utf-8")
    low = ref.lower()
    if low.endswith(".json"):
        import json

        return json.loads(text)
    if low.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text)
    return text.strip("\n")


def _expand_vars(variables: dict) -> list:
    """promptfoo runs a test once per combination of ARRAY-valued vars; a list
    var used to be sent to the model as its Python repr."""
    import itertools

    keys = [k for k, v in variables.items() if isinstance(v, list)]
    if not keys:
        return [dict(variables)]
    combos = []
    for values in itertools.product(*(variables[k] for k in keys)):
        combo = dict(variables)
        combo.update(zip(keys, values))
        combos.append(combo)
    return combos


def _provider_model(provider) -> "str | None":
    """The model name in a promptfoo provider reference ("openai:gpt-4o",
    {"id": "openai:chat:gpt-4o"}), else None."""
    pid = provider.get("id") if isinstance(provider, dict) else provider
    if not isinstance(pid, str) or ":" not in pid:
        return None
    return pid.split(":")[-1] or None


def config_from_promptfoo_dict(
    data, model=None, run=None, base_url=None, base_dir=None
) -> MutEvalConfig:
    """Build a MutEvalConfig from an already-parsed promptfoo config dict.

    Emits one eval per translatable assertion TYPE, warns about skipped types,
    and *drops* (does not fail on) a case whose assertions are all unsupported —
    failing closed only if nothing in the whole suite is translatable. ``model``
    is auto-read from the ``providers:`` block when not given explicitly. External
    ``tests: file://…`` (CSV/JSONL/JSON/YAML) are loaded relative to ``base_dir``.

    Fidelity: ``{{var}}`` in assertion values, ``file://`` vars / values,
    ``defaultTest.vars``, and array vars are handled as promptfoo does. What
    muteval can NOT reproduce — output transforms, test ``threshold`` / assertion
    ``weight`` scoring, extra prompts — is listed in one warning, never silently.
    """
    base_dir = Path(base_dir) if base_dir is not None else Path(".")
    notes: list = []
    prompts = data.get("prompts")
    if isinstance(prompts, list) and len(prompts) > 1:
        notes.append(
            f"{len(prompts)} prompts: muteval mutates only the first (run once per "
            "prompt to cover the others)"
        )
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
    default_test = default_test or {}
    default_asserts = default_test.get("assert") or []
    default_vars = default_test.get("vars") or {}
    default_options = default_test.get("options") or {}
    grader_model = _provider_model(default_options.get("provider"))

    transformed = weighted = thresholded = 0
    raw_cases = []
    for tst in tests:
        options = {**default_options, **(tst.get("options") or {})}
        if options.get("transform") or options.get("postprocess"):
            # The assertions grade the TRANSFORMED output, which muteval can't
            # compute: grading the raw output would change their meaning.
            transformed += 1
            continue
        if tst.get("threshold") is not None:
            thresholded += 1
        merged_vars = {**default_vars, **(tst.get("vars") or {})}
        merged_vars = {k: _load_file_value(v, base_dir) for k, v in merged_vars.items()}
        asserts = []
        for a in list(default_asserts) + list(tst.get("assert") or []):
            if not isinstance(a, dict):
                continue
            a = dict(a)
            if a.get("transform"):
                a["type"] = f"{a.get('type', '')} (with transform)"  # unsupported
            if a.get("weight") not in (None, 1):
                weighted += 1
            if "value" in a:
                a["value"] = _load_file_value(a["value"], base_dir)
            if a.get("provider") and not a.get("_grader"):
                a["_grader"] = _provider_model(a.get("provider"))
            asserts.append(a)
        for combo in _expand_vars(merged_vars):
            case = dict(combo)
            case["_asserts"] = asserts
            raw_cases.append(case)
    if transformed:
        notes.append(
            f"{transformed} test(s) with an output transform were dropped (muteval "
            "can't run promptfoo transforms, and the assertions grade the "
            "transformed output)"
        )
    if thresholded or weighted:
        notes.append(
            "test `threshold` / assertion `weight` scoring is not reproduced: "
            "muteval requires EVERY assertion to pass"
        )
    if not raw_cases:
        if transformed:
            raise ValueError(
                f"every promptfoo test ({transformed}) uses an output transform "
                "(options.transform / postprocess), which muteval can't run — and the "
                "assertions grade the transformed output. Remove the transform or "
                "point a muteval config at the post-transform behavior."
            )
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
    for note in notes:
        print(f"muteval: promptfoo fidelity — {note}.", file=sys.stderr)

    resolved = _resolve_model(data, model, base_url) if run is None else model
    if run is None:
        run = _make_run(resolved, base_url)
    # The llm-rubric grader: the assertion's / defaultTest's provider, else the
    # model under test (a hard-coded gpt-4o-mini broke on a non-OpenAI base_url).
    grader = grader_model or resolved

    types = sorted(supported)
    evals = [_type_eval(t, base_url, grader) for t in types]
    names = [f"promptfoo:{t}" for t in types]
    return MutEvalConfig(
        prompt=prompt,
        cases=cases,
        run=run,
        evals=evals,
        eval_names=names,
        model_under_test=resolved,
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
