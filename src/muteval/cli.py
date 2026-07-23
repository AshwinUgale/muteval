"""muteval command line.

Two ways to run:

  1) Zero-config (no Python file) — muteval owns the model call:
       muteval run --prompt-file system.txt --cases cases.jsonl \
         --model gpt-4o-mini --check contains:8080 \
         --judge "the answer is grounded in the context" --fail-under 75

     Add a retrieval corpus to mutation-test RAG (drops/clears docs):
       muteval run --prompt-file system.txt --cases cases.jsonl \
         --context-file knowledge.txt --judge "grounded in the context"

  2) Full control — point at a Python config (custom run/pipeline/metrics):
       muteval run --config muteval_config.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

from muteval import __version__
from muteval.config import MutEvalConfig, load_config
from muteval.mutators import OPERATORS
from muteval.report import format_report
from muteval.runner import (
    BASELINE_ERRORED,
    BASELINE_FAILED,
    BUDGET_EXCEEDED,
    NO_EVALUATED_MUTANTS,
    NO_MUTANTS,
    PARTIAL_ERRORS,
    VALID,
    run_mutation_testing,
    select_mutants,
)
from muteval.system import System

_STARTER_CONFIG = '''"""muteval config scaffold — edit the TODOs, then run:

    muteval run --config muteval_config.py
"""

from muteval import MutEvalConfig
from muteval import checks


SYSTEM_PROMPT = """You are a support assistant for an online store.
- You must always cite the order ID in your reply.
- Do not promise refunds; a manager must approve them.
- Reply in a polite, professional tone.
"""


def run(prompt: str, case: dict) -> str:
    # A tiny stand-in "model" so this runs with NO API key: it reflects the
    # prompt, so mutating a rule changes the output (that is how muteval finds
    # eval gaps). TODO: replace with a call to YOUR real LLM/app.
    p = prompt.lower()
    reply = []
    if "polite" in p:
        reply.append("Hi there!")
    if "cite the order id" in p:
        reply.append(f"Order {case['order_id']}:")
    if "do not promise refunds" in p:
        reply.append("I can't promise a refund; a manager will review it.")
    else:
        reply.append("Sure, refunding you now!")
    return " ".join(reply)


config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,
    cases=[{"input": "where is my order?", "order_id": "A123"}],
    run=run,
    # DELIBERATELY incomplete — muteval will surface the gaps (nothing here
    # checks the refund rule or the tone). Add checks to kill the survivors.
    evals=[checks.contains_case("order_id")],
)
'''


_RAG_STARTER_CONFIG = '''"""muteval RAG scaffold (System mode) — edit the four blocks, then:

    muteval check --config muteval_config.py   # validate wiring + baseline first
    muteval run   --config muteval_config.py

The mutation target here is the retrieved CONTEXT (System mode). This runs with
NO API key using a tiny mock retriever; replace the TODO with your real pipeline.
"""

from muteval import MutEvalConfig, System
from muteval import checks


# ---- 2. YOUR CONTEXT: the retrieval corpus muteval will mutate ----
CONTEXT = (
    "doc-1 :: The Orbit X ships with a 24-month warranty from the purchase date.",
    "doc-2 :: Support replies to tickets within one business day.",
    "doc-3 :: Returns are accepted within 30 days if the device is undamaged.",
)

SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know."
)


def _overlap(a: str, b: str) -> int:
    wa = {w for w in a.lower().split() if len(w) > 2}
    wb = {w for w in b.lower().split() if len(w) > 2}
    return len(wa & wb)


# ---- 1. YOUR PIPELINE: run(system, case) -> a FRESH output string ----
def run(system, case):
    # TODO: replace this mock with YOUR real pipeline:
    #   - retrieve from system.context (the possibly-mutated corpus)
    #   - generate with your LLM using system.prompt
    docs = system.context or ()
    if not docs:
        return "I don't know."
    best = max(docs, key=lambda d: _overlap(case["question"], d))
    return best.split("::", 1)[-1].strip()   # mock "LLM": echo the top doc


# ---- 4. YOUR CASES ----
CASES = [
    {"question": "How long is the Orbit X warranty?", "expected": "24-month"},
    {"question": "How fast does support reply?", "expected": "one business day"},
]


# ---- 3. YOUR EVALS: grade the output ----
config = MutEvalConfig(
    system=System(prompt=SYSTEM_PROMPT, context=CONTEXT),
    cases=CASES,
    run=run,
    evals=[
        checks.contains_case("expected"),   # the answer must contain the expected fact
        # An LLM judge on ANY OpenAI-compatible endpoint (needs OPENAI_API_KEY):
        # checks.llm_judge(
        #     "the answer is grounded in the retrieved context and does not invent facts",
        #     base_url="https://api.groq.com/openai/v1",   # or OpenAI/Gemini/GitHub Models/Ollama
        #     model="openai/gpt-oss-20b", input_key="question",
        # ),
    ],
    # DELIBERATELY thin — muteval will surface what these evals miss (e.g. nothing
    # here checks the grounding/abstention rule). Uncomment the judge to close gaps.
)
'''


# --- zero-config helpers -----------------------------------------------------


def _load_cases(path: str) -> List[Any]:
    """Load cases from a .jsonl (one JSON object per line) or .json (a list).

    Tolerates a UTF-8 BOM (PowerShell's `Out-File -Encoding utf8` writes one).
    """
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    try:
        doc = json.loads(text)
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            return [doc]
    except json.JSONDecodeError:
        pass
    cases = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {i}: invalid JSON ({exc})") from exc
    return cases


def _load_context(args: argparse.Namespace) -> List[str]:
    """Build the shared retrieval corpus (a list of docs) from --context /
    --context-file. A context file is split into docs on blank lines."""
    docs: List[str] = list(args.context or [])
    if args.context_file:
        raw = Path(args.context_file).read_text(encoding="utf-8-sig")
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        docs += parts or ([raw.strip()] if raw.strip() else [])
    return docs


def _check_from_spec(spec: str, threshold: float, model: str):
    """Turn a --check string like 'contains:8080' into a muteval eval."""
    from muteval import checks

    name, _, arg = spec.partition(":")
    name = name.strip().lower()
    arg = arg.strip()

    if name == "contains":
        return checks.contains(arg)
    if name in ("not_contains", "notcontains"):
        return checks.not_contains(arg)
    if name in ("contains_case", "contains_key"):
        return checks.contains_case(arg)
    if name == "regex":
        return checks.regex_matches(arg)
    if name == "is_json":
        return checks.is_json()
    if name == "equals":
        return checks.equals(arg or "expected")
    if name == "judge":
        if not arg:
            raise ValueError("judge:<rubric> needs a rubric, e.g. judge:is it polite")
        return checks.llm_judge(arg, threshold=threshold, model=model)
    raise ValueError(
        f"unknown check '{name}'. Use one of: contains, not_contains, "
        "contains_case, regex, is_json, equals, judge"
    )


def _config_from_flags(args: argparse.Namespace) -> MutEvalConfig:
    from muteval.runners import openai_run

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig")
    else:
        prompt = args.prompt
    if not prompt or not prompt.strip():
        raise ValueError("--prompt/--prompt-file is empty")

    cases = _load_cases(args.cases)

    specs = list(args.check or [])
    specs += [f"judge:{r}" for r in (args.judge or [])]
    if not specs:
        raise ValueError(
            "provide at least one --check (e.g. --check contains:8080) "
            "or --judge \"...\""
        )
    evals = [_check_from_spec(s, args.threshold, args.model) for s in specs]
    names = [s.split(":", 1)[0] for s in specs]

    if getattr(args, "target", None):
        from muteval.runners import callable_run

        run = callable_run(args.target)
    elif getattr(args, "endpoint", None):
        from muteval.runners import http_run

        run = http_run(args.endpoint)
    else:
        run = openai_run(model=args.model)
    context_docs = _load_context(args)
    common = dict(
        cases=cases, run=run, evals=evals, eval_names=names,
        runs_per_mutant=args.runs_per_mutant,
        scope_include=args.scope_include, scope_exclude=args.scope_exclude,
    )
    if context_docs or args.mutate_model:
        # System mode: the corpus is mutable (drop_context_doc / clear_context)
        # and/or the model is mutable (downgrade_model). openai_run is
        # System-aware, so it honors both.
        sys_obj = System(
            prompt=prompt,
            context=tuple(context_docs) if context_docs else None,
            model=args.model if args.mutate_model else None,
        )
        return MutEvalConfig(system=sys_obj, **common)
    return MutEvalConfig(prompt=prompt, **common)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muteval",
        description="Mutation testing for LLM eval suites — measure whether "
        "your evals would actually catch a regression.",
    )
    parser.add_argument("--version", action="version", version=f"muteval {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    run = sub.add_parser("run", help="Run mutation testing against an eval suite.")
    run.add_argument(
        "--config", "-c",
        help="Path to a Python file defining a module-level `config`. Use this "
        "for custom run()/pipelines/metrics.",
    )
    run.add_argument(
        "--promptfoo",
        metavar="promptfooconfig.yaml",
        help="Ingest an existing promptfoo config (prompt + tests + assertions) "
        "and run muteval against it — no muteval config file needed.",
    )
    run.add_argument("--prompt", help="The system prompt to mutate (inline).")
    run.add_argument("--prompt-file", help="File containing the system prompt to mutate.")
    run.add_argument("--cases", help="Cases file (.jsonl or .json list).")
    run.add_argument(
        "--target",
        metavar="pkg.mod:fn",
        help="Use your own function as the system under test, imported by dotted "
        "path and called as fn(prompt, case) -> str. No run() wrapper needed.",
    )
    run.add_argument(
        "--endpoint",
        metavar="URL",
        help="Drive an HTTP endpoint as the system: POSTs {prompt, case} JSON, "
        "reads the text output. Test a deployed pipeline without importing it.",
    )
    run.add_argument(
        "--context", action="append",
        help="A retrieved-context doc (repeatable). Enables RAG mutation "
        "(drop_context_doc / clear_context) over a shared corpus.",
    )
    run.add_argument(
        "--context-file",
        help="File of retrieved context; split into docs on blank lines. "
        "Enables RAG mutation.",
    )
    run.add_argument(
        "--mutate-model", action="store_true",
        help="Also test a model downgrade (downgrade_model): swaps --model for a "
        "weaker one and checks whether your evals notice.",
    )
    run.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini).")
    run.add_argument(
        "--check", action="append",
        help="A built-in check, repeatable: contains:TXT, not_contains:TXT, "
        "contains_case:KEY, regex:PAT, is_json, equals, judge:<rubric>.",
    )
    run.add_argument("--judge", action="append", help="LLM-judge rubric (repeatable). Sugar for --check judge:<rubric>.")
    run.add_argument("--scope-include", help="Only mutate prompt lines matching this regex.")
    run.add_argument("--scope-exclude", help="Never mutate prompt lines matching this regex.")
    run.add_argument("--threshold", type=float, default=0.7, help="Judge pass threshold (default 0.7).")
    run.add_argument("--runs-per-mutant", type=int, default=1, help="Runs per mutant (default 1).")
    run.add_argument(
        "--operators", nargs="+", choices=list(OPERATORS.keys()), default=None,
        help="Subset of mutation operators (default: all).",
    )
    run.add_argument("--max-mutants", type=int, default=None, help="Cap the number of mutants (head).")
    run.add_argument("--sample", type=int, default=None, help="Randomly sample N mutants (cheap runs).")
    run.add_argument("--seed", type=int, default=None, help="Seed for --sample (reproducible).")
    run.add_argument(
        "--cache", metavar="PATH", default=None,
        help="Memoize run outputs + eval outcomes in a sqlite file, so an "
        "identical re-run makes zero model/judge calls. Ignored when "
        "--runs-per-mutant > 1 (non-deterministic).",
    )
    run.add_argument(
        "--concurrency", type=int, default=1, metavar="N",
        help="Evaluate N mutants in parallel (thread pool). Cuts wall-clock on "
        "API-bound suites. Default 1 (sequential).",
    )
    run.add_argument(
        "--manifest", metavar="PATH", default=None,
        help="Write a reproducible-run manifest (version, model, seed, operators, "
        "config fingerprint, result) — commit it beside a real run to make it auditable.",
    )
    run.add_argument(
        "--max-calls", type=int, default=None, metavar="N",
        help="Cap the number of model + judge calls (cache hits / skipped judges "
        "don't count). Fails closed (exit 2) before overspending.",
    )
    run.add_argument(
        "--fail-under", type=float, default=None, metavar="PCT",
        help="Exit non-zero if the mutation score is below this percent (gate CI).",
    )
    run.add_argument(
        "--fail-on-severity", choices=["high", "medium", "low"], default=None,
        help="Exit non-zero if any real survivor is at or above this severity "
        "(e.g. 'high' fails on any unguarded high-severity gap, even if the "
        "overall score looks fine).",
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate the run WITHOUT calling the model.",
    )
    run.add_argument(
        "--allow-empty", action="store_true",
        help="Treat a run that generates NO mutants as a pass (exit 0) instead "
        "of an invalid run. Baseline failures/errors are never rescued by this.",
    )
    run.add_argument(
        "--max-error-rate", type=float, default=None, metavar="FRAC",
        help="Fraction of mutants allowed to error before the run is INVALID "
        "(default 0.0 = fail closed on any errored mutant). Overrides the "
        "config value.",
    )
    run.add_argument(
        "--allow-mutant-errors", action="store_true",
        help="Tolerate any number of errored mutants (equivalent to "
        "--max-error-rate 1.0). Score is computed over the survivors.",
    )
    run.add_argument("--json", metavar="PATH", default=None,
        help="Write machine-readable results (score, CI, survivors) to a JSON file.")
    run.add_argument("--badge", metavar="PATH", default=None,
        help="Write a shields.io endpoint JSON for the eval-coverage badge.")
    run.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    init = sub.add_parser("init", help="Scaffold a starter muteval_config.py you can edit.")
    init.add_argument("--path", "-p", default="muteval_config.py", help="Where to write the scaffold.")
    init.add_argument("--force", action="store_true", help="Overwrite if it exists.")
    init.add_argument(
        "--template", "-t", choices=["basic", "rag"], default="basic",
        help="basic = prompt-only support-bot; rag = System-mode (mutates retrieved context).",
    )

    probe = sub.add_parser(
        "probe", help="Rate your eval suite's quality (a report card of probes)."
    )
    probe.add_argument("--config", "-c", required=True, help="Path to a config file.")
    probe.add_argument(
        "--probes", nargs="+", default=None,
        help="Subset of probes to run (default: all).",
    )
    probe.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    probe.add_argument(
        "--html", metavar="PATH", default=None,
        help="Also write the report card as a standalone HTML page.",
    )

    check = sub.add_parser(
        "check",
        help="Doctor: validate a config's wiring (and baseline) BEFORE a full run.",
    )
    check.add_argument("--config", "-c", required=True, help="Path to a config file.")
    check.add_argument(
        "--operators", nargs="+", choices=list(OPERATORS.keys()), default=None,
        help="Operators to check mutant generation for (default: all).",
    )
    check.add_argument(
        "--no-model", action="store_true",
        help="Only run the 0-call structural checks (skip run()/evals).",
    )
    check.add_argument(
        "--full", action="store_true",
        help="Exercise run()/evals on EVERY case (a true baseline), not just the first.",
    )
    check.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    results = sub.add_parser(
        "results", help="Show the ranked survivors from the last run (no re-run).",
    )
    results.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    show = sub.add_parser(
        "show", help="Show one survivor from the last run: details + output diff.",
    )
    show.add_argument("id", type=int, help="Survivor id (from `muteval results`).")
    show.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    report = sub.add_parser(
        "report", help="Render a shareable HTML report from a run.",
    )
    report.add_argument("--html", required=True, metavar="PATH", help="Output HTML file.")
    report.add_argument(
        "--json", metavar="PATH", default=None,
        help="Input run JSON (default: the last run at .muteval/last_run.json).",
    )

    label = sub.add_parser(
        "label",
        help="Emit a worksheet (case/output/machine-verdict) to hand-label for "
        "the human-agreement probe.",
    )
    label.add_argument("--config", "-c", required=True, help="Path to a config file.")
    label.add_argument(
        "--out", metavar="PATH", default=str(Path(".muteval") / "labels.csv"),
        help="Where to write the worksheet CSV (default: .muteval/labels.csv).",
    )
    return parser


def _format_checks(results, use_color: bool = True) -> str:
    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = ["", c("muteval check — config doctor", "1"), ""]
    for r in results:
        tag = c("PASS", "32") if r.ok else c("FAIL", "1;31")
        line = f"  [{tag}] {r.name}"
        if r.detail:
            line += c(f"  — {r.detail}", "2" if r.ok else "33")
        lines.append(line)
    lines.append("")
    from muteval.doctor import all_ok

    if all_ok(results):
        lines.append(c("✓ Ready — nothing blocking. Run: muteval run --config <file>", "32"))
    else:
        lines.append(c("✗ Not ready — fix the FAIL row(s) above, then re-check.", "1;31"))
    return "\n".join(lines)


def _load_run_config(args: argparse.Namespace) -> MutEvalConfig:
    if args.config:
        return load_config(args.config)
    if getattr(args, "promptfoo", None):
        from muteval.adapters.promptfoo import from_promptfoo

        return from_promptfoo(args.promptfoo, model=args.model)
    if args.prompt or args.prompt_file:
        if not args.cases:
            raise ValueError("--cases is required in zero-config mode")
        return _config_from_flags(args)
    raise ValueError(
        "nothing to run: pass --config FILE, --promptfoo FILE, or zero-config "
        "flags (--prompt/--prompt-file + --cases + --check/--judge)"
    )


_LAST_RUN = Path(".muteval") / "last_run.json"


def _save_last_run(result) -> None:
    """Persist the last run's machine-readable summary for `results` / `show`.
    Best-effort: never fail a run over persistence."""
    from muteval.report import result_to_dict

    try:
        _LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
        _LAST_RUN.write_text(
            json.dumps(result_to_dict(result), indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _write_label_worksheet(config, out) -> int:
    """Run the system + evals over every case and write a hand-labeling
    worksheet CSV (one row per case×eval) with a blank human_label column."""
    import csv

    from muteval.evals import coerce_outcome
    from muteval.runner import _eval_label

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_index", "case", "output", "eval", "machine_verdict", "human_label"])
        for ci, case in enumerate(config.cases):
            output = config.invoke(config.system, case)
            for idx, ev in enumerate(config.evals):
                label = _eval_label(config, idx)
                verdict = "pass" if coerce_outcome(ev(output, case)).passed else "fail"
                w.writerow([ci, str(case)[:200], str(output)[:500], label, verdict, ""])
                n += 1
    return n


def _load_last_run() -> Optional[dict]:
    if not _LAST_RUN.exists():
        return None
    try:
        return json.loads(_LAST_RUN.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_SEV_TAG = {"high": ("[HIGH]", "31"), "medium": ("[MED ]", "33"), "low": ("[LOW ]", "2")}


def _format_results(data: dict, use_color: bool = True) -> str:
    def c(t, code):
        return f"\033[{code}m{t}\033[0m" if use_color else t

    survs = data.get("survivors", [])
    lines = ["", c("muteval — survivors from the last run", "1")]
    eff = data.get("effective_score")
    if eff is not None:
        lines.append(c(f"effective score {eff * 100:.0f}%  ", "2") + f"({len(survs)} real survivor(s))")
    lines.append("")
    if not survs:
        lines.append(c("✓ No survivors saved — your evals caught everything.", "32"))
        return "\n".join(lines)
    for s in survs:
        tag, code = _SEV_TAG.get(s.get("severity") or "medium", ("[MED ]", "33"))
        lines.append(
            f"  {c(str(s['id']).rjust(3), '1')} {c(tag, code)} "
            f"[{s['operator']}] {s['description']}"
        )
    lines.append("")
    lines.append(c("Inspect one:  muteval show <id>", "2"))
    return "\n".join(lines)


def _format_show(s: dict, use_color: bool = True) -> str:
    import difflib

    def c(t, code):
        return f"\033[{code}m{t}\033[0m" if use_color else t

    tag, code = _SEV_TAG.get(s.get("severity") or "medium", ("[MED ]", "33"))
    lines = [
        "",
        c(f"muteval — survivor #{s['id']}  {tag}", "1"),
        f"  operator:    {s['operator']}",
        f"  description: {s['description']}",
        f"  fix:         {c(s.get('fix') or '(none)', '36')}",
        "",
    ]
    base, mut = s.get("baseline_output"), s.get("mutant_output")
    if base is None or mut is None:
        lines.append(c("  (no output diff captured — inert or output unchanged)", "2"))
        return "\n".join(lines)
    lines.append(c("  output diff (baseline → mutant):", "1"))
    diff = difflib.unified_diff(
        base.splitlines(), mut.splitlines(),
        fromfile="baseline", tofile="mutant", lineterm="",
    )
    for ln in diff:
        col = "32" if ln.startswith("+") else "31" if ln.startswith("-") else "2"
        lines.append("  " + c(ln, col))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        dest = Path(args.path)
        if dest.exists() and not args.force:
            print(f"muteval: {dest} already exists (use --force).", file=sys.stderr)
            return 2
        scaffold = _RAG_STARTER_CONFIG if args.template == "rag" else _STARTER_CONFIG
        dest.write_text(scaffold, encoding="utf-8")
        print(
            f"muteval: wrote {dest} ({args.template} template).\n"
            f"  Validate it:  muteval check --config {dest}\n"
            f"  Then run it:  muteval run   --config {dest}"
        )
        return 0

    if args.command == "run":
        try:
            config = _load_run_config(args)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: {exc}", file=sys.stderr)
            return 2

        # CLI overrides for the error budget (config value is the default).
        if args.allow_mutant_errors:
            config.max_error_rate = 1.0
        elif args.max_error_rate is not None:
            if not 0.0 <= args.max_error_rate <= 1.0:
                print("muteval: --max-error-rate must be in [0, 1]", file=sys.stderr)
                return 2
            config.max_error_rate = args.max_error_rate

        if args.dry_run:
            # Use the SAME selection path as a real run so the counts can't drift.
            mutants = select_mutants(
                config, operators=args.operators, sample=args.sample,
                seed=args.seed, max_mutants=args.max_mutants,
            )
            ctx = config.system.context or ()
            print(
                "muteval dry-run OK:\n"
                f"  prompt:  {len(config.system.prompt)} chars\n"
                f"  context: {len(ctx)} doc(s)\n"
                f"  cases:   {len(config.cases)}\n"
                f"  evals:   {', '.join(config.eval_names) or len(config.evals)}\n"
                f"  mutants that would run: {len(mutants)}"
            )
            return 0

        cache = None
        if args.cache:
            from muteval.cache import Cache

            cache = Cache(args.cache)
        result = run_mutation_testing(
            config, operators=args.operators, max_mutants=args.max_mutants,
            sample=args.sample, seed=args.seed, cache=cache,
            concurrency=args.concurrency, max_calls=args.max_calls,
        )
        if cache is not None:
            cache.close()
        print(format_report(result, use_color=not args.no_color))
        _save_last_run(result)  # for `muteval results` / `muteval show <id>`

        if args.manifest:
            from muteval.report import run_manifest

            try:
                Path(args.manifest).write_text(
                    json.dumps(run_manifest(result, config, operators=args.operators,
                                            seed=args.seed), indent=2),
                    encoding="utf-8",
                )
                print(f"muteval: wrote manifest {args.manifest}")
            except OSError as exc:
                print(f"muteval: could not write manifest: {exc}", file=sys.stderr)

        # JSON is always safe to write — it carries "status" and None-aware
        # scores, so it is useful precisely when the run is invalid.
        if args.json:
            from muteval.report import result_to_dict

            Path(args.json).write_text(
                json.dumps(result_to_dict(result), indent=2), encoding="utf-8"
            )

        # Validity gate. An invalid or empty run has NO trustworthy score, so we
        # fail closed (exit 2) BEFORE writing a badge or applying score/severity
        # gates — a green CI must never come from a vacuous run.
        if result.status != VALID:
            if result.status == NO_MUTANTS and args.allow_empty:
                return 0
            reason = {
                BASELINE_ERRORED: "baseline suite ERRORED on the original system",
                BASELINE_FAILED: "baseline suite did not pass on the original system",
                NO_MUTANTS: "no mutants were generated "
                "(use --allow-empty to treat as a pass)",
                NO_EVALUATED_MUTANTS: "every mutant errored — no verdict produced",
                BUDGET_EXCEEDED: "hit --max-calls before finishing "
                "(raise --max-calls or narrow with --sample)",
                PARTIAL_ERRORS: f"{result.errored}/{result.total} mutant(s) errored "
                f"({result.error_rate * 100:.0f}% > allowed "
                f"{config.max_error_rate * 100:.0f}%); raise --max-error-rate "
                "or --allow-mutant-errors to accept",
            }.get(result.status, result.status)
            print(
                f"muteval: INVALID — {reason}. No mutation score produced.",
                file=sys.stderr,
            )
            return 2

        # From here the run is VALID and result.score is a real number.
        if args.badge:
            from muteval.report import badge_dict

            Path(args.badge).write_text(
                json.dumps(badge_dict(result)), encoding="utf-8"
            )

        failed = False
        if args.fail_under is not None and result.score * 100 < args.fail_under:
            print(
                f"\nmuteval: FAIL — score {result.score * 100:.0f}% is below "
                f"--fail-under {args.fail_under:.0f}%",
                file=sys.stderr,
            )
            failed = True

        if args.fail_on_severity is not None:
            from muteval.severity import MEDIUM, severity_rank

            threshold = severity_rank(args.fail_on_severity)
            offending = [
                o for o in result.real_survivors
                if severity_rank(o.severity or MEDIUM) <= threshold
            ]
            if offending:
                print(
                    f"\nmuteval: FAIL — {len(offending)} survivor(s) at or above "
                    f"'{args.fail_on_severity}' severity "
                    f"(--fail-on-severity {args.fail_on_severity}).",
                    file=sys.stderr,
                )
                failed = True

        return 1 if failed else 0

    if args.command == "probe":
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: {exc}", file=sys.stderr)
            return 2
        from muteval.probes import run_probes
        from muteval.report import format_probe_card

        results = run_probes(config, probes=args.probes)
        print(format_probe_card(results, use_color=not args.no_color))
        if args.html:
            from muteval.report import format_probe_card_html

            try:
                Path(args.html).write_text(format_probe_card_html(results), encoding="utf-8")
                print(f"muteval: wrote {args.html}")
            except OSError as exc:
                print(f"muteval: could not write {args.html}: {exc}", file=sys.stderr)
                return 2
        return 0 if all(r.ok for r in results) else 1

    if args.command == "check":
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: {exc}", file=sys.stderr)
            return 2
        from muteval.doctor import all_ok, run_checks

        check_results = run_checks(
            config, operators=args.operators, use_model=not args.no_model, full=args.full
        )
        print(_format_checks(check_results, use_color=not args.no_color))
        return 0 if all_ok(check_results) else 2

    if args.command == "results":
        data = _load_last_run()
        if not data:
            print("muteval: no saved run. Run `muteval run ...` first.", file=sys.stderr)
            return 2
        print(_format_results(data, use_color=not args.no_color))
        return 0

    if args.command == "show":
        data = _load_last_run()
        if not data:
            print("muteval: no saved run. Run `muteval run ...` first.", file=sys.stderr)
            return 2
        match = next(
            (s for s in data.get("survivors", []) if s.get("id") == args.id), None
        )
        if match is None:
            print(
                f"muteval: no survivor with id {args.id} (see `muteval results`).",
                file=sys.stderr,
            )
            return 2
        print(_format_show(match, use_color=not args.no_color))
        return 0

    if args.command == "label":
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: {exc}", file=sys.stderr)
            return 2
        try:
            n = _write_label_worksheet(config, args.out)
        except Exception as exc:  # noqa: BLE001 - surface run/eval failures clearly
            print(f"muteval: could not build worksheet: {exc}", file=sys.stderr)
            return 2
        print(
            f"muteval: wrote {n} rows to {args.out}. Fill the 'human_label' column "
            f"(pass/fail), then run `muteval probe` to see human agreement."
        )
        return 0

    if args.command == "report":
        if args.json:
            try:
                data = json.loads(Path(args.json).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"muteval: could not read {args.json}: {exc}", file=sys.stderr)
                return 2
        else:
            data = _load_last_run()
        if not data:
            print(
                "muteval: no run to report. Run `muteval run ...` first, or pass "
                "--json PATH.",
                file=sys.stderr,
            )
            return 2
        from muteval.report import format_report_html

        try:
            Path(args.html).write_text(format_report_html(data), encoding="utf-8")
        except OSError as exc:
            print(f"muteval: could not write {args.html}: {exc}", file=sys.stderr)
            return 2
        print(f"muteval: wrote {args.html}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
