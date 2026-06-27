"""muteval command line.

Two ways to run:

  1) Zero-config (no Python file) — muteval owns the model call:
       muteval run --prompt-file system.txt --cases cases.jsonl \
         --model gpt-4o-mini --check contains:8080 \
         --judge "the answer is grounded in the context" --fail-under 75

  2) Full control — point at a Python config (custom run/pipeline/metrics):
       muteval run --config muteval_config.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from muteval import __version__
from muteval.config import MutEvalConfig, load_config
from muteval.mutators import OPERATORS
from muteval.report import format_report
from muteval.runner import run_mutation_testing

_STARTER_CONFIG = '''"""muteval config scaffold — edit the TODOs, then run:

    muteval run --config muteval_config.py
"""

from muteval import MutEvalConfig
from muteval import checks


SYSTEM_PROMPT = """You are a helpful support assistant.
- You must always cite the order ID in your answer.
- Do not promise refunds.
"""


def run(prompt: str, case: dict) -> str:
    # TODO: call your real LLM/app with `prompt` and return its text output.
    return f"Order {case['order_id']}: here is your status."


config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,
    cases=[{"input": "where is my order?", "order_id": "A123"}],
    run=run,
    evals=[
        checks.contains_case("order_id"),
        checks.not_contains("refund"),
    ],
)
'''


# --- zero-config helpers -----------------------------------------------------


def _load_cases(path: str) -> List[Any]:
    """Load cases from a .jsonl (one JSON object per line) or .json (a list)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    # Try a single JSON document first (a list, or one object).
    try:
        doc = json.loads(text)
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            return [doc]
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL.
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
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
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

    return MutEvalConfig(
        prompt=prompt,
        cases=cases,
        run=openai_run(model=args.model),
        evals=evals,
        eval_names=names,
        runs_per_mutant=args.runs_per_mutant,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muteval",
        description="Mutation testing for LLM eval suites — measure whether "
        "your evals would actually catch a regression.",
    )
    parser.add_argument("--version", action="version", version=f"muteval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run mutation testing against an eval suite.")
    # Full-control mode:
    run.add_argument(
        "--config", "-c",
        help="Path to a Python file defining a module-level `config` "
        "(a muteval.MutEvalConfig). Use this for custom run()/pipelines/metrics.",
    )
    # Zero-config mode:
    run.add_argument("--prompt", help="The system prompt to mutate (inline).")
    run.add_argument("--prompt-file", help="File containing the system prompt to mutate.")
    run.add_argument("--cases", help="Cases file (.jsonl or .json list).")
    run.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini).")
    run.add_argument(
        "--check", action="append",
        help="A built-in check, repeatable. e.g. contains:8080, not_contains:refund, "
        "contains_case:order_id, regex:ORD-\\d+, is_json, equals, judge:<rubric>.",
    )
    run.add_argument(
        "--judge", action="append",
        help="LLM-as-judge rubric, repeatable. Sugar for --check judge:<rubric>.",
    )
    run.add_argument("--threshold", type=float, default=0.7, help="Judge pass threshold (default 0.7).")
    run.add_argument("--runs-per-mutant", type=int, default=1, help="Runs per mutant (default 1).")
    # Shared:
    run.add_argument(
        "--operators", nargs="+", choices=list(OPERATORS.keys()), default=None,
        help="Subset of mutation operators to use (default: all).",
    )
    run.add_argument("--max-mutants", type=int, default=None, help="Cap the number of mutants.")
    run.add_argument(
        "--fail-under", type=float, default=None, metavar="PCT",
        help="Exit non-zero if the mutation score is below this percent (gate CI).",
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate the run (prompt, cases, evals) WITHOUT calling "
        "the model. Cheap way to check your setup.",
    )
    run.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    init = sub.add_parser("init", help="Scaffold a starter muteval_config.py you can edit.")
    init.add_argument("--path", "-p", default="muteval_config.py", help="Where to write the scaffold.")
    init.add_argument("--force", action="store_true", help="Overwrite if it exists.")
    return parser


def _load_run_config(args: argparse.Namespace) -> MutEvalConfig:
    if args.config:
        return load_config(args.config)
    if args.prompt or args.prompt_file:
        if not args.cases:
            raise ValueError("--cases is required in zero-config mode")
        return _config_from_flags(args)
    raise ValueError(
        "nothing to run: pass --config FILE, or zero-config flags "
        "(--prompt/--prompt-file + --cases + --check/--judge)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init":
        dest = Path(args.path)
        if dest.exists() and not args.force:
            print(f"muteval: {dest} already exists (use --force).", file=sys.stderr)
            return 2
        dest.write_text(_STARTER_CONFIG, encoding="utf-8")
        print(f"muteval: wrote {dest}. Edit it, then: muteval run --config {dest}")
        return 0

    if args.command == "run":
        try:
            config = _load_run_config(args)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: {exc}", file=sys.stderr)
            return 2

        if args.dry_run:
            n_mut = len(
                __import__("muteval.mutators", fromlist=["generate_mutants"])
                .generate_mutants(config.system, operators=args.operators)
            )
            print(
                f"muteval dry-run OK:\n"
                f"  prompt: {len(config.system.prompt)} chars\n"
                f"  cases:  {len(config.cases)}\n"
                f"  evals:  {', '.join(config.eval_names) or len(config.evals)}\n"
                f"  mutants that would run: {n_mut}"
                + (f" (capped to {args.max_mutants})" if args.max_mutants else "")
            )
            return 0

        result = run_mutation_testing(
            config, operators=args.operators, max_mutants=args.max_mutants
        )
        print(format_report(result, use_color=not args.no_color))

        if args.fail_under is not None and result.score * 100 < args.fail_under:
            print(
                f"\nmuteval: FAIL — score {result.score * 100:.0f}% is below "
                f"--fail-under {args.fail_under:.0f}%",
                file=sys.stderr,
            )
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
