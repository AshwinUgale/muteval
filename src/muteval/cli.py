"""Command-line interface: ``muteval run --config path/to/muteval_config.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from muteval import __version__
from muteval.config import load_config
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
    # This stub just echoes so the scaffold runs out of the box.
    return f"Order {case['order_id']}: here is your status."


config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,
    cases=[
        {"input": "where is my order?", "order_id": "A123"},
    ],
    run=run,
    evals=[
        checks.contains_case("order_id"),   # answer must cite the order id
        checks.not_contains("refund"),      # never promise a refund
    ],
)
'''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muteval",
        description="Mutation testing for LLM eval suites — measure whether "
        "your evals would actually catch a regression.",
    )
    parser.add_argument("--version", action="version", version=f"muteval {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run mutation testing against an eval suite.")
    run.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a Python file defining a module-level `config` "
        "(a muteval.MutEvalConfig).",
    )
    run.add_argument(
        "--operators",
        nargs="+",
        choices=list(OPERATORS.keys()),
        default=None,
        help="Subset of mutation operators to use (default: all).",
    )
    run.add_argument(
        "--max-mutants",
        type=int,
        default=None,
        help="Cap the number of mutants (useful for fast/cheap runs).",
    )
    run.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="PCT",
        help="Exit non-zero if the mutation score is below this percent "
        "(e.g. 75). Use this to gate CI.",
    )
    run.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    init = sub.add_parser(
        "init", help="Scaffold a starter muteval_config.py you can edit."
    )
    init.add_argument(
        "--path",
        "-p",
        default="muteval_config.py",
        help="Where to write the scaffold (default: ./muteval_config.py).",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the file if it already exists.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init":
        dest = Path(args.path)
        if dest.exists() and not args.force:
            print(
                f"muteval: {dest} already exists (use --force to overwrite).",
                file=sys.stderr,
            )
            return 2
        dest.write_text(_STARTER_CONFIG, encoding="utf-8")
        print(f"muteval: wrote {dest}. Edit the TODOs, then:\n  muteval run --config {dest}")
        return 0

    if args.command == "run":
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            print(f"muteval: error loading config: {exc}", file=sys.stderr)
            return 2

        result = run_mutation_testing(
            config,
            operators=args.operators,
            max_mutants=args.max_mutants,
        )
        print(format_report(result, use_color=not args.no_color))

        if args.fail_under is not None:
            score_pct = result.score * 100
            if score_pct < args.fail_under:
                print(
                    f"\nmuteval: FAIL — score {score_pct:.0f}% is below "
                    f"--fail-under {args.fail_under:.0f}%",
                    file=sys.stderr,
                )
                return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
