"""`muteval check` — validate a config's wiring cheaply, before a full run.

Runs layered checks in order (cheapest first). Structural checks cost 0 model
calls; the model-calling checks run on a single case by default, so a wiring or
compatibility bug costs ~1 call, not a whole run. It also surfaces **per-eval
baseline diagnostics** — the score/verdict of each eval on the ORIGINAL system —
so a red baseline shows *which* eval failed and why, instead of an opaque
"baseline failed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from muteval.config import MutEvalConfig
from muteval.evals import coerce_outcome
from muteval.runner import select_mutants


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False  # a failed fatal check stops the remaining (dependent) checks


def _has_fatal_failure(results: List[CheckResult]) -> bool:
    return any(r.fatal and not r.ok for r in results)


def all_ok(results: List[CheckResult]) -> bool:
    return all(r.ok for r in results)


def run_checks(
    config: MutEvalConfig,
    operators: Optional[List[str]] = None,
    use_model: bool = True,
    full: bool = False,
) -> List[CheckResult]:
    """Validate a config layer by layer and return one CheckResult per layer.

    - ``use_model=False`` runs only the 0-call structural checks.
    - ``full=False`` (default) exercises run()/evals on ONE case (cheap); ``full``
      runs every case (a true baseline over the whole suite).
    """
    results: List[CheckResult] = []

    # --- structural checks (0 model calls) ---------------------------------
    results.append(CheckResult("config loaded", True, "config object is valid"))

    n_cases = len(config.cases or [])
    results.append(
        CheckResult("cases present", n_cases > 0, f"{n_cases} case(s)", fatal=True)
    )
    n_evals = len(config.evals or [])
    results.append(
        CheckResult("evals present", n_evals > 0, f"{n_evals} eval(s)", fatal=True)
    )

    try:
        mutants = select_mutants(config, operators=operators)
        ok = len(mutants) > 0
        results.append(
            CheckResult(
                "mutants generate",
                ok,
                f"{len(mutants)} mutant(s) would run"
                if ok
                else "no mutants — prompt too short, or operators/scope filtered them all out",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult("mutants generate", False, f"{type(exc).__name__}: {exc}", fatal=True)
        )

    if _has_fatal_failure(results) or not use_model:
        return results

    # --- model checks (1 call by default) ----------------------------------
    cases = list(config.cases) if full else list(config.cases)[:1]

    try:
        first_output = config.invoke(config.system, cases[0])
        ok = isinstance(first_output, str)
        results.append(
            CheckResult(
                "run() returns text",
                ok,
                f"got {type(first_output).__name__}, {len(first_output)} chars"
                if ok
                else f"run() returned {type(first_output).__name__}, expected str",
                fatal=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult("run() returns text", False, f"{type(exc).__name__}: {exc}", fatal=True)
        )

    if _has_fatal_failure(results):
        return results

    # --- per-eval baseline diagnostics -------------------------------------
    baseline_ok = True
    for i, case in enumerate(cases):
        try:
            output = first_output if i == 0 else config.invoke(config.system, case)
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(f"run() on case[{i}]", False, f"{type(exc).__name__}: {exc}"))
            baseline_ok = False
            continue
        for j, ev in enumerate(config.evals):
            label = (
                config.eval_names[j]
                if j < len(config.eval_names)
                else getattr(ev, "__name__", f"eval[{j}]")
            )
            try:
                outcome = coerce_outcome(ev(output, case), name=label)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CheckResult(
                        f"eval '{label}' on case[{i}]",
                        False,
                        f"raised {type(exc).__name__}: {exc} (wiring/parse bug)",
                    )
                )
                baseline_ok = False
                continue
            score = f", score={outcome.score:.2f}" if outcome.score is not None else ""
            results.append(
                CheckResult(
                    f"eval '{label}' on case[{i}]",
                    outcome.passed,
                    ("passed" if outcome.passed else "FAILED on the ORIGINAL system") + score,
                )
            )
            if not outcome.passed:
                baseline_ok = False

    scope = "all cases" if full else "the first case"
    results.append(
        CheckResult(
            "baseline passes on original system",
            baseline_ok,
            f"green over {scope} — ready to run"
            if baseline_ok
            else "RED — evals don't pass on the unmutated system; muteval will refuse to score. "
            "Fix the failing eval(s) above (format mismatch? noisy judge? wrong threshold?).",
        )
    )
    return results
