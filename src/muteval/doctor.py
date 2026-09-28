"""`muteval check` — validate a config's wiring cheaply, before a full run.

Runs layered checks in order (cheapest first). Structural checks cost 0 model
calls; the model-calling checks run on a single case by default, so a wiring or
compatibility bug costs ~2 calls (the second is a repeat that checks whether the
system and its LLM judges are deterministic), not a whole run. It also surfaces **per-eval
baseline diagnostics** — the score/verdict of each eval on the ORIGINAL system —
so a red baseline shows *which* eval failed and why, instead of an opaque
"baseline failed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from muteval.config import MutEvalConfig
from muteval.evals import coerce_outcome
from muteval.mutators import REGRESSION
from muteval.runner import select_mutants


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False  # a failed fatal check stops the remaining (dependent) checks
    # A failed WARN check is shown but doesn't make the config "not ready": the
    # run is valid, but its numbers need care (e.g. a nondeterministic system).
    warn: bool = False


def _has_fatal_failure(results: List[CheckResult]) -> bool:
    return any(r.fatal and not r.ok for r in results)


def all_ok(results: List[CheckResult]) -> bool:
    return all(r.ok or r.warn for r in results)


def _noise_checks(config: MutEvalConfig, case, first_output) -> List[CheckResult]:
    """Is the system deterministic on one case, and is each LLM judge stable on
    one identical output? Non-fatal: reported as WARN with what to set."""
    out: List[CheckResult] = []
    try:
        second = config.invoke(config.system, case)
    except Exception as exc:  # noqa: BLE001 - a flaky call is itself a finding
        return [
            CheckResult(
                "run() is repeatable",
                False,
                f"a second call raised {type(exc).__name__}: {exc}",
                warn=True,
            )
        ]
    key = getattr(config, "output_key", None)
    try:
        same = (key(first_output) == key(second)) if key else first_output == second
    except Exception:  # noqa: BLE001
        same = first_output == second
    if same:
        out.append(
            CheckResult(
                "run() is repeatable", True, "two calls on case[0] gave the same output"
            )
        )
    elif config.runs_per_mutant > 1 or config.baseline_runs > 1 or key:
        out.append(
            CheckResult(
                "run() is repeatable",
                True,
                "output varies between calls — handled by your runs_per_mutant / "
                "baseline_runs / output_key settings",
            )
        )
    else:
        out.append(
            CheckResult(
                "run() is repeatable",
                False,
                "two calls on case[0] gave DIFFERENT outputs: with one sample per "
                "mutant, wording drift looks like a behavior change and noise looks "
                "like a kill. Set output_key= (the part that IS the behavior) and "
                "baseline_runs=3, or an odd runs_per_mutant=3",
                warn=True,
            )
        )

    for j, ev in enumerate(config.evals):
        if not getattr(ev, "is_llm", False):
            continue  # rule-based checks are deterministic on identical input
        label = config.eval_names[j] if j < len(config.eval_names) else f"eval[{j}]"
        try:
            a = coerce_outcome(ev(first_output, case)).passed
            b = coerce_outcome(ev(first_output, case)).passed
        except Exception:  # noqa: BLE001 - the per-eval diagnostics report it
            continue
        if a != b and config.runs_per_mutant == 1:
            out.append(
                CheckResult(
                    f"judge '{label}' is stable",
                    False,
                    "gave different verdicts on the SAME output — at runs_per_mutant=1 "
                    "its noise is counted as kills. Use an odd runs_per_mutant (3), or a "
                    "steadier judge/rubric",
                    warn=True,
                )
            )
        else:
            out.append(
                CheckResult(
                    f"judge '{label}' is stable",
                    True,
                    "same verdict twice on the same output"
                    if a == b
                    else "verdicts vary — handled by your runs_per_mutant",
                )
            )
    return out


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
        scored = sum(1 for m in mutants if m.intent == REGRESSION)
        robust = len(mutants) - scored
        if scored:
            detail = f"{len(mutants)} mutant(s) would run"
            if robust:
                detail += (
                    f" ({scored} scored, {robust} robustness — reported, not scored)"
                )
        elif robust:
            # Only meaning-preserving operators: the run can't produce a score.
            detail = (
                f"only {robust} robustness mutant(s) (paraphrase/reorder) — reported, "
                "never scored, so there'd be no mutation score; add regression operators"
            )
        else:
            detail = (
                "no mutants — prompt too short, or operators/scope filtered them all out"
            )
        results.append(CheckResult("mutants generate", scored > 0, detail))
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(
                "mutants generate", False, f"{type(exc).__name__}: {exc}", fatal=True
            )
        )

    if _has_fatal_failure(results) or not use_model:
        return results

    # --- model checks (1 call by default) ----------------------------------
    cases = list(config.cases) if full else list(config.cases)[:1]

    try:
        first_output = config.invoke(config.system, cases[0])
        # Text, or structured output your evals read (the {"final", "trace"}
        # agent bridge that checks.on_final / checks.tracelint consume).
        ok = isinstance(first_output, (str, dict, list))
        if isinstance(first_output, str):
            detail = f"got str, {len(first_output)} chars"
        elif ok:
            detail = f"got {type(first_output).__name__} (fine if your evals read it)"
        else:
            detail = f"run() returned {type(first_output).__name__}, expected str/dict"
        results.append(CheckResult("run() returns output", ok, detail, fatal=True))
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(
                "run() returns output", False, f"{type(exc).__name__}: {exc}", fatal=True
            )
        )

    if _has_fatal_failure(results):
        return results

    # --- noise check (1 extra model call + 1 extra call per LLM judge) --------
    # With one sample per mutant, muteval can't see noise: a flaky judge's kill
    # looks like detection, and free-text wording drift looks like a behavior
    # change. Catch it here, before a paid run, as a WARNING (the run is still
    # valid; its numbers need the settings named below).
    results.extend(_noise_checks(config, cases[0], first_output))

    # --- per-eval baseline diagnostics -------------------------------------
    baseline_ok = True
    for i, case in enumerate(cases):
        try:
            output = first_output if i == 0 else config.invoke(config.system, case)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(f"run() on case[{i}]", False, f"{type(exc).__name__}: {exc}")
            )
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
                    ("passed" if outcome.passed else "FAILED on the ORIGINAL system")
                    + score,
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
