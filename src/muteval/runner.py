"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original system.
     (If it doesn't, mutation results are meaningless — you have failing evals
     before we even break anything.)
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a blind spot in your evals).
  3. Mutation score = killed / evaluated mutants. Higher is better.

For survivors, the runner also records the *near miss* — the eval that came
closest to catching the regression (smallest score-over-threshold margin) — so
the report can flag "this one almost failed."

Resilience: a single eval call raising (API timeout, rate limit, network blip)
must NOT abort the whole run. Such a mutant is recorded as "errored" and
excluded from the score, and the run continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from muteval.config import MutEvalConfig
from muteval.evals import EvalOutcome, coerce_outcome
from muteval.mutators import Mutant, generate_mutants
from muteval.system import System


@dataclass
class MutantOutcome:
    mutant: Mutant
    killed: bool
    failing_eval: Optional[str] = None
    errored: bool = False
    error: Optional[str] = None
    # Near-miss info for survivors: the eval that came closest to catching it.
    closest_eval: Optional[str] = None
    min_margin: Optional[float] = None


@dataclass
class _SuiteRun:
    """One pass of the eval suite over all cases."""

    failing_eval: Optional[str]  # None if the whole suite passed
    outcomes: List[EvalOutcome]  # all outcomes if passed; up to the failure otherwise


@dataclass
class MutationResult:
    baseline_passed: bool
    baseline_error: Optional[str] = None
    outcomes: List[MutantOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def evaluated(self) -> int:
        """Mutants that produced a clean verdict (excludes errored ones)."""
        return sum(1 for o in self.outcomes if not o.errored)

    @property
    def killed(self) -> int:
        return sum(1 for o in self.outcomes if o.killed and not o.errored)

    @property
    def errored(self) -> int:
        return sum(1 for o in self.outcomes if o.errored)

    @property
    def survivors(self) -> List[MutantOutcome]:
        return [o for o in self.outcomes if not o.killed and not o.errored]

    @property
    def score(self) -> float:
        """Mutation score in [0, 1] over evaluated mutants. 1.0 if none."""
        if self.evaluated == 0:
            return 1.0
        return self.killed / self.evaluated


def _eval_label(config: MutEvalConfig, idx: int) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    ev = config.evals[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def _run_suite(system: System, config: MutEvalConfig) -> _SuiteRun:
    """Run the eval suite once over all cases.

    Short-circuits on the first failing eval (cheap — important for paid LLM
    judges). Returns the failing eval's label, or None if everything passed,
    plus the outcomes collected so far. May raise if config.run or an eval
    raises — callers decide how to handle that.
    """
    collected: List[EvalOutcome] = []
    for case in config.cases:
        output = config.invoke(system, case)
        for idx, ev in enumerate(config.evals):
            label = _eval_label(config, idx)
            outcome = coerce_outcome(ev(output, case), name=label)
            collected.append(outcome)
            if not outcome.passed:
                return _SuiteRun(failing_eval=label, outcomes=collected)
    return _SuiteRun(failing_eval=None, outcomes=collected)


def _near_miss(outcomes: List[EvalOutcome]) -> tuple[Optional[str], Optional[float]]:
    """Of the passing outcomes that expose a margin, find the closest call."""
    margins = [
        (o.name, o.margin) for o in outcomes if o.margin is not None and o.passed
    ]
    if not margins:
        return None, None
    name, margin = min(margins, key=lambda nm: nm[1])
    return name, margin


def run_mutation_testing(
    config: MutEvalConfig,
    operators: List[str] | None = None,
    max_mutants: Optional[int] = None,
) -> MutationResult:
    """Run mutation testing for the given config and return a MutationResult."""
    # Baseline — resilient: an error here shouldn't lose the whole run.
    baseline_passed = False
    baseline_error: Optional[str] = None
    try:
        baseline_passed = _run_suite(config.system, config).failing_eval is None
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        baseline_error = f"{type(exc).__name__}: {exc}"

    mutants = generate_mutants(config.system, operators=operators)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]

    result = MutationResult(
        baseline_passed=baseline_passed, baseline_error=baseline_error
    )

    for mutant in mutants:
        try:
            suite_run = _SuiteRun(failing_eval=None, outcomes=[])
            for _ in range(config.runs_per_mutant):
                suite_run = _run_suite(mutant.system, config)
                if suite_run.failing_eval is not None:
                    break
            killed = suite_run.failing_eval is not None
            closest_eval = min_margin = None
            if not killed:
                closest_eval, min_margin = _near_miss(suite_run.outcomes)
            result.outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    killed=killed,
                    failing_eval=suite_run.failing_eval,
                    closest_eval=closest_eval,
                    min_margin=min_margin,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # A flaky eval call (timeout, rate limit, API error) must not nuke
            # the whole run. Record this mutant as errored and keep going.
            result.outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    killed=False,
                    errored=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return result
