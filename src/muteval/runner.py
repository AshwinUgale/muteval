"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original prompt.
     (If it doesn't, mutation results are meaningless — you have failing evals
     before we even break anything.)
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a blind spot in your evals).
  3. Mutation score = killed / evaluated mutants. Higher is better.

Resilience: a single eval call raising (API timeout, rate limit, network blip)
must NOT abort the whole run. Such a mutant is recorded as "errored" and
excluded from the score, and the run continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from muteval.config import MutEvalConfig
from muteval.mutators import Mutant, generate_mutants


@dataclass
class MutantOutcome:
    mutant: Mutant
    killed: bool
    failing_eval: Optional[str] = None
    errored: bool = False
    error: Optional[str] = None


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


def _suite_outcome(prompt: str, config: MutEvalConfig) -> Optional[str]:
    """Run the eval suite once over all cases.

    Returns None if the whole suite passes, or the name of the first failing
    eval if anything fails. May raise if config.run or an eval raises — callers
    decide how to handle that.
    """
    for case in config.cases:
        output = config.run(prompt, case)
        for idx, ev in enumerate(config.evals):
            if not ev(output, case):
                return _eval_label(config, idx)
    return None


def _suite_passes(prompt: str, config: MutEvalConfig) -> bool:
    return _suite_outcome(prompt, config) is None


def _eval_label(config: MutEvalConfig, idx: int) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    ev = config.evals[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


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
        baseline_passed = _suite_passes(config.prompt, config)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        baseline_error = f"{type(exc).__name__}: {exc}"

    mutants = generate_mutants(config.prompt, operators=operators)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]

    result = MutationResult(
        baseline_passed=baseline_passed, baseline_error=baseline_error
    )

    for mutant in mutants:
        try:
            failing_eval: Optional[str] = None
            for _ in range(config.runs_per_mutant):
                failing_eval = _suite_outcome(mutant.prompt, config)
                if failing_eval is not None:
                    break
            result.outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    killed=failing_eval is not None,
                    failing_eval=failing_eval,
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
