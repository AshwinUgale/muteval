"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original prompt.
     (If it doesn't, mutation results are meaningless — you have failing evals
     before we even break anything.)
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a blind spot in your evals).
  3. Mutation score = killed / total mutants. Higher is better.
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
    failing_eval: Optional[str] = None  # which eval caught it (first failure)


@dataclass
class MutationResult:
    baseline_passed: bool
    outcomes: List[MutantOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def killed(self) -> int:
        return sum(1 for o in self.outcomes if o.killed)

    @property
    def survivors(self) -> List[MutantOutcome]:
        return [o for o in self.outcomes if not o.killed]

    @property
    def score(self) -> float:
        """Mutation score in [0, 1]. Returns 1.0 if there were no mutants."""
        if not self.outcomes:
            return 1.0
        return self.killed / self.total


def _suite_outcome(prompt: str, config: MutEvalConfig) -> Optional[str]:
    """Run the eval suite once over all cases.

    Returns None if the whole suite passes, or the name of the first failing
    eval if anything fails.
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
    baseline_passed = _suite_passes(config.prompt, config)

    mutants = generate_mutants(config.prompt, operators=operators)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]

    result = MutationResult(baseline_passed=baseline_passed)

    for mutant in mutants:
        # A mutant is killed if the suite fails on ANY of runs_per_mutant runs.
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

    return result
