"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original system, and
     we record the baseline OUTPUTS for each case.
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a potential blind spot in your evals).
  3. For survivors, we diff the mutant's outputs against the baseline outputs:
       - output CHANGED but evals still passed  -> a REAL coverage gap.
       - output UNCHANGED (identical text)       -> an INERT/equivalent mutant;
         no output-based eval could have caught it, so it is NOT counted as an
         eval blind spot. (Standard mutation testing excludes equivalent mutants.)
  4. Mutation score = killed / evaluated. The *effective* score additionally
     drops inert survivors from the denominator — the honest "of the mutants
     that actually degraded the output, how many did the evals catch?"

Resilience: a single eval/model call raising (timeout, rate limit, blip) must
NOT abort the whole run. Such a mutant is recorded as "errored" and excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from muteval.config import MutEvalConfig
from muteval.evals import EvalOutcome, coerce_outcome
from muteval.mutators import Mutant, generate_mutants
from muteval.severity import severity_of
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
    # Did this mutant actually change the system's output vs baseline?
    #   True  -> output changed (a survivor here is a real coverage gap)
    #   False -> output identical (inert / equivalent mutant)
    #   None  -> unknown (e.g. killed before all cases ran, or baseline errored)
    output_changed: Optional[bool] = None
    # Ranked danger of this mutation: 'high' | 'medium' | 'low'.
    severity: Optional[str] = None
    # Fraction of runs in which the suite caught this mutant (judge-noise signal).
    kill_rate: Optional[float] = None


@dataclass
class _SuiteRun:
    """One pass of the eval suite over all cases."""

    failing_eval: Optional[str]  # None if the whole suite passed
    outcomes: List[EvalOutcome]  # all outcomes if passed; up to the failure otherwise
    outputs: List[str]  # the system output for each case run (in order)


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
    def inert_survivors(self) -> List[MutantOutcome]:
        """Survivors whose output was IDENTICAL to baseline — equivalent mutants,
        not eval blind spots (no output-based eval could catch them)."""
        return [o for o in self.survivors if o.output_changed is False]

    @property
    def real_survivors(self) -> List[MutantOutcome]:
        """Survivors that actually changed the output but evals didn't catch —
        genuine coverage gaps. (Includes survivors with unknown diff status.)"""
        return [o for o in self.survivors if o.output_changed is not False]

    @property
    def high_severity_survivors(self) -> List[MutantOutcome]:
        """Real coverage gaps ranked HIGH — the dangerous ones."""
        from muteval.severity import HIGH
        return [o for o in self.real_survivors if o.severity == HIGH]

    @property
    def score_ci(self):
        """Wilson 95% CI on the raw mutation score (killed / evaluated)."""
        from muteval.stats import wilson_interval

        return wilson_interval(self.killed, self.evaluated)

    @property
    def effective_score_ci(self):
        """Wilson 95% CI on the effective score (excludes inert mutants)."""
        from muteval.stats import wilson_interval

        return wilson_interval(self.killed, max(self.evaluated - len(self.inert_survivors), 0))

    @property
    def flaky(self) -> List[MutantOutcome]:
        """Mutants whose verdict flipped between runs (0 < kill_rate < 1)."""
        return [
            o for o in self.outcomes
            if o.kill_rate is not None and 0.0 < o.kill_rate < 1.0
        ]

    @property
    def score(self) -> float:
        """Mutation score in [0, 1] over evaluated mutants. 1.0 if none."""
        if self.evaluated == 0:
            return 1.0
        return self.killed / self.evaluated

    @property
    def effective_score(self) -> float:
        """Mutation score with inert (output-unchanged) survivors removed from
        the denominator — the honest 'of mutants that degraded the output, how
        many did the evals catch?'. Falls back to ``score`` if nothing to drop."""
        effective = self.evaluated - len(self.inert_survivors)
        if effective <= 0:
            return 1.0
        return self.killed / effective


def _eval_label(config: MutEvalConfig, idx: int) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    ev = config.evals[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def _run_suite(system: System, config: MutEvalConfig) -> _SuiteRun:
    """Run the eval suite once over all cases.

    Short-circuits on the first failing eval (cheap — important for paid LLM
    judges). Returns the failing eval's label (or None), the outcomes collected,
    and the system outputs produced (one per case run).
    """
    collected: List[EvalOutcome] = []
    outputs: List[str] = []
    for case in config.cases:
        output = config.invoke(system, case)
        outputs.append(output)
        for idx, ev in enumerate(config.evals):
            label = _eval_label(config, idx)
            outcome = coerce_outcome(ev(output, case), name=label)
            collected.append(outcome)
            if not outcome.passed:
                return _SuiteRun(failing_eval=label, outcomes=collected, outputs=outputs)
    return _SuiteRun(failing_eval=None, outcomes=collected, outputs=outputs)


def _near_miss(outcomes: List[EvalOutcome]) -> tuple[Optional[str], Optional[float]]:
    """Of the passing outcomes that expose a margin, find the closest call."""
    margins = [
        (o.name, o.margin) for o in outcomes if o.margin is not None and o.passed
    ]
    if not margins:
        return None, None
    name, margin = min(margins, key=lambda nm: nm[1])
    return name, margin


def _diff_outputs(baseline: List[str], mutant: List[str]) -> Optional[bool]:
    """True if any case's output changed; False if all identical; None if we
    can't compare (lengths differ / baseline missing)."""
    if not baseline or len(baseline) != len(mutant):
        return None
    return any(a != b for a, b in zip(baseline, mutant))


def run_mutation_testing(
    config: MutEvalConfig,
    operators: List[str] | None = None,
    max_mutants: Optional[int] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
) -> MutationResult:
    """Run mutation testing for the given config and return a MutationResult."""
    if operators is None:
        operators = getattr(config, "operators", None)

    # Baseline — resilient: an error here shouldn't lose the whole run.
    baseline_passed = False
    baseline_error: Optional[str] = None
    baseline_outputs: List[str] = []
    try:
        baseline_run = _run_suite(config.system, config)
        baseline_passed = baseline_run.failing_eval is None
        baseline_outputs = baseline_run.outputs
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        baseline_error = f"{type(exc).__name__}: {exc}"

    mutants = generate_mutants(
        config.system, operators=operators, scope=getattr(config, "scope", None)
    )
    if sample is not None and 0 <= sample < len(mutants):
        import random

        mutants = random.Random(seed).sample(mutants, sample)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]

    result = MutationResult(
        baseline_passed=baseline_passed, baseline_error=baseline_error
    )

    for mutant in mutants:
        try:
            runs = [
                _run_suite(mutant.system, config)
                for _ in range(config.runs_per_mutant)
            ]
            fails = sum(1 for r in runs if r.failing_eval is not None)
            kill_rate = fails / len(runs)
            killed = kill_rate >= config.kill_threshold
            # A representative run consistent with the (majority) verdict.
            rep = next(
                (r for r in runs if (r.failing_eval is not None) == killed), runs[0]
            )
            closest_eval = min_margin = None
            output_changed: Optional[bool] = None
            if not killed:
                closest_eval, min_margin = _near_miss(rep.outcomes)
                output_changed = _diff_outputs(baseline_outputs, rep.outputs)
            result.outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    killed=killed,
                    failing_eval=rep.failing_eval,
                    closest_eval=closest_eval,
                    min_margin=min_margin,
                    output_changed=output_changed,
                    severity=severity_of(mutant),
                    kill_rate=kill_rate,
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
                    severity=severity_of(mutant),
                )
            )

    return result
