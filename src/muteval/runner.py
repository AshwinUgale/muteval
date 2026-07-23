"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original system, and
     we record the baseline OUTPUTS for each case.
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a potential blind spot in your evals).
  3. For survivors, we diff the mutant's outputs against the baseline outputs:
       - output CHANGED but evals still passed  -> a REAL coverage gap.
       - output UNCHANGED (identical text on this run) -> an OBSERVATIONALLY
         UNCHANGED mutant; no output-based eval could have caught it on the
         samples we saw, so it is NOT counted as an eval blind spot. (This is a
         weaker claim than the classic "equivalent mutant": for a stochastic
         system, identical output on a few samples does not PROVE equivalence —
         see docs/LIMITATIONS.md. Raise runs_per_mutant to harden it.)
  4. Mutation score = killed / evaluated. The *effective* score additionally
     drops these observationally-unchanged survivors from the denominator:
     "of the mutants that actually changed the output we observed, how many did
     the evals catch?"

Resilience: a single eval/model call raising (timeout, rate limit, blip) must
NOT abort the whole run. Such a mutant is recorded as "errored" and excluded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, cast

from muteval.config import MutEvalConfig
from muteval.evals import EvalOutcome, coerce_outcome
from muteval.mutators import Mutant, generate_mutants
from muteval.severity import severity_of
from muteval.system import System

# A run is only VALID when the baseline passed AND >= 1 mutant produced a
# clean verdict. Anything else is invalid/empty — NOT a score of any kind.
VALID = "valid"
BASELINE_FAILED = "baseline_failed"
BASELINE_ERRORED = "baseline_errored"
NO_MUTANTS = "no_mutants"
NO_EVALUATED_MUTANTS = "no_evaluated_mutants"
# Some (but not all) mutants errored, above the allowed error budget. The score
# is computed over a shrunken denominator, so it is NOT trustworthy for CI.
PARTIAL_ERRORS = "partial_errors"
# The run hit --max-calls before finishing: incomplete, so no trustworthy score.
BUDGET_EXCEEDED = "budget_exceeded"


class BudgetExceeded(Exception):
    """Raised when a run exceeds its --max-calls budget (fail closed)."""


class _Budget:
    """Thread-safe counter of ACTUAL model/judge calls (cache hits and skipped
    judges don't count). Raises BudgetExceeded when the cap is passed."""

    def __init__(self, max_calls: Optional[int]):
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()

    def charge(self) -> None:
        if self.max_calls is None:
            return
        with self._lock:
            self.calls += 1
            if self.calls > self.max_calls:
                raise BudgetExceeded(
                    f"exceeded --max-calls={self.max_calls} (made {self.calls} calls)"
                )


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
    # For survivors: a sample of the FIRST case whose output changed vs baseline,
    # so `muteval show` can render the baseline-vs-mutant diff.
    baseline_output: Optional[str] = None
    mutant_output: Optional[str] = None


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
    status: str = VALID
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
        """Survivors whose output was IDENTICAL to baseline on the samples we ran
        — observationally unchanged, so not eval blind spots (no output-based
        eval could catch them here). NOTE: for a stochastic system this is not
        proof of true equivalence; raise runs_per_mutant to harden it."""
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
    def error_rate(self) -> float:
        """Fraction of generated mutants that errored (0.0 when none generated)."""
        if self.total == 0:
            return 0.0
        return self.errored / self.total

    @property
    def score(self) -> Optional[float]:
        """Mutation score over evaluated mutants, or None when there is no
        evidence (0 evaluated) — no evidence is NOT a perfect score."""
        if self.evaluated == 0:
            return None
        return self.killed / self.evaluated

    @property
    def effective_score(self) -> Optional[float]:
        """Observed-degradation score: mutants whose OUTPUT changed but the
        evals missed. Excludes single-sample "unchanged" survivors. None when
        there is nothing to score. (Not provably exact for stochastic judges;
        see LIMITATIONS.)"""
        effective = self.evaluated - len(self.inert_survivors)
        if effective <= 0:
            return None
        return self.killed / effective


def _eval_label(config: MutEvalConfig, idx: int) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    ev = config.evals[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def _ordered_evals(config: MutEvalConfig):
    """(orig_idx, ev, label) with CHEAP (rule-based) checks before LLM judges.

    Combined with the short-circuit below, a cheap check that already fails a
    mutant means the expensive judge is never called. Order is stable within a
    cost tier, so reporting stays deterministic.
    """
    items = [(i, ev, _eval_label(config, i)) for i, ev in enumerate(config.evals)]
    return sorted(items, key=lambda t: bool(getattr(t[1], "is_llm", False)))


def _run_suite(system: System, config: MutEvalConfig, cache=None, baseline=None, budget=None) -> _SuiteRun:
    """Run the eval suite once over all cases.

    Three cost savers, all safe for deterministic suites:
    * **cheap-checks-first** — rule-based evals run before LLM judges, so the
      short-circuit skips the judge when a cheap check already fails the mutant.
    * **short-circuit** — stop at the first failing eval (a mutant is already
      killed).
    * **skip-unchanged** — when ``baseline`` is given and a case's output is
      byte-identical to the baseline's, reuse the baseline's (passing) outcomes
      instead of re-running the evals (0 judge calls for inert mutants).
    Plus the optional ``cache`` (memoizes across whole runs).
    """
    collected: List[EvalOutcome] = []
    outputs: List[str] = []
    ordered = _ordered_evals(config)
    base_outputs, base_by_case = baseline if baseline else (None, None)
    for ci, case in enumerate(config.cases):
        output = cache.get_output(system, case) if cache is not None else None
        if output is None:
            if budget is not None:
                budget.charge()  # a real model call
            output = config.invoke(system, case)
            if cache is not None:
                cache.set_output(system, case, output)
        outputs.append(output)
        # Skip-unchanged: identical output => deterministic evals reproduce the
        # baseline's passing outcomes; reuse them, run no evals for this case.
        if (
            base_outputs is not None
            and ci < len(base_outputs)
            and output == base_outputs[ci]
        ):
            collected.extend(base_by_case[ci])
            continue
        for idx, ev, label in ordered:
            outcome = cache.get_outcome(system, case, label) if cache is not None else None
            if outcome is None:
                if budget is not None and getattr(ev, "is_llm", False):
                    budget.charge()  # a real (paid) judge call
                outcome = coerce_outcome(ev(output, case), name=label)
                if cache is not None:
                    cache.set_outcome(system, case, label, outcome)
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


def select_mutants(
    config: MutEvalConfig,
    operators: List[str] | None = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    max_mutants: Optional[int] = None,
) -> List[Mutant]:
    """Generate the mutants a run would execute, applying (in order) operator
    selection, scope, sampling, and the max-mutants cap. Shared by the real
    runner AND ``--dry-run`` so the two can never drift apart.

    ``operators=None`` falls back to ``config.operators`` (then to all operators).
    """
    selected = operators if operators is not None else getattr(config, "operators", None)
    # config.operators is untyped (Any); generate_mutants wants str|Callable ops.
    ops = cast("List[str | Callable] | None", selected)
    mutants = generate_mutants(
        config.system, operators=ops, scope=getattr(config, "scope", None)
    )
    if sample is not None and 0 <= sample < len(mutants):
        import random

        mutants = random.Random(seed).sample(mutants, sample)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]
    return mutants


def _evaluate_mutant(mutant, config, cache, baseline_arg, baseline_outputs, budget=None) -> MutantOutcome:
    """Evaluate a single mutant into a MutantOutcome. Pure w.r.t. the mutant, so
    it is safe to run concurrently across a thread pool."""
    try:
        runs = [
            _run_suite(mutant.system, config, cache=cache, baseline=baseline_arg, budget=budget)
            for _ in range(config.runs_per_mutant)
        ]
        fails = sum(1 for r in runs if r.failing_eval is not None)
        kill_rate = fails / len(runs)
        if config.kill_threshold is None:
            killed = fails * 2 > len(runs)  # strict majority; ties survive
        else:
            killed = kill_rate >= config.kill_threshold
        rep = next(
            (r for r in runs if (r.failing_eval is not None) == killed), runs[0]
        )
        closest_eval = min_margin = None
        output_changed: Optional[bool] = None
        sample_base = sample_mut = None
        if not killed:
            closest_eval, min_margin = _near_miss(rep.outcomes)
            # Capture the first case whose output changed, for `muteval show`.
            for i, mo in enumerate(rep.outputs):
                if i < len(baseline_outputs) and baseline_outputs[i] != mo:
                    sample_base, sample_mut = baseline_outputs[i], mo
                    break
            # Aggregate output-change evidence across EVERY surviving run so
            # raising runs_per_mutant actually hardens equivalence detection: any
            # observed change -> changed; any incomplete/absent comparison ->
            # unknown (None); "unchanged" only when every comparable run matched.
            survivor_runs = [r for r in runs if r.failing_eval is None]
            diffs = [_diff_outputs(baseline_outputs, r.outputs) for r in survivor_runs]
            if any(d is True for d in diffs):
                output_changed = True
            elif not diffs or any(d is None for d in diffs):
                output_changed = None
            else:
                output_changed = False
        return MutantOutcome(
            mutant=mutant, killed=killed, failing_eval=rep.failing_eval,
            closest_eval=closest_eval, min_margin=min_margin,
            output_changed=output_changed, severity=severity_of(mutant),
            kill_rate=kill_rate, baseline_output=sample_base, mutant_output=sample_mut,
        )
    except BudgetExceeded:
        raise  # budget is a hard stop, not a per-mutant error
    except Exception as exc:  # noqa: BLE001
        # A flaky eval call (timeout, rate limit, API error) must not nuke the
        # whole run. Record this mutant as errored and keep going.
        return MutantOutcome(
            mutant=mutant, killed=False, errored=True,
            error=f"{type(exc).__name__}: {exc}", severity=severity_of(mutant),
        )


def run_mutation_testing(
    config: MutEvalConfig,
    operators: List[str] | None = None,
    max_mutants: Optional[int] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    cache=None,
    concurrency: int = 1,
    max_calls: Optional[int] = None,
) -> MutationResult:
    """Run mutation testing for the given config and return a MutationResult.

    ``max_calls`` caps the number of ACTUAL model + judge calls (cache hits and
    skipped judges don't count). Exceeding it fails closed with status
    ``budget_exceeded`` — no trustworthy score.

    ``cache`` (a ``muteval.cache.Cache``) memoizes run outputs + eval outcomes so
    an identical re-run makes zero model/judge calls. It is disabled when
    ``runs_per_mutant > 1`` (those repeats exist to observe non-determinism, which
    a cache would erase).

    ``concurrency`` > 1 evaluates mutants across a thread pool (order preserved),
    cutting wall-clock on API-bound suites.
    """
    # Caching assumes determinism; a noisy (multi-run) suite must not be cached.
    if cache is not None and config.runs_per_mutant > 1:
        cache = None
    budget = _Budget(max_calls)
    # Baseline — resilient: an error here shouldn't lose the whole run.
    baseline_passed = False
    baseline_error: Optional[str] = None
    baseline_outputs: List[str] = []
    # The baseline gets one shot per attempt; a flaky judge (timeout / API error)
    # must not poison the whole run, so retry a few times on *exceptions* only.
    # A clean pass/fail verdict is a real result and is NOT retried.
    for _ in range(max(config.runs_per_mutant, 3)):
        try:
            baseline_run = _run_suite(config.system, config, cache=cache, budget=budget)
            baseline_passed = baseline_run.failing_eval is None
            baseline_outputs = baseline_run.outputs
            baseline_error = None
            break
        except BudgetExceeded as exc:
            # Budget hit during the baseline: incomplete, fail closed.
            result = MutationResult(baseline_passed=False, baseline_error=str(exc))
            result.status = BUDGET_EXCEEDED
            return result
        except Exception as exc:  # noqa: BLE001 - transient judge/API errors
            baseline_error = f"{type(exc).__name__}: {exc}"
            continue

    result = MutationResult(
        baseline_passed=baseline_passed, baseline_error=baseline_error
    )
    # BASELINE GATE: an invalid baseline makes every downstream number
    # meaningless (a failing eval fails on every mutant too, faking 100%).
    if baseline_error is not None:
        result.status = BASELINE_ERRORED
        return result
    if not baseline_passed:
        result.status = BASELINE_FAILED
        return result

    mutants = select_mutants(
        config, operators=operators, sample=sample, seed=seed, max_mutants=max_mutants
    )

    if not mutants:
        result.status = NO_MUTANTS
        return result

    # Skip-unchanged optimization: give each mutant run the baseline's per-case
    # outputs + (passing) outcomes so cases whose output didn't change reuse them
    # and call no judges. Only for deterministic runs (runs_per_mutant == 1); a
    # noisy suite must re-run the judges to observe the noise.
    baseline_arg = None
    if config.runs_per_mutant == 1 and baseline_outputs:
        n_evals = len(config.evals)
        oc = baseline_run.outcomes
        if len(oc) == len(config.cases) * n_evals:  # baseline ran fully (it passed)
            base_by_case = [oc[i * n_evals:(i + 1) * n_evals] for i in range(len(config.cases))]
            baseline_arg = (baseline_outputs, base_by_case)

    def _worker(mutant: Mutant) -> MutantOutcome:
        return _evaluate_mutant(mutant, config, cache, baseline_arg, baseline_outputs, budget)

    concurrency = max(1, int(concurrency or 1))
    try:
        if concurrency > 1 and len(mutants) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                # ex.map preserves input order, so outcomes stay deterministic.
                result.outcomes.extend(ex.map(_worker, mutants))
        else:
            for mutant in mutants:
                result.outcomes.append(_worker(mutant))
    except BudgetExceeded:
        # Hit --max-calls partway: the run is incomplete, so no trustworthy score.
        result.status = BUDGET_EXCEEDED
        return result

    # Validity: no evidence at all is invalid; too many errors is invalid too
    # (a score over a shrunken denominator is not trustworthy — fail closed).
    if result.evaluated == 0:
        result.status = NO_EVALUATED_MUTANTS
    elif result.error_rate > config.max_error_rate:
        result.status = PARTIAL_ERRORS
    return result
