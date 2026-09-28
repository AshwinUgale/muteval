"""The mutation-testing engine.

Flow:
  1. Establish a baseline: the eval suite must PASS on the original system, and
     we record the baseline OUTPUTS for each case.
  2. For each mutant, run the eval suite. The mutant is "killed" if the suite
     FAILS (your evals detected the degradation) and "survives" if it still
     PASSES (a potential blind spot in your evals).
  3. For survivors, we diff the mutant's outputs against the baseline outputs
     (under ``config.output_key`` when set — the part of the output that IS the
     behavior — and against every baseline sample when ``baseline_runs`` > 1):
       - output CHANGED but evals still passed  -> a REAL coverage gap.
       - output UNCHANGED (matches a baseline sample) -> an OBSERVATIONALLY
         UNCHANGED mutant; no output-based eval could have caught it on the
         samples we saw, so it is NOT counted as an eval blind spot. (This is a
         weaker claim than the classic "equivalent mutant": for a stochastic
         system, identical output on a few samples does not PROVE equivalence —
         see docs/LIMITATIONS.md.)
       - UNDETERMINED (unseen output on a case where the baseline itself
         varied) -> still counted as a gap (conservative), flagged as such.
  4. Mutation score = killed / resolved, over REGRESSION mutants only.
     Meaning-preserving ("robustness") mutants — paraphrase, reorder — are
     never scored; the report lists the ones that flipped a verdict. The
     *effective* score additionally drops observationally-unchanged survivors
     from the denominator: "of the mutants that actually changed the behavior we
     observed, how many did the evals catch?"

Resilience: a single eval/model call raising (timeout, rate limit, blip) must
NOT abort the whole run. Such a mutant is recorded as "errored" and excluded.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple, cast

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
# Mutants produced verdicts, but every one TIED over runs_per_mutant: nothing
# resolved, so there is no score (not a vacuous one).
NO_CONFIDENT_SCORE = "no_confident_score"
# More tied (unresolved) mutants than config.max_unresolved_rate allows: the
# score is over a shrunken denominator (1 resolved of 17 is not "100%").
PARTIAL_UNRESOLVED = "partial_unresolved"
# Only meaning-preserving (robustness) operators ran: reported, never scored.
NO_SCORED_MUTANTS = "no_scored_mutants"


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
    # A tied verdict over runs_per_mutant (judge straddled 50%): neither killed
    # nor a confident survivor. Excluded from the score's numerator AND
    # denominator, and reported as its own rate. Only ever True under strict
    # majority (kill_threshold is None) with an even runs_per_mutant.
    unresolved: bool = False
    failing_eval: Optional[str] = None
    errored: bool = False
    error: Optional[str] = None
    # Near-miss info for survivors: the eval that came closest to catching it.
    closest_eval: Optional[str] = None
    min_margin: Optional[float] = None
    # Did this mutant actually change the system's output vs baseline (every
    # baseline sample, under config.output_key)? Judged on the runs that agree
    # with the verdict.
    #   True  -> changed (a survivor here is a real coverage gap)
    #   False -> matches a baseline sample: an inert survivor, or a NOISE kill
    #   None  -> undetermined (noisy baseline case, unkeyable output, ...)
    output_changed: Optional[bool] = None
    # Ranked danger of this mutation: 'high' | 'medium' | 'low'.
    severity: Optional[str] = None
    # Fraction of runs in which the suite caught this mutant (judge-noise signal).
    kill_rate: Optional[float] = None
    # Distinct evals that caught this mutant across its runs. For a FLAKY mutant
    # (0 < kill_rate < 1), these are the ambiguous rubric dimensions — the eval
    # question is unstable, not just the judge.
    caught_by: Tuple[str, ...] = ()
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
    # The case as run() left it, per case (a run() may write side-channel data
    # such as case["used_context"] for an eval to read). Skip-unchanged only
    # reuses a baseline verdict when BOTH the output and this state match.
    states: List[str] = field(default_factory=list)


@dataclass
class _Fingerprints:
    """Fingerprints of ``config.run`` and each eval (by index), computed ONCE
    per run before anything executes, so cache keys don't drift as stateful
    evals mutate themselves."""

    run: str
    evals: Dict[int, str]


def _fingerprints(config: MutEvalConfig) -> "_Fingerprints":
    from muteval.fingerprint import fingerprint

    return _Fingerprints(
        run=fingerprint(config.run),
        evals={i: fingerprint(ev) for i, ev in enumerate(config.evals)},
    )


def _isolate(case: Any) -> Any:
    """A private copy of ``case`` for one (system, case, run) cell, shared by
    run() and the evals of that cell only. Without it a run() that writes into
    the case leaks that write to other mutants — and, under --concurrency, to
    other threads mid-evaluation. Uncopyable cases are used as-is."""
    try:
        return copy.deepcopy(case)
    except Exception:  # noqa: BLE001 - a case holding a client/lock/socket
        return case


@dataclass
class MutationResult:
    baseline_passed: bool
    baseline_error: Optional[str] = None
    status: str = VALID
    outcomes: List[MutantOutcome] = field(default_factory=list)
    # Positive control: did the suite REJECT an obviously-bad output (blank /
    # nonsense)? True = the suite can distinguish something (harness is scoring).
    # False = it passed the baseline AND garbage, so its verdicts may be vacuous
    # (or it's a guardrail-only suite, where that's expected). None = not checked
    # (only LLM judges, which we don't call for the control, or it errored).
    canary_caught: Optional[bool] = None
    # Provenance so scores are comparable across time: the model under test (when
    # known) and the judge model(s) muteval could introspect (its own llm_judge /
    # grounded). A model bump silently replaces the "coin" behind the score.
    model_under_test: Optional[str] = None
    judge_models: Tuple[str, ...] = ()
    # Signatures the user has ACCEPTED (a survivor they've ruled "untested by
    # design"). Accepted survivors are split out of the actionable set and don't
    # trip --fail-on-severity, so a decided gap stops resurfacing as noise. The
    # mutation score is unchanged — the eval still doesn't cover it.
    accepted: FrozenSet[str] = frozenset()
    # With more than one baseline sample (runs_per_mutant or baseline_runs > 1):
    # how many cases' baseline output (under config.output_key) varied between
    # samples. On those cases an unseen mutant output can't be told apart from
    # sampling noise. None = not measured (a single sample).
    noisy_cases: Optional[int] = None
    # Fraction of the graded baseline runs (runs_per_mutant of them) the ORIGINAL
    # system passed. The original must survive the same kill rule as a mutant,
    # or its own noise would be counted as kills.
    baseline_pass_rate: Optional[float] = None
    # Cache provenance: how many output/outcome lookups were SERVED from --cache
    # (None = no cache in use), and why a requested cache was disabled.
    cache_hits: Optional[int] = None
    cache_note: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def _scored(self) -> List[MutantOutcome]:
        """Outcomes that count toward the score: REGRESSION mutants only.
        Meaning-preserving (robustness) mutants are reported, never scored."""
        from muteval.mutators import REGRESSION

        return [o for o in self.outcomes if o.mutant.intent == REGRESSION]

    @property
    def regression_total(self) -> int:
        """Regression (scored) mutants generated, errored or not."""
        return len(self._scored)

    @property
    def robustness(self) -> List[MutantOutcome]:
        """Meaning-preserving mutants (paraphrase, reorder, ...) with a clean
        verdict. Surviving is the healthy outcome for these."""
        from muteval.mutators import ROBUSTNESS

        return [
            o for o in self.outcomes if o.mutant.intent == ROBUSTNESS and not o.errored
        ]

    @property
    def brittle(self) -> List[MutantOutcome]:
        """Robustness mutants that flipped an eval verdict: a meaning-preserving
        edit was 'caught', so an eval keys on wording/order (or the system is
        sensitive to it). Not a coverage gap and not scored."""
        return [o for o in self.robustness if o.killed and not o.unresolved]

    @property
    def undetermined_survivors(self) -> List[MutantOutcome]:
        """Real survivors whose output change couldn't be established (e.g. the
        baseline itself varied on those cases). Still counted as gaps, the
        conservative choice, but the report says why they're uncertain."""
        return [o for o in self.real_survivors if o.output_changed is None]

    @property
    def flaky_by_eval(self) -> Dict[str, int]:
        """For flaky mutants, how many flipped on each eval — the ambiguous rubric
        dimensions. A dimension with many flips is a bug in the eval *question*;
        rewrite it before adding runs."""
        counts: Dict[str, int] = {}
        for o in self.flaky:
            for label in o.caught_by:
                counts[label] = counts.get(label, 0) + 1
        return counts

    @property
    def evaluated(self) -> int:
        """Regression mutants that produced a clean (non-errored) verdict —
        includes unresolved ties. The SCORE denominator is ``resolved``."""
        return sum(1 for o in self._scored if not o.errored)

    @property
    def unresolved(self) -> int:
        """Mutants whose verdict tied over ``runs_per_mutant`` (judge straddled
        50%) — excluded from the score's numerator and denominator."""
        return sum(1 for o in self._scored if o.unresolved and not o.errored)

    @property
    def unresolved_rate(self) -> float:
        """Fraction of evaluated (regression) mutants whose verdict tied."""
        return self.unresolved / self.evaluated if self.evaluated else 0.0

    @property
    def resolved(self) -> int:
        """Mutants with a confident killed/survived verdict — the score
        denominator (evaluated minus unresolved ties)."""
        return sum(1 for o in self._scored if not o.errored and not o.unresolved)

    @property
    def killed(self) -> int:
        return sum(1 for o in self._scored if o.killed and not o.errored)

    @property
    def errored(self) -> int:
        """Errored mutants of ANY intent (an error is an error: it counts
        toward the error budget either way)."""
        return sum(1 for o in self.outcomes if o.errored)

    @property
    def survivors(self) -> List[MutantOutcome]:
        """Confident survivors — evals missed a mutant that got a resolved
        verdict. Excludes unresolved ties (not a confident coverage gap) and
        robustness mutants (surviving a meaning-preserving edit is healthy)."""
        return [
            o for o in self._scored if not o.killed and not o.errored and not o.unresolved
        ]

    @property
    def inert_survivors(self) -> List[MutantOutcome]:
        """Survivors whose output was IDENTICAL to baseline on the samples we ran
        — observationally unchanged, so not eval blind spots (no output-based
        eval could catch them here). NOTE: for a stochastic system this is not
        proof of true equivalence; raise runs_per_mutant to harden it."""
        return [o for o in self.survivors if o.output_changed is False]

    @property
    def noise_kills(self) -> List[MutantOutcome]:
        """Kills whose outputs matched what the ORIGINAL system itself produced
        (a baseline sample, under ``config.output_key``): the mutant didn't
        change behavior, so the eval failing is noise (a flaky judge, or the
        original's own failure mode), not a caught regression. The mirror image
        of ``inert_survivors``; both leave the effective score."""
        return [
            o
            for o in self._scored
            if o.killed and not o.errored and o.output_changed is False
        ]

    @property
    def real_survivors(self) -> List[MutantOutcome]:
        """Survivors that actually changed the output but evals didn't catch —
        genuine coverage gaps. (Includes survivors with unknown diff status.)"""
        return [o for o in self.survivors if o.output_changed is not False]

    @property
    def accepted_survivors(self) -> List[MutantOutcome]:
        """Real survivors the user marked accepted (untested by design)."""
        return [o for o in self.real_survivors if o.mutant.signature in self.accepted]

    @property
    def new_survivors(self) -> List[MutantOutcome]:
        """Actionable coverage gaps — real survivors NOT accepted by the user."""
        return [o for o in self.real_survivors if o.mutant.signature not in self.accepted]

    @property
    def high_severity_survivors(self) -> List[MutantOutcome]:
        """NEW (unaccepted) coverage gaps ranked HIGH — what a gate blocks on."""
        from muteval.severity import HIGH

        return [o for o in self.new_survivors if o.severity == HIGH]

    @property
    def score_ci(self):
        """Wilson 95% CI on the raw mutation score (killed / resolved)."""
        from muteval.stats import wilson_interval

        return wilson_interval(self.killed, self.resolved)

    @property
    def effective_counts(self) -> Tuple[int, int]:
        """(killed, denominator) for the effective score: drop inert survivors
        AND noise kills — mutants that didn't change observed behavior, whether
        the evals 'caught' them or not."""
        noise = len(self.noise_kills)
        return (
            self.killed - noise,
            self.resolved - len(self.inert_survivors) - noise,
        )

    @property
    def effective_score_ci(self):
        """Wilson 95% CI on the effective score (excludes inert survivors, noise
        kills and unresolved ties)."""
        from muteval.stats import wilson_interval

        k, n = self.effective_counts
        return wilson_interval(k, max(n, 0))

    @property
    def flaky(self) -> List[MutantOutcome]:
        """Mutants whose verdict flipped between runs (0 < kill_rate < 1)."""
        return [
            o
            for o in self.outcomes
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
        """Mutation score over RESOLVED mutants (killed / resolved), or None when
        there is no evidence (0 resolved, e.g. every verdict tied) — no evidence
        is NOT a perfect score."""
        if self.resolved == 0:
            return None
        return self.killed / self.resolved

    @property
    def effective_score(self) -> Optional[float]:
        """Observed-degradation score: of the mutants that CHANGED observed
        behavior, the fraction the evals caught. Excludes unchanged survivors
        (inert), unchanged kills (noise) and unresolved ties. None when there is
        nothing to score. (Not provably exact for stochastic systems; see
        LIMITATIONS.)"""
        k, n = self.effective_counts
        if n <= 0:
            return None
        return k / n


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


def _run_suite(
    system: System,
    config: MutEvalConfig,
    cache=None,
    baseline=None,
    budget=None,
    fps: "Optional[_Fingerprints]" = None,
) -> _SuiteRun:
    """Run the eval suite once over all cases.

    Three cost savers, all safe for deterministic suites:
    * **cheap-checks-first** — rule-based evals run before LLM judges, so the
      short-circuit skips the judge when a cheap check already fails the mutant.
    * **short-circuit** — stop at the first failing eval (a mutant is already
      killed).
    * **skip-unchanged** — when ``baseline`` is given and a case's output AND
      the case state run() left behind are identical to the baseline's, reuse
      the baseline's (passing) outcomes instead of re-running the evals (0 judge
      calls for inert mutants).
    Plus the optional ``cache`` (memoizes across whole runs; needs ``fps``).
    """
    from muteval.cache import _case_repr

    collected: List[EvalOutcome] = []
    outputs: List[str] = []
    states: List[str] = []
    ordered = _ordered_evals(config)
    base_outputs, base_by_case, base_states = baseline if baseline else (None, None, None)
    use_cache = cache is not None and fps is not None
    for ci, case in enumerate(config.cases):
        before = _case_repr(case)
        c = _isolate(case)
        hit = cache.lookup_output(system, case, fps.run) if use_cache else None
        if hit is not None:
            output, post = hit
            if post is not None:  # replay run()'s writes into the case
                c = post
        else:
            if budget is not None:
                budget.charge()  # a real model call
            output = config.invoke(system, c)
            if use_cache:
                after_run = _case_repr(c)
                cache.store_output(
                    system,
                    case,
                    output,
                    fps.run,
                    post_run_case=c if after_run != before else None,
                )
        outputs.append(output)
        state = _case_repr(c)
        states.append(state)
        # Skip-unchanged: identical output AND identical case state => the
        # evals see exactly what they saw on the baseline, so a deterministic
        # suite reproduces its passing outcomes; reuse them, run no evals.
        if (
            base_outputs is not None
            and ci < len(base_outputs)
            and output == base_outputs[ci]
            and state == base_states[ci]
        ):
            collected.extend(base_by_case[ci])
            continue
        for idx, ev, label in ordered:
            outcome = (
                cache.get_outcome(output, c, fps.evals[idx], label) if use_cache else None
            )
            if outcome is None:
                if budget is not None and getattr(ev, "is_llm", False):
                    budget.charge()  # a real (paid) judge call
                outcome = coerce_outcome(ev(output, c), name=label)
                if use_cache:
                    cache.set_outcome(output, c, fps.evals[idx], outcome)
            collected.append(outcome)
            if not outcome.passed:
                return _SuiteRun(
                    failing_eval=label, outcomes=collected, outputs=outputs, states=states
                )
    return _SuiteRun(
        failing_eval=None, outcomes=collected, outputs=outputs, states=states
    )


# Obviously-bad outputs a discriminating suite should reject: a blank and a short
# nonsense string. Used as a positive control (see MutationResult.canary_caught).
_CANARY_OUTPUTS = ("", "lorem ipsum dolor sit amet consectetur")


def _canary_caught(config: MutEvalConfig) -> Optional[bool]:
    """Positive control for the harness/suite: feed the RULE-BASED evals a blank
    and a nonsense output (no model or judge call — free) and report whether the
    suite rejects any of them.

    Returns True if at least one eval fails on at least one canary — the suite
    can tell *something* apart, so a 0% mutation score reflects the mutations, not
    a harness that isn't scoring. False means it passed the baseline AND pure
    garbage (a vacuous suite — or a guardrail-only one, where this is expected).
    None when there are no rule-based evals to check cheaply, or it errored.
    """
    cheap = [
        (ev, label)
        for _idx, ev, label in _ordered_evals(config)
        if not getattr(ev, "is_llm", False)
    ]
    if not cheap:
        return None  # only LLM judges — don't spend API calls on the control
    try:
        for case in config.cases:
            for output in _CANARY_OUTPUTS:
                for ev, label in cheap:
                    if not coerce_outcome(ev(output, case), name=label).passed:
                        return True
        return False
    except Exception:  # noqa: BLE001 - the control must never break a real run
        return None


def _near_miss(outcomes: List[EvalOutcome]) -> tuple[Optional[str], Optional[float]]:
    """Of the passing outcomes that expose a margin, find the closest call."""
    margins = [(o.name, o.margin) for o in outcomes if o.margin is not None and o.passed]
    if not margins:
        return None, None
    name, margin = min(margins, key=lambda nm: nm[1])
    return name, margin


# Marks an output whose config.output_key raised: no verdict on that case.
_UNKEYABLE = object()


def _key(config: MutEvalConfig, output):
    key = getattr(config, "output_key", None)
    if key is None:
        return output
    try:
        return key(output)
    except Exception:  # noqa: BLE001 - a bad key means "can't tell", not a crash
        return _UNKEYABLE


@dataclass
class _BaselineProfile:
    """Per case: the keyed baseline samples, and whether they disagreed."""

    samples: List[list]
    noisy: List[bool]


def _profile(config: MutEvalConfig, samples_by_case: List[List[str]]) -> _BaselineProfile:
    # A sample whose key raised says nothing about the baseline's spread: drop
    # it rather than letting one bad sample mark the whole case undetermined.
    keyed = [
        [k for k in (_key(config, out) for out in outs) if k is not _UNKEYABLE]
        for outs in samples_by_case
    ]
    noisy = [any(k != ks[0] for k in ks[1:]) for ks in keyed]
    return _BaselineProfile(samples=keyed, noisy=noisy)


def _diff_outputs(
    config: MutEvalConfig, profile: Optional[_BaselineProfile], mutant: List[str]
) -> Optional[bool]:
    """Did the mutant change BEHAVIOR (the output under ``config.output_key``)?

    Per case: False if the keyed output matches ANY baseline sample (not
    evidence of change); True if it's unseen and the baseline was stable on that
    case; None if it's unseen but the baseline itself varied there (can't tell a
    change from sampling noise) or the key raised. Across cases, any True wins,
    then any None. None overall when we can't compare at all.

    ``mutant`` may be SHORTER than the case list: a killed run stops at its
    first failing case, so only the outputs it actually produced are compared.
    """
    if (
        profile is None
        or not profile.samples
        or not mutant
        or len(mutant) > len(profile.samples)
    ):
        return None
    verdicts: List[Optional[bool]] = []
    for base, noisy, out in zip(profile.samples, profile.noisy, mutant):
        k = _key(config, out)
        if k is _UNKEYABLE or not base:
            verdicts.append(None)
        elif any(k == b for b in base):
            verdicts.append(False)
        else:
            verdicts.append(None if noisy else True)
    if any(v is True for v in verdicts):
        return True
    if any(v is None for v in verdicts):
        return None
    return False


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
    if sample is not None and sample < 0:
        raise ValueError(f"sample must be >= 0 (got {sample})")
    if max_mutants is not None and max_mutants < 0:
        # mutants[:-1] silently dropped the last mutant.
        raise ValueError(f"max_mutants must be >= 0 (got {max_mutants})")
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


def _verdict(fails: int, n: int, config: MutEvalConfig) -> Tuple[bool, bool]:
    """(killed, unresolved) for ``fails`` failing runs out of ``n``. The SAME
    rule judges every mutant and the baseline: the original system must survive
    what a mutant would be killed by, or its own noise is counted as kills."""
    if config.kill_threshold is not None:
        return fails / n >= config.kill_threshold, False
    # Strict majority. A dead-even split (only possible with an even n) hasn't
    # earned a binary verdict — the judge straddled 50% — so it is UNRESOLVED:
    # excluded from the score, never silently defaulted to "survived".
    if n > 1 and fails * 2 == n:
        return False, True
    return fails * 2 > n, False


def _aggregate_change(
    config: MutEvalConfig, profile: _BaselineProfile, runs: List[_SuiteRun]
) -> Optional[bool]:
    """Did these runs' outputs change behavior vs the baseline? Any observed
    change -> True; any undetermined comparison -> None; False only when every
    run matched a baseline sample."""
    diffs = [_diff_outputs(config, profile, r.outputs) for r in runs]
    if any(d is True for d in diffs):
        return True
    if not diffs or any(d is None for d in diffs):
        return None
    return False


def _evaluate_mutant(
    mutant,
    config,
    cache,
    baseline_arg,
    baseline_outputs,
    budget=None,
    profile=None,
    fps=None,
) -> MutantOutcome:
    """Evaluate a single mutant into a MutantOutcome. Each (system, case, run)
    cell works on a private copy of the case (see ``_isolate``), so mutants can
    run concurrently across a thread pool. EVALS are shared: a stateful eval
    must be thread-safe under --concurrency (the deepeval adapter copies its
    metric per call for exactly this reason)."""
    try:
        runs = [
            _run_suite(
                mutant.system,
                config,
                cache=cache,
                baseline=baseline_arg,
                budget=budget,
                fps=fps,
            )
            for _ in range(config.runs_per_mutant)
        ]
        fails = sum(1 for r in runs if r.failing_eval is not None)
        kill_rate = fails / len(runs)
        caught_by = tuple(
            sorted({r.failing_eval for r in runs if r.failing_eval is not None})
        )
        killed, unresolved = _verdict(fails, len(runs), config)
        rep = next((r for r in runs if (r.failing_eval is not None) == killed), runs[0])
        closest_eval = min_margin = None
        sample_base = sample_mut = None
        if profile is None:
            profile = _profile(config, [[o] for o in baseline_outputs])
        # Output-change evidence from the runs that AGREE with the verdict (the
        # failing runs for a kill, the passing runs for a survivor). NOTE: for
        # free-text output, more runs mean more chances to see a wording change
        # — set config.output_key (+ baseline_runs) so "change" means a change
        # in behavior, not in phrasing.
        agreeing = [r for r in runs if (r.failing_eval is not None) == killed]
        output_changed = _aggregate_change(config, profile, agreeing)
        if not killed:
            closest_eval, min_margin = _near_miss(rep.outcomes)
            # Capture the first case whose output changed, for `muteval show`.
            for i, mo in enumerate(rep.outputs):
                if i < len(baseline_outputs) and baseline_outputs[i] != mo:
                    sample_base, sample_mut = baseline_outputs[i], mo
                    break
        return MutantOutcome(
            mutant=mutant,
            killed=killed,
            unresolved=unresolved,
            failing_eval=rep.failing_eval,
            closest_eval=closest_eval,
            min_margin=min_margin,
            output_changed=output_changed,
            severity=severity_of(mutant),
            kill_rate=kill_rate,
            caught_by=caught_by,
            baseline_output=sample_base,
            mutant_output=sample_mut,
        )
    except BudgetExceeded:
        raise  # budget is a hard stop, not a per-mutant error
    except Exception as exc:  # noqa: BLE001
        # A flaky eval call (timeout, rate limit, API error) must not nuke the
        # whole run. Record this mutant as errored and keep going.
        return MutantOutcome(
            mutant=mutant,
            killed=False,
            errored=True,
            error=f"{type(exc).__name__}: {exc}",
            severity=severity_of(mutant),
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
    canary: bool = False,
    accepted: Optional[Iterable[str]] = None,
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
    # Caching assumes determinism; a noisy (multi-run) suite must not be cached,
    # and baseline sampling exists to observe variance a cache would erase.
    cache_note: Optional[str] = None
    if cache is not None and (config.runs_per_mutant > 1 or config.baseline_runs > 1):
        cache = None
        cache_note = "disabled: repeated runs exist to observe noise a cache would erase"
    fps = _fingerprints(config) if cache is not None else None
    hits_before = cache.hits if cache is not None else 0
    budget = _Budget(max_calls)
    # Baseline — graded runs_per_mutant times and judged by the SAME rule as a
    # mutant (see _verdict): if a flaky judge would "kill" the unmodified
    # original, every kill it hands a mutant is noise, so the run is invalid.
    # Each graded run is retried a few times on *exceptions* only (a transient
    # judge/API blip must not poison the run); a clean pass/fail verdict is a
    # real result and is NOT retried.
    baseline_error: Optional[str] = None
    graded: List[_SuiteRun] = []
    for _ in range(config.runs_per_mutant):
        attempt_error: Optional[str] = None
        for _attempt in range(3):
            try:
                graded.append(
                    _run_suite(config.system, config, cache=cache, budget=budget, fps=fps)
                )
                attempt_error = None
                break
            except BudgetExceeded as exc:
                # Budget hit during the baseline: incomplete, fail closed.
                result = MutationResult(baseline_passed=False, baseline_error=str(exc))
                result.status = BUDGET_EXCEEDED
                return result
            except Exception as exc:  # noqa: BLE001 - transient judge/API errors
                attempt_error = f"{type(exc).__name__}: {exc}"
        if attempt_error is not None:
            baseline_error = attempt_error
            break

    baseline_passed = False
    baseline_outputs: List[str] = []
    baseline_run: Optional[_SuiteRun] = None
    base_fails = 0
    if baseline_error is None:
        base_fails = sum(1 for r in graded if r.failing_eval is not None)
        base_killed, base_tied = _verdict(base_fails, len(graded), config)
        baseline_passed = not base_killed and not base_tied
        # The reference outputs (skip-unchanged, `muteval show` diffs) come from
        # a run that fully PASSED, so its per-case outcomes are all passing.
        baseline_run = next((r for r in graded if r.failing_eval is None), None)
        if baseline_run is not None:
            baseline_outputs = baseline_run.outputs

    result = MutationResult(
        baseline_passed=baseline_passed, baseline_error=baseline_error
    )
    if graded and baseline_error is None:
        result.baseline_pass_rate = (len(graded) - base_fails) / len(graded)
    result.cache_note = cache_note
    # Provenance (recorded regardless of outcome): the model under test, and any
    # judge model muteval can introspect (its own llm_judge/grounded).
    result.model_under_test = config.system.model if config.system else None
    result.judge_models = tuple(
        sorted({m for ev in config.evals if (m := getattr(ev, "judge_model", None))})
    )
    result.accepted = frozenset(accepted or ()) | frozenset(
        config.accepted_survivors or ()
    )
    # BASELINE GATE: an invalid baseline makes every downstream number
    # meaningless (a failing eval fails on every mutant too, faking 100%).
    if baseline_error is not None:
        result.status = BASELINE_ERRORED
        return result
    if not baseline_passed:
        result.status = BASELINE_FAILED
        return result

    # Positive control (opt-in): the baseline passed — but does the suite reject
    # anything at all? A suite that passes pure garbage isn't scoring (see the
    # report). Off by default because it calls the rule-based evals an extra time,
    # which would perturb the cache's zero-calls-on-rerun guarantee.
    if canary:
        result.canary_caught = _canary_caught(config)

    mutants = select_mutants(
        config, operators=operators, sample=sample, seed=seed, max_mutants=max_mutants
    )

    if not mutants:
        result.status = NO_MUTANTS
        return result

    # Baseline variance: every graded baseline run already sampled the
    # ORIGINAL system's outputs, so they all count; config.baseline_runs tops the
    # total up with output-only samples (no evals, so no judge calls). "Did the
    # mutant change behavior?" is then judged against the baseline's own spread,
    # not one sample. A transient error only loses that sample; the budget binds.
    samples_by_case: List[List[str]] = [[] for _ in config.cases]
    for r in graded:
        for ci, out in enumerate(r.outputs):  # a failing run stops early
            samples_by_case[ci].append(out)
    try:
        for _ in range(max(0, config.baseline_runs - len(graded))):
            for ci, case in enumerate(config.cases):
                budget.charge()
                try:
                    samples_by_case[ci].append(config.invoke(config.system, case))
                except Exception:  # noqa: BLE001
                    continue
    except BudgetExceeded:
        result.status = BUDGET_EXCEEDED
        return result
    profile = _profile(config, samples_by_case)
    if max(len(graded), config.baseline_runs) > 1:
        result.noisy_cases = sum(profile.noisy)

    # Skip-unchanged optimization: give each mutant run the baseline's per-case
    # outputs + (passing) outcomes so cases whose output didn't change reuse them
    # and call no judges. Only for deterministic runs (runs_per_mutant == 1); a
    # noisy suite must re-run the judges to observe the noise.
    baseline_arg = None
    if config.runs_per_mutant == 1 and baseline_outputs:
        n_evals = len(config.evals)
        oc = baseline_run.outcomes
        if len(oc) == len(config.cases) * n_evals:  # baseline ran fully (it passed)
            base_by_case = [
                oc[i * n_evals : (i + 1) * n_evals] for i in range(len(config.cases))
            ]
            baseline_arg = (baseline_outputs, base_by_case, baseline_run.states)

    def _worker(mutant: Mutant) -> MutantOutcome:
        return _evaluate_mutant(
            mutant, config, cache, baseline_arg, baseline_outputs, budget, profile, fps
        )

    concurrency = max(1, int(concurrency or 1))
    try:
        if concurrency > 1 and len(mutants) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [ex.submit(_worker, m) for m in mutants]
                try:
                    # Collected in submission order, so outcomes stay deterministic.
                    for f in futures:
                        result.outcomes.append(f.result())
                except BudgetExceeded:
                    # Stop QUEUED mutants from starting; without this the pool
                    # drained every remaining mutant (and its untagged judges)
                    # after the budget was already spent.
                    for f in futures:
                        f.cancel()
                    raise
        else:
            for mutant in mutants:
                result.outcomes.append(_worker(mutant))
    except BudgetExceeded:
        # Hit --max-calls partway: the run is incomplete, so no trustworthy score.
        result.status = BUDGET_EXCEEDED
        return result

    if cache is not None:
        result.cache_hits = cache.hits - hits_before

    # Validity: no evidence at all is invalid; too many errors or too many
    # unresolved ties is invalid too (a score over a shrunken denominator is not
    # trustworthy — fail closed).
    if result.evaluated == 0:
        # "Only robustness ran" means no REGRESSION mutant was generated at all —
        # not merely that every regression mutant errored (that's no-evaluated).
        result.status = (
            NO_EVALUATED_MUTANTS if result.regression_total else NO_SCORED_MUTANTS
        )
    elif result.error_rate > config.max_error_rate:
        result.status = PARTIAL_ERRORS
    elif result.resolved == 0:
        result.status = NO_CONFIDENT_SCORE
    elif result.unresolved_rate > config.max_unresolved_rate:
        result.status = PARTIAL_UNRESOLVED
    return result
