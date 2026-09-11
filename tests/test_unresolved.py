"""Unresolved (tied) verdicts as a first-class outcome.

Motivated by a reader: under strict majority a dead-even split (fails*2 ==
len(runs)) silently defaulted to "survived". That isn't an earned verdict — the
judge straddled 50%. muteval now marks it UNRESOLVED and excludes it from the
score's numerator AND denominator (CI on the resolved set), reporting it
separately.
"""

from muteval import MutEvalConfig, run_mutation_testing
from muteval.mutators import Mutant
from muteval.report import format_report, result_to_dict
from muteval.runner import MutantOutcome, MutationResult
from muteval.system import System


def _outcome(killed=False, unresolved=False, errored=False):
    return MutantOutcome(
        mutant=Mutant(
            operator="remove_emphasis", description="d", system=System(prompt="p")
        ),
        killed=killed,
        unresolved=unresolved,
        errored=errored,
        output_changed=True,
    )


def test_unresolved_excluded_from_score_denominator():
    r = MutationResult(
        baseline_passed=True,
        outcomes=[
            _outcome(killed=True),  # killed
            _outcome(),  # survived (real gap)
            _outcome(unresolved=True),  # tie
            _outcome(errored=True),  # errored
        ],
    )
    assert r.killed == 1
    assert r.evaluated == 3  # killed + survived + unresolved (all non-errored)
    assert r.resolved == 2  # excludes the tie
    assert r.unresolved == 1
    assert r.score == 0.5  # killed / resolved (NOT killed/evaluated == 1/3)
    assert len(r.survivors) == 1  # the tie is NOT a confident survivor
    d = result_to_dict(r)
    assert d["resolved"] == 2 and d["unresolved"] == 1


def test_all_ties_give_no_confident_score():
    r = MutationResult(baseline_passed=True, outcomes=[_outcome(unresolved=True)])
    assert r.resolved == 0
    assert r.score is None
    assert "NO CONFIDENT SCORE" in format_report(r, use_color=False)


class _FlakyEval:
    """Returns verdicts from a fixed pattern to simulate judge noise."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def __call__(self, output, case):
        v = self.pattern[self.i] if self.i < len(self.pattern) else True
        self.i += 1
        return v


def test_tie_is_marked_unresolved_end_to_end():
    # baseline passes (True); the one mutant fails 1 of 2 runs -> tied -> unresolved.
    cfg = MutEvalConfig(
        prompt="You **must** cite the order ID.",
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[_FlakyEval([True, False, True])],
        runs_per_mutant=2,
    )
    r = run_mutation_testing(cfg, operators=["remove_emphasis"])
    assert r.baseline_passed
    assert r.unresolved == 1
    assert r.resolved == 0
    o = r.outcomes[0]
    assert o.unresolved is True and o.killed is False
    # A tie is not a survivor, so it never shows up as a coverage gap.
    assert r.survivors == []
