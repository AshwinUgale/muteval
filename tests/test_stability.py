"""Phase 1.1: statistical stability — majority-vote verdicts + Wilson CI."""

from muteval import MutEvalConfig, run_mutation_testing
from muteval.stats import min_samples_for_lower_bound, wilson_interval


# --- the stats helper -------------------------------------------------------

def test_wilson_matches_the_19_of_20_intuition():
    lo, hi = wilson_interval(19, 20)
    assert 0.74 < lo < 0.79        # ~76%
    assert 0.97 < hi <= 1.0        # ~99%


def test_wilson_unknown_when_no_samples():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_min_samples_to_defend_a_claim():
    # observing ~95%, you need many cases to defend ">90%".
    assert min_samples_for_lower_bound(0.95, 0.90) > 50


# --- majority-vote aggregation (fixes the any-run-kills bias) ---------------

class _FlakyEval:
    """Returns verdicts from a fixed pattern to simulate judge noise."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def __call__(self, output, case):
        v = self.pattern[self.i] if self.i < len(self.pattern) else True
        self.i += 1
        return v


def _cfg(pattern):
    # remove_emphasis on a bolded prompt yields exactly ONE mutant.
    return MutEvalConfig(
        prompt="You **must** cite the order ID.",
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[_FlakyEval(pattern)],
        runs_per_mutant=3,
    )


def test_killed_only_when_failing_majority_of_runs():
    # baseline passes, then the mutant fails 2 of 3 runs -> killed.
    r = run_mutation_testing(_cfg([True, False, False, True]),
                             operators=["remove_emphasis"])
    assert r.baseline_passed
    o = r.outcomes[0]
    assert abs(o.kill_rate - 2 / 3) < 1e-9
    assert o.killed is True
    assert len(r.flaky) == 1        # it flipped between runs


def test_survives_when_failing_only_a_minority():
    # baseline passes, mutant fails 1 of 3 -> NOT killed (old code would kill it).
    r = run_mutation_testing(_cfg([True, False, True, True]),
                             operators=["remove_emphasis"])
    o = r.outcomes[0]
    assert abs(o.kill_rate - 1 / 3) < 1e-9
    assert o.killed is False
    assert len(r.flaky) == 1


def test_score_ci_is_well_formed():
    r = run_mutation_testing(_cfg([True, False, False, True]),
                             operators=["remove_emphasis"])
    lo, hi = r.score_ci
    assert 0.0 <= lo <= hi <= 1.0
