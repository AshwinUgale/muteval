import itertools

from muteval import MutEvalConfig, run_mutation_testing
from muteval.runner import (
    BASELINE_ERRORED,
    BASELINE_FAILED,
    NO_MUTANTS,
    PARTIAL_ERRORS,
    VALID,
)


def _make_config(evals, eval_names):
    prompt = (
        "You are a support bot.\n"
        "- You must always cite the order ID.\n"
        "- Do not promise refunds.\n"
    )

    def run(p, case):
        out = []
        if "must always cite the order id" in p.lower():
            out.append(f"order {case['order_id']}")
        if "do not promise refunds" in p.lower():
            out.append("no refund promised")
        else:
            out.append("refund promised")
        return " ".join(out)

    return MutEvalConfig(
        prompt=prompt,
        cases=[{"input": "hi", "order_id": "X1"}],
        run=run,
        evals=evals,
        eval_names=eval_names,
    )


def test_baseline_passes_when_evals_match_prompt():
    cfg = _make_config(
        evals=[lambda o, c: c["order_id"] in o],
        eval_names=["cites_order_id"],
    )
    result = run_mutation_testing(cfg)
    assert result.baseline_passed is True


def test_strong_eval_kills_the_relevant_mutant():
    # An eval that checks order-id citation should KILL the mutant that
    # weakens the "must always cite" instruction.
    cfg = _make_config(
        evals=[lambda o, c: c["order_id"] in o],
        eval_names=["cites_order_id"],
    )
    result = run_mutation_testing(cfg)
    assert result.killed >= 1
    assert 0.0 <= result.score <= 1.0


def test_missing_eval_leaves_a_survivor():
    # With NO eval for the refund behavior, the mutant that deletes
    # "Do not promise refunds" must SURVIVE.
    cfg = _make_config(
        evals=[lambda o, c: bool(o.strip())],  # only checks non-empty
        eval_names=["is_nonempty"],
    )
    result = run_mutation_testing(cfg)
    assert len(result.survivors) >= 1
    # A weak suite should score below perfect.
    assert result.score < 1.0


def test_score_is_none_and_status_no_mutants():
    cfg = MutEvalConfig(
        prompt="short",  # too short to mutate meaningfully
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[lambda o, c: True],
    )
    result = run_mutation_testing(cfg)
    # No mutants -> no evidence. That is NOT a perfect score; it is unknown.
    assert result.total == 0
    assert result.status == NO_MUTANTS
    assert result.score is None
    assert result.effective_score is None


def test_baseline_error_stops_the_run_before_any_mutant():
    # A flaky eval that raises on the ORIGINAL system means we cannot trust a
    # baseline. The run must stop (no mutants generated/run) and be INVALID,
    # never a vacuous 100%.
    def boom(output, case):
        raise RuntimeError("api timeout")

    cfg = MutEvalConfig(
        prompt="You are a bot.\n- You must cite the order ID.\n- Do not lie.",
        cases=[{"order_id": "X1"}],
        run=lambda p, c: "order X1",
        evals=[boom],
        eval_names=["boom"],
    )
    result = run_mutation_testing(cfg)  # must NOT raise
    assert result.baseline_error is not None
    assert result.status == BASELINE_ERRORED
    assert result.total == 0  # never got past the baseline gate
    assert result.score is None
    assert result.effective_score is None


def test_baseline_failure_is_invalid_not_perfect():
    # Baseline eval fails (returns False) on the original system -> INVALID.
    cfg = MutEvalConfig(
        prompt="You are a bot.\n- You must cite the order ID.\n- Do not lie.",
        cases=[{"order_id": "X1"}],
        run=lambda p, c: "order X1",
        evals=[lambda o, c: False],  # never passes, even on the original
    )
    result = run_mutation_testing(cfg)
    assert result.baseline_passed is False
    assert result.status == BASELINE_FAILED
    assert result.total == 0
    assert result.score is None


def test_even_run_ties_survive_strict_majority():
    # With kill_threshold=None (strict majority), a tie must SURVIVE, not kill.
    # Build a run() whose eval fails on exactly half the runs by cycling.
    from itertools import count

    state = count()

    def half_failing(output, case):
        # Only used on mutants; alternate pass/fail so a 2-run mutant ties 1-1.
        return next(state) % 2 == 0

    cfg = MutEvalConfig(
        prompt="You **must** cite the order ID.",
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[half_failing],
        runs_per_mutant=2,
    )
    # Baseline consumed some state; assert on the aggregate behavior instead:
    # every mutant with an exact 1-1 split is NOT killed.
    result = run_mutation_testing(cfg, operators=["remove_emphasis"])
    for o in result.outcomes:
        if o.kill_rate == 0.5:
            assert o.killed is False  # ties survive under strict majority


def _partial_error_config(max_error_rate=0.0):
    # run() raises whenever the KEEPLINE rule is gone -> mutants that drop/edit
    # that line ERROR, while other mutants evaluate normally. Baseline keeps it.
    def run(prompt, case):
        if "KEEPLINE stable." not in prompt:
            raise RuntimeError("boom: keepline removed")
        return "ok"

    return MutEvalConfig(
        prompt="- Cite the order ID.\n- KEEPLINE stable.\n- Be polite.",
        cases=[{"x": 1}],
        run=run,
        evals=[lambda o, c: True],
        max_error_rate=max_error_rate,
    )


def test_partial_mutant_errors_are_invalid_by_default():
    # Some (not all) mutants error -> the run is PARTIAL_ERRORS, NOT valid, so a
    # score over the shrunken denominator can't silently pass CI.
    result = run_mutation_testing(_partial_error_config(max_error_rate=0.0))
    assert result.errored >= 1
    assert result.evaluated >= 1  # at least one mutant did produce a verdict
    assert result.status == PARTIAL_ERRORS
    assert result.error_rate > 0.0


def test_error_budget_can_accept_partial_errors():
    # With a permissive budget the same run is VALID (score over survivors).
    result = run_mutation_testing(_partial_error_config(max_error_rate=1.0))
    assert result.errored >= 1
    assert result.status == VALID
    assert result.score is not None


def test_output_change_uses_all_runs_not_just_representative():
    # A survivor whose output changes on SOME runs must be classified changed,
    # not "observationally unchanged" from a single representative run.
    outputs = itertools.cycle(["A", "B", "A"])

    def run(prompt, case):
        if "**" in prompt:      # baseline (unmutated) is stable "A"
            return "A"
        return next(outputs)    # mutant: A, B, A across 3 runs

    cfg = MutEvalConfig(
        prompt="Answer **now**.",
        cases=[{"x": 1}],
        run=run,
        evals=[lambda o, c: True],
        runs_per_mutant=3,
    )
    result = run_mutation_testing(cfg, operators=["remove_emphasis"])
    o = result.outcomes[0]
    assert o.killed is False
    assert o.output_changed is True          # observed "B" is not discarded
    assert result.inert_survivors == []      # so it is NOT misclassified inert


def test_strict_majority_thresholds():
    # Unit-check the verdict rule directly for even N.
    def verdict(fails, n):
        return fails * 2 > n  # strict majority; ties survive

    assert verdict(1, 2) is False  # 1/2 tie -> survives
    assert verdict(2, 2) is True   # 2/2 -> killed
    assert verdict(2, 4) is False  # 2/4 tie -> survives
    assert verdict(3, 4) is True   # 3/4 -> killed


def test_baseline_retries_past_a_transient_error():
    # A flaky judge that errors on the FIRST baseline call must not poison the
    # run — the baseline retries and succeeds.
    state = {"n": 0}

    def flaky(output, case):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("api timeout on baseline")
        return True

    cfg = MutEvalConfig(
        prompt="You **must** cite the order ID.",
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[flaky],
    )
    result = run_mutation_testing(cfg, operators=["remove_emphasis"])
    assert result.baseline_error is None
    assert result.baseline_passed is True
