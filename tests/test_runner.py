from muteval import MutEvalConfig, run_mutation_testing


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


def test_score_is_one_when_no_mutants():
    cfg = MutEvalConfig(
        prompt="short",  # too short to mutate meaningfully
        cases=[{"x": 1}],
        run=lambda p, c: "ok",
        evals=[lambda o, c: True],
    )
    result = run_mutation_testing(cfg)
    # No mutants -> nothing to catch -> score defined as 1.0
    assert result.score == 1.0


def test_eval_exception_is_contained_not_fatal():
    # A flaky eval (timeout/API error) must mark the mutant errored and let the
    # run finish, not crash the whole process.
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
    assert result.errored == result.total
    assert result.evaluated == 0
    # Errored mutants are excluded from the score denominator.
    assert result.score == 1.0


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
