"""End-to-end tests for the generalized runner: system-mode + near-miss."""

from muteval import MutEvalConfig, System, run_mutation_testing
from muteval.evals import EvalOutcome


def test_system_mode_context_mutation_is_killed():
    # A RAG system whose answer depends on retrieved context. An eval that
    # checks the key fact appears should KILL context-dropping mutants.
    system = System(
        prompt="Answer using only the retrieved context.",
        context=("The server listens on port 8080.",),
    )

    def run(sys_, case):
        # The "model" can only state the fact if it's still in the context.
        joined = " ".join(sys_.context or [])
        return joined if "8080" in joined else "I don't know."

    cfg = MutEvalConfig(
        system=system,
        cases=[{"question": "what port?"}],
        run=run,
        evals=[lambda o, c: "8080" in o],
        eval_names=["mentions_port"],
    )
    result = run_mutation_testing(cfg, operators=["drop_context_doc", "clear_context"])
    assert result.baseline_passed is True
    # Dropping the only doc must be caught by the eval.
    assert result.killed >= 1
    assert result.score == 1.0


def test_system_mode_survivor_when_eval_ignores_context():
    system = System(
        prompt="Answer using the retrieved context.",
        context=("The server listens on port 8080.",),
    )

    def run(sys_, case):
        joined = " ".join(sys_.context or [])
        return joined if "8080" in joined else "I don't know."

    cfg = MutEvalConfig(
        system=system,
        cases=[{"question": "what port?"}],
        run=run,
        evals=[lambda o, c: bool(o.strip())],  # only checks non-empty
        eval_names=["nonempty"],
    )
    result = run_mutation_testing(cfg, operators=["clear_context"])
    # The eval doesn't depend on retrieval, so the regression survives.
    assert len(result.survivors) >= 1


def test_near_miss_margin_recorded_for_survivor():
    # A scored eval that passes by a hair on the mutant should record the margin.
    def run(prompt, case):
        return "answer"

    def scored_eval(output, case):
        # Always passes, but with a thin margin to expose near-miss reporting.
        return EvalOutcome(passed=True, score=0.71, threshold=0.70, name="faith")

    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n- Do not promise refunds.\n"
        "- Always greet the user.",
        cases=[{"order_id": "X1"}],
        run=run,
        evals=[scored_eval],
    )
    result = run_mutation_testing(cfg)
    survivors = result.survivors
    assert survivors  # weak eval -> survivors exist
    near = [o for o in survivors if o.min_margin is not None]
    assert near
    assert near[0].closest_eval == "faith"
    assert round(near[0].min_margin, 4) == round(0.71 - 0.70, 4)


def test_config_rejects_prompt_and_system_together():
    try:
        MutEvalConfig(
            prompt="p",
            system=System(prompt="p"),
            cases=[{"x": 1}],
            run=lambda s, c: "o",
            evals=[lambda o, c: True],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for prompt+system")
