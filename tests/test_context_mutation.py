"""Context (RAG) mutation in System mode: corrupted retrieval should survive a
weak eval and rank HIGH severity."""

from muteval import MutEvalConfig, System, run_mutation_testing
from muteval.severity import HIGH


def _run(system, case):
    docs = system.context or ("",)
    return f"Per the docs: {docs[0]}"


def test_corrupted_context_survives_weak_eval_as_high_severity():
    sys = System(
        prompt="Answer using only the retrieved context.",
        context=(
            "The server listens on port 8080 by default.",
            "Override with the PORT environment variable.",
        ),
        model="mock",
    )
    cfg = MutEvalConfig(
        system=sys,
        cases=[{"question": "what port?"}],
        run=_run,
        evals=[lambda o, c: bool(o.strip())],  # weak: only checks non-empty
    )
    result = run_mutation_testing(
        cfg, operators=["corrupt_context_doc", "drop_context_doc"]
    )
    # The corrupted/dropped retrieval changes the answer, the weak eval misses it.
    assert len(result.real_survivors) >= 1
    # And those retrieval-poisoning survivors rank HIGH.
    assert any(o.severity == HIGH for o in result.real_survivors)
