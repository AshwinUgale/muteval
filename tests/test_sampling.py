"""A5: deterministic mutant sampling."""

from muteval import MutEvalConfig, run_mutation_testing

_PROMPT = (
    "- You must cite the order ID.\n- Do not promise refunds.\n"
    "- Always greet the user.\n- Never reveal another customer's data.\n"
    "- Be concise.\n"
)


def _cfg():
    return MutEvalConfig(
        prompt=_PROMPT, cases=[{"input": "x"}], run=lambda p, c: p,
        evals=[lambda o, c: True],
    )


def test_sample_limits_mutant_count():
    full = run_mutation_testing(_cfg())
    sampled = run_mutation_testing(_cfg(), sample=3, seed=1)
    assert sampled.total == 3
    assert full.total > 3


def test_sample_is_deterministic_with_seed():
    a = run_mutation_testing(_cfg(), sample=4, seed=7)
    b = run_mutation_testing(_cfg(), sample=4, seed=7)
    assert [o.mutant.description for o in a.outcomes] == [
        o.mutant.description for o in b.outcomes
    ]


def test_sample_larger_than_population_is_safe():
    res = run_mutation_testing(_cfg(), sample=10_000, seed=1)
    assert res.total > 0  # no error; returns all available
