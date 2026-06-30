"""A3: bring-your-own operators (register_operator + callables)."""

from muteval import MutEvalConfig, register_operator, run_mutation_testing
from muteval.mutators import Mutant, OPERATORS, generate_mutants
from muteval.system import as_system


def _shout(target):
    s = as_system(target)
    return [Mutant("shout", "uppercased the prompt", s.with_prompt(s.prompt.upper()))]


def test_generate_mutants_accepts_callable_operator():
    ms = generate_mutants("be quiet please", operators=[_shout])
    assert len(ms) == 1
    assert ms[0].operator == "shout"
    assert ms[0].prompt == "BE QUIET PLEASE"


def test_register_operator_then_select_by_name():
    register_operator("shout_test", _shout)
    assert "shout_test" in OPERATORS
    ms = generate_mutants("keep calm", operators=["shout_test"])
    assert ms[0].prompt == "KEEP CALM"
    del OPERATORS["shout_test"]  # keep global registry clean for other tests


def test_config_operators_field_used_by_runner():
    cfg = MutEvalConfig(
        prompt="be nice",
        cases=[{"input": "x"}],
        run=lambda p, c: p,                      # echo the (mutated) prompt
        evals=[lambda o, c: o == o.lower()],     # passes only if output is lowercase
        operators=[_shout],                      # config-level custom operator
    )
    result = run_mutation_testing(cfg)            # no explicit operators -> uses config.operators
    assert result.total == 1                      # only the shout mutant
    assert result.killed == 1                     # uppercased output fails the lowercase eval
