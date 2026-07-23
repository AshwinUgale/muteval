"""v0.5 P0-9: the plugin API contract (docs/PLUGINS.md).

A third-party plugin that satisfies these contracts must keep working. This test
exercises all three extension points — a custom operator, a custom eval, and a
custom probe — through the public surface, and cleans up the global registries.
"""

from __future__ import annotations

import pytest

from muteval import EvalOutcome, MutEvalConfig, System, as_system, run_mutation_testing
from muteval.adapters.base import case_get, scorer_to_eval
from muteval.mutators import OPERATORS, Mutant, generate_mutants, register_operator
from muteval.probes.base import PROBES, ProbeResult, register_probe, run_probes


@pytest.fixture
def clean_registries():
    ops_before, probes_before = dict(OPERATORS), dict(PROBES)
    yield
    OPERATORS.clear(); OPERATORS.update(ops_before)
    PROBES.clear(); PROBES.update(probes_before)


# --- 1. custom operator ------------------------------------------------------

def test_custom_operator_contract(clean_registries):
    def shout(target):
        s = as_system(target)
        return [Mutant(operator="shout", description="UPPERCASED the prompt",
                       system=s.with_prompt(s.prompt.upper()))]

    register_operator("shout", shout)
    sys = System(prompt="be quiet please", model="gpt-4o-mini")

    # Selected by name...
    mutants = generate_mutants(sys, operators=["shout"])
    assert any(m.operator == "shout" and m.system.prompt == "BE QUIET PLEASE" for m in mutants)
    # ...and passed straight through as a callable (no registration needed).
    mutants2 = generate_mutants(sys, operators=[shout])
    assert any(m.operator == "shout" for m in mutants2)


def test_custom_operator_runs_end_to_end(clean_registries):
    def shout(target):
        s = as_system(target)
        return [Mutant(operator="shout", description="uppercased",
                       system=s.with_prompt(s.prompt.upper()))]

    cfg = MutEvalConfig(
        system=System(prompt="please stay calm and be helpful", model="gpt-4o-mini"),
        cases=[{"q": "hi"}],
        run=lambda system, case: system.prompt,  # echo the (mutated) prompt
        evals=[lambda o, c: o.islower()],  # baseline lower passes; SHOUT survives/killed
        eval_names=["is_lower"],
        operators=[shout],
    )
    result = run_mutation_testing(cfg)
    assert result.status == "valid"
    assert any(o.mutant.operator == "shout" for o in result.outcomes)


# --- 2. custom eval / adapter helpers ---------------------------------------

def test_custom_eval_contract():
    # Plain bool eval.
    plain = lambda output, case: "8080" in output
    assert bool(plain("port 8080", {})) is True

    # EvalOutcome-returning eval (carries score/threshold).
    ev = scorer_to_eval(lambda o, c: 1.0 if "$" in o else 0.0, threshold=0.5, name="price")
    out = ev("costs $5", {})
    assert isinstance(out, EvalOutcome) and out.passed and out.name == "price"

    # case_get reads dict or object cases.
    assert case_get({"topic": "x"}, "topic") == "x"

    class C:
        topic = "y"

    assert case_get(C(), "topic") == "y"


def test_is_llm_tag_is_respected_by_ordering():
    from muteval.runner import _ordered_evals

    judge = lambda o, c: True
    judge.is_llm = True
    cheap = lambda o, c: True
    cfg = MutEvalConfig(
        prompt="answer.", cases=[{"q": 1}], run=lambda p, c: "x",
        evals=[judge, cheap], eval_names=["judge", "cheap"],
    )
    labels = [lbl for _, _, lbl in _ordered_evals(cfg)]
    assert labels.index("cheap") < labels.index("judge")


# --- 3. custom probe ---------------------------------------------------------

def test_custom_probe_contract(clean_registries):
    def has_negative_case(config):
        ok = any(isinstance(c, dict) and c.get("unanswerable") for c in config.cases)
        return ProbeResult(
            name="has_negative_case", ok=ok,
            summary="has an unanswerable case" if ok else "no negative case",
        )

    register_probe("has_negative_case", has_negative_case)
    cfg = MutEvalConfig(
        prompt="answer.", cases=[{"q": "x"}, {"q": "y", "unanswerable": True}],
        run=lambda p, c: "x", evals=[lambda o, c: True], eval_names=["e"],
    )
    results = run_probes(cfg, probes=["has_negative_case"])
    assert len(results) == 1
    assert isinstance(results[0], ProbeResult) and results[0].ok is True
