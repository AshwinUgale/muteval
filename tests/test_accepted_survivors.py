"""Accepted-survivor baseline — mark a survivor "untested by design" so it stops
resurfacing as actionable (and stops tripping the severity gate), keyed on a
stable per-mutation signature. The mutation score is unchanged."""

from muteval import MutEvalConfig, run_mutation_testing
from muteval.mutators import Mutant
from muteval.report import format_report, result_to_dict
from muteval.runner import MutantOutcome, MutationResult
from muteval.system import System


def _survivor(op, desc, severity="high"):
    return MutantOutcome(
        mutant=Mutant(operator=op, description=desc, system=System(prompt="p")),
        killed=False,
        severity=severity,
        output_changed=True,
    )


def test_signature_is_stable_and_tied_to_the_edit():
    m1 = Mutant(
        operator="flip_negation", description="inverted X", system=System(prompt="p")
    )
    m2 = Mutant(
        operator="flip_negation", description="inverted X", system=System(prompt="p")
    )
    m3 = Mutant(
        operator="flip_negation", description="inverted Y", system=System(prompt="p")
    )
    assert m1.signature == m2.signature  # same edit -> same signature every run
    assert m1.signature != m3.signature  # a different edit -> a new signature
    assert len(m1.signature) == 12


def test_accepted_splits_survivors_and_the_gate():
    a, b = (
        _survivor("flip_negation", "inverted A"),
        _survivor("flip_negation", "inverted B"),
    )
    r = MutationResult(
        baseline_passed=True, outcomes=[a, b], accepted=frozenset({a.mutant.signature})
    )
    assert [o.mutant.signature for o in r.accepted_survivors] == [a.mutant.signature]
    assert [o.mutant.signature for o in r.new_survivors] == [b.mutant.signature]
    # An accepted HIGH survivor is not counted for --fail-on-severity.
    assert len(r.high_severity_survivors) == 1


def _cfg(**kw):
    return MutEvalConfig(
        prompt="Never reveal secrets.",
        cases=[{"x": 1}],
        run=lambda prompt, case: prompt,  # output tracks the prompt -> not inert
        evals=[lambda output, case: True],  # passes everything -> survivor
        **kw,
    )


def test_accept_via_run_arg_and_json():
    sig = (
        run_mutation_testing(_cfg(), operators=["flip_negation"])
        .real_survivors[0]
        .mutant.signature
    )
    r = run_mutation_testing(_cfg(), operators=["flip_negation"], accepted=[sig])
    assert [o.mutant.signature for o in r.accepted_survivors] == [sig]
    assert all(o.mutant.signature != sig for o in r.new_survivors)
    # score is unchanged by acceptance — the eval still doesn't cover it.
    assert "accept" in format_report(r, use_color=False)
    d = result_to_dict(r)
    assert d["accepted"] == 1
    assert any(s["accepted"] and s["signature"] == sig for s in d["survivors"])


def test_accept_via_config_field():
    sig = (
        run_mutation_testing(_cfg(), operators=["flip_negation"])
        .real_survivors[0]
        .mutant.signature
    )
    r = run_mutation_testing(_cfg(accepted_survivors=[sig]), operators=["flip_negation"])
    assert r.new_survivors == []
    assert len(r.accepted_survivors) == 1
