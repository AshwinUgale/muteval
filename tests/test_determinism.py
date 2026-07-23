"""v0.4 'Provably honest': determinism.

Same config + same seed must produce a byte-identical machine-readable result.
A reproducible number is a prerequisite for a defensible one. Uses a fully
offline (mock-model) config so the only source of variation is muteval's own RNG
(mutant sampling), which the seed must fully control.
"""

from __future__ import annotations

import json

from muteval import EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.report import result_to_dict

SYSTEM = System(
    prompt=(
        "You are a support assistant.\n"
        "- Always cite the source document inline.\n"
        "- Do not invent facts that are not in the notes.\n"
        "- If the answer is not in the notes, say you don't know.\n"
        "- Keep answers concise and professional.\n"
        "- Greet the user politely."
    ),
    model="gpt-4o-mini",
)

CASES = [{"q": "q1", "gt": "8080"}, {"q": "q2", "gt": "keys"}, {"q": "q3", "gt": "1986"}]


def _run(system, case):
    cite = "cite the source" in system.prompt.lower()
    ans = f"The answer is {case['gt']}"
    return ans + (" [1](doc)" if cite else "") + "."


def _correct(output, case):
    return EvalOutcome(passed=case["gt"] in output, name="correct")


def _make_config():
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=_run, evals=[_correct], eval_names=["correct"]
    )


def test_same_seed_same_result_dict():
    r1 = run_mutation_testing(_make_config(), sample=6, seed=123)
    r2 = run_mutation_testing(_make_config(), sample=6, seed=123)
    assert json.dumps(result_to_dict(r1), sort_keys=True) == json.dumps(
        result_to_dict(r2), sort_keys=True
    )


def test_no_sampling_is_deterministic():
    r1 = run_mutation_testing(_make_config())
    r2 = run_mutation_testing(_make_config())
    assert json.dumps(result_to_dict(r1), sort_keys=True) == json.dumps(
        result_to_dict(r2), sort_keys=True
    )


def test_different_seed_may_select_different_mutants_but_stays_valid():
    # Different seeds are allowed to differ; both must be internally consistent.
    for seed in (1, 2, 7, 99):
        r = run_mutation_testing(_make_config(), sample=5, seed=seed)
        d = result_to_dict(r)
        assert d["killed"] <= d["evaluated"] <= d["total"]
        if d["score"] is not None:
            assert 0.0 <= d["score"] <= 1.0
