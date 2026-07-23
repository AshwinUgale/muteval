"""v0.5 P0-5: parallel mutant evaluation (--concurrency N).

Running mutants across a thread pool cuts wall-clock on API-bound suites. It must
produce the SAME result as a sequential run (order preserved) and be safe to
combine with a shared --cache.
"""

from __future__ import annotations

import json

from muteval import Cache, EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.report import result_to_dict

SYSTEM = System(
    prompt=(
        "You are a support assistant.\n"
        "- Always cite the source document.\n"
        "- Do not invent facts.\n"
        "- Say you don't know if the notes lack the answer.\n"
        "- Greet the user politely."
    ),
    model="gpt-4o-mini",
)
CASES = [{"gt": "8080"}, {"gt": "1986"}, {"gt": "keys"}]


def _run(system, case):
    cite = "cite the source" in system.prompt.lower()
    return f"The answer is {case['gt']}" + (" [1](doc)" if cite else "") + "."


def _correct(output, case):
    return EvalOutcome(passed=case["gt"] in output, name="correct")


def _config():
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=_run, evals=[_correct], eval_names=["correct"]
    )


def test_parallel_matches_sequential():
    seq = result_to_dict(run_mutation_testing(_config(), concurrency=1))
    par = result_to_dict(run_mutation_testing(_config(), concurrency=4))
    assert json.dumps(seq, sort_keys=True) == json.dumps(par, sort_keys=True)


def test_concurrency_zero_or_negative_is_treated_as_serial():
    for c in (0, -3, None):
        r = run_mutation_testing(_config(), concurrency=c)
        assert r.status == "valid" and r.total > 0


def test_cache_is_thread_safe_under_concurrency(tmp_path):
    calls = {"n": 0}

    def counting_run(system, case):
        calls["n"] += 1
        return _run(system, case)

    def make():
        return MutEvalConfig(
            system=SYSTEM, cases=CASES, run=counting_run,
            evals=[_correct], eval_names=["correct"],
        )

    cache = Cache(str(tmp_path / "c.sqlite"))
    r1 = run_mutation_testing(make(), cache=cache, concurrency=4)
    assert r1.status == "valid"
    assert r1.errored == 0  # no thread/sqlite errors
    calls["n"] = 0
    r2 = run_mutation_testing(make(), cache=cache, concurrency=4)
    assert calls["n"] == 0  # fully served from the (thread-safe) cache
    cache.close()
    assert json.dumps(result_to_dict(r1), sort_keys=True) == json.dumps(
        result_to_dict(r2), sort_keys=True
    )
