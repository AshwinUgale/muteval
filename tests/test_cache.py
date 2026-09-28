"""v0.5 P0-3: caching / incremental runs.

The cost of mutation testing is re-running the suite (model + judges) per mutant.
The cache memoizes run outputs and eval outcomes so an identical re-run makes
ZERO model/judge calls — the release-plan gate for this feature.
"""

from __future__ import annotations

import json

from muteval import Cache, EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.report import result_to_dict


def _counting_config(calls):
    def run(system, case):
        calls["run"] += 1
        cite = "cite" in system.prompt.lower()
        return f"answer {case['gt']}" + (" [1](doc)" if cite else "")

    def correct(output, case):
        calls["eval"] += 1
        return EvalOutcome(passed=case["gt"] in output, name="correct")

    return MutEvalConfig(
        system=System(
            prompt="Answer the question. Always cite the source.", model="gpt-4o-mini"
        ),
        cases=[{"gt": "8080"}, {"gt": "1986"}],
        run=run,
        evals=[correct],
        eval_names=["correct"],
    )


def test_second_identical_run_makes_zero_calls(tmp_path):
    calls = {"run": 0, "eval": 0}
    cache = Cache(str(tmp_path / "c.sqlite"))

    r1 = run_mutation_testing(_counting_config(calls), cache=cache)
    assert calls["run"] > 0 and calls["eval"] > 0  # first run does real work

    calls["run"] = calls["eval"] = 0
    r2 = run_mutation_testing(_counting_config(calls), cache=cache)
    assert calls["run"] == 0, "cached re-run must not call the model"
    assert calls["eval"] == 0, "cached re-run must not call the judges"
    cache.close()

    # And the cached result is identical to the fresh one.
    assert _scored(r1) == _scored(r2)


def test_cache_disabled_for_nondeterministic_suites(tmp_path):
    calls = {"run": 0, "eval": 0}
    cache = Cache(str(tmp_path / "c.sqlite"))
    cfg = _counting_config(calls)
    cfg.runs_per_mutant = 2  # noisy suite -> cache must be bypassed

    run_mutation_testing(cfg, cache=cache)
    calls["run"] = calls["eval"] = 0
    run_mutation_testing(cfg, cache=cache)
    # Still calls on the "re-run" because caching is disabled at runs_per_mutant>1.
    assert calls["run"] > 0
    cache.close()


def test_cache_roundtrips_output_and_outcome(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    sys_a = System(prompt="A", model="m")
    sys_b = System(prompt="B", model="m")
    case = {"gt": "x"}

    assert cache.get_output(sys_a, case) is None  # miss
    cache.set_output(sys_a, case, "hello")
    assert cache.get_output(sys_a, case) == "hello"  # hit
    assert cache.get_output(sys_b, case) is None  # different system -> miss

    # Outcomes are keyed on (output, case, eval FINGERPRINT) — not the label.
    oc = EvalOutcome(passed=True, score=0.71, threshold=0.70, name="judge")
    assert cache.get_outcome("hello", case, "fp1", "judge") is None
    cache.set_outcome("hello", case, "fp1", oc)
    got = cache.get_outcome("hello", case, "fp1", "judge")
    assert (
        got.passed and got.score == 0.71 and got.threshold == 0.70 and got.name == "judge"
    )
    assert got.margin == 0.71 - 0.70  # scored fields survive the roundtrip
    assert cache.get_outcome("hello", case, "fp2", "judge") is None  # edited eval
    assert cache.get_outcome("other", case, "fp1", "judge") is None  # other output
    assert cache.hits > 0 and cache.misses > 0
    cache.close()


def _scored(result):
    """The result JSON minus cache provenance (hits legitimately differ between
    the run that filled the cache and the run it served)."""
    d = result_to_dict(result)
    d.pop("cache", None)
    return json.dumps(d, sort_keys=True)
