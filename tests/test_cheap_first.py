"""v0.5 P0-4: cheap-checks-first + skip-unchanged.

Two cost savers that keep the same result while making far fewer LLM-judge calls:
  * rule-based checks run before judges, so the short-circuit skips the judge
    when a cheap check already kills the mutant;
  * a mutant case whose output is unchanged from baseline reuses the baseline's
    (passing) outcomes instead of re-running the judge.
"""

from __future__ import annotations

from muteval import EvalOutcome, MutEvalConfig, System, checks, run_mutation_testing
from muteval.runner import _ordered_evals, _run_suite


def test_builtin_llm_judge_is_tagged():
    ev = checks.llm_judge("is it grounded?", threshold=0.7)
    assert getattr(ev, "is_llm", False) is True
    # rule-based checks are cheap (untagged).
    assert getattr(checks.contains("x"), "is_llm", False) is False


def test_ordered_evals_puts_cheap_before_llm():
    def cheap(o, c):
        return True

    def judge(o, c):
        return True

    judge.is_llm = True
    cfg = MutEvalConfig(
        prompt="answer.", cases=[{"q": 1}], run=lambda p, c: "x",
        evals=[judge, cheap], eval_names=["judge", "cheap"],  # judge listed FIRST
    )
    labels = [label for _, _, label in _ordered_evals(cfg)]
    assert labels.index("cheap") < labels.index("judge")


def test_short_circuit_skips_judge_when_cheap_check_fails():
    calls = {"judge": 0}

    def judge(o, c):
        calls["judge"] += 1
        return EvalOutcome(passed=True, name="judge")

    judge.is_llm = True

    def cheap(o, c):
        return EvalOutcome(passed=False, name="cheap")  # always fails

    cfg = MutEvalConfig(
        prompt="answer the question.", cases=[{"q": 1}], run=lambda p, c: "out",
        evals=[judge, cheap], eval_names=["judge", "cheap"],  # judge first, but cheaper wins
    )
    r = _run_suite(cfg.system, cfg)
    assert r.failing_eval == "cheap"
    assert calls["judge"] == 0  # cheap ran first, failed, judge never called


def test_skip_unchanged_skips_judges_for_inert_mutants():
    calls = {"judge": 0}

    def run(system, case):
        return "constant output"  # every mutant produces IDENTICAL output -> inert

    def judge(output, case):
        calls["judge"] += 1
        return EvalOutcome(passed=True, name="judge")

    judge.is_llm = True

    cfg = MutEvalConfig(
        system=System(
            prompt="Line one.\nLine two to mutate.\nLine three here.", model="gpt-4o-mini"
        ),
        cases=[{"q": "a"}, {"q": "b"}],
        run=run,
        evals=[judge],
        eval_names=["judge"],
    )
    result = run_mutation_testing(cfg)
    # The judge runs only for the baseline's 2 cases; every mutant is inert and
    # reuses the baseline outcomes -> zero further judge calls.
    assert calls["judge"] == 2
    assert result.total > 0  # mutants were generated and evaluated
