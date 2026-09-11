"""Provenance + per-eval flaky attribution (reader feedback on the series).

- The flaky list is a bug report about the eval *question*: attribute flips to the
  eval that caused them, so an ambiguous rubric dimension is rewritten, not just
  re-sampled.
- Judge drift is silent: a model bump replaces the coin a majority vote stabilizes,
  so pin the judge model (and the model under test) in the result.
"""

from muteval import MutEvalConfig, checks, run_mutation_testing
from muteval.mutators import Mutant
from muteval.report import result_to_dict
from muteval.runner import MutantOutcome, MutationResult
from muteval.system import System


def _m():
    return Mutant(operator="remove_emphasis", description="d", system=System(prompt="p"))


def test_judge_model_recorded_only_for_builtin_judge():
    built_in = checks.llm_judge("good", model="gpt-4o-mini")
    assert getattr(built_in, "judge_model", None) == "gpt-4o-mini"
    # A caller-supplied judge hides its model, so muteval doesn't claim one.
    custom = checks.llm_judge("good", judge=lambda prompt: 1.0)
    assert getattr(custom, "judge_model", None) is None


def test_flaky_by_eval_attributes_flips_to_the_dimension():
    def out(kill_rate, caught_by):
        return MutantOutcome(
            mutant=_m(),
            killed=False,
            kill_rate=kill_rate,
            caught_by=caught_by,
            output_changed=True,
        )

    r = MutationResult(
        baseline_passed=True,
        outcomes=[
            out(0.5, ("judge_faithfulness",)),  # flaky
            out(0.5, ("judge_faithfulness",)),  # flaky
            out(0.5, ("judge_tone",)),  # flaky
            out(1.0, ("judge_faithfulness",)),  # not flaky -> excluded
        ],
    )
    assert r.flaky_by_eval == {"judge_faithfulness": 2, "judge_tone": 1}


def test_provenance_collected_end_to_end():
    # A deterministic custom judge (no API) that advertises its model, so we can
    # exercise the runner's provenance collection without a key.
    judge = checks.llm_judge("grounded", judge=lambda prompt: 1.0)
    judge.judge_model = "gpt-4o-mini"  # simulate a judge exposing its model
    cfg = MutEvalConfig(
        system=System(prompt="You **must** cite the id.", model="gpt-4o"),
        cases=[{"x": 1}],
        run=lambda system, case: "ok",
        evals=[judge],
    )
    r = run_mutation_testing(cfg, operators=["remove_emphasis"])
    assert r.model_under_test == "gpt-4o"
    assert r.judge_models == ("gpt-4o-mini",)
    d = result_to_dict(r)
    assert d["model_under_test"] == "gpt-4o"
    assert d["judge_models"] == ["gpt-4o-mini"]
    assert isinstance(d["flaky_by_eval"], dict)
