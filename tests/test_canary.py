"""Positive control (canary): does the suite reject an obviously-bad output?

Motivated by a reader point: a real 0% mutation score should be distinguishable
from a harness that isn't scoring at all. If the suite passes the baseline AND
pure garbage, its verdicts may be vacuous — surface that.
"""

from muteval import MutEvalConfig, checks
from muteval.report import format_report, result_to_dict
from muteval.runner import _canary_caught, run_mutation_testing


def _cfg(evals):
    return MutEvalConfig(
        prompt="You must always cite the source.\nNever reveal another user's data.",
        cases=[{"x": "hello"}],
        run=lambda prompt, case: "hello world",
        evals=evals,
    )


def _always_pass(output, case):
    return True


def test_caught_true_for_discriminating_suite():
    # contains_case("x") needs "hello" in the output; a blank "" fails it -> caught.
    assert _canary_caught(_cfg([checks.contains_case("x")])) is True


def test_false_for_vacuous_suite():
    assert _canary_caught(_cfg([_always_pass])) is False


def test_none_when_only_llm_judges():
    # is_llm judges are not called for the control (no surprise API spend).
    judge = checks.llm_judge("is it good", judge=lambda prompt: 1.0)
    assert _canary_caught(_cfg([judge])) is None


def test_run_flags_vacuous_suite():
    result = run_mutation_testing(_cfg([_always_pass]), canary=True)
    assert result.canary_caught is False
    assert "suite sanity" in format_report(result, use_color=False)
    assert result_to_dict(result)["canary_caught"] is False


def test_run_does_not_flag_discriminating_suite():
    # A real check on the output; the run stays clean of the sanity warning.
    result = run_mutation_testing(_cfg([checks.contains_case("x")]), canary=True)
    assert result.canary_caught is True


def test_off_by_default():
    # Not requested -> not computed -> no interference with the default path.
    result = run_mutation_testing(_cfg([_always_pass]))
    assert result.canary_caught is None
    assert "suite sanity" not in format_report(result, use_color=False)
