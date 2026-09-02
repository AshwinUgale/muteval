"""weaken_numeric_threshold — loosen a numeric constraint in the prompt.

Motivated by a reader point: the mutation score is confounded by the operator
mix, so a discriminating suite needs operators aimed at behaviors the current
checks say nothing about — a loosened numeric threshold is one (a faithfulness /
relevancy suite has no opinion on it).
"""

from types import SimpleNamespace

from muteval.mutators import OPERATORS, generate_mutants
from muteval.severity import MEDIUM, severity_of
from muteval.suggest import suggest_eval
from muteval.system import System


def test_registered():
    assert "weaken_numeric_threshold" in OPERATORS


def test_loosens_upper_bound():
    muts = generate_mutants(
        System(prompt="Summarize in at most 3 sentences."),
        operators=["weaken_numeric_threshold"],
    )
    assert len(muts) == 1
    assert "at most 6 sentences" in muts[0].system.prompt
    assert "3 -> 6" in muts[0].description


def test_loosens_lower_bound():
    muts = generate_mutants(
        System(prompt="Provide at least 4 citations."),
        operators=["weaken_numeric_threshold"],
    )
    assert len(muts) == 1
    assert "at least 2 citations" in muts[0].system.prompt


def test_time_window_is_a_bound():
    muts = generate_mutants(
        System(prompt="Respond within 24 hours."),
        operators=["weaken_numeric_threshold"],
    )
    assert len(muts) == 1
    assert "within 48 hours" in muts[0].system.prompt


def test_skips_numbers_without_a_bound_word():
    # a version string / bare number with no threshold context -> no mutant
    assert (
        generate_mutants(
            System(prompt="You are running on model gpt-4o. Answer the question."),
            operators=["weaken_numeric_threshold"],
        )
        == []
    )


def test_severity_is_medium():
    m = generate_mutants(
        System(prompt="Use at most 5 tools."), operators=["weaken_numeric_threshold"]
    )[0]
    assert severity_of(m) == MEDIUM


def test_suggested_fix_mentions_the_limit():
    s = suggest_eval(
        SimpleNamespace(
            mutant=SimpleNamespace(
                operator="weaken_numeric_threshold",
                description="loosened threshold 3 -> 6",
            )
        )
    )
    assert "numeric limit" in s
