"""A4: operator factories (parametrized operators) + register_operator combo."""

import warnings

import pytest

from muteval import make_downgrade_model, make_weaken_modals, register_operator
from muteval.mutators import OPERATORS, generate_mutants
from muteval.system import System


def test_make_weaken_modals_custom_pairs():
    op = make_weaken_modals([("shall", "may")])
    ms = op("You shall not pass.")
    assert len(ms) == 1
    assert "may" in ms[0].prompt


def test_make_downgrade_model_custom_ladder():
    op = make_downgrade_model(["big", "small", "tiny"])
    ms = op(System(prompt="p", model="big"))
    assert {m.system.model for m in ms} == {"small", "tiny"}
    assert op(System(prompt="p", model="tiny")) == []


def test_make_downgrade_model_refuses_off_ladder_model():
    # A model NOT in the supplied ladder must NOT be mapped onto it (that could
    # be an UPGRADE). Warn and emit nothing — same rule as the built-in.
    op = make_downgrade_model(["gpt-4o", "gpt-4o-mini"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ms = op(System(prompt="p", model="tiny-local"))
    assert ms == []
    assert any("not in the supplied ladder" in str(w.message) for w in caught)


def test_make_downgrade_model_validates_ladder():
    with pytest.raises(ValueError):
        make_downgrade_model(["only-one"])          # needs >= 2
    with pytest.raises(ValueError):
        make_downgrade_model(["a", "a"])            # no duplicates


def test_factory_registers_and_runs_by_name():
    register_operator("weaken_shall", make_weaken_modals([("shall", "may")]))
    try:
        ms = generate_mutants("You shall comply.", operators=["weaken_shall"])
        assert ms and "may" in ms[0].prompt
    finally:
        del OPERATORS["weaken_shall"]
