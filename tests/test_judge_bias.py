"""v0.6: the judge-bias panel must flag biased judges and clear a fair one.

Validated on constructed synthetic judges whose bias is known by construction.
"""

from __future__ import annotations

from muteval.probes.judge_bias import (
    BiasPanel,
    position_bias,
    run_judge_bias_panel,
    self_preference,
    verbosity_bias,
)


# --- synthetic judges (a, b, case) -> "A"|"B"|"tie" --------------------------

def fair_judge(a, b, case):
    """Picks the answer containing the CORRECT token — content only."""
    ac, bc = "CORRECT" in a, "CORRECT" in b
    if ac and not bc:
        return "A"
    if bc and not ac:
        return "B"
    return "tie"


def position_judge(a, b, case):
    return "A"  # always the first-shown answer -> pure position bias


def verbosity_judge(a, b, case):
    if len(a) == len(b):
        return "tie"
    return "A" if len(a) > len(b) else "B"  # always the longer answer


def selfpref_judge(a, b, case):
    am, bm = "[MINE]" in a, "[MINE]" in b
    if am and not bm:
        return "A"
    if bm and not am:
        return "B"
    return "tie"


# content differs, so a content-judge has a clear (order-invariant) winner.
POSITION_PAIRS = [
    ("the CORRECT answer", "a wrong answer", {}),
    ("nope, not this", "here is the CORRECT one", {}),
    ("CORRECT: 8080", "maybe 9090", {}),
]

# same substance; `long` is `short` padded with filler.
VERBOSITY_PAIRS = [
    ("The CORRECT answer is 42.", "The CORRECT answer is 42. Furthermore, as the documentation thoroughly explains, this holds.", {}),
    ("CORRECT: port 8080.", "CORRECT: port 8080. To be comprehensive and helpful, note this is the documented default value.", {}),
]

# own-model output is tagged; content is otherwise equal.
SELF_PAIRS = [
    ("The answer is 42. [MINE]", "The answer is 42.", {}, ""),
    ("Port 8080. [MINE]", "Port 8080.", {}, ""),
]


def test_position_bias_flags_position_judge_only():
    assert position_bias(fair_judge, POSITION_PAIRS) == 0.0
    assert position_bias(position_judge, POSITION_PAIRS) == 1.0
    # a verbosity judge is order-invariant on length -> no position bias
    assert position_bias(verbosity_judge, POSITION_PAIRS) in (0.0, None)


def test_verbosity_bias_flags_verbosity_judge_only():
    # fair judge calls padded-equal answers a tie -> not assessed / no pref
    assert verbosity_bias(fair_judge, VERBOSITY_PAIRS) in (None, 0.0)
    assert verbosity_bias(verbosity_judge, VERBOSITY_PAIRS) == 1.0


def test_self_preference_flags_selfpref_judge():
    assert self_preference(fair_judge, SELF_PAIRS) in (None, 0.0)
    assert self_preference(selfpref_judge, SELF_PAIRS) == 1.0
    # not assessed without labeled pairs
    assert self_preference(fair_judge, None) is None


def test_panel_ok_only_for_the_fair_judge():
    fair = run_judge_bias_panel(fair_judge, POSITION_PAIRS, VERBOSITY_PAIRS, SELF_PAIRS)
    assert isinstance(fair, BiasPanel)
    assert fair.ok(threshold=0.1) is True

    biased = run_judge_bias_panel(position_judge, POSITION_PAIRS, VERBOSITY_PAIRS, SELF_PAIRS)
    assert biased.ok(threshold=0.1) is False
    assert biased.position_bias == 1.0
