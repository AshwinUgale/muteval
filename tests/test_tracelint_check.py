"""checks.tracelint() + checks.on_final() — the {"final","trace"} bridge.

The whole story of the muteval<->tracelint seam in four assertions, offline and
keyless. A tool output mutated into a 200-carrying-`declined` fault leaves the
final answer looking clean; the user's semantic eval misses it; only tracelint's
declared-failure contract (`failure_when`) kills the mutant.

Skips cleanly when the optional `muteval[tracelint]` extra isn't installed.
"""

import json

import pytest

from muteval import checks

pytest.importorskip("tracelint")  # requires the muteval[tracelint] extra


# Tool ground truth: charge_card declares what failure looks like in its result,
# so a domain failure returned as transport success (status:ok body:declined) is
# caught structurally instead of slipping through.
REG = {
    "tools": {
        "charge_card": {
            "metadata": {
                "side_effecting": True,
                "failure_when": {"pointer": "/status", "in": ["declined", "failed"]},
            }
        }
    }
}

_STEPS_OK = [
    {"type": "message", "role": "user", "content": "Charge $50 for order A100."},
    {
        "type": "tool_call",
        "call_id": "c1",
        "name": "charge_card",
        "args": {"order_id": "A100", "amount": 50},
    },
    {
        "type": "tool_result",
        "call_id": "c1",
        "content": {"status": "succeeded", "receipt": "R1"},
        "status": "ok",
    },
    {"type": "message", "role": "assistant", "content": "Your payment was successful."},
]

_STEPS_DENIED = [
    {"type": "message", "role": "user", "content": "Charge $50 for order A100."},
    {
        "type": "tool_call",
        "call_id": "c1",
        "name": "charge_card",
        "args": {"order_id": "A100", "amount": 50},
    },
    {
        # The fault: transport succeeded (status ok), but the body says declined.
        "type": "tool_result",
        "call_id": "c1",
        "content": {"status": "declined", "reason": "insufficient_funds"},
        "status": "ok",
    },
    # ...and the agent naively proceeds — final is identical to baseline. That
    # identical answer is the honest silent failure.
    {"type": "message", "role": "assistant", "content": "Your payment was successful."},
]

BASELINE = {
    "final": "Your payment was successful.",
    "trace": {"run_id": "b", "steps": _STEPS_OK, "final": "Your payment was successful."},
}

MUTANT = {
    "final": "Your payment was successful.",
    "trace": {
        "run_id": "m",
        "steps": _STEPS_DENIED,
        "final": "Your payment was successful.",
    },
}


def test_baseline_is_clean():
    # Baseline trace has no fault -> nothing flagged -> baseline stays green.
    assert checks.tracelint(registry=REG)(BASELINE, {}).passed is True


def test_mutant_killed_by_declared_contract():
    # The declined charge trips R2a (hard_event); fail_on="hard" catches it.
    assert checks.tracelint(registry=REG)(MUTANT, {}).passed is False


def test_mutant_survives_without_contract():
    # No registry -> no failure_when -> the declined fault is invisible: it
    # survives. This is the gap the declared contract closes.
    assert checks.tracelint(registry=None)(MUTANT, {}).passed is True


def test_semantic_eval_misses_the_fault():
    # The final answer still reads clean, so the user's output-grading eval
    # passes on the mutant — the honest contrast that makes the survivor real.
    assert checks.on_final(checks.contains("successful"))(MUTANT, {}).passed is True


def test_bridge_accepts_json_string():
    # run() may return the bridge object as a JSON string, not just a dict.
    assert checks.tracelint(registry=REG)(json.dumps(MUTANT), {}).passed is False
    assert (
        checks.on_final(checks.contains("successful"))(json.dumps(MUTANT), {}).passed
        is True
    )


def test_fail_on_defect_would_miss_the_hard_event():
    # A declared-failure is a hard_event, NOT a hard_defect: fail_on="defect"
    # (has_hard_defect only) wrongly lets the mutant survive. This is exactly why
    # "hard" is the default.
    assert checks.tracelint(registry=REG, fail_on="defect")(MUTANT, {}).passed is True


def test_invalid_fail_on_rejected():
    with pytest.raises(ValueError):
        checks.tracelint(fail_on="nonsense")
