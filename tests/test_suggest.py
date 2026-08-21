"""Suggested-eval-per-survivor: each survivor gets an actionable starter check."""

from types import SimpleNamespace

from muteval.suggest import suggest_eval


def _survivor(operator, description):
    return SimpleNamespace(
        mutant=SimpleNamespace(operator=operator, description=description)
    )


def test_context_corruption_suggests_correctness_eval():
    s = suggest_eval(_survivor("corrupt_context_doc", "corrupted retrieved doc #1"))
    assert "CORRECTNESS" in s


def test_downgrade_model_suggests_quality_floor():
    s = suggest_eval(_survivor("downgrade_model", "downgraded model to gpt-3.5"))
    assert "quality-floor" in s


def test_dropped_line_suggestion_names_the_rule():
    s = suggest_eval(
        _survivor("drop_instruction_lines", 'dropped line: "You must cite the order ID."')
    )
    assert "llm_judge" in s
    assert "cite the order ID" in s


def test_unknown_operator_gives_generic_suggestion():
    s = suggest_eval(_survivor("mystery_op", "did something"))
    assert "checks the behavior" in s


def test_tool_fault_suggests_tracelint_and_gate_holds():
    # A tool-fault survivor names the deterministic check that kills it...
    fix = suggest_eval(
        _survivor("deny_tool_output", "tool output #1 returned a domain failure")
    )
    assert "checks.tracelint" in fix
    assert "failure_when" in fix
    # ...and the suggestion never leaks onto prompt/context survivors.
    prompt_fix = suggest_eval(
        _survivor("drop_instruction_lines", 'dropped line: "You must cite the order ID."')
    )
    assert "checks.tracelint" not in prompt_fix
