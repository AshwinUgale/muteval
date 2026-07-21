"""Turn a survivor into a concrete starter eval that would catch it.

A survivor tells you a gap exists; this tells you what to DO about it — a starter
check (an LLM-judge rubric for prompt rules, or a structural eval for
context/tool/model mutations) that would kill that mutant. Operator-aware.
"""

from __future__ import annotations

import re

# Structural mutations imply a specific *kind* of missing eval.
_STRUCTURAL = {
    "drop_context_doc": 'add a groundedness eval — the answer must contain/cite the '
    'retrieved fact (e.g. checks.contains("<key fact>"))',
    "truncate_context_doc": "add a groundedness eval for the retrieved fact that was cut",
    "clear_context": 'add an eval that the answer refuses / says "I don\'t know" when '
    "no context is retrieved",
    "corrupt_context_doc": "add a CORRECTNESS eval against ground truth — faithfulness "
    "happily passes a wrong-but-grounded answer",
    "swap_context_doc": "add an eval tying the answer to the expected source/fact",
    "downgrade_model": 'add a quality-floor eval (e.g. checks.llm_judge("the answer is '
    'complete and accurate")) so a weaker model is caught',
    "drop_tool_output": "add an eval that the answer actually reflects the tool result",
    "corrupt_tool_output": "add a correctness eval on tool-derived facts",
    "swap_tool_output": "add an eval tying the answer to the correct tool result",
}


def _short(s: str, n: int = 52) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _quoted(desc: str):
    m = re.search(r'"([^"]{6,}?)"', desc or "")
    return m.group(1).strip() if m else None


def _near(desc: str):
    m = re.search(r"near:\s*(.+?)\)\s*$", desc or "")
    return m.group(1).strip() if m else None


def suggest_eval(outcome) -> str:
    """A one-line starter eval that would catch this survivor."""
    op = getattr(outcome.mutant, "operator", "")
    desc = getattr(outcome.mutant, "description", "") or ""

    if op in _STRUCTURAL:
        return _STRUCTURAL[op]

    if op in ("drop_instruction_lines", "delete_sentences"):
        phrase = _quoted(desc)
        if phrase:
            return f'add checks.llm_judge("the reply still follows: {_short(phrase)}")'

    if op in ("flip_negation", "weaken_modals"):
        near = _near(desc)
        if near:
            return (
                f'add an eval for the rule near "{_short(near)}" '
                "(e.g. a checks.llm_judge for that behavior)"
            )

    if op == "truncate_prompt":
        return "add checks for the instructions in the dropped tail of the prompt"
    if op == "drop_few_shot_example":
        return "add an eval that the format the examples taught still holds"
    if op == "remove_emphasis":
        return "add an eval for the de-emphasized behavior"

    return "add an eval that checks the behavior this mutation changed"
