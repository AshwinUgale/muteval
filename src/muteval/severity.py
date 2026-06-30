"""Severity ranking for mutants — surface the dangerous coverage gaps first.

A survivor where the evals missed an inverted safety rule matters far more than
one where they missed a reordered sentence. Without ranking, the survivor list
is flat and the scary gaps drown in the cosmetic ones.

Severity = the operator's inherent destructiveness, escalated one level when the
mutated text touches safety/correctness-critical language (never / must / refund
/ PII / cite / "I don't know" / ...). Both tables are plain data — override
``OPERATOR_SEVERITY`` or ``CRITICAL_PATTERNS`` for your domain.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

_RANK = {HIGH: 0, MEDIUM: 1, LOW: 2}
_ESCALATE = {LOW: MEDIUM, MEDIUM: HIGH, HIGH: HIGH}

# How destructive is each *kind* of mutation, independent of content?
OPERATOR_SEVERITY = {
    # inverting or replacing meaning / facts — most dangerous
    "flip_negation": HIGH,
    "corrupt_context_doc": HIGH,
    "swap_context_doc": HIGH,
    "corrupt_tool_output": HIGH,
    "swap_tool_output": HIGH,
    "clear_context": HIGH,
    "drop_tool_output": HIGH,
    "downgrade_model": HIGH,
    # removing or weakening instructions / context — meaningful
    "drop_instruction_lines": MEDIUM,
    "delete_sentences": MEDIUM,
    "drop_context_doc": MEDIUM,
    "truncate_prompt": MEDIUM,
    "truncate_context_doc": MEDIUM,
    "drop_few_shot_example": MEDIUM,
    "weaken_modals": MEDIUM,
    # cosmetic / ordering — least likely to matter
    "remove_emphasis": LOW,
    "shuffle_context": LOW,
    "duplicate_context_doc": LOW,
}

# If a mutation's changed text matches any of these, bump severity one level.
CRITICAL_PATTERNS = [
    r"never", r"always", r"must", r"do\s*n.?t", r"cannot", r"refus",
    r"refund", r"privac", r"\bpii\b", r"secur", r"safe", r"polic",
    r"confidential", r"password", r"credential", r"medical", r"legal",
    r"hallucin", r"\bcite\b", r"\bsource", r"don.?t know", r"author",
    r"\bdelete\b", r"customer", r"\bdata\b", r"comply", r"complian",
]
_CRITICAL_RE = re.compile("|".join(CRITICAL_PATTERNS), re.IGNORECASE)


def severity_of(mutant, extra_critical: Optional[Iterable[str]] = None) -> str:
    """Severity for one mutant: operator base, escalated on critical content.

    Content is read from the mutant's human description (which includes the
    changed snippet), so this needs no separate diff.
    """
    base = OPERATOR_SEVERITY.get(getattr(mutant, "operator", ""), MEDIUM)
    text = getattr(mutant, "description", "") or ""
    hit = bool(_CRITICAL_RE.search(text))
    if not hit and extra_critical:
        hit = any(re.search(t, text, re.IGNORECASE) for t in extra_critical)
    return _ESCALATE[base] if hit else base


def severity_rank(sev: str) -> int:
    """Sort key — HIGH sorts first."""
    return _RANK.get(sev, 1)
