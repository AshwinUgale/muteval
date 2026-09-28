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
    "deny_tool_output": HIGH,
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
    "weaken_numeric_threshold": MEDIUM,
    # cosmetic / ordering — least likely to matter
    "remove_emphasis": LOW,
    "shuffle_context": LOW,
    "duplicate_context_doc": LOW,
    # meaning-preserving ("robustness") operators — never scored as coverage
    # gaps (see mutators.OPERATOR_INTENT); listed so they don't fall back to MEDIUM
    "paraphrase_instruction": LOW,
    "swap_adjacent_instructions": LOW,
}

# If the text a mutation acted on (its line / sentence / doc — ``Mutant.focus``)
# is about safety or correctness, bump severity one level. CONTENT words only,
# with word boundaries: the modal words themselves (never / always / must /
# do not / cannot) used to be here, but every weaken/flip/drop of a modal line
# contains them, so they escalated nearly everything to HIGH ("Never use emojis"
# ranked with corrupted retrieval) and matched inside words ("mustard",
# "nevertheless", "tornado nothing").
CRITICAL_PATTERNS = [
    r"\brefus",
    r"\brefund",
    r"\bprivacy\b|\bprivate\b",
    r"\bpii\b",
    r"\bpersonal (?:data|information|details)",
    r"\b(?:customer|user|patient|client)s?['’]? (?:data|information|details|records)",
    r"\bsecur(?:e|ity)\b",
    r"\b(?:un)?safe(?:ty)?\b",
    r"\bpolic(?:y|ies)\b",
    r"\bconfidential",
    r"\bpassword",
    r"\bcredential",
    r"\bsecrets?\b",
    r"\bmedical\b|\bdiagnos",
    r"\blegal(?:ly)?\b",
    r"\bhallucinat",
    # hallucination rules said plainly ("do not guess", "never invent facts")
    r"\bguess(?:es|ing)?\b|\binvent(?:s|ed|ing)?\b|\bfabricat"
    r"|\bmake (?:things |anything )?up\b",
    r"\bcit(?:e|es|ing|ation|ations)\b",
    r"\bsources?\b",
    r"(?:don['’]?t|do not) know",
    r"\b(?:un)?authori[sz]",
    r"\bdelet(?:e|es|ing|ion)\b",
    r"\bcompl(?:y|ies|iance|iant)\b",
    r"\binjection\b|\bjailbreak",
    r"\bpayments?\b|\bcharge[sd]?\b",
]
_CRITICAL_RE = re.compile("|".join(CRITICAL_PATTERNS), re.IGNORECASE)


def severity_of(mutant, extra_critical: Optional[Iterable[str]] = None) -> str:
    """Severity for one mutant: operator base, escalated on critical content.

    Content is read from ``mutant.focus`` — the ORIGINAL text the mutation
    acted on (the weakened line, the dropped sentence, the corrupted doc) —
    not the description, which also carries the operator's trigger word and
    neighboring text. Custom mutants without a focus fall back to the
    description.
    """
    base = OPERATOR_SEVERITY.get(getattr(mutant, "operator", ""), MEDIUM)
    text = getattr(mutant, "focus", "") or getattr(mutant, "description", "") or ""
    hit = bool(_CRITICAL_RE.search(text))
    if not hit and extra_critical:
        hit = any(re.search(t, text, re.IGNORECASE) for t in extra_critical)
    return _ESCALATE[base] if hit else base


def severity_rank(sev: str) -> int:
    """Sort key — HIGH sorts first."""
    return _RANK.get(sev, 1)
