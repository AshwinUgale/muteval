"""Mutation operators.

Each operator takes the original prompt and yields zero or more ``Mutant``s —
a deliberately degraded version of the prompt plus a human description of what
was broken. The runner then checks whether the user's eval suite catches the
degradation.

v0 operators are rule-based and deterministic (given a seed) so results are
reproducible and require no API calls to generate. LLM-driven semantic
mutations and non-prompt targets (retrieved context, tool outputs) are on the
roadmap — see ROADMAP in the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class Mutant:
    """A single degraded version of the prompt."""

    operator: str  # which mutation operator produced this
    description: str  # human-readable "what was broken"
    prompt: str  # the mutated prompt text


# --- Operators ---------------------------------------------------------------

# Pairs of (strong -> weak) wordings. Case-insensitive, whole-word matches.
_MODAL_WEAKENINGS = [
    ("must not", "should avoid"),
    ("must", "should"),
    ("never", "rarely"),
    ("always", "usually"),
    ("required", "optional"),
    ("do not", "try not to"),
    ("don't", "try not to"),
    ("strictly", "ideally"),
    ("ensure", "consider"),
    ("only", "preferably"),
]


def weaken_modals(prompt: str) -> List[Mutant]:
    """Soften strong instructions (MUST -> should, never -> rarely, ...).

    Each replaceable occurrence becomes its own mutant so a single missed
    instruction is isolated.
    """
    mutants: List[Mutant] = []
    for strong, weak in _MODAL_WEAKENINGS:
        pattern = re.compile(rf"\b{re.escape(strong)}\b", re.IGNORECASE)
        for match in pattern.finditer(prompt):
            start, end = match.span()
            mutated = prompt[:start] + weak + prompt[end:]
            snippet = _context_snippet(prompt, start, end)
            mutants.append(
                Mutant(
                    operator="weaken_modals",
                    description=f'weakened "{match.group(0)}" -> "{weak}" (near: {snippet})',
                    prompt=mutated,
                )
            )
    return mutants


def drop_instruction_lines(prompt: str) -> List[Mutant]:
    """Delete a single instruction line/bullet at a time.

    Models "someone trimmed the prompt and silently removed a capability."
    """
    lines = prompt.splitlines()
    mutants: List[Mutant] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _is_instruction_line(stripped):
            continue
        remaining = lines[:i] + lines[i + 1 :]
        mutated = "\n".join(remaining)
        mutants.append(
            Mutant(
                operator="drop_instruction_lines",
                description=f'dropped line: "{_truncate(stripped)}"',
                prompt=mutated,
            )
        )
    return mutants


def delete_sentences(prompt: str) -> List[Mutant]:
    """Delete a single sentence at a time (for prose-style prompts)."""
    sentences = _split_sentences(prompt)
    if len(sentences) < 2:
        return []
    mutants: List[Mutant] = []
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) < 12:
            continue
        remaining = sentences[:i] + sentences[i + 1 :]
        mutated = " ".join(s.strip() for s in remaining).strip()
        mutants.append(
            Mutant(
                operator="delete_sentences",
                description=f'deleted sentence: "{_truncate(sentence.strip())}"',
                prompt=mutated,
            )
        )
    return mutants


# Pairs that INVERT meaning — a stronger regression than mere weakening.
_NEGATION_FLIPS = [
    ("must not", "must"),
    ("should not", "should"),
    ("cannot", "can"),
    ("can not", "can"),
    ("do not", "do"),
    ("don't", "do"),
    ("never", "always"),
    ("always", "never"),
]


def flip_negation(prompt: str) -> List[Mutant]:
    """Invert a rule (do not -> do, never -> always).

    A meaning-inverting mutation — a far more dangerous regression than merely
    weakening a modal, so any eval worth its salt should catch it.
    """
    mutants: List[Mutant] = []
    for src, dst in _NEGATION_FLIPS:
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        for match in pattern.finditer(prompt):
            start, end = match.span()
            mutated = prompt[:start] + dst + prompt[end:]
            snippet = _context_snippet(prompt, start, end)
            mutants.append(
                Mutant(
                    operator="flip_negation",
                    description=f'inverted "{match.group(0)}" -> "{dst}" (near: {snippet})',
                    prompt=mutated,
                )
            )
    return mutants


def truncate_prompt(prompt: str) -> List[Mutant]:
    """Cut off the tail of the prompt (lossy truncation).

    Models a prompt that got clipped — by a token budget, a bad edit, or
    context-window pressure — silently dropping its later instructions.
    """
    lines = prompt.splitlines()
    if len(lines) < 4:
        return []
    mutants: List[Mutant] = []
    for frac in (0.5, 0.75):
        keep = max(1, int(len(lines) * frac))
        if keep >= len(lines):
            continue
        mutated = "\n".join(lines[:keep])
        dropped = len(lines) - keep
        mutants.append(
            Mutant(
                operator="truncate_prompt",
                description=(
                    f"truncated prompt — dropped the last {dropped} of "
                    f"{len(lines)} lines"
                ),
                prompt=mutated,
            )
        )
    return mutants


# Markers that signal a few-shot example/demonstration block.
_EXAMPLE_MARKER = re.compile(
    r"(?im)(\bexample\b|input:|output:|^\s*q:|^\s*a:|user:|assistant:)"
)


def drop_few_shot_example(prompt: str) -> List[Mutant]:
    """Remove a single few-shot example block at a time.

    For few-shot prompts: drops one demonstration so you can see whether your
    evals notice degraded in-context guidance.
    """
    blocks = re.split(r"\n\s*\n", prompt)
    if len(blocks) < 2:
        return []
    mutants: List[Mutant] = []
    for i, block in enumerate(blocks):
        if not _EXAMPLE_MARKER.search(block):
            continue
        remaining = blocks[:i] + blocks[i + 1 :]
        mutated = "\n\n".join(remaining).strip()
        mutants.append(
            Mutant(
                operator="drop_few_shot_example",
                description=f'dropped example block: "{_truncate(block.strip())}"',
                prompt=mutated,
            )
        )
    return mutants


def remove_emphasis(prompt: str) -> List[Mutant]:
    """Strip emphasis cues (**bold**, IMPORTANT:/CRITICAL: markers).

    Tests whether your evals are sensitive to the *salience* of instructions,
    not just their presence.
    """
    mutated = re.sub(r"\*\*(.+?)\*\*", r"\1", prompt)
    mutated = re.sub(r"__(.+?)__", r"\1", mutated)
    mutated = re.sub(
        r"(?im)^\s*(IMPORTANT|CRITICAL|NOTE|WARNING|ATTENTION)\b:?\s*", "", mutated
    )
    if mutated == prompt:
        return []
    return [
        Mutant(
            operator="remove_emphasis",
            description="removed emphasis cues (bold / IMPORTANT / CRITICAL markers)",
            prompt=mutated,
        )
    ]


# Registry of all operators. Keyed by name so they can be selected/filtered.
OPERATORS: Dict[str, Callable[[str], List[Mutant]]] = {
    "weaken_modals": weaken_modals,
    "flip_negation": flip_negation,
    "drop_instruction_lines": drop_instruction_lines,
    "delete_sentences": delete_sentences,
    "truncate_prompt": truncate_prompt,
    "drop_few_shot_example": drop_few_shot_example,
    "remove_emphasis": remove_emphasis,
}


def generate_mutants(
    prompt: str,
    operators: List[str] | None = None,
) -> List[Mutant]:
    """Run the selected operators and return a de-duplicated list of mutants."""
    selected = operators or list(OPERATORS.keys())
    seen = set()
    mutants: List[Mutant] = []
    for name in selected:
        op = OPERATORS.get(name)
        if op is None:
            raise ValueError(
                f"Unknown operator '{name}'. Available: {list(OPERATORS)}"
            )
        for mutant in op(prompt):
            # Skip no-ops and exact-duplicate prompts.
            if mutant.prompt == prompt or mutant.prompt in seen:
                continue
            seen.add(mutant.prompt)
            mutants.append(mutant)
    return mutants


# --- helpers -----------------------------------------------------------------


def _is_instruction_line(stripped: str) -> bool:
    if len(stripped) < 8:
        return False
    bullet = re.match(r"^([-*+]|\d+[.)])\s+", stripped)
    return bool(bullet) or stripped.endswith((".", ":", "!"))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p for p in parts if p.strip()]


def _context_snippet(text: str, start: int, end: int, width: int = 24) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = text[left:right].replace("\n", " ").strip()
    return _truncate(snippet, 60)


def _truncate(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
