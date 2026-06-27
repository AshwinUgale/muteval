"""Mutation operators.

Each operator takes the mutation target (a ``System`` — or a bare prompt string,
which is promoted to ``System(prompt=...)`` for backward compatibility) and
yields zero or more ``Mutant``s: a deliberately degraded ``System`` plus a
human description of what was broken. The runner then checks whether the user's
eval suite catches the degradation.

Prompt operators are rule-based and deterministic so results are reproducible
and need no API calls. Context operators (``drop_context_doc``, ``clear_context``)
mutate the *retrieved context* of a RAG system and only fire when the target
actually carries context — the first step of the roadmap beyond prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from muteval.system import System, Target, as_system


@dataclass(frozen=True)
class Mutant:
    """A single degraded version of the system under test."""

    operator: str  # which mutation operator produced this
    description: str  # human-readable "what was broken"
    system: System  # the fully mutated system
    target: str = "prompt"  # which part of the system was mutated

    @property
    def prompt(self) -> str:
        """The mutated prompt (back-compat shortcut for ``system.prompt``)."""
        return self.system.prompt


# --- Prompt operators --------------------------------------------------------

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


def weaken_modals(target: Target) -> List[Mutant]:
    """Soften strong instructions (MUST -> should, never -> rarely, ...).

    Each replaceable occurrence becomes its own mutant so a single missed
    instruction is isolated.
    """
    system = as_system(target)
    prompt = system.prompt
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
                    system=system.with_prompt(mutated),
                )
            )
    return mutants


def drop_instruction_lines(target: Target) -> List[Mutant]:
    """Delete a single instruction line/bullet at a time.

    Models "someone trimmed the prompt and silently removed a capability."
    """
    system = as_system(target)
    lines = system.prompt.splitlines()
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
                system=system.with_prompt(mutated),
            )
        )
    return mutants


def delete_sentences(target: Target) -> List[Mutant]:
    """Delete a single sentence at a time (for prose-style prompts)."""
    system = as_system(target)
    sentences = _split_sentences(system.prompt)
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
                system=system.with_prompt(mutated),
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


def flip_negation(target: Target) -> List[Mutant]:
    """Invert a rule (do not -> do, never -> always).

    A meaning-inverting mutation — a far more dangerous regression than merely
    weakening a modal, so any eval worth its salt should catch it.
    """
    system = as_system(target)
    prompt = system.prompt
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
                    system=system.with_prompt(mutated),
                )
            )
    return mutants


def truncate_prompt(target: Target) -> List[Mutant]:
    """Cut off the tail of the prompt (lossy truncation).

    Models a prompt that got clipped — by a token budget, a bad edit, or
    context-window pressure — silently dropping its later instructions.
    """
    system = as_system(target)
    lines = system.prompt.splitlines()
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
                system=system.with_prompt(mutated),
            )
        )
    return mutants


# Markers that signal a few-shot example/demonstration block.
_EXAMPLE_MARKER = re.compile(
    r"(?im)(\bexample\b|input:|output:|^\s*q:|^\s*a:|user:|assistant:)"
)


def drop_few_shot_example(target: Target) -> List[Mutant]:
    """Remove a single few-shot example block at a time.

    For few-shot prompts: drops one demonstration so you can see whether your
    evals notice degraded in-context guidance.
    """
    system = as_system(target)
    blocks = re.split(r"\n\s*\n", system.prompt)
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
                system=system.with_prompt(mutated),
            )
        )
    return mutants


def remove_emphasis(target: Target) -> List[Mutant]:
    """Strip emphasis cues (**bold**, IMPORTANT:/CRITICAL: markers).

    Tests whether your evals are sensitive to the *salience* of instructions,
    not just their presence.
    """
    system = as_system(target)
    prompt = system.prompt
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
            system=system.with_prompt(mutated),
        )
    ]


# --- Context operators (RAG) -------------------------------------------------
# These only fire when the target actually carries retrieved context, so they
# are no-ops for plain prompt-only systems (and never affect legacy configs).


def drop_context_doc(target: Target) -> List[Mutant]:
    """Drop a single retrieved document at a time.

    Models a retriever that silently lost a relevant doc. If your suite still
    passes, your evals don't actually depend on retrieval quality.
    """
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    mutants: List[Mutant] = []
    for i, doc in enumerate(docs):
        remaining = docs[:i] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="drop_context_doc",
                description=f'dropped retrieved doc #{i + 1}: "{_truncate(doc)}"',
                system=system.replace(context=tuple(remaining)),
                target="context",
            )
        )
    return mutants


def clear_context(target: Target) -> List[Mutant]:
    """Remove ALL retrieved context (simulate total retrieval failure)."""
    system = as_system(target)
    if not system.context:
        return []
    return [
        Mutant(
            operator="clear_context",
            description=(
                f"cleared all retrieved context "
                f"(dropped {len(system.context)} doc(s))"
            ),
            system=system.replace(context=()),
            target="context",
        )
    ]


# Registry of all operators. Keyed by name so they can be selected/filtered.
OPERATORS: Dict[str, Callable[[Target], List[Mutant]]] = {
    "weaken_modals": weaken_modals,
    "flip_negation": flip_negation,
    "drop_instruction_lines": drop_instruction_lines,
    "delete_sentences": delete_sentences,
    "truncate_prompt": truncate_prompt,
    "drop_few_shot_example": drop_few_shot_example,
    "remove_emphasis": remove_emphasis,
    "drop_context_doc": drop_context_doc,
    "clear_context": clear_context,
}


def generate_mutants(
    target: Target,
    operators: List[str] | None = None,
) -> List[Mutant]:
    """Run the selected operators and return a de-duplicated list of mutants."""
    original = as_system(target)
    original_key = original.key()
    selected = operators or list(OPERATORS.keys())
    seen = set()
    mutants: List[Mutant] = []
    for name in selected:
        op = OPERATORS.get(name)
        if op is None:
            raise ValueError(
                f"Unknown operator '{name}'. Available: {list(OPERATORS)}"
            )
        for mutant in op(original):
            mkey = mutant.system.key()
            # Skip no-ops (identical to the original) and exact duplicates.
            if mkey == original_key or mkey in seen:
                continue
            seen.add(mkey)
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
