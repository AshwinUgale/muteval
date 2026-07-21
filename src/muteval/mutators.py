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
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

from muteval.scope import Scope, filter_mutants
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


# --- More context operators (B2): corrupt / swap / shuffle / duplicate / truncate

_IRRELEVANT_DOC = (
    "Reminder: the office cafeteria serves lunch from 12:00 to 13:00 on weekdays."
)


def _corrupt_doc(doc: str) -> "str | None":
    """Deterministic, rule-based corruption: change the first number, else flip a
    polarity verb. Returns a plausible-but-wrong doc, or None if uncorruptible."""
    m = re.search(r"\d+", doc)
    if m:
        n = int(m.group(0))
        wrong = str(n + 1 if n != 0 else 9)
        return doc[: m.start()] + wrong + doc[m.end() :]
    flip = re.search(r"\b(is|are|was|were|can|will|does|do)\b", doc)
    if flip:
        return doc[: flip.end()] + " not" + doc[flip.end() :]
    return None


def corrupt_context_doc(target: Target) -> List[Mutant]:
    """Inject a plausible-but-wrong fact into one retrieved doc at a time.

    Tests whether your evals catch a retriever returning subtly wrong content —
    the answer may now be confidently incorrect."""
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    mutants: List[Mutant] = []
    for i, doc in enumerate(docs):
        bad = _corrupt_doc(doc)
        if bad is None or bad == doc:
            continue
        new = docs[:i] + [bad] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="corrupt_context_doc",
                description=f'corrupted retrieved doc #{i + 1}: "{_truncate(bad)}"',
                system=system.replace(context=tuple(new)),
                target="context",
            )
        )
    return mutants


def swap_context_doc(target: Target) -> List[Mutant]:
    """Replace one retrieved doc with an irrelevant one (a bad retrieval hit)."""
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    mutants: List[Mutant] = []
    for i in range(len(docs)):
        if docs[i] == _IRRELEVANT_DOC:
            continue
        new = docs[:i] + [_IRRELEVANT_DOC] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="swap_context_doc",
                description=f"swapped retrieved doc #{i + 1} for an irrelevant doc",
                system=system.replace(context=tuple(new)),
                target="context",
            )
        )
    return mutants


def shuffle_context(target: Target) -> List[Mutant]:
    """Reverse the order of retrieved docs (tests position sensitivity)."""
    system = as_system(target)
    if not system.context or len(system.context) < 2:
        return []
    reordered = tuple(reversed(system.context))
    if reordered == system.context:
        return []
    return [
        Mutant(
            operator="shuffle_context",
            description=f"reversed the order of {len(system.context)} retrieved docs",
            system=system.replace(context=reordered),
            target="context",
        )
    ]


def duplicate_context_doc(target: Target) -> List[Mutant]:
    """Duplicate one retrieved doc (adds redundant noise to the context)."""
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    mutants: List[Mutant] = []
    for i, doc in enumerate(docs):
        new = docs[: i + 1] + [doc] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="duplicate_context_doc",
                description=f'duplicated retrieved doc #{i + 1}: "{_truncate(doc)}"',
                system=system.replace(context=tuple(new)),
                target="context",
            )
        )
    return mutants


def truncate_context_doc(target: Target) -> List[Mutant]:
    """Clip the tail of one retrieved doc (a chunk that got cut off)."""
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    mutants: List[Mutant] = []
    for i, doc in enumerate(docs):
        words = doc.split()
        if len(words) < 6:
            continue
        keep = max(1, len(words) // 2)
        clipped = " ".join(words[:keep])
        if clipped == doc:
            continue
        new = docs[:i] + [clipped] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="truncate_context_doc",
                description=f"truncated retrieved doc #{i + 1} to its first {keep} words",
                system=system.replace(context=tuple(new)),
                target="context",
            )
        )
    return mutants


# --- Model-swap operator (B3) ------------------------------------------------

# Strong -> weak ladder. downgrade_model emits a mutant for each model STRICTLY
# WEAKER than the System's current model, using this known ordering. It only
# fires when System.model is set AND the current model is on a known ladder —
# we never *guess* that an arbitrary model is stronger/weaker than another.
_MODEL_LADDER = ("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")


def downgrade_model(target: Target) -> List[Mutant]:
    """Swap the model for a weaker one (does your suite notice a cheaper model?).

    No-op unless ``System.model`` is set (so it never fires on prompt-only runs).

    Conservative by design: if the current model is NOT on muteval's known
    ladder (``_MODEL_LADDER``), we do NOT invent an ordering — guessing that
    e.g. ``gpt-3.5-turbo`` is a "downgrade" from an unknown model could be flat
    wrong. Instead we warn and emit nothing; pass your own strong->weak ladder
    via ``make_downgrade_model([...])`` to test provider-specific downgrades.
    """
    system = as_system(target)
    current = system.model
    if not current:
        return []
    if current not in _MODEL_LADDER:
        warnings.warn(
            f"downgrade_model: model {current!r} is not on muteval's known "
            f"ladder {_MODEL_LADDER}; refusing to guess a downgrade. Use "
            f"make_downgrade_model([...]) with your own strong->weak ladder.",
            stacklevel=2,
        )
        return []
    weaker = _MODEL_LADDER[_MODEL_LADDER.index(current) + 1 :]
    mutants: List[Mutant] = []
    for model in weaker:
        mutants.append(
            Mutant(
                operator="downgrade_model",
                description=f"downgraded model {current} -> {model}",
                system=system.replace(model=model),
                target="model",
            )
        )
    return mutants


# --- Tool-output operators (B4, agents) --------------------------------------
# System.tools is treated as a tuple of tool OUTPUTS (strings). These fire only
# when tools are present. Note: the built-in openai_run does not inject tools —
# agent pipelines consume system.tools via their own run(system, case).

_IRRELEVANT_TOOL = "tool_result: {\"status\": \"ok\", \"data\": \"unrelated\"}"


def drop_tool_output(target: Target) -> List[Mutant]:
    """Drop one tool output at a time (a tool silently returned nothing)."""
    system = as_system(target)
    if not system.tools:
        return []
    tools = list(system.tools)
    mutants: List[Mutant] = []
    for i in range(len(tools)):
        new = tools[:i] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="drop_tool_output",
                description=f'dropped tool output #{i + 1}: "{_truncate(str(tools[i]))}"',
                system=system.replace(tools=tuple(new)),
                target="tools",
            )
        )
    return mutants


def corrupt_tool_output(target: Target) -> List[Mutant]:
    """Corrupt one tool output (a tool returned a plausible-but-wrong result)."""
    system = as_system(target)
    if not system.tools:
        return []
    tools = list(system.tools)
    mutants: List[Mutant] = []
    for i, tool in enumerate(tools):
        bad = _corrupt_doc(str(tool))
        if bad is None or bad == str(tool):
            continue
        new = tools[:i] + [bad] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="corrupt_tool_output",
                description=f'corrupted tool output #{i + 1}: "{_truncate(bad)}"',
                system=system.replace(tools=tuple(new)),
                target="tools",
            )
        )
    return mutants


def swap_tool_output(target: Target) -> List[Mutant]:
    """Replace one tool output with an irrelevant one (wrong tool / stale call)."""
    system = as_system(target)
    if not system.tools:
        return []
    tools = list(system.tools)
    mutants: List[Mutant] = []
    for i in range(len(tools)):
        if tools[i] == _IRRELEVANT_TOOL:
            continue
        new = tools[:i] + [_IRRELEVANT_TOOL] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="swap_tool_output",
                description=f"swapped tool output #{i + 1} for an irrelevant result",
                system=system.replace(tools=tuple(new)),
                target="tools",
            )
        )
    return mutants


# --- Operator factories (A4): parametrize built-in operators ----------------
# Combine with register_operator to add a tuned variant, e.g.
#   register_operator("weaken_modals_eu", make_weaken_modals([("shall","may")]))


def make_weaken_modals(pairs: "List[tuple]") -> "Callable[[Target], List[Mutant]]":
    """Build a weaken_modals-style operator with custom (strong, weak) pairs."""

    def op(target: Target) -> List[Mutant]:
        system = as_system(target)
        prompt = system.prompt
        mutants: List[Mutant] = []
        for strong, weak in pairs:
            pattern = re.compile(rf"\b{re.escape(strong)}\b", re.IGNORECASE)
            for match in pattern.finditer(prompt):
                start, end = match.span()
                mutated = prompt[:start] + weak + prompt[end:]
                mutants.append(
                    Mutant(
                        operator="weaken_modals",
                        description=f'weakened "{match.group(0)}" -> "{weak}"',
                        system=system.with_prompt(mutated),
                    )
                )
        return mutants

    op.__name__ = "weaken_modals_custom"
    return op


def make_downgrade_model(ladder: "List[str]") -> "Callable[[Target], List[Mutant]]":
    """Build a downgrade_model operator with a custom strong->weak model ladder.

    Conservative, like the built-in: if the current model is NOT in ``ladder``,
    no downgrade can be inferred (guessing could produce an *upgrade*), so it
    warns and emits nothing.
    """
    if len(ladder) < 2:
        raise ValueError("model ladder must contain at least two models")
    if len(ladder) != len(set(ladder)):
        raise ValueError("model ladder must not contain duplicates")

    def op(target: Target) -> List[Mutant]:
        system = as_system(target)
        current = system.model
        if not current:
            return []
        if current not in ladder:
            warnings.warn(
                f"downgrade_model: model {current!r} is not in the supplied "
                f"ladder {tuple(ladder)}; no downgrade can be inferred.",
                stacklevel=2,
            )
            return []
        weaker = ladder[ladder.index(current) + 1 :]
        return [
            Mutant(
                operator="downgrade_model",
                description=f"downgraded model {current} -> {m}",
                system=system.replace(model=m),
                target="model",
            )
            for m in weaker
        ]

    op.__name__ = "downgrade_model_custom"
    return op


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
    "corrupt_context_doc": corrupt_context_doc,
    "swap_context_doc": swap_context_doc,
    "shuffle_context": shuffle_context,
    "duplicate_context_doc": duplicate_context_doc,
    "truncate_context_doc": truncate_context_doc,
    "downgrade_model": downgrade_model,
    "drop_tool_output": drop_tool_output,
    "corrupt_tool_output": corrupt_tool_output,
    "swap_tool_output": swap_tool_output,
}


def register_operator(name: str, fn: "Callable[[Target], List[Mutant]]") -> "Callable[[Target], List[Mutant]]":
    """Register a custom mutation operator under ``name`` so it runs by default
    and can be selected via ``--operators name`` / ``operators=[name]``.

    The operator is ``fn(target) -> list[Mutant]`` (``target`` is a System or a
    bare prompt string; use ``as_system(target)``). Returns ``fn`` so it can be
    used as a decorator. Bring-your-own operators never touch the eval suite —
    they only produce mutated Systems, preserving muteval's orthogonality.
    """
    OPERATORS[name] = fn
    return fn


def generate_mutants(
    target: Target,
    operators: "List[str | Callable] | None" = None,
    scope: "Scope | None" = None,
) -> List[Mutant]:
    """Run the selected operators and return a de-duplicated list of mutants.

    ``operators`` items may be registered operator NAMES (str) or operator
    CALLABLES (``fn(target) -> list[Mutant]``) for bring-your-own operators.
    """
    original = as_system(target)
    original_key = original.key()
    selected = operators if operators is not None else list(OPERATORS.keys())
    seen = set()
    mutants: List[Mutant] = []
    for item in selected:
        if callable(item):
            op = item
        else:
            op = OPERATORS.get(item)
            if op is None:
                raise ValueError(
                    f"Unknown operator '{item}'. Available: {list(OPERATORS)}"
                )
        for mutant in op(original):
            mkey = mutant.system.key()
            # Skip no-ops (identical to the original) and exact duplicates.
            if mkey == original_key or mkey in seen:
                continue
            seen.add(mkey)
            mutants.append(mutant)
    return filter_mutants(original.prompt, mutants, scope)


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
