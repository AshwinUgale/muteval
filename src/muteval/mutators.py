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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from muteval.scope import Scope, filter_mutants
from muteval.system import System, Target, as_system


@dataclass(frozen=True)
class Mutant:
    """A single degraded version of the system under test."""

    operator: str  # which mutation operator produced this
    description: str  # human-readable "what was broken"
    system: System  # the fully mutated system
    target: str = "prompt"  # which part of the system was mutated
    # The ORIGINAL text this mutation acted on (its line / sentence / doc), for
    # severity ranking. Not part of the signature. "" = unknown (custom ops).
    focus: str = field(default="", compare=False)

    @property
    def prompt(self) -> str:
        """The mutated prompt (back-compat shortcut for ``system.prompt``)."""
        return self.system.prompt

    @property
    def signature(self) -> str:
        """A stable id for this specific mutation (operator + the exact edit),
        used to ACCEPT a survivor so it doesn't resurface as actionable on later
        runs. Tied to the change text, so editing that part of the prompt yields a
        new signature (the accepted mutation correctly re-surfaces); a deterministic
        operator produces the same signature every run."""
        import hashlib

        return hashlib.sha256(f"{self.operator}|{self.description}".encode()).hexdigest()[
            :12
        ]

    @property
    def intent(self) -> str:
        """``"regression"`` (scored) or ``"robustness"`` (meaning-preserving,
        reported separately, never scored). See ``OPERATOR_INTENT``."""
        return intent_of(self.operator)


# --- Prompt operators --------------------------------------------------------

# Pairs of (strong -> weak) wordings. Case-insensitive, whole-word matches.
_MODAL_WEAKENINGS = [
    # Longer phrases first: a shorter pair never re-matches inside a span a
    # longer one already weakened ("must" inside "must not" used to yield a
    # second, overlapping mutant).
    ("must not", "should not"),  # "should avoid share X" wasn't English
    ("must", "should"),
    ("never", "rarely"),
    ("always", "usually"),
    ("required to", "encouraged to"),  # "You are optional to" wasn't English
    ("required", "optional"),
    ("do not", "try not to"),
    ("don't", "try not to"),
    ("strictly", "ideally"),
    ("ensure", "consider"),
    ("only", "preferably"),
]


# Weakenings that only make sense addressed to an agent: "try not to" needs an
# imperative ("Do not X" -> "Try not to X"). Applied to a descriptive clause
# ("small changes do not make a PR out-of-scope") it yields nonsense ("small
# changes try not to make ..."), a mutant no real edit could produce.
_IMPERATIVE_ONLY = {"do not", "don't"}

# "only" as a restriction ("use only X", "only answer ...") weakens to
# "preferably"; as a determiner or in a fixed phrase it doesn't: "the only
# exception", "not only ... but", "if and only if".
_ONLY_SKIP_BEFORE = re.compile(
    r"\b(?:the|a|an|and|not|our|your|its|their|my|his|her|one)\s+\Z", re.IGNORECASE
)
_ONLY_SKIP_AFTER = re.compile(r"\A\s+if\b", re.IGNORECASE)


def _overlaps(used: "List[Tuple[int, int]]", start: int, end: int) -> bool:
    return any(s < end and start < e for s, e in used)


def _weaken(system: System, pairs, snippets: bool) -> List[Mutant]:
    prompt = system.prompt
    mutants: List[Mutant] = []
    used: List[Tuple[int, int]] = []
    for strong, weak in pairs:
        pattern = re.compile(rf"\b{_apostrophes(re.escape(strong))}\b", re.IGNORECASE)
        for match in pattern.finditer(prompt):
            start, end = match.span()
            if _overlaps(used, start, end) or _in_quotes(prompt, start):
                continue
            key = strong.lower().replace("’", "'")
            if key in _IMPERATIVE_ONLY and not _is_imperative_at(prompt, start):
                continue
            if key == "only" and (
                _ONLY_SKIP_BEFORE.search(prompt[:start])
                or _ONLY_SKIP_AFTER.search(prompt[end:])
            ):
                continue
            used.append((start, end))
            mutated = prompt[:start] + _match_case(match.group(0), weak) + prompt[end:]
            # Description keeps the canonical (lower-case) replacement so a
            # mutant's signature is stable across case-preservation changes.
            description = f'weakened "{match.group(0)}" -> "{weak}"'
            if snippets:
                description += f" (near: {_context_snippet(prompt, start, end)})"
            mutants.append(
                Mutant(
                    operator="weaken_modals",
                    description=description,
                    system=system.with_prompt(mutated),
                    focus=_line_at(prompt, start),
                )
            )
    return mutants


def weaken_modals(target: Target) -> List[Mutant]:
    """Soften strong instructions (MUST -> should, never -> rarely, ...).

    Each replaceable occurrence becomes its own mutant so a single missed
    instruction is isolated. ``do not`` / ``don't`` are only weakened where they
    are imperative (see ``_IMPERATIVE_ONLY``); letter case is preserved.
    """
    return _weaken(as_system(target), _MODAL_WEAKENINGS, snippets=True)


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
                focus=stripped,
            )
        )
    return mutants


_NUMBERED = re.compile(r"^\d+[.)]\s")


def swap_adjacent_instructions(target: Target) -> List[Mutant]:
    """Swap adjacent instruction lines to expose order-sensitive prompts.

    A ROBUSTNESS operator: it assumes an unordered rule list. Numbered items
    (ordered steps) are never swapped — reordering steps is a real change, and
    "2." above "1." isn't a paraphrase of anything."""
    system = as_system(target)
    lines = system.prompt.splitlines()
    mutants: List[Mutant] = []
    for i in range(len(lines) - 1):
        first = lines[i].strip()
        second = lines[i + 1].strip()
        if not (_is_instruction_line(first) and _is_instruction_line(second)):
            continue
        if _NUMBERED.match(first) or _NUMBERED.match(second):
            continue
        swapped = lines[:]
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        mutants.append(
            Mutant(
                operator="swap_adjacent_instructions",
                description=(
                    "swapped adjacent instruction lines: "
                    f'"{_truncate(first)}" before "{_truncate(second)}"'
                ),
                system=system.with_prompt("\n".join(swapped)),
                focus=f"{first}\n{second}",
            )
        )
    return mutants


# Meaning-preserving rewrites. Every replacement must stay grammatical for any
# verb that follows: "do not X" -> "never X" works for every X, where the old
# "avoid" produced "avoid follow ...". "in order to" -> "to" keeps the
# infinitive ("for" did not). Deletions are tidied afterwards (spacing, the
# capital letter that started the sentence).
_PARAPHRASES = [
    # not before "not": deleting "you should" from "you should not X" leaves "not X".
    (r"\b(?:you should|please|kindly)\b(?!\s+not\b)", ""),
    # "make sure to X" -> "be sure to X"; "ensure X" -> "make sure X". (The old
    # "verify that" broke on "make sure to ..." -> "verify that to ...".)
    (r"\bmake sure\b", "be sure"),
    (r"\bensure\b", "make sure"),
    (r"\b(?:do not|don't)\b", "never"),
    (r"\b(?:in order to)\b", "to"),
    (r"\b(?:at this point|currently)\b", ""),
    (r"\b(?:it is important to)\b", ""),
    (r"\b(?:a number of|several)\b", "some"),
    (r"\b(?:the fact that)\b", "that"),
    (r"\b(?:in the event that)\b", "if"),
    (r"\b(?:has the ability to)\b", "can"),
]


def paraphrase_instruction(target: Target) -> List[Mutant]:
    """Reword a single instruction line while preserving meaning.

    Uses simple synonym swaps and phrasing changes to expose evals that
    depend on exact wording rather than semantic content. This is a ROBUSTNESS
    operator (see ``OPERATOR_INTENT``): a survivor is the healthy outcome and a
    kill means something reacted to wording alone, so it is never scored as a
    coverage gap.
    """
    system = as_system(target)
    lines = system.prompt.splitlines()
    mutants: List[Mutant] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _is_instruction_line(stripped):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        for pattern, replacement in _PARAPHRASES:
            rewritten = _rewrite_outside_quotes(stripped, pattern, replacement)
            # The rule didn't fire: tidying alone isn't a paraphrase.
            if rewritten == stripped:
                continue
            new_line = _tidy_line(rewritten, stripped)
            if new_line != stripped:
                new_lines = lines.copy()
                new_lines[i] = indent + new_line
                mutants.append(
                    Mutant(
                        operator="paraphrase_instruction",
                        description=f'paraphrased line: "{_truncate(stripped)}" -> "{_truncate(new_line)}"',
                        system=system.with_prompt("\n".join(new_lines)),
                        focus=stripped,
                    )
                )
    return mutants


def _rewrite_outside_quotes(line: str, pattern: str, replacement: str) -> str:
    """Apply one paraphrase rule to ``line``, never inside a quoted literal (a
    required phrase such as ``say "I don't know"`` is not a paraphrase target).
    Deleting a capitalized word mid-line ("Be brief. Please use JSON.")
    re-capitalizes the word that now starts the sentence."""
    out: List[str] = []
    pos = 0
    cap_next = False
    for m in re.finditer(pattern, line, flags=re.IGNORECASE):
        if _in_quotes(line, m.start()):
            continue
        segment = line[pos : m.start()]
        out.append(_capitalize_first(segment) if cap_next else segment)
        rep = _match_case(m.group(0), replacement)
        cap_next = not rep and m.group(0)[:1].isupper() and m.start() > 0
        out.append(rep)
        pos = m.end()
    tail = line[pos:]
    out.append(_capitalize_first(tail) if cap_next else tail)
    return "".join(out)


def _capitalize_first(text: str) -> str:
    for k, ch in enumerate(text):
        if ch.isalpha():
            return text[:k] + ch.upper() + text[k + 1 :]
        if ch not in " ,;:":
            return text
    return text


def delete_sentences(target: Target) -> List[Mutant]:
    """Delete a single sentence at a time (for prose-style prompts).

    Only the sentence's own span is removed; every other byte of the prompt
    (line breaks, bullets, indentation) is left as it was, so the mutant is ONE
    change. A sentence never crosses a line break, so an unpunctuated line (a
    heading, an input template) can't be glued onto its neighbours.
    """
    system = as_system(target)
    lines = system.prompt.splitlines()
    spans = [
        (i, prefix, body, s, e)
        for i, line in enumerate(lines)
        for prefix, body in [_split_bullet(line)]
        for s, e in _sentence_spans(body)
    ]
    if len(spans) < 2:
        return []
    mutants: List[Mutant] = []
    for i, prefix, body, s, e in spans:
        sentence = body[s:e]
        # Skip fragments, headings ("## Output format") and lead-ins ("Follow
        # these steps:") — deleting those isn't dropping an instruction.
        if (
            len(sentence) < 12
            or sentence.endswith(":")
            or prefix.strip() == ""
            and (body.lstrip().startswith("#"))
        ):
            continue
        left, right = body[:s].rstrip(), body[e:].lstrip()
        rest = left + (" " if left and right else "") + right
        if rest.strip():
            new_lines = lines[:i] + [prefix + rest] + lines[i + 1 :]
        else:  # the line held only this sentence: drop the whole line
            new_lines = lines[:i] + lines[i + 1 :]
        mutants.append(
            Mutant(
                operator="delete_sentences",
                description=f'deleted sentence: "{_truncate(sentence)}"',
                system=system.with_prompt("\n".join(new_lines)),
                focus=sentence,
            )
        )
    return mutants


# Pairs that INVERT meaning — a stronger regression than mere weakening.
# Apostrophes match both ' and ’ (see _apostrophes).
_NEGATION_FLIPS = [
    ("must not", "must"),
    ("mustn't", "must"),
    ("should not", "should"),
    ("shouldn't", "should"),
    ("cannot", "can"),
    ("can not", "can"),
    ("can't", "can"),
    ("will not", "will"),
    ("won't", "will"),
    ("does not", "does"),
    ("doesn't", "does"),
    ("do not", "do"),
    ("don't", "do"),
    ("never", "always"),
    ("always", "never"),
]


def flip_negation(target: Target) -> List[Mutant]:
    """Invert a rule (do not -> do, never -> always).

    A meaning-inverting mutation — a far more dangerous regression than merely
    weakening a modal, so any eval worth its salt should catch it. Unlike
    ``weaken_modals`` this also applies to DESCRIPTIVE rules ("small changes do
    not make a PR out-of-scope" -> "... do make ..."): that is a grammatical,
    real inversion of the rule. Letter case is preserved.
    """
    system = as_system(target)
    prompt = system.prompt
    mutants: List[Mutant] = []
    used: List[Tuple[int, int]] = []
    for src, dst in _NEGATION_FLIPS:
        pattern = re.compile(rf"\b{_apostrophes(re.escape(src))}\b", re.IGNORECASE)
        for match in pattern.finditer(prompt):
            start, end = match.span()
            if _overlaps(used, start, end) or _in_quotes(prompt, start):
                continue  # never flip a quoted literal ('say "I don't know"')
            # "not always" -> "not never" is nonsense, not an inversion.
            if src in ("never", "always") and re.search(
                r"\bnot\s+\Z", prompt[:start], re.IGNORECASE
            ):
                continue
            used.append((start, end))
            mutated = prompt[:start] + _match_case(match.group(0), dst) + prompt[end:]
            snippet = _context_snippet(prompt, start, end)
            mutants.append(
                Mutant(
                    operator="flip_negation",
                    description=f'inverted "{match.group(0)}" -> "{dst}" (near: {snippet})',
                    system=system.with_prompt(mutated),
                    focus=_line_at(prompt, start),
                )
            )
    return mutants


def truncate_prompt(target: Target) -> List[Mutant]:
    """Cut off the tail of the prompt (lossy truncation).

    Models a prompt that got clipped — by a token budget, a bad edit, or
    context-window pressure — silently dropping its later instructions.

    Lines carrying input placeholders (``{{var}}``, ``{var}``, ``${VAR}``, ...)
    are never cut — only the tail of the INSTRUCTION lines is, wherever they sit
    (above, around or below the input block). Cutting the input template means
    the model never sees the input, so every eval fails: a guaranteed kill that
    says nothing about eval coverage but still inflates the score.
    """
    system = as_system(target)
    lines = system.prompt.splitlines()
    candidates = [i for i, line in enumerate(lines) if not _PLACEHOLDER_RE.search(line)]
    has_inputs = len(candidates) < len(lines)
    if len(candidates) < 4:
        return []
    mutants: List[Mutant] = []
    for frac in (0.5, 0.75):
        keep = max(1, int(len(candidates) * frac))
        if keep >= len(candidates):
            continue
        dropped = set(candidates[keep:])
        mutated = "\n".join(line for i, line in enumerate(lines) if i not in dropped)
        cut = [lines[i].strip() for i in sorted(dropped) if lines[i].strip()]
        noun = "instruction lines (input lines kept)" if has_inputs else "lines"
        description = (
            f"truncated prompt — dropped the last {len(dropped)} of "
            f"{len(candidates) if has_inputs else len(lines)} {noun}, from "
            f'"{_truncate(cut[0] if cut else "", 40)}"'
        )
        mutants.append(
            Mutant(
                operator="truncate_prompt",
                description=description,
                system=system.with_prompt(mutated),
                focus="\n".join(cut),
            )
        )
    return mutants


# A demonstration line: "Label: content" at the start of a line.
_LABEL_LINE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z ]{0,24}?)[ \t]*:[ \t]*\S", re.MULTILINE
)
# Known input/output label pairs of a demonstration.
_IO_PAIRS = (
    ({"input"}, {"output"}),
    ({"q", "question"}, {"a", "answer"}),
    ({"user", "human", "customer"}, {"assistant", "ai", "agent", "bot"}),
)


def _demo_labels(block: str) -> "frozenset[str]":
    return frozenset(m.group(1).strip().lower() for m in _LABEL_LINE.finditer(block))


def drop_few_shot_example(target: Target) -> List[Mutant]:
    """Remove a single few-shot example block at a time.

    For few-shot prompts: drops one demonstration so you can see whether your
    evals notice degraded in-context guidance. A block counts as a
    demonstration only if it is SHAPED like one: at least two "Label: content"
    lines, whose label set either repeats across blocks (Review:/Label: ...
    Review:/Label:) or is a known input/output pair (Input/Output, Q/A,
    User/Assistant). A keyword match ("Format your output: ...", "For example,
    ...") used to drop instructions and miss the real demos. The rest of the
    prompt is left byte-identical (it used to be re-joined and stripped).
    """
    system = as_system(target)
    prompt = system.prompt
    # Blocks separated by blank lines, as (start, end) spans in the original.
    spans = [m.span() for m in re.finditer(r"[^\n]+(?:\n(?![ \t]*\n)[^\n]*)*", prompt)]
    spans = [(s, e) for s, e in spans if prompt[s:e].strip()]
    if len(spans) < 2:
        return []
    labels = [_demo_labels(prompt[s:e]) for s, e in spans]
    counts: Dict[frozenset, int] = {}
    for lb in labels:
        if len(lb) >= 2:
            counts[lb] = counts.get(lb, 0) + 1

    def is_demo(lb: "frozenset[str]") -> bool:
        if len(lb) < 2:
            return False
        if counts.get(lb, 0) >= 2:
            return True
        return any(lb & ins and lb & outs for ins, outs in _IO_PAIRS)

    mutants: List[Mutant] = []
    for (s, e), lb in zip(spans, labels):
        if not is_demo(lb):
            continue
        block = prompt[s:e]
        # Remove the block and the blank-line separator AFTER it (or before it,
        # for the last block); nothing else changes.
        after = re.match(r"\n[ \t]*\n+", prompt[e:])
        if after:
            mutated = prompt[:s] + prompt[e + after.end() :]
        else:
            before = re.search(r"\n[ \t]*\n+\Z", prompt[:s])
            cut = before.start() if before else s
            mutated = prompt[:cut] + prompt[e:]
        mutants.append(
            Mutant(
                operator="drop_few_shot_example",
                description=f'dropped example block: "{_truncate(block.strip())}"',
                system=system.with_prompt(mutated),
                focus=block.strip(),
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
    removed: List[str] = []

    def unbold(m: "re.Match[str]") -> str:
        inner = m.group(2)
        # __init__ / __name__ are identifiers, not bold text.
        if m.group(1) == "__" and re.fullmatch(r"[a-z_][a-z0-9_]*", inner):
            return m.group(0)
        removed.append(f"{m.group(1)}bold{m.group(1)}")
        return inner

    mutated = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", unbold, prompt)

    # Label markers: UPPERCASE only, followed by ":" or "!" (the old
    # case-insensitive match deleted "Note that ..." / "Important details").
    # Indentation and bullets stay; blank lines are never touched.
    def unlabel(m: "re.Match[str]") -> str:
        removed.append(f"{m.group(2)}:")
        return m.group(1)

    mutated = re.sub(
        r"(?m)^([ \t]*(?:[-*+][ \t]+)?)(IMPORTANT|CRITICAL|NOTE|WARNING|ATTENTION)"
        r"[ \t]*[:!][ \t]*",
        unlabel,
        mutated,
    )

    # ALL-CAPS emphasis ("NEVER share", "you MUST") -> normal case. YES/NO are
    # left alone: they're usually required output tokens, not emphasis.
    def uncaps(m: "re.Match[str]") -> str:
        word = m.group(0)
        removed.append(f'"{word}"')
        lowered = word.lower()
        before = m.string[: m.start()]
        at_start = not before.strip() or re.search(
            r"(?:[.!?:]|^[ \t]*[-*+])[ \t]*\Z", before, re.M
        )
        return lowered[:1].upper() + lowered[1:] if at_start else lowered

    mutated = re.sub(
        r"\b(?:DO NOT|DON'T|NEVER|ALWAYS|MUST|NOT|ONLY|CANNOT|REQUIRED|STRICTLY"
        r"|IMPORTANT|CRITICAL)\b",
        uncaps,
        mutated,
    )
    if mutated == prompt:
        return []
    counts: Dict[str, int] = {}
    for r in removed:
        counts[r] = counts.get(r, 0) + 1
    summary = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in counts.items())
    changed = [a for a, b in zip(prompt.splitlines(), mutated.splitlines()) if a != b]
    return [
        Mutant(
            operator="remove_emphasis",
            description=f"removed emphasis: {_truncate(summary, 60)}",
            system=system.with_prompt(mutated),
            focus="\n".join(line.strip() for line in changed),
        )
    ]


# A numeric BOUND is decided by the phrase attached to the number — directly
# before it ("at most 3", "under 50", "limit it to 3") or directly after it
# ("3 or fewer", "50 words max") — not by any bound word within 40 chars. The
# old window read "no fewer than 3" as an UPPER bound (it contains "fewer
# than") and tightened it to 6, and read "at least 2 and at most 4" as one bound.
# Negated compounds are checked first.
_BOUND_BEFORE = [
    (re.compile(r"\b(?:no|not)\s+(?:fewer|less)\s+than\s+\Z", re.I), "lower"),
    (re.compile(r"\b(?:no|not)\s+(?:more|greater|longer)\s+than\s+\Z", re.I), "upper"),
    (
        re.compile(
            r"\b(?:at\s+least|minimum(?:\s+of)?|min\.?|more\s+than|greater\s+than"
            r"|over|above|exceed(?:ing)?)\s+\Z",
            re.I,
        ),
        "lower",
    ),
    (
        re.compile(
            r"\b(?:at\s+most|up\s+to|maximum(?:\s+of)?|max\.?|fewer\s+than|less\s+than"
            r"|within|under|below|limit(?:ed)?\s+(?:\w+\s+)?to)\s+\Z",
            re.I,
        ),
        "upper",
    ),
]
_BOUND_AFTER = [
    (
        re.compile(
            r"\A\s+(?:\w+\s+)?(?:or\s+(?:fewer|less)|at\s+most|max(?:imum)?)\b", re.I
        ),
        "upper",
    ),
    (
        re.compile(r"\A\s+(?:\w+\s+)?(?:or\s+more|at\s+least|min(?:imum)?)\b", re.I),
        "lower",
    ),
]
# One number token: grouped thousands ("1,000") or a decimal ("0.5") is ONE
# number (the old \b\d+\b turned "0.5" into "0.10" and "1,000" into "1,1").
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w]|[.,]\d)")


def _bound_direction(prompt: str, start: int, end: int) -> "str | None":
    line_start = prompt.rfind("\n", 0, start) + 1
    line_end = prompt.find("\n", end)
    before = prompt[line_start:start]
    after = prompt[end : line_end if line_end != -1 else len(prompt)]
    for rx, direction in _BOUND_BEFORE:
        if rx.search(before):
            return direction
    for rx, direction in _BOUND_AFTER:
        if rx.search(after):
            return direction
    return None


def _loosen(token: str, direction: str) -> str:
    grouped = "," in token
    value = float(token.replace(",", "")) if "." in token else int(token.replace(",", ""))
    if direction == "upper":
        new = value * 2 if value > 0 else 1
    elif isinstance(value, int):
        new = value // 2 if value > 1 else 0
    else:
        new = value / 2
    if grouped:
        return f"{int(new):,}"
    if isinstance(new, float):
        return f"{new:g}"
    return str(new)


def weaken_numeric_threshold(target: Target) -> List[Mutant]:
    """Loosen a numeric threshold in the prompt (make a constraint more permissive).

    An upper bound is increased ("at most 3" -> "at most 6"), a lower bound is
    decreased ("at least 5" -> "at least 2"). Models a common silent regression —
    a limit that got loosened — that output-grading and reference-free evals
    usually say nothing about. Only fires on a number with a bound phrase
    attached (see ``_BOUND_BEFORE`` / ``_BOUND_AFTER``), so it skips versions,
    years, list markers ("1. Keep ...") and other incidental digits.
    """
    system = as_system(target)
    prompt = system.prompt
    mutants: List[Mutant] = []
    for match in _NUMBER.finditer(prompt):
        start, end = match.span()
        direction = _bound_direction(prompt, start, end)
        if direction is None:
            continue
        old = match.group(0)
        new = _loosen(old, direction)
        if new == old:
            continue
        mutated = prompt[:start] + new + prompt[end:]
        snippet = _context_snippet(prompt, start, end)
        mutants.append(
            Mutant(
                operator="weaken_numeric_threshold",
                description=f"loosened threshold {old} -> {new} (near: {snippet})",
                system=system.with_prompt(mutated),
                focus=_line_at(prompt, start),
            )
        )
    return mutants


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
                focus=str(doc),
            )
        )
    return mutants


def clear_context(target: Target) -> List[Mutant]:
    """Remove ALL retrieved context (simulate total retrieval failure)."""
    system = as_system(target)
    if not system.context:
        return []
    docs = list(system.context)
    return [
        Mutant(
            operator="clear_context",
            # Names the first doc so an accepted signature doesn't survive a
            # completely different context.
            description=(
                f"cleared all retrieved context (dropped {len(docs)} doc(s), "
                f'starting "{_truncate(docs[0], 40)}")'
            ),
            system=system.replace(context=()),
            target="context",
            focus="\n".join(str(d) for d in docs),
        )
    ]


# --- More context operators (B2): corrupt / swap / shuffle / duplicate / truncate

_IRRELEVANT_DOC = (
    "Reminder: the office cafeteria serves lunch from 12:00 to 13:00 on weekdays."
)


# A FACT number: standalone, not glued to letters/hyphens/underscores — so an
# id like "doc-1", "B2B", "v2" or "ORD-9" is never the thing corrupted (changing
# "doc-1" to "doc-2" altered a label, not a fact, so the mutant was inert).
_FACT_NUMBER = re.compile(r"(?<![\w\-])\d+(?:\.\d+)?(?![\w\-])")
_POLARITY_VERB = re.compile(
    r"\b(is|are|was|were|can|will|does|do|should|must)\b(?!['’]t)", re.IGNORECASE
)


def _corrupt_doc(doc: str) -> "tuple[str, str, str] | None":
    """Deterministic, rule-based corruption of ONE fact. Returns
    ``(corrupted_doc, before, after)`` — the edited token, for the description —
    or None if nothing is safely corruptible.

    1. Change the first FACT number (not an id; see ``_FACT_NUMBER``).
    2. Else flip a polarity verb: remove an existing "not" ("is not open" ->
       "is open") or add one ("is open" -> "is not open"). Never "do not not"
       or "can not't".
    """
    m = _FACT_NUMBER.search(doc)
    if m:
        old = m.group(0)
        if "." in old:
            new = f"{float(old) + 1:g}"
        else:
            n = int(old)
            new = str(n + 1 if n != 0 else 9)
        return doc[: m.start()] + new + doc[m.end() :], old, new
    for verb in _POLARITY_VERB.finditer(doc):
        rest = doc[verb.end() :]
        negated = re.match(r"\s+not\b", rest, re.IGNORECASE)
        if negated:
            before = verb.group(0) + negated.group(0)
            return doc[: verb.end()] + rest[negated.end() :], before, verb.group(0)
        return (
            doc[: verb.end()] + " not" + rest,
            verb.group(0),
            verb.group(0) + " not",
        )
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
        if not isinstance(doc, str):
            continue  # opaque doc (dict/object): nothing safe to corrupt in place
        corrupted = _corrupt_doc(doc)
        if corrupted is None or corrupted[0] == doc:
            continue
        bad, before, after = corrupted
        new = docs[:i] + [bad] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="corrupt_context_doc",
                # Name the edited token: a change deep in a long doc was
                # invisible in a truncated preview of the doc.
                description=(
                    f'corrupted retrieved doc #{i + 1}: "{before}" -> "{after}" in '
                    f'"{_truncate(doc, 50)}"'
                ),
                system=system.replace(context=tuple(new)),
                target="context",
                focus=doc,
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
        # A string stand-in for a structured (dict) doc would change its TYPE —
        # a guaranteed crash/kill, not a bad retrieval hit.
        if not isinstance(docs[i], str) or docs[i] == _IRRELEVANT_DOC:
            continue
        new = docs[:i] + [_IRRELEVANT_DOC] + docs[i + 1 :]
        mutants.append(
            Mutant(
                operator="swap_context_doc",
                description=(
                    f'swapped retrieved doc #{i + 1} ("{_truncate(docs[i], 40)}") '
                    "for an irrelevant doc"
                ),
                system=system.replace(context=tuple(new)),
                target="context",
                focus=docs[i],
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
            description=(
                f"reversed the order of {len(system.context)} retrieved docs "
                f'(first was "{_truncate(system.context[0], 40)}")'
            ),
            system=system.replace(context=reordered),
            target="context",
            focus="\n".join(str(d) for d in system.context),
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
                focus=str(doc),
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
        if not isinstance(doc, str):
            continue  # a structured doc can't be clipped as text
        words = list(re.finditer(r"\S+", doc))
        if len(words) < 6:
            continue
        keep = max(1, len(words) // 2)
        # Cut at the character offset where word #keep ends, so the kept half
        # keeps its own line breaks (re-joining words reflowed the whole doc).
        clipped = doc[: words[keep - 1].end()]
        if clipped == doc:
            continue
        new = docs[:i] + [clipped] + docs[i + 1 :]
        cut = " ".join(w.group(0) for w in words[keep:])
        mutants.append(
            Mutant(
                operator="truncate_context_doc",
                description=(
                    f"truncated retrieved doc #{i + 1} to its first {keep} words "
                    f'(cut "{_truncate(cut, 40)}")'
                ),
                system=system.replace(context=tuple(new)),
                target="context",
                focus=cut,
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

_IRRELEVANT_TOOL = 'tool_result: {"status": "ok", "data": "unrelated"}'
_IRRELEVANT_TOOL_DICT = {"status": "ok", "data": "unrelated"}


def _same_type(original: Any, as_text: str, as_dict: dict) -> Any:
    """A stand-in of the SAME type as ``original`` (a string for a string, a
    dict for a dict), or None. Swapping a dict tool output for a string changed
    its type — a guaranteed crash/kill in the agent, not the fault being
    modeled."""
    if isinstance(original, str):
        return as_text
    if isinstance(original, dict):
        return dict(as_dict)
    return None


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
                focus=str(tools[i]),
            )
        )
    return mutants


def _corrupt_tool(tool: Any) -> "tuple[Any, str, str] | None":
    """Corrupt one fact in a tool output, keeping its type: a string via
    ``_corrupt_doc``; a dict by bumping its first numeric field."""
    if isinstance(tool, str):
        return _corrupt_doc(tool)
    if isinstance(tool, dict):
        for key, value in tool.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bumped = value + 1 if value != 0 else 9
                return {**tool, key: bumped}, f"{key}: {value}", f"{key}: {bumped}"
    return None


def corrupt_tool_output(target: Target) -> List[Mutant]:
    """Corrupt one tool output (a tool returned a plausible-but-wrong result)."""
    system = as_system(target)
    if not system.tools:
        return []
    tools = list(system.tools)
    mutants: List[Mutant] = []
    for i, tool in enumerate(tools):
        corrupted = _corrupt_tool(tool)
        if corrupted is None or corrupted[0] == tool:
            continue
        bad, before, after = corrupted
        new = tools[:i] + [bad] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="corrupt_tool_output",
                description=(
                    f'corrupted tool output #{i + 1}: "{before}" -> "{after}" in '
                    f'"{_truncate(str(tool), 50)}"'
                ),
                system=system.replace(tools=tuple(new)),
                target="tools",
                focus=str(tool),
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
        stand_in = _same_type(tools[i], _IRRELEVANT_TOOL, _IRRELEVANT_TOOL_DICT)
        if stand_in is None or tools[i] == stand_in:
            continue
        new = tools[:i] + [stand_in] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="swap_tool_output",
                description=(
                    f'swapped tool output #{i + 1} ("{_truncate(str(tools[i]), 40)}") '
                    "for an irrelevant result"
                ),
                system=system.replace(tools=tuple(new)),
                target="tools",
                focus=str(tools[i]),
            )
        )
    return mutants


# A domain failure returned as transport SUCCESS: HTTP 200 whose body says it
# failed. Structured-error detection (status/error fields) is blind to this — an
# agent that trusts the transport status proceeds as if the call worked. Only a
# declared failure contract (e.g. tracelint's failure_when) catches it.
_DENIED_TOOL_OUTPUT = '{"status": "declined", "reason": "insufficient_funds"}'
_DENIED_TOOL_OUTPUT_DICT = {"status": "declined", "reason": "insufficient_funds"}


def deny_tool_output(target: Target) -> List[Mutant]:
    """Turn one tool output into a domain FAILURE returned as transport success
    (a 200 carrying ``{"status": "declined"}``).

    Models the hardest tool fault to catch: the call "succeeded" at the transport
    layer but failed in its body, so exception/status-based error handling never
    fires and a naive agent proceeds as if it worked. If your eval suite still
    passes, it can't see a declined charge that the agent reported as success.
    """
    system = as_system(target)
    if not system.tools:
        return []
    tools = list(system.tools)
    mutants: List[Mutant] = []
    for i in range(len(tools)):
        stand_in = _same_type(tools[i], _DENIED_TOOL_OUTPUT, _DENIED_TOOL_OUTPUT_DICT)
        if stand_in is None or tools[i] == stand_in:
            continue
        new = tools[:i] + [stand_in] + tools[i + 1 :]
        mutants.append(
            Mutant(
                operator="deny_tool_output",
                description=(
                    f"tool output #{i + 1} returned a domain failure "
                    f'(HTTP 200 + status:declined) instead of "{_truncate(str(tools[i]), 40)}"'
                ),
                system=system.replace(tools=tuple(new)),
                target="tools",
                focus=str(tools[i]),
            )
        )
    return mutants


# --- Operator factories (A4): parametrize built-in operators ----------------
# Combine with register_operator to add a tuned variant, e.g.
#   register_operator("weaken_modals_eu", make_weaken_modals([("shall","may")]))


def make_weaken_modals(pairs: "List[tuple]") -> "Callable[[Target], List[Mutant]]":
    """Build a weaken_modals-style operator with custom (strong, weak) pairs."""

    def op(target: Target) -> List[Mutant]:
        return _weaken(as_system(target), pairs, snippets=False)

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
    "swap_adjacent_instructions": swap_adjacent_instructions,
    "paraphrase_instruction": paraphrase_instruction,
    "delete_sentences": delete_sentences,
    "truncate_prompt": truncate_prompt,
    "drop_few_shot_example": drop_few_shot_example,
    "remove_emphasis": remove_emphasis,
    "weaken_numeric_threshold": weaken_numeric_threshold,
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
    "deny_tool_output": deny_tool_output,
}

# What a mutant is FOR. A "regression" operator injects a degradation the evals
# should catch: kill = good, survivor = coverage gap (scored). A "robustness"
# operator makes a meaning-preserving edit: survival is the healthy outcome and a
# kill means an eval (or the system) reacted to wording/order alone. Robustness
# mutants are never scored; the report lists the ones that flipped a verdict.
# Operators not listed here are "regression".
REGRESSION = "regression"
ROBUSTNESS = "robustness"
OPERATOR_INTENT: Dict[str, str] = {
    "paraphrase_instruction": ROBUSTNESS,
    "swap_adjacent_instructions": ROBUSTNESS,
}


def intent_of(operator: str) -> str:
    """``"regression"`` (scored) or ``"robustness"`` (meaning-preserving)."""
    return OPERATOR_INTENT.get(operator, REGRESSION)


def register_operator(
    name: str, fn: "Callable[[Target], List[Mutant]]", intent: str = REGRESSION
) -> "Callable[[Target], List[Mutant]]":
    """Register a custom mutation operator under ``name`` so it runs by default
    and can be selected via ``--operators name`` / ``operators=[name]``.

    The operator is ``fn(target) -> list[Mutant]`` (``target`` is a System or a
    bare prompt string; use ``as_system(target)``). Returns ``fn`` so it can be
    used as a decorator. Bring-your-own operators never touch the eval suite —
    they only produce mutated Systems, preserving muteval's orthogonality.

    ``intent="robustness"`` marks a meaning-preserving operator (see
    ``OPERATOR_INTENT``): its mutants are reported but never scored.
    """
    if intent not in (REGRESSION, ROBUSTNESS):
        raise ValueError(f"intent must be {REGRESSION!r} or {ROBUSTNESS!r}")
    OPERATORS[name] = fn
    if intent == ROBUSTNESS:
        OPERATOR_INTENT[name] = ROBUSTNESS
    else:
        OPERATOR_INTENT.pop(name, None)
    return fn


def generate_mutants(
    target: Target,
    operators: "List[str | Callable] | None" = None,
    scope: "Scope | None" = None,
) -> List[Mutant]:
    """Run the selected operators and return a de-duplicated list of mutants.

    ``operators`` items may be registered operator NAMES (str) or operator
    CALLABLES (``fn(target) -> list[Mutant]``) for bring-your-own operators.

    A mutant whose prompt lost any of the original's input placeholders
    (``{{var}}`` / ``{var}`` / ``${VAR}``) is dropped: the model would never see
    the input, so every eval fails — a guaranteed kill that measures nothing.
    """
    from muteval.severity import severity_of, severity_rank

    original = as_system(target)
    original_key = original.key()
    inputs = _placeholders(original.prompt)
    selected = operators if operators is not None else list(OPERATORS.keys())
    index: Dict[tuple, int] = {}
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
            if mkey == original_key:  # a no-op
                continue
            if inputs and not inputs <= _placeholders(mutant.system.prompt):
                continue
            if mkey in index:
                # The same mutated system from two operators (dropping the only
                # doc == clearing the context): keep the MORE severe framing,
                # rather than whichever operator happened to run first.
                pos = index[mkey]
                if severity_rank(severity_of(mutant)) < severity_rank(
                    severity_of(mutants[pos])
                ):
                    mutants[pos] = mutant
                continue
            index[mkey] = len(mutants)
            mutants.append(mutant)
    return filter_mutants(original.prompt, mutants, scope)


# --- helpers -----------------------------------------------------------------


def _is_instruction_line(stripped: str) -> bool:
    """Is this line an instruction (droppable / swappable / paraphrasable)?
    Headings ("## Rules", "Rules:") and short lead-ins ("Follow these steps:")
    aren't — dropping one removes a label, not a capability. A longer line
    ending in ":" ("Classify the sentiment of the review below:") still is."""
    if len(stripped) < 8 or stripped.startswith("#"):
        return False
    if stripped.endswith(":") and len(stripped.split()) < 4:
        return False
    bullet = re.match(r"^([-*+]|\d+[.)])\s+", stripped)
    return bool(bullet) or stripped.endswith((".", ":", "!"))


# Input placeholders a prompt template fills in: {{var}} (mustache / jinja /
# promptfoo); {var}, {0}, {}, {case.q}, {q!r}, {q:>10} (str.format); ${VAR} /
# $VAR (shell / JS / string.Template); %(q)s / %s (%-formatting).
_PLACEHOLDER_RE = re.compile(
    r"\{\{[^{}]+\}\}"
    r"|\{(?:[A-Za-z_][\w.\[\]]*|\d*)(?:![rsa])?(?::[^{}]*)?\}"
    r"|\$\{[A-Za-z_]\w*\}"
    r"|\$[A-Za-z_]\w*"
    r"|%\([A-Za-z_]\w*\)[sdrifx]"
    r"|%[sd]\b"
)


def _placeholders(text: str) -> "set[str]":
    return {" ".join(m.split()) for m in _PLACEHOLDER_RE.findall(text)}


# What may precede an IMPERATIVE "do not": the start of the text / a line / a
# bullet, sentence or clause punctuation ("If unsure, do not guess"), "please" /
# "then", optionally followed by an emphasis marker ("- **Do not** reveal").
# NOT a bare "you": "You do not have access to X" describes, it doesn't command.
_IMPERATIVE_LEAD = re.compile(
    r"(?:(?:\A|\n)[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?|[.!?:;,][ \t]+|[\"'(][ \t]*"
    r"|\b(?:please|then)[ \t]+)(?:\*\*|__|\*|_)?\Z",
    re.IGNORECASE,
)
_LEADING_BULLET = re.compile(r"\A[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?")


def _is_imperative_at(text: str, start: int) -> bool:
    """Is the phrase starting at ``start`` addressed to the reader (a command)?"""
    return bool(_IMPERATIVE_LEAD.search(text[:start]))


def _match_case(original: str, replacement: str) -> str:
    """Carry ``original``'s case onto ``replacement``: ALL CAPS stays all caps,
    a capitalised word stays capitalised (sentence-initial "Do not" -> "Try not
    to", not "try not to")."""
    if not replacement:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _tidy_line(rewritten: str, original: str) -> str:
    """Clean up after a phrase was deleted from a line: collapse doubled spaces,
    drop space before punctuation, and re-capitalise the first word when the
    original line started with a capital (after any bullet marker)."""
    bullet, orig_body = _split_bullet(original)
    body = rewritten[len(bullet) :] if rewritten.startswith(bullet) else rewritten
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+([,.;:!?])", r"\1", body)
    # "Respond in JSON, please." minus "please" -> "Respond in JSON."
    body = re.sub(r"[,;]([.!?])", r"\1", body)
    body = body.lstrip(" \t,;:")
    if orig_body[:1].isupper() and body[:1].islower():
        body = body[:1].upper() + body[1:]
    return (bullet + body).rstrip()


def _split_bullet(line: str) -> "tuple[str, str]":
    """Split a line into its (indent + bullet marker) prefix and its body."""
    prefix = _LEADING_BULLET.match(line).group(0)  # always matches (all-optional)
    return prefix, line[len(prefix) :]


# A sentence within ONE line: from a non-space char to terminal punctuation
# followed by whitespace/end-of-line, or to the end of the line.
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)")


# A "sentence" ending in one of these isn't finished ("Use JSON, e.g. a list").
_ABBREVIATION_END = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|approx|incl|mr|mrs|ms|dr|no|fig)\.\Z", re.IGNORECASE
)


def _sentence_spans(body: str) -> "List[tuple[int, int]]":
    spans: List[Tuple[int, int]] = []
    for m in _SENTENCE_RE.finditer(body):
        if not m.group(0).strip():
            continue
        if spans and _ABBREVIATION_END.search(body[spans[-1][0] : spans[-1][1]]):
            spans[-1] = (spans[-1][0], m.end())  # continue the abbreviated sentence
        else:
            spans.append(m.span())
    return spans


def _line_at(text: str, pos: int) -> str:
    """The (stripped) line containing ``pos`` — a mutation's severity focus."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : end if end != -1 else len(text)].strip()


def _in_quotes(text: str, pos: int) -> bool:
    """Is ``pos`` inside a double-quoted literal on its line? A quoted phrase
    ('say "I don't know"') is a required string, not an instruction to rewrite.
    Straight quotes pair up; curly quotes open/close explicitly."""
    line = text[text.rfind("\n", 0, pos) + 1 : pos]
    if line.count('"') % 2 == 1:
        return True
    return line.count("“") > line.count("”")


def _apostrophes(escaped: str) -> str:
    """Let an escaped pattern's apostrophe match both ' and ’ ("Don’t")."""
    return escaped.replace("'", "['’]").replace("\\'", "['’]")


def _context_snippet(text: str, start: int, end: int, width: int = 24) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = text[left:right].replace("\n", " ").strip()
    return _truncate(snippet, 60)


def _truncate(text: Any, limit: int = 70) -> str:
    text = " ".join(str(text).split())  # docs / tool outputs may be dicts
    return text if len(text) <= limit else text[: limit - 1] + "…"
