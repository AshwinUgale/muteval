"""A2: limit *which parts* of the prompt may be mutated.

Two ways to scope, combinable, applied as a POST-generation filter (so we never
touch the operators themselves):

* **Inline markers** — wrap mutable regions in ``[[mutate]] ... [[/mutate]]``.
  The markers are stripped from the actual prompt (the model never sees them);
  only changes landing inside a marked region are kept.
* **Line-level regex** — ``include`` keeps only mutants whose changed line(s)
  match the pattern; ``exclude`` drops mutants whose changed line(s) match it.

Only ``target == "prompt"`` mutants are scoped; context/model/tool mutants pass
through untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Pattern, Tuple

_OPEN = "[[mutate]]"
_CLOSE = "[[/mutate]]"


def strip_markers(text: str) -> Tuple[str, Optional[List[Tuple[int, int]]]]:
    """Remove ``[[mutate]]``/``[[/mutate]]`` markers, returning the clean text
    and the mutable (start, end) char ranges in clean coordinates. If there are
    no markers, returns ``(text, None)`` meaning "everything is mutable"."""
    if _OPEN not in text:
        if _CLOSE in text:
            raise ValueError(
                f"scope marker {_CLOSE} without a matching {_OPEN}: it would be "
                "sent to the model verbatim"
            )
        return text, None
    clean_parts: List[str] = []
    ranges: List[Tuple[int, int]] = []
    pos = 0
    out_len = 0
    while True:
        o = text.find(_OPEN, pos)
        if o == -1:
            clean_parts.append(text[pos:])
            break
        clean_parts.append(text[pos:o])
        out_len += o - pos
        c = text.find(_CLOSE, o + len(_OPEN))
        if c == -1:  # unterminated marker: treat rest as mutable
            region = text[o + len(_OPEN) :]
            clean_parts.append(region)
            ranges.append((out_len, out_len + len(region)))
            out_len += len(region)
            break
        region = text[o + len(_OPEN) : c]
        if _OPEN in region:
            raise ValueError(
                f"nested {_OPEN} markers aren't supported: close the first region "
                f"with {_CLOSE} before opening another"
            )
        clean_parts.append(region)
        ranges.append((out_len, out_len + len(region)))
        out_len += len(region)
        pos = c + len(_CLOSE)
    clean = "".join(clean_parts)
    if _CLOSE in clean:
        raise ValueError(
            f"scope marker {_CLOSE} without a matching {_OPEN}: it would be sent to "
            "the model verbatim"
        )
    return clean, (ranges or None)


def _changed_span(a: str, b: str) -> Optional[Tuple[int, int]]:
    """The (start, end) char range in ``a`` that differs from ``b``."""
    if a == b:
        return None
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    ja, jb = len(a), len(b)
    while ja > i and jb > i and a[ja - 1] == b[jb - 1]:
        ja -= 1
        jb -= 1
    return (i, max(i, ja))


def _common_affixes(a, b) -> Tuple[int, int]:
    """(prefix, suffix) lengths shared by sequences ``a`` and ``b``, with the
    suffix never overlapping the prefix — so a single edit gets its canonical
    LEFTMOST span (deleting "Never guess. " from "Be kind. Never guess. Be
    brief." is exactly that sentence, not a shifted ". Never guess" alignment)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return i, j


def _changed_hunks(a: str, b: str) -> List[Tuple[int, int]]:
    """Per-edit changed char ranges in ``a``. A pure insertion yields a
    zero-width range (i1 == i2) at the insertion point.

    The common prefix/suffix is trimmed first and only the changed MIDDLE is
    diffed: mutants are local edits, so this is linear in practice (a
    char-level SequenceMatcher over the whole prompt took minutes per run on a
    5k-char prompt). Separate hunks are still reported per edit, so a mutant
    that edits two marker regions (protected text between them) is judged
    region-by-region, not as one span that straddles the protected text."""
    if a == b:
        return []
    pre, suf = _common_affixes(a, b)
    mid_a, mid_b = a[pre : len(a) - suf], b[pre : len(b) - suf]
    if not mid_a or not mid_b:  # a pure deletion or insertion
        return [(pre, pre + len(mid_a))]
    hunks: List[Tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(
        None, mid_a, mid_b, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            hunks.append((pre + i1, pre + i2))
    return hunks


def _core(text: str, start: int, end: int) -> Tuple[int, int]:
    """Shrink a changed span past the whitespace/newlines at its edges: deleting
    a marked line also deletes its line break, which sits just OUTSIDE a
    tightly-wrapped ``[[mutate]]line[[/mutate]]`` region."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _affected_lines(original: str, mutant: str) -> List[str]:
    """The ORIGINAL lines a mutation touched (removed or rewritten).

    include/exclude select which of YOUR lines may be mutated, so they're
    matched against the original text only — matching the mutated text let
    ``--scope-include never`` keep an "Always -> never" flip, and
    ``--scope-exclude should`` drop a "must -> should" weakening. A pure
    insertion (no original line touched) is judged by the lines it adds.

    Occurrence-aware (line-level SequenceMatcher on the trimmed middle): if a
    line appears twice and one copy is removed, that's detected."""
    a, b = original.split("\n"), mutant.split("\n")
    pre, suf = _common_affixes(a, b)
    mid_a, mid_b = a[pre : len(a) - suf], b[pre : len(b) - suf]
    removed: List[str] = []
    added: List[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, mid_a, mid_b, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        removed.extend(mid_a[i1:i2])
        added.extend(mid_b[j1:j2])
    return removed or added


@dataclass
class Scope:
    ranges: Optional[List[Tuple[int, int]]] = None  # marker regions (char)
    include: Optional[Pattern] = None  # keep if a changed line matches
    exclude: Optional[Pattern] = None  # drop if a changed line matches

    def is_active(self) -> bool:
        return bool(self.ranges or self.include or self.exclude)

    def keep(self, original_prompt: str, mutant_prompt: str) -> bool:
        if original_prompt == mutant_prompt:
            return False
        if self.ranges is not None:
            hunks = [
                _core(original_prompt, s, e)
                for s, e in _changed_hunks(original_prompt, mutant_prompt)
            ]
            # EVERY changed hunk must be fully contained in a marked region.
            # (Checking one first-to-last envelope would wrongly reject a mutant
            # that makes separate valid edits in two regions with protected text
            # between them; a mere overlap would wrongly accept one that straddles
            # a boundary and edits protected text.)
            for start, end in hunks:
                if start == end:  # pure insertion: the point must sit in a region
                    if not any(s <= start <= e for (s, e) in self.ranges):
                        return False
                elif not any(s <= start and end <= e for (s, e) in self.ranges):
                    return False
        if self.include is not None or self.exclude is not None:
            lines = _affected_lines(original_prompt, mutant_prompt)
            if self.include is not None and not any(
                self.include.search(ln) for ln in lines
            ):
                return False
            if self.exclude is not None and any(self.exclude.search(ln) for ln in lines):
                return False
        return True


def make_scope(
    ranges: Optional[List[Tuple[int, int]]] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
) -> Optional[Scope]:
    """Build a Scope from optional marker ranges + include/exclude regex strings.
    Returns None if nothing scopes anything."""
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    scope = Scope(ranges=ranges, include=inc, exclude=exc)
    return scope if scope.is_active() else None


def filter_mutants(original_prompt: str, mutants, scope: Optional[Scope]):
    """Keep only prompt-target mutants allowed by ``scope``; pass others through."""
    if scope is None or not scope.is_active():
        return mutants
    kept = []
    for m in mutants:
        if getattr(m, "target", "prompt") != "prompt":
            kept.append(m)  # context/model/tool mutants are not prompt-scoped
        elif scope.keep(original_prompt, m.prompt):
            kept.append(m)
    return kept
