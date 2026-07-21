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
            region = text[o + len(_OPEN):]
            clean_parts.append(region)
            ranges.append((out_len, out_len + len(region)))
            out_len += len(region)
            break
        region = text[o + len(_OPEN):c]
        clean_parts.append(region)
        ranges.append((out_len, out_len + len(region)))
        out_len += len(region)
        pos = c + len(_CLOSE)
    return "".join(clean_parts), (ranges or None)


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


def _changed_hunks(a: str, b: str) -> List[Tuple[int, int]]:
    """Per-edit changed char ranges in ``a`` (SequenceMatcher opcodes). A pure
    insertion yields a zero-width range (i1 == i2) at the insertion point.

    Unlike a single ``_changed_span`` envelope, this returns each edit
    separately, so a mutant that edits two marker regions (with protected text
    between them) can be judged region-by-region instead of as one span that
    straddles the protected text."""
    hunks: List[Tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            hunks.append((i1, i2))
    return hunks


def _affected_lines(original: str, mutant: str) -> List[str]:
    """Lines added or removed between original and mutant (the lines a line-level
    mutation actually touched).

    Uses ``difflib.SequenceMatcher`` opcodes so the diff is OCCURRENCE-aware: if
    a line ("- Do not lie.") appears twice and a mutation removes ONE copy, we
    detect it. A set-based diff would miss that (the line still exists elsewhere)
    and would also falsely ignore a changed line whose text happens to collide
    with an unrelated line. Only the lines inside changed hunks are returned."""
    a, b = original.split("\n"), mutant.split("\n")
    touched: List[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        touched.extend(a[i1:i2])  # removed / replaced-from lines
        touched.extend(b[j1:j2])  # added / replaced-to lines
    return touched


@dataclass
class Scope:
    ranges: Optional[List[Tuple[int, int]]] = None       # marker regions (char)
    include: Optional[Pattern] = None                     # keep if a changed line matches
    exclude: Optional[Pattern] = None                     # drop if a changed line matches

    def is_active(self) -> bool:
        return bool(self.ranges or self.include or self.exclude)

    def keep(self, original_prompt: str, mutant_prompt: str) -> bool:
        hunks = _changed_hunks(original_prompt, mutant_prompt)
        if not hunks:
            return False
        if self.ranges is not None:
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
            if self.include is not None and not any(self.include.search(ln) for ln in lines):
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
