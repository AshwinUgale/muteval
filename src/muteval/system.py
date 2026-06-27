"""The mutation *target*: what muteval degrades.

v0 of muteval mutated a single prompt string. That made the roadmap items —
mutating retrieved context (RAG) and tool outputs (agents) — impossible to
express, because there was nowhere to put them.

``System`` generalizes the target into a small immutable bundle of everything
that defines the system under test: the prompt, the retrieved ``context``, the
``tools`` and their outputs, the ``model`` choice, and an open ``extra`` dict
for anything else. Mutation operators take a ``System`` and return degraded
``System``s; the runner reruns the user's eval suite against each one.

Backward compatibility is the whole point of ``as_system``: anywhere a bare
prompt string used to be accepted, a string is silently promoted to
``System(prompt=...)``, so every existing config and operator keeps working.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class System:
    """An immutable snapshot of the system under test.

    Attributes:
        prompt: The system prompt (the original v0 mutation target).
        context: Retrieved documents fed to the system (RAG). ``None`` means
            "this system has no retrieval"; an empty tuple means "retrieval ran
            and returned nothing" — a meaningful difference for mutation.
        tools: Tool schemas and/or tool outputs available to the system
            (agents). Opaque to the core; operators/adapters interpret them.
        model: The model identifier, for model-swap mutants.
        extra: Escape hatch for anything else a custom operator/run needs.
    """

    prompt: str = ""
    context: Optional[Tuple[str, ...]] = None
    tools: Optional[Tuple[Any, ...]] = None
    model: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def with_prompt(self, prompt: str) -> "System":
        """Return a copy with a new prompt (leaves every other field intact)."""
        return dataclasses.replace(self, prompt=prompt)

    def with_context(self, context: Optional[Sequence[str]]) -> "System":
        """Return a copy with new retrieved context."""
        ctx = tuple(context) if context is not None else None
        return dataclasses.replace(self, context=ctx)

    def replace(self, **changes: Any) -> "System":
        """Return a copy with arbitrary fields replaced."""
        if "context" in changes and changes["context"] is not None:
            changes["context"] = tuple(changes["context"])
        return dataclasses.replace(self, **changes)

    def key(self) -> tuple:
        """A hashable signature used to dedupe mutants and detect no-ops."""
        return (
            self.prompt,
            self.context,
            repr(self.tools),
            self.model,
            repr(sorted(self.extra.items())) if self.extra else "",
        )


# A mutation target accepts either a System or a bare prompt string.
Target = Union[System, str]


def as_system(target: Target) -> System:
    """Normalize a target to a ``System``.

    A bare string is promoted to ``System(prompt=string)`` so legacy code that
    passes a prompt keeps working unchanged.
    """
    if isinstance(target, System):
        return target
    if isinstance(target, str):
        return System(prompt=target)
    raise TypeError(
        f"mutation target must be a str or muteval.System, got {type(target).__name__}"
    )
