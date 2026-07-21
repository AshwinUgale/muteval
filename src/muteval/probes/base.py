"""Probe interface for the eval evaluator (Phase 2).

A probe rates an eval SUITE along one dimension (mutation coverage is the
flagship; probes add other lenses — statistical adequacy, judge reliability,
discrimination, ...). Each probe takes the MutEvalConfig and returns a
ProbeResult. Probes are pluggable via a registry, mirroring mutation OPERATORS.

There is deliberately NO single composite score — the output is a report card of
separately-interpretable signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ProbeResult:
    name: str
    ok: bool                       # did the suite pass this quality check?
    summary: str                   # one-line headline (the number)
    detail: Optional[str] = None   # what it means / what to do about it
    metrics: Dict[str, Any] = field(default_factory=dict)


Probe = Callable[..., ProbeResult]
PROBES: Dict[str, Probe] = {}


def register_probe(name: str, fn: Probe) -> Probe:
    PROBES[name] = fn
    return fn


def run_probes(config, probes: Optional[List[str]] = None) -> List[ProbeResult]:
    """Run the selected probes (default: all) and return their results."""
    selected = probes or list(PROBES)
    results: List[ProbeResult] = []
    for name in selected:
        fn = PROBES.get(name)
        if fn is None:
            raise ValueError(f"Unknown probe '{name}'. Available: {list(PROBES)}")
        results.append(fn(config))
    return results
