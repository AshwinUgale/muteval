"""Discrimination probe: can each metric tell a GOOD output from a BAD one?

An eval can be reliable and well-sampled and still be useless — if it scores good
and bad answers about the same, it isn't measuring quality. This probe takes
per-case exemplars (`case["good"]` and `case["bad"]`, lists of output strings),
runs each eval on both, and measures the separation (mean good score - mean bad
score). A small gap means the metric doesn't discriminate.

Opt-in: if no case provides good/bad exemplars, the probe reports "not assessed"
rather than faking a signal (auto-generating "bad" outputs is unreliable).
"""

from __future__ import annotations

from muteval.evals import coerce_outcome
from muteval.probes.base import ProbeResult


def _eval_name(config, idx, ev) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def _score(ev, output, case) -> float:
    oc = coerce_outcome(ev(output, case))
    if oc.score is not None:
        return float(oc.score)
    return 1.0 if oc.passed else 0.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def discrimination(config, min_gap: float = 0.3) -> ProbeResult:
    good = {}   # eval name -> [scores on good outputs]
    bad = {}
    have = False
    for case in config.cases:
        goods = case.get("good") if isinstance(case, dict) else None
        bads = case.get("bad") if isinstance(case, dict) else None
        if not goods or not bads:
            continue
        have = True
        for idx, ev in enumerate(config.evals):
            name = _eval_name(config, idx, ev)
            try:
                for o in goods:
                    good.setdefault(name, []).append(_score(ev, o, case))
                for o in bads:
                    bad.setdefault(name, []).append(_score(ev, o, case))
            except Exception:  # noqa: BLE001 - skip a broken eval
                continue

    if not have:
        return ProbeResult(
            name="discrimination",
            ok=True,  # optional — not-assessed shouldn't fail the card
            summary="not assessed (no good/bad exemplars provided)",
            detail="add per-case 'good' and 'bad' example outputs to enable this probe.",
            metrics={"assessed": False},
        )

    gaps = {n: _mean(good[n]) - _mean(bad.get(n, [])) for n in good}
    worst_name = min(gaps, key=gaps.get)
    worst_gap = gaps[worst_name]
    ok = worst_gap >= min_gap

    if ok:
        summary = f"all evals separate good from bad (min gap {worst_gap:.2f})"
        detail = f">= the {min_gap:.2f} target — the metrics discriminate."
    else:
        g = _mean(good[worst_name])
        b = _mean(bad.get(worst_name, []))
        summary = (
            f"'{worst_name}' barely separates: good {g:.2f} vs bad {b:.2f} "
            f"(gap {worst_gap:.2f})"
        )
        detail = (
            f"gap < {min_gap:.2f} — this metric doesn't tell good from bad, so its "
            f"pass/fail is close to meaningless. Fix or replace it."
        )

    return ProbeResult(
        name="discrimination",
        ok=ok,
        summary=summary,
        detail=detail,
        metrics={"assessed": True, "gaps": gaps, "min_gap": worst_gap},
    )
