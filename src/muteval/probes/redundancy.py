"""Redundancy probe: do your metrics measure different things, or the same thing?

Running five metrics that all correlate ~1.0 is wasted cost and false coverage —
you think you have five checks, but you really have one. This probe collects each
eval's score across the cases and correlates them pairwise; highly-correlated
pairs are flagged as redundant.

Needs >= 2 evals and >= 3 cases whose scores actually VARY (constant evals carry
no information to correlate). Otherwise it reports "not assessed".
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
    return float(oc.score) if oc.score is not None else (1.0 if oc.passed else 0.0)


def _var(v):
    n = len(v)
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / n


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def _na(reason):
    return ProbeResult(
        name="redundancy", ok=True,
        summary=f"not assessed ({reason})",
        detail="needs >= 2 evals and >= 3 cases whose scores vary.",
        metrics={"assessed": False},
    )


def redundancy(config, max_corr: float = 0.9) -> ProbeResult:
    if len(config.evals) < 2:
        return _na("need >= 2 evals")

    vectors = {}
    for case in config.cases:
        try:
            output = config.invoke(config.system, case)
        except Exception:  # noqa: BLE001
            continue
        for idx, ev in enumerate(config.evals):
            name = _eval_name(config, idx, ev)
            try:
                vectors.setdefault(name, []).append(_score(ev, output, case))
            except Exception:  # noqa: BLE001
                pass

    lengths = {len(v) for v in vectors.values()}
    if not vectors or min(lengths) < 3:
        return _na("< 3 evaluable cases")

    # drop constant (zero-variance) evals — nothing to correlate.
    active = {n: v for n, v in vectors.items() if _var(v) > 1e-12}
    if len(active) < 2:
        return _na("evals don't vary across cases")

    names = list(active)
    worst_pair, worst_abs = (names[0], names[1]), 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = _pearson(active[names[i]], active[names[j]])
            if abs(r) > worst_abs:
                worst_abs, worst_pair = abs(r), (names[i], names[j])

    ok = worst_abs <= max_corr
    a, b = worst_pair
    if ok:
        summary = f"metrics are distinct (max correlation {worst_abs:.2f})"
        detail = f"<= {max_corr:.2f} — each eval adds independent signal."
    else:
        summary = f"'{a}' and '{b}' correlate {worst_abs:.2f} — redundant"
        detail = (
            f"> {max_corr:.2f}: these measure nearly the same thing, so one is "
            f"wasted cost and inflates your sense of coverage. Drop or differentiate one."
        )

    return ProbeResult(
        name="redundancy", ok=ok, summary=summary, detail=detail,
        metrics={"assessed": True, "max_corr": worst_abs, "pair": list(worst_pair)},
    )
