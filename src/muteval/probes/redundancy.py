"""Redundancy probe: do your metrics measure different things, or the same thing?

Running five metrics that all move together is wasted cost and false coverage —
you think you have five checks, but you really have one. This probe collects each
eval's score across the cases, correlates them, and flags **redundant families**
(groups of metrics that all measure the same construct).

It defaults to **Spearman** (rank) correlation, which catches monotonic
redundancy that a plain Pearson misses — two judges whose scores agree in rank but
on different (or saturated) scales are redundant for catching regressions, and
Pearson can rate them well below 1.0. Families are found by connected components
over the "|correlation| > threshold" graph, so transitive overlap (A~B, B~C) is
reported as one family "keep 1 of {A, B, C}" instead of loose pairs.

Needs >= 2 evals and >= 3 cases whose scores actually VARY (constant evals carry
no information to correlate). Otherwise it reports "not assessed".

Method + threshold references: sklearn's multicollinearity recipe (Spearman +
clustering); "Agreement Metrics for LLM-as-Judge" (arXiv 2606.00093) on how
reporting several correlated metrics is an "illusion of corroborating evidence".
See docs/PRIOR-ART.md.
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


def _ranks(v):
    """Average (fractional) ranks, so ties are handled correctly."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _redundant_families(names, active, corr, threshold):
    """Connected components over the |corr| > threshold graph. Returns
    (families with >= 2 members, max |corr| seen, the worst pair)."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_abs = 0.0
    worst = (names[0], names[1]) if len(names) >= 2 else (names[0], names[0])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr(active[names[i]], active[names[j]])
            if abs(r) > max_abs:
                max_abs, worst = abs(r), (names[i], names[j])
            if abs(r) > threshold:
                parent[find(names[i])] = find(names[j])

    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    families = [sorted(g) for g in groups.values() if len(g) >= 2]
    return families, max_abs, worst


def _na(reason):
    return ProbeResult(
        name="redundancy", ok=True,
        summary=f"not assessed ({reason})",
        detail="needs >= 2 evals and >= 3 cases whose scores vary.",
        metrics={"assessed": False},
    )


def redundancy(config, max_corr: float = 0.9, method: str = "spearman") -> ProbeResult:
    if len(config.evals) < 2:
        return _na("need >= 2 evals")

    vectors: dict = {}
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

    corr = _spearman if method == "spearman" else _pearson
    names = list(active)
    families, max_abs, worst = _redundant_families(names, active, corr, max_corr)

    ok = not families
    if ok:
        summary = f"metrics are distinct (max {method} correlation {max_abs:.2f})"
        detail = f"<= {max_corr:.2f} — each eval adds independent signal."
    else:
        fam_str = "; ".join("{" + ", ".join(f) + "}" for f in families)
        summary = f"{len(families)} redundant metric family(ies): {fam_str}"
        detail = (
            f"members correlate > {max_corr:.2f} ({method}) — each family measures "
            "nearly one construct, so it's wasted cost and inflates your sense of "
            "coverage. Keep one metric per family, drop or differentiate the rest."
        )

    return ProbeResult(
        name="redundancy", ok=ok, summary=summary, detail=detail,
        metrics={
            "assessed": True, "method": method, "max_corr": max_abs,
            "worst_pair": list(worst), "families": families,
        },
    )
