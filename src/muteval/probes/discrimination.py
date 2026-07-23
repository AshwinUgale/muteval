"""Discrimination probe: can each metric tell a GOOD output from a BAD one?

An eval can be reliable and well-sampled and still be useless — if it scores good
and bad answers about the same, it isn't measuring quality. This probe takes
per-case exemplars (`case["good"]` and `case["bad"]`, lists of output strings),
runs each eval on both, and measures **separability**.

The headline is **AUC** (the probability the metric ranks a random good answer
above a random bad one), computed dependency-free as the Mann–Whitney statistic
U / (n_good · n_bad). It is scale-free — a length metric outputting 0–4000 and a
judge outputting 0–10 are directly comparable — and interpretable: 1.0 = perfect
separation, 0.5 = a coin flip (no discrimination). Ties (good == bad) count as
0.5, i.e. as failure to discriminate. We also report **Cohen's d** (magnitude of
separation in SD units) and a normal-approximation significance p-value, since
exemplar sets are usually tiny and a big-looking gap can be noise.

This is the Classical Test Theory *item-discrimination index* applied to evals;
AUC/rank measures are what the NLG/MT meta-evaluation field converged on over raw
score gaps. See docs/PRIOR-ART.md.

Opt-in: if no case provides good/bad exemplars, the probe reports "not assessed"
rather than faking a signal.
"""

from __future__ import annotations

import math
from typing import Dict

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


def _auc(good, bad):
    """P(good ranks above bad) via Mann-Whitney U; ties count 0.5. Also returns U."""
    u = 0.0
    for g in good:
        for b in bad:
            u += 1.0 if g > b else (0.5 if g == b else 0.0)
    denom = len(good) * len(bad)
    return (u / denom if denom else 0.5), u


def _cohens_d(good, bad):
    ng, nb = len(good), len(bad)
    if ng < 2 or nb < 2:
        return None
    mg, mb = _mean(good), _mean(bad)
    vg = sum((x - mg) ** 2 for x in good) / (ng - 1)
    vb = sum((x - mb) ** 2 for x in bad) / (nb - 1)
    sp = math.sqrt(((ng - 1) * vg + (nb - 1) * vb) / (ng + nb - 2))
    if sp == 0:
        return None  # zero within-group spread — separation is total or none
    return (mg - mb) / sp


def _mw_pvalue(good, bad, u):
    """Two-sided Mann-Whitney p via the normal approximation (supporting signal)."""
    ng, nb = len(good), len(bad)
    mu = ng * nb / 2.0
    sd = math.sqrt(ng * nb * (ng + nb + 1) / 12.0)
    if sd == 0:
        return 1.0
    z = (u - mu) / sd
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))


def discrimination(config, min_auc: float = 0.7) -> ProbeResult:
    good: Dict[str, list] = {}
    bad: Dict[str, list] = {}
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
            name="discrimination", ok=True,
            summary="not assessed (no good/bad exemplars provided)",
            detail="add per-case 'good' and 'bad' example outputs to enable this probe.",
            metrics={"assessed": False},
        )

    stats = {}
    for n in good:
        g, b = good[n], bad.get(n, [])
        auc, u = _auc(g, b)
        stats[n] = {
            "auc": auc,
            "gap": _mean(g) - _mean(b),
            "cohen_d": _cohens_d(g, b),
            "p": _mw_pvalue(g, b, u),
            "n_good": len(g), "n_bad": len(b),
        }

    worst = min(stats, key=lambda n: stats[n]["auc"])
    w = stats[worst]
    ok = w["auc"] >= min_auc

    if ok:
        summary = f"all evals separate good from bad (min AUC {w['auc']:.2f})"
        detail = f">= the {min_auc:.2f} target — the metrics discriminate quality."
    else:
        d_str = f", d={w['cohen_d']:.2f}" if w["cohen_d"] is not None else ""
        summary = (
            f"'{worst}' barely separates good from bad "
            f"(AUC {w['auc']:.2f}{d_str}, p={w['p']:.2f})"
        )
        detail = (
            f"AUC < {min_auc:.2f} (0.5 = coin flip) — this metric can't tell good "
            f"from bad, so its pass/fail is close to meaningless. Fix or replace it."
        )

    return ProbeResult(
        name="discrimination", ok=ok, summary=summary, detail=detail,
        metrics={
            "assessed": True, "worst_eval": worst, "min_auc": w["auc"],
            "stats": stats,
            # back-compat: flat gap map kept for existing consumers
            "gaps": {n: stats[n]["gap"] for n in stats},
        },
    )
