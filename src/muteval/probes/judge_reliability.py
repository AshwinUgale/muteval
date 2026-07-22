"""Judge-reliability probe: does the eval give the same verdict when re-run?

LLM-as-judge metrics are non-deterministic: run the same metric on the same
output twice and it can flip pass<->fail. This probe holds the output FIXED
(generated once) and re-runs each eval `runs` times. It reports two things:

* the **verdict-flip rate** (plain-language: how often the verdict changed), and
* **Krippendorff's alpha** across the runs — a chance-corrected reliability
  coefficient (each re-run treated as a "rater"). alpha=1 is perfect
  self-consistency, ~0 is chance, <0 is systematic disagreement. This is the
  statistic the large-scale LLM-judge reliability studies use; see docs/PRIOR-ART.md.

Rule-based checks are perfectly stable (0% flips, alpha=1). Noisy judges are not.

NOTE (reliability != validity): a judge can be perfectly self-consistent yet
systematically wrong. This probe measures stability only. Directional-bias tests
(position / verbosity / self-preference) need a structured judge abstraction and
are future work.

Cost: re-runs each eval `runs` times, so for LLM judges it costs `runs`x the calls.
"""

from __future__ import annotations

from collections import defaultdict

from muteval.evals import coerce_outcome
from muteval.probes.base import ProbeResult


def _eval_name(config, idx, ev) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def _krippendorff_alpha_nominal(items):
    """Nominal Krippendorff's alpha over a list of items, each a list of ratings
    (one per run/rater). Dependency-free. Returns 1.0 when there is no variation
    (perfect agreement) and can go negative for systematic disagreement."""
    o = defaultdict(float)
    for ratings in items:
        m = len(ratings)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(ratings[i], ratings[j])] += w
    values = {v for pair in o for v in pair}
    if len(values) < 2:
        return 1.0  # everyone agreed on one value across the board
    n_c = {v: sum(o[(v, k)] for k in values) for v in values}
    n = sum(n_c.values())
    if n <= 1:
        return 1.0
    d_o = sum(o[(c, k)] for c in values for k in values if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in values for k in values if c != k) / (n - 1)
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def judge_reliability(config, runs: int = 3, target_flip_rate: float = 0.05) -> ProbeResult:
    total = flipped = 0
    worst = {}                 # eval name -> flips
    matrices = {}              # eval name -> [per-case list of `runs` verdicts]
    for case in config.cases:
        try:
            output = config.invoke(config.system, case)  # fix the output once
        except Exception:  # noqa: BLE001
            continue
        for idx, ev in enumerate(config.evals):
            name = _eval_name(config, idx, ev)
            try:
                verdicts = [int(bool(coerce_outcome(ev(output, case)))) for _ in range(runs)]
            except Exception:  # noqa: BLE001 - a broken eval isn't a flip; skip
                continue
            total += 1
            matrices.setdefault(name, []).append(verdicts)
            if len(set(verdicts)) > 1:
                flipped += 1
                worst[name] = worst.get(name, 0) + 1

    rate = flipped / total if total else 0.0
    alpha_by_eval = {n: _krippendorff_alpha_nominal(m) for n, m in matrices.items()}
    min_alpha = min(alpha_by_eval.values()) if alpha_by_eval else 1.0
    ok = total > 0 and rate <= target_flip_rate

    if not total:
        summary = "no (eval, case) pairs could be evaluated"
        detail = "every eval errored — can't assess reliability."
    else:
        summary = (
            f"{flipped}/{total} verdicts flipped across {runs} runs "
            f"= {rate * 100:.0f}% flaky (Krippendorff alpha {min_alpha:.2f})"
        )
        if ok:
            detail = f"stable (<= {target_flip_rate * 100:.0f}% flip target; alpha near 1)."
        else:
            top = max(worst, key=worst.get)
            detail = (
                f"non-deterministic — worst: '{top}' (alpha {alpha_by_eval[top]:.2f}). "
                f"A single run can't be trusted; average over more runs or use a "
                f"stabler judge (e.g. temperature 0, or a jury)."
            )

    return ProbeResult(
        name="judge_reliability",
        ok=ok,
        summary=summary,
        detail=detail,
        metrics={
            "pairs": total, "flipped": flipped, "flip_rate": rate, "runs": runs,
            "by_eval": worst, "alpha_by_eval": alpha_by_eval, "min_alpha": min_alpha,
        },
    )
