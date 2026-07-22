"""Statistical-adequacy probe: does your suite have enough cases to trust its
number?

A pass rate over few cases has a wide confidence interval — 19/20 is not "95%",
it's 95% with a Wilson interval of ~[76%, 99%]. This probe runs the suite on the
original system, computes the Wilson CI on the pass rate, and flags suites whose
interval is too wide to trust — with the number of cases you'd need for a tight
estimate.
"""

from __future__ import annotations

from muteval.evals import coerce_outcome
from muteval.probes.base import ProbeResult
from muteval.stats import interval, min_samples_for_precision


def _case_passes(config, case) -> bool:
    output = config.invoke(config.system, case)
    for ev in config.evals:
        if not coerce_outcome(ev(output, case)):
            return False
    return True


def statistical_adequacy(
    config, target_margin: float = 0.15, method: str = "wilson"
) -> ProbeResult:
    """`method`: 'wilson' (default) or 'jeffreys' (Beta-Binomial; degrades better
    at very small n — Bowyer et al. 2025, Brown/Cai/DasGupta 2001)."""
    passed = evaluated = 0
    for case in config.cases:
        try:
            ok_case = _case_passes(config, case)
        except Exception:  # noqa: BLE001 - a flaky case shouldn't break the probe
            continue
        evaluated += 1
        passed += 1 if ok_case else 0

    n = evaluated
    lo, hi = interval(passed, n, method=method)
    half = (hi - lo) / 2 if n else 1.0
    rate = passed / n if n else 0.0
    ok = n > 0 and half <= target_margin
    need = min_samples_for_precision(rate, target_margin) if n else None

    if not n:
        summary = "no cases could be evaluated"
        detail = "every case errored — can't assess sample adequacy."
    else:
        summary = (
            f"{passed}/{n} cases pass = {rate * 100:.0f}% "
            f"[95% CI {lo * 100:.0f}-{hi * 100:.0f}%]"
        )
        if ok:
            detail = (
                f"sample adequate: +/-{half * 100:.0f}% is within the "
                f"+/-{target_margin * 100:.0f}% target."
            )
        else:
            detail = (
                f"CI too wide (+/-{half * 100:.0f}%) to trust the number — add "
                f"cases (~{need} for +/-{target_margin * 100:.0f}%)."
            )

    return ProbeResult(
        name="statistical_adequacy",
        ok=ok,
        summary=summary,
        detail=detail,
        metrics={
            "n": n, "passed": passed, "pass_rate": rate,
            "ci_low": lo, "ci_high": hi, "cases_needed": need,
            "target_margin": target_margin,
        },
    )
