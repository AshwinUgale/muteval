"""v0.6: threshold-calibration probe — is the pass/fail line in the right place?

A scored eval (LLM-judge, similarity, ...) turns a number into a verdict via a
threshold. Pick that line badly and the eval is decorative: too low and bad
answers pass; too high and good answers fail. This probe takes per-case GOOD and
BAD exemplars (``case["good"]`` / ``case["bad"]``, as the discrimination probe
does), scores them, and checks where the configured threshold sits relative to
the separation point.

A well-calibrated threshold sits ABOVE every bad-example score and AT/BELOW every
good-example score. We report, per scored eval: the good/bad score ranges, whether
they separate, the recommended midpoint, and a verdict — ``ok`` /
``too_lenient`` (a bad answer would pass) / ``too_strict`` (a good answer would
fail). Opt-in: without exemplars, or without any scored eval, it reports
"not assessed" rather than inventing a signal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from muteval.evals import coerce_outcome
from muteval.probes.base import ProbeResult
from muteval.probes.discrimination import _eval_name


def _score_and_threshold(ev, output, case):
    oc = coerce_outcome(ev(output, case))
    score = float(oc.score) if oc.score is not None else (1.0 if oc.passed else 0.0)
    return score, oc.threshold


def _verdict(good_scores: List[float], bad_scores: List[float], threshold: float) -> Dict[str, Any]:
    max_bad = max(bad_scores)
    min_good = min(good_scores)
    separates = min_good > max_bad
    v = "ok"
    if threshold <= max_bad:
        v = "too_lenient"   # a bad-example score is >= threshold -> would pass
    elif threshold > min_good:
        v = "too_strict"    # a good-example score is < threshold -> would fail
    return {
        "threshold": round(threshold, 3),
        "max_bad": round(max_bad, 3),
        "min_good": round(min_good, 3),
        "separates": separates,
        "recommended": round((max_bad + min_good) / 2, 3) if separates else None,
        "verdict": v,
    }


def threshold_calibration(config) -> ProbeResult:
    good: Dict[str, List[float]] = {}
    bad: Dict[str, List[float]] = {}
    thr: Dict[str, Optional[float]] = {}
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
                    s, t = _score_and_threshold(ev, o, case)
                    good.setdefault(name, []).append(s)
                    thr[name] = t
                for o in bads:
                    s, _ = _score_and_threshold(ev, o, case)
                    bad.setdefault(name, []).append(s)
            except Exception:  # noqa: BLE001 - skip a broken eval
                continue

    if not have:
        return ProbeResult(
            name="threshold_calibration", ok=True,
            summary="not assessed (no good/bad exemplars provided)",
            detail="add per-case 'good' and 'bad' example outputs to enable this probe.",
            metrics={"assessed": False},
        )

    results = {}
    for name in good:
        if name in bad and thr.get(name) is not None:
            results[name] = _verdict(good[name], bad[name], thr[name])  # scored evals only

    if not results:
        return ProbeResult(
            name="threshold_calibration", ok=True,
            summary="not assessed (no scored eval with a threshold)",
            detail="only evals returning a score + threshold can be calibrated.",
            metrics={"assessed": False},
        )

    bad_ones = [n for n, r in results.items() if r["verdict"] != "ok"]
    ok = not bad_ones
    if ok:
        summary = f"{len(results)} threshold(s) well-calibrated"
    else:
        summary = f"{len(bad_ones)}/{len(results)} threshold(s) miscalibrated: " + ", ".join(
            f"{n} ({results[n]['verdict']}; recommend ~{results[n]['recommended']})"
            for n in bad_ones
        )
    return ProbeResult(
        name="threshold_calibration", ok=ok, summary=summary,
        detail="A good threshold sits above every bad-example score and at/below "
        "every good-example score.",
        metrics={"assessed": True, "evals": results},
    )
