"""Judge-reliability probe: does the eval give the same verdict when re-run?

LLM-as-judge metrics are non-deterministic: run the same metric on the same
output twice and it can flip pass<->fail. This probe holds the output FIXED
(generated once) and re-runs each eval `runs` times, measuring the verdict-flip
rate. Rule-based checks are perfectly stable (0% flips); noisy judges are not.

Cost note: this re-runs each eval `runs` times, so for LLM-judge metrics it costs
`runs`x the judge calls. Rule-based checks are free.
"""

from __future__ import annotations

from muteval.evals import coerce_outcome
from muteval.probes.base import ProbeResult


def _eval_name(config, idx, ev) -> str:
    if idx < len(config.eval_names):
        return config.eval_names[idx]
    return getattr(ev, "__name__", f"eval[{idx}]")


def judge_reliability(config, runs: int = 3, target_flip_rate: float = 0.05) -> ProbeResult:
    total = flipped = 0
    worst = {}  # eval name -> flips
    for case in config.cases:
        try:
            output = config.invoke(config.system, case)  # fix the output once
        except Exception:  # noqa: BLE001
            continue
        for idx, ev in enumerate(config.evals):
            name = _eval_name(config, idx, ev)
            try:
                verdicts = [bool(coerce_outcome(ev(output, case))) for _ in range(runs)]
            except Exception:  # noqa: BLE001 - a broken eval isn't a flip; skip
                continue
            total += 1
            if len(set(verdicts)) > 1:
                flipped += 1
                worst[name] = worst.get(name, 0) + 1

    rate = flipped / total if total else 0.0
    ok = total > 0 and rate <= target_flip_rate

    if not total:
        summary = "no (eval, case) pairs could be evaluated"
        detail = "every eval errored — can't assess reliability."
    else:
        summary = (
            f"{flipped}/{total} verdicts flipped across {runs} runs "
            f"= {rate * 100:.0f}% flaky"
        )
        if ok:
            detail = f"stable (<= {target_flip_rate * 100:.0f}% flip target)."
        else:
            top = max(worst, key=worst.get)
            detail = (
                f"non-deterministic — worst: '{top}'. A single run can't be "
                f"trusted; average over more runs or use a stabler judge."
            )

    return ProbeResult(
        name="judge_reliability",
        ok=ok,
        summary=summary,
        detail=detail,
        metrics={"pairs": total, "flipped": flipped, "flip_rate": rate,
                 "runs": runs, "by_eval": worst},
    )
