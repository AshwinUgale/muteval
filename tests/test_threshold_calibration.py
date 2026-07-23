"""v0.6: threshold-calibration probe validation.

Constructed so the good/bad separation point is known, then check the probe
flags a threshold on the wrong side of it and clears a well-placed one.
"""

from __future__ import annotations

from muteval import EvalOutcome, MutEvalConfig
from muteval.probes.threshold_calibration import threshold_calibration


def _scored_eval(threshold):
    # Score = 0.9 if the output contains 'GOOD', else 0.2. Good exemplars score
    # 0.9, bad exemplars 0.2, so the separation point is ~0.55.
    def ev(output, case):
        score = 0.9 if "GOOD" in output else 0.2
        return EvalOutcome(passed=score >= threshold, score=score, threshold=threshold, name="judge")

    return ev


def _cfg(threshold):
    return MutEvalConfig(
        prompt="answer the question.",
        cases=[{"question": "q", "good": ["a GOOD answer"], "bad": ["a weak answer"]}],
        run=lambda p, c: "x",
        evals=[_scored_eval(threshold)],
        eval_names=["judge"],
    )


def test_well_placed_threshold_is_ok():
    r = threshold_calibration(_cfg(0.55))  # between 0.2 (bad) and 0.9 (good)
    assert r.ok is True
    assert r.metrics["evals"]["judge"]["verdict"] == "ok"
    assert r.metrics["evals"]["judge"]["recommended"] == 0.55


def test_too_lenient_threshold_is_flagged():
    r = threshold_calibration(_cfg(0.15))  # below the bad score 0.2 -> bad passes
    assert r.ok is False
    assert r.metrics["evals"]["judge"]["verdict"] == "too_lenient"


def test_too_strict_threshold_is_flagged():
    r = threshold_calibration(_cfg(0.95))  # above the good score 0.9 -> good fails
    assert r.ok is False
    assert r.metrics["evals"]["judge"]["verdict"] == "too_strict"


def test_not_assessed_without_exemplars():
    cfg = MutEvalConfig(
        prompt="answer.", cases=[{"question": "q"}], run=lambda p, c: "x",
        evals=[_scored_eval(0.5)], eval_names=["judge"],
    )
    r = threshold_calibration(cfg)
    assert r.metrics.get("assessed") is False


def test_registered_and_runs_in_panel():
    from muteval.probes import PROBES, run_probes

    assert "threshold_calibration" in PROBES
    results = run_probes(_cfg(0.55), probes=["threshold_calibration"])
    assert results and results[0].name == "threshold_calibration"
