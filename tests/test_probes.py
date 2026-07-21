"""Phase 2: the eval evaluator — probe interface + statistical-adequacy probe."""

from muteval import MutEvalConfig
from muteval.probes import PROBES, run_probes
from muteval.probes.statistical_adequacy import statistical_adequacy
from muteval.stats import min_samples_for_precision


def _cfg(n_cases):
    return MutEvalConfig(
        prompt="You are a bot.",
        cases=[{"i": i} for i in range(n_cases)],
        run=lambda p, c: "ok",
        evals=[lambda o, c: True],
    )


def test_few_cases_flagged_statistically_inadequate():
    r = statistical_adequacy(_cfg(2))
    assert r.ok is False
    assert r.metrics["n"] == 2
    assert "add cases" in (r.detail or "")


def test_many_cases_are_adequate():
    r = statistical_adequacy(_cfg(60))
    assert r.ok is True
    assert r.metrics["pass_rate"] == 1.0


def test_run_probes_runs_registered_probes():
    results = run_probes(_cfg(2))
    assert "statistical_adequacy" in PROBES
    assert any(x.name == "statistical_adequacy" for x in results)


def test_unknown_probe_raises():
    try:
        run_probes(_cfg(2), probes=["does_not_exist"])
    except ValueError as exc:
        assert "Unknown probe" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown probe")


def test_tighter_margin_needs_more_samples():
    assert min_samples_for_precision(1.0, 0.05) > min_samples_for_precision(1.0, 0.20)


def test_deterministic_evals_are_reliable():
    from muteval.probes.judge_reliability import judge_reliability

    r = judge_reliability(_cfg(3))
    assert r.ok is True
    assert r.metrics["flip_rate"] == 0.0


def test_flaky_judge_is_flagged():
    from muteval.probes.judge_reliability import judge_reliability

    class _Cycler:  # verdict flips every call -> non-deterministic judge
        def __init__(self):
            self.i = 0

        def __call__(self, output, case):
            v = self.i % 2 == 0
            self.i += 1
            return v

    cfg = MutEvalConfig(
        prompt="You are a bot.",
        cases=[{"i": 0}],
        run=lambda p, c: "ok",
        evals=[_Cycler()],
        eval_names=["cycler"],
    )
    r = judge_reliability(cfg)  # 3 runs -> [T, F, T] -> flipped
    assert r.ok is False
    assert r.metrics["flip_rate"] > 0


def _disc_cfg(evals):
    return MutEvalConfig(
        prompt="You are a bot.",
        cases=[{
            "i": 0,
            "good": ["a good answer", "another good one"],
            "bad": ["", "wrong nonsense"],
        }],
        run=lambda p, c: "ok",
        evals=evals,
        eval_names=["ev"],
    )


def test_discriminating_eval_passes():
    from muteval.probes.discrimination import discrimination

    r = discrimination(_disc_cfg([lambda o, c: "good" in o]))
    assert r.ok is True
    assert r.metrics["assessed"] is True


def test_nondiscriminating_eval_is_flagged():
    from muteval.probes.discrimination import discrimination

    r = discrimination(_disc_cfg([lambda o, c: True]))  # passes good AND bad
    assert r.ok is False


def test_discrimination_not_assessed_without_exemplars():
    from muteval.probes.discrimination import discrimination

    r = discrimination(_cfg(2))  # no good/bad on cases
    assert r.ok is True
    assert r.metrics["assessed"] is False


def _score_cfg(score_lists):
    from muteval.evals import EvalOutcome

    n = len(score_lists[0])
    evals = [
        (lambda sl: (lambda o, c: EvalOutcome(passed=True, score=sl[c["i"]])))(sl)
        for sl in score_lists
    ]
    return MutEvalConfig(
        prompt="You are a bot.",
        cases=[{"i": i} for i in range(n)],
        run=lambda p, c: "ok",
        evals=evals,
        eval_names=[f"e{i}" for i in range(len(score_lists))],
    )


def test_redundant_metrics_are_flagged():
    from muteval.probes.redundancy import redundancy

    r = redundancy(_score_cfg([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]))  # identical
    assert r.ok is False
    assert r.metrics["max_corr"] > 0.99


def test_independent_metrics_ok():
    from muteval.probes.redundancy import redundancy

    r = redundancy(_score_cfg([[0, 0, 1, 1], [0, 1, 0, 1]]))  # pearson r == 0
    assert r.ok is True


def test_redundancy_not_assessed_with_one_eval():
    from muteval.probes.redundancy import redundancy

    r = redundancy(_score_cfg([[0, 1, 2, 3]]))
    assert r.metrics["assessed"] is False
