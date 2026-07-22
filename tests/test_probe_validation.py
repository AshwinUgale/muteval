"""Trust gate: each probe is validated on CONSTRUCTED ground truth.

A probe only earns its place in the eval-evaluator once it demonstrably fires on
a known-bad input and stays quiet on a known-good one. All deterministic, no keys.
See docs/PLAN-probe-validation.md.
"""

import importlib.util
from pathlib import Path

import pytest

from muteval.probes.discrimination import discrimination
from muteval.probes.judge_reliability import judge_reliability
from muteval.probes.redundancy import redundancy
from muteval.probes.statistical_adequacy import statistical_adequacy
from muteval.stats import interval

_DIR = Path(__file__).resolve().parent.parent / "validation" / "probe_validation"


def _load(name):
    path = _DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- redundancy ------------------------------------------------------------
_vr = _load("validate_redundancy.py")


def test_redundancy_flags_the_redundant_family():
    r = redundancy(_vr.redundant_config())            # spearman (default)
    assert r.ok is False
    fam = next(f for f in r.metrics["families"] if "lin" in f)
    assert {"lin", "dup", "mono"}.issubset(set(fam))  # all three grouped
    assert "indep" not in fam                          # independent left alone


def test_spearman_catches_monotonic_that_pearson_misses():
    cfg = _vr.redundant_config()
    sp_fam = next(f for f in redundancy(cfg).metrics["families"] if "lin" in f)
    assert "mono" in sp_fam                            # Spearman groups mono with lin

    pe = redundancy(cfg, method="pearson")
    pe_lin_fam = next((f for f in pe.metrics["families"] if "lin" in f), [])
    assert "mono" not in pe_lin_fam                    # Pearson misses the nonlinear dup


def test_redundancy_passes_on_distinct_metrics():
    r = redundancy(_vr.distinct_config())
    assert r.ok is True
    assert r.metrics["families"] == []


# --- discrimination --------------------------------------------------------
_vd = _load("validate_discrimination.py")


def test_discrimination_passes_on_a_separating_metric():
    r = discrimination(_vd.discriminating_config())
    assert r.ok is True
    assert r.metrics["stats"]["parsed"]["auc"] == 1.0


def test_discrimination_flags_a_nonseparating_metric():
    r = discrimination(_vd.nondiscriminating_config())
    assert r.ok is False
    assert abs(r.metrics["stats"]["parsed"]["auc"] - 0.5) < 1e-9  # coin flip


def test_auc_catches_overlap_that_raw_mean_gap_misses():
    r = discrimination(_vd.overlapping_large_gap_config())
    assert r.ok is False                                   # AUC flags it
    assert r.metrics["stats"]["parsed"]["auc"] < 0.7
    # ...even though the raw mean gap is large (the old probe would have passed)
    assert r.metrics["gaps"]["parsed"] > 1.0


# --- statistical_adequacy --------------------------------------------------
_vs = _load("validate_statistical_adequacy.py")


def test_adequacy_flags_a_too_small_suite():
    r = statistical_adequacy(_vs.small_config())
    assert r.ok is False
    assert r.metrics["cases_needed"] > r.metrics["n"]      # tells you how many more


def test_adequacy_passes_a_large_suite():
    assert statistical_adequacy(_vs.large_config()).ok is True


@pytest.mark.parametrize("method", ["wilson", "jeffreys"])
def test_interval_shrinks_monotonically_and_clamps_endpoints(method):
    # width strictly decreases as n grows at a fixed rate
    widths = [
        (lambda lo, hi: hi - lo)(*interval(round(0.9 * n), n, method=method))
        for n in (5, 10, 40, 160)
    ]
    assert all(b < a for a, b in zip(widths, widths[1:])), (method, widths)
    # endpoints clamp: 0 successes -> low 0; n successes -> high 1
    assert interval(0, 20, method=method)[0] == 0.0
    assert interval(20, 20, method=method)[1] == 1.0


# --- judge_reliability -----------------------------------------------------
_vj = _load("validate_judge_reliability.py")


def test_reliability_passes_a_deterministic_judge():
    r = judge_reliability(_vj.reliable_config())
    assert r.ok is True
    assert r.metrics["flip_rate"] == 0.0
    assert r.metrics["min_alpha"] == 1.0        # perfect chance-corrected agreement


def test_reliability_flags_a_coin_flip_judge():
    r = judge_reliability(_vj.noisy_config())
    assert r.ok is False
    assert r.metrics["flip_rate"] > 0.5
    assert r.metrics["min_alpha"] < 0.5         # alpha ~ 0 (chance-level)


def test_alpha_separates_reliable_from_noisy():
    rel = judge_reliability(_vj.reliable_config()).metrics["min_alpha"]
    noisy = judge_reliability(_vj.noisy_config()).metrics["min_alpha"]
    assert rel > noisy                          # chance-correction distinguishes them
